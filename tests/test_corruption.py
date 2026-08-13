"""Corruption operator tests.

The identity-at-severity-0 test is the most important test in the repository. If an
operator perturbs its input at severity 0, then every retention curve is measured against
a baseline that is itself corrupted, every AUDC is wrong by an unknown amount, and
nothing else in the results is trustworthy. It is also exactly the kind of bug that
produces plausible-looking plots, which is why it gets an explicit, exhaustive test
rather than a spot check.
"""

from __future__ import annotations

import pytest
import torch

from wfb.corruption import registry
from wfb.corruption.base import Corruption, CorruptionContext
from wfb.corruption.sweeps import (
    DEFAULT_SEVERITIES,
    graded_axis,
    removal_grid,
    smoke_grid,
    standard_grid,
    unique_plans,
)
from wfb.types import CorruptionPlan, CorruptionSpec, DatasetBundle, Modality

ALL_OPERATORS = registry.available()


def make_context(
    modality: Modality, bundle: DatasetBundle, generator: torch.Generator
) -> CorruptionContext:
    """Context with real train statistics for ``modality``."""
    return CorruptionContext(modality=modality, stats=bundle.stats[modality], generator=generator)


def operator_for(name: str) -> tuple[Corruption, Modality]:
    """Instantiate ``name`` with defaults, paired with a modality it accepts."""
    cls = registry.get(name)
    return cls(), cls.applies_to[0]


# --------------------------------------------------------------- the core invariant


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_severity_zero_is_exact_identity(
    name: str, tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator, modality = operator_for(name)
    x = tiny_bundle["test"].features[modality][:6].clone()
    out = operator(x, 0.0, make_context(modality, tiny_bundle, generator))
    assert torch.equal(out, x), f"{name} is not an identity at severity 0"


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_operator_never_mutates_its_input(
    name: str, tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator, modality = operator_for(name)
    x = tiny_bundle["test"].features[modality][:6].clone()
    reference = x.clone()
    operator(x, 0.7, make_context(modality, tiny_bundle, generator))
    assert torch.equal(x, reference), f"{name} modified its input in place"


@pytest.mark.parametrize("name", ALL_OPERATORS)
@pytest.mark.parametrize("severity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_shape_is_preserved(
    name: str,
    severity: float,
    tiny_bundle: DatasetBundle,
    generator: torch.Generator,
) -> None:
    operator, modality = operator_for(name)
    x = tiny_bundle["test"].features[modality][:6]
    out = operator(x, severity, make_context(modality, tiny_bundle, generator))
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), f"{name} produced non-finite values"


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_operator_does_not_touch_global_rng(
    name: str, tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    """Operators must draw only from the context generator.

    If they used the global RNG, evaluating model B after model A would corrupt the
    inputs differently, silently invalidating the paired significance tests.
    """
    operator, modality = operator_for(name)
    x = tiny_bundle["test"].features[modality][:6]
    before = torch.random.get_rng_state()
    operator(x, 0.6, make_context(modality, tiny_bundle, generator))
    assert torch.equal(before, torch.random.get_rng_state()), f"{name} used the global RNG"


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_same_generator_seed_gives_identical_output(name: str, tiny_bundle: DatasetBundle) -> None:
    operator, modality = operator_for(name)
    x = tiny_bundle["test"].features[modality][:6]
    first = operator(x, 0.6, make_context(modality, tiny_bundle, torch.Generator().manual_seed(7)))
    second = operator(x, 0.6, make_context(modality, tiny_bundle, torch.Generator().manual_seed(7)))
    assert torch.equal(first, second)


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_damage_is_monotone_in_severity(name: str, mosi_synthetic: DatasetBundle) -> None:
    """Higher severity must not do *less* damage, averaged over the batch.

    Checked in expectation over 8 seeds — individual draws of a stochastic operator can
    of course go either way, but a systematic inversion means the severity mapping is
    upside down.
    """
    operator, modality = operator_for(name)
    if name in {"none", "shift"}:  # identity, and a quantised (step-wise) operator
        pytest.skip(f"{name} has no continuous severity response")
    x = mosi_synthetic["test"].features[modality][:32]

    def damage(severity: float) -> float:
        total = 0.0
        for seed in range(8):
            ctx = CorruptionContext(
                modality=modality,
                stats=mosi_synthetic.stats[modality],
                generator=torch.Generator().manual_seed(seed),
            )
            total += float((operator(x, severity, ctx) - x).abs().mean())
        return total / 8

    low, mid, high = damage(0.25), damage(0.55), damage(1.0)
    assert low <= mid + 1e-6, f"{name}: damage fell from severity 0.25 to 0.55"
    assert mid <= high + 1e-6, f"{name}: damage fell from severity 0.55 to 1.0"


# --------------------------------------------------------------- specific behaviours


def test_zero_removal_at_full_severity_zeroes_the_modality(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator = registry.get("zero")()
    x = tiny_bundle["test"].features[Modality.AUDIO][:4]
    out = operator(x, 1.0, make_context(Modality.AUDIO, tiny_bundle, generator))
    assert torch.allclose(out, torch.zeros_like(out))


def test_mean_removal_at_full_severity_gives_the_train_mean(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator = registry.get("mean")()
    x = tiny_bundle["test"].features[Modality.VISUAL][:4]
    out = operator(x, 1.0, make_context(Modality.VISUAL, tiny_bundle, generator))
    expected = tiny_bundle.stats[Modality.VISUAL].mean.expand_as(out)
    assert torch.allclose(out, expected, atol=1e-5)


def test_mask_falls_back_to_mean_without_a_learned_token(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    x = tiny_bundle["test"].features[Modality.TEXT][:4]
    ctx = make_context(Modality.TEXT, tiny_bundle, generator)
    masked = registry.get("mask")()(x, 1.0, ctx)
    meaned = registry.get("mean")()(x, 1.0, ctx)
    assert torch.allclose(masked, meaned)


def test_mask_uses_the_learned_token_when_provided(tiny_bundle: DatasetBundle) -> None:
    x = tiny_bundle["test"].features[Modality.TEXT][:4]
    token = torch.full((x.shape[-1],), 3.5)
    ctx = CorruptionContext(
        modality=Modality.TEXT,
        stats=tiny_bundle.stats[Modality.TEXT],
        mask_vectors={Modality.TEXT: token},
    )
    out = registry.get("mask")()(x, 1.0, ctx)
    assert torch.allclose(out, token.expand_as(out))


def test_frame_dropout_rate_matches_severity(mosi_synthetic: DatasetBundle) -> None:
    """The fraction of zeroed frames should track severity to within sampling noise."""
    operator = registry.get("frame_dropout")()
    x = mosi_synthetic["test"].features[Modality.AUDIO][:64]
    ctx = CorruptionContext(
        modality=Modality.AUDIO,
        stats=mosi_synthetic.stats[Modality.AUDIO],
        generator=torch.Generator().manual_seed(3),
    )
    out = operator(x, 0.5, ctx)
    dropped = (out.abs().sum(dim=-1) == 0).float().mean()
    assert 0.44 < float(dropped) < 0.56


def test_burst_dropout_loses_a_contiguous_run(mosi_synthetic: DatasetBundle) -> None:
    operator = registry.get("burst_dropout")()
    x = mosi_synthetic["test"].features[Modality.AUDIO][:1]
    ctx = CorruptionContext(
        modality=Modality.AUDIO,
        stats=mosi_synthetic.stats[Modality.AUDIO],
        generator=torch.Generator().manual_seed(11),
    )
    out = operator(x, 0.4, ctx)
    lost = (out[0].abs().sum(dim=-1) == 0).nonzero().flatten()
    assert lost.numel() > 0
    assert torch.equal(lost, torch.arange(int(lost[0]), int(lost[-1]) + 1)), "gap is not contiguous"


def test_asr_error_rate_scales_with_severity(mosi_synthetic: DatasetBundle) -> None:
    """More severity must change more of the token sequence."""
    operator = registry.get("asr_error")()
    x = mosi_synthetic["test"].features[Modality.TEXT][:32]

    def changed_fraction(severity: float) -> float:
        ctx = CorruptionContext(
            modality=Modality.TEXT,
            stats=mosi_synthetic.stats[Modality.TEXT],
            generator=torch.Generator().manual_seed(5),
        )
        out = operator(x, severity, ctx)
        return float(((out - x).abs().sum(dim=-1) > 1e-6).float().mean())

    assert changed_fraction(0.25) < changed_fraction(1.0)


def test_temporal_shift_moves_frames_by_the_stated_amount(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator = registry.get("shift")(max_shift=4, circular=True)
    x = tiny_bundle["test"].features[Modality.AUDIO][:2]
    out = operator(x, 1.0, make_context(Modality.AUDIO, tiny_bundle, generator))
    assert torch.allclose(out, torch.roll(x, shifts=4, dims=-2))


def test_blur_reduces_temporal_variance(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator = registry.get("blur")()
    x = tiny_bundle["test"].features[Modality.VISUAL][:8]
    out = operator(x, 1.0, make_context(Modality.VISUAL, tiny_bundle, generator))
    assert float(out.var(dim=-2).mean()) < float(x.var(dim=-2).mean())


def test_word_shuffle_preserves_the_bag_of_tokens(
    tiny_bundle: DatasetBundle, generator: torch.Generator
) -> None:
    operator = registry.get("word_shuffle")()
    x = tiny_bundle["test"].features[Modality.TEXT][:4]
    out = operator(x, 1.0, make_context(Modality.TEXT, tiny_bundle, generator))
    assert torch.allclose(out.sum(dim=1), x.sum(dim=1), atol=1e-4)


# --------------------------------------------------------------- plans and registry


def test_plan_application_only_touches_named_modalities(
    tiny_bundle: DatasetBundle, sample_features: dict[Modality, torch.Tensor]
) -> None:
    plan = CorruptionPlan.single(Modality.AUDIO, "zero", 1.0)
    out = registry.apply_plan(sample_features, plan, tiny_bundle.stats)
    assert torch.allclose(out[Modality.AUDIO], torch.zeros_like(out[Modality.AUDIO]))
    assert torch.equal(out[Modality.TEXT], sample_features[Modality.TEXT])
    assert torch.equal(out[Modality.VISUAL], sample_features[Modality.VISUAL])


def test_clean_plan_is_a_no_op(
    tiny_bundle: DatasetBundle, sample_features: dict[Modality, torch.Tensor]
) -> None:
    out = registry.apply_plan(sample_features, CorruptionPlan.clean(), tiny_bundle.stats)
    for modality, tensor in sample_features.items():
        assert torch.equal(out[modality], tensor)


def test_plans_compose_in_order(
    tiny_bundle: DatasetBundle, sample_features: dict[Modality, torch.Tensor]
) -> None:
    plan = CorruptionPlan(
        (
            CorruptionSpec(Modality.AUDIO, "gaussian_noise", 0.5),
            CorruptionSpec(Modality.AUDIO, "zero", 1.0),
        )
    )
    out = registry.apply_plan(
        sample_features, plan, tiny_bundle.stats, torch.Generator().manual_seed(0)
    )
    # Zeroing last must win regardless of what came before it.
    assert torch.allclose(out[Modality.AUDIO], torch.zeros_like(out[Modality.AUDIO]))


def test_plan_key_and_hash_are_stable_and_order_independent() -> None:
    a = CorruptionPlan(
        (
            CorruptionSpec(Modality.AUDIO, "gaussian_noise", 0.5),
            CorruptionSpec(Modality.TEXT, "asr_error", 0.25),
        )
    )
    b = CorruptionPlan(
        (
            CorruptionSpec(Modality.AUDIO, "gaussian_noise", 0.5),
            CorruptionSpec(Modality.TEXT, "asr_error", 0.25),
        )
    )
    assert a.key() == b.key()
    assert a.hash() == b.hash()
    assert CorruptionPlan.clean().key() == "clean"


def test_plan_roundtrips_through_json_form() -> None:
    plan = CorruptionPlan.single(Modality.TEXT, "asr_error", 0.3, max_wer=0.5)
    restored = CorruptionPlan.from_dict(plan.to_dict())
    assert restored.key() == plan.key()
    assert restored.specs[0].params["max_wer"] == 0.5


def test_severity_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="severity"):
        CorruptionSpec(Modality.AUDIO, "zero", 1.5)


def test_operator_rejects_a_modality_it_does_not_apply_to() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        registry.build(CorruptionSpec(Modality.AUDIO, "asr_error", 0.5))


def test_unknown_operator_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown corruption"):
        registry.get("does_not_exist")


def test_plan_generator_is_deterministic_per_sample() -> None:
    plan = CorruptionPlan.single(Modality.AUDIO, "gaussian_noise", 0.5)
    a = registry.plan_generator(plan, 17).get_state()
    b = registry.plan_generator(plan, 17).get_state()
    c = registry.plan_generator(plan, 18).get_state()
    assert torch.equal(a, b), "same (plan, index) must give the same RNG state"
    assert not torch.equal(a, c), "different samples must be corrupted differently"


# --------------------------------------------------------------- sweep construction


def test_standard_grid_covers_the_seven_subset_lattice() -> None:
    names = {axis.name for axis in removal_grid(("zero",))}
    expected = {
        "remove.T.zero",
        "remove.A.zero",
        "remove.V.zero",
        "remove.TA.zero",
        "remove.TV.zero",
        "remove.AV.zero",
        "remove.TAV.zero",
    }
    assert expected == names


def test_every_axis_starts_at_the_clean_anchor() -> None:
    for axis in standard_grid():
        assert axis.severities[0] == 0.0
        assert axis.plans[0].is_clean


def test_unique_plans_deduplicates_the_shared_clean_anchor() -> None:
    axes = standard_grid()
    total = sum(len(axis.plans) for axis in axes)
    deduplicated = unique_plans(axes)
    assert len(deduplicated) < total
    assert "clean" in deduplicated


def test_axis_rejects_a_ladder_that_does_not_start_clean() -> None:
    with pytest.raises(ValueError, match="first severity must be 0"):
        graded_axis(Modality.AUDIO, "gaussian_noise", (0.2, 0.4))


def test_smoke_grid_is_small_enough_for_ci() -> None:
    assert len(unique_plans(smoke_grid())) <= 12


def test_default_severity_ladder_is_the_documented_one() -> None:
    assert DEFAULT_SEVERITIES[0] == 0.0
    assert DEFAULT_SEVERITIES[-1] == 1.0
    assert len(DEFAULT_SEVERITIES) == 6


def test_catalogue_describes_every_operator() -> None:
    entries = registry.catalogue()
    assert {e["name"] for e in entries} == set(ALL_OPERATORS)
    for entry in entries:
        assert entry["applies_to"], f"{entry['name']} declares no modalities"
