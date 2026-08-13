"""Architecture tests: shapes, gradients, and the ability to actually learn."""

from __future__ import annotations

import pytest
import torch

from wfb.models import (
    MODEL_REGISTRY,
    SOPHISTICATION_ORDER,
    DataSpec,
    ModelConfig,
    available_models,
    build_model,
    model_summary,
)
from wfb.types import DatasetBundle, Modality

ARCHITECTURES = ["text_only", "audio_only", "visual_only", "early", "late", "tfn", "lmf", "mult"]


def make_config(name: str) -> ModelConfig:
    """A small, fast config for tests. MulT needs hidden divisible by heads."""
    if name == "mult":
        return ModelConfig(name=name, hidden=16, heads=4, layers=1, encoder="transformer")
    return ModelConfig(name=name, hidden=16, layers=1, tensor_dim=6, post_fusion_dim=16)


@pytest.fixture
def spec(tiny_bundle: DatasetBundle) -> DataSpec:
    return DataSpec.from_bundle(tiny_bundle)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_forward_produces_the_right_shape(
    name: str, spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    model = build_model(make_config(name), spec)
    out = model(sample_features)
    assert out.prediction.shape == (4,), f"{name} should emit one scalar per sample"
    assert torch.isfinite(out.prediction).all()


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_gradients_reach_every_parameter(name: str, spec: DataSpec) -> None:
    """A parameter with no gradient is a disconnected subgraph — a silent architecture bug.

    Uses an *odd* batch size deliberately. The L1 gradient is ±1 per sample, so with an
    even batch the output bias can cancel to exactly zero by coincidence and the test
    would report a wiring bug that is not there.
    """
    torch.manual_seed(0)
    model = build_model(make_config(name), spec)
    features = {m: torch.randn(5, spec.seq_len, d) for m, d in spec.dims.items()}
    loss = model.loss(model(features).prediction, torch.randn(5))
    loss.backward()
    missing = [
        pname
        for pname, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or float(param.grad.abs().sum()) == 0.0)
    ]
    # Mask tokens are unused unless modality dropout is on, so they are legitimately dry.
    missing = [m for m in missing if "mask_tokens" not in m]
    assert not missing, f"{name}: no gradient reached {missing}"


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_model_can_overfit_a_single_batch(
    name: str, spec: DataSpec, tiny_bundle: DatasetBundle
) -> None:
    """The cheapest possible check that an architecture is capable of learning at all.

    A model that cannot drive the loss down on eight memorised samples has a wiring bug,
    and no amount of hyperparameter tuning later will reveal it as clearly as this does.

    The learning rate is 5e-3 rather than something brisker because the two
    transformer-based architectures genuinely stall above ~1e-2 — that is a property of
    the optimiser, not a defect, and a test that used a rate they cannot tolerate would
    be reporting a wiring bug that does not exist.
    """
    torch.manual_seed(0)
    model = build_model(make_config(name), spec)
    features = {m: t[:8].clone() for m, t in tiny_bundle["train"].features.items()}
    labels = tiny_bundle["train"].labels[:8].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    first = float(model.loss(model(features).prediction, labels).detach())
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.loss(model(features).prediction, labels)
        loss.backward()
        optimizer.step()
    final = float(loss.detach())
    assert final < first * 0.5, f"{name} failed to overfit 8 samples: {first:.3f} -> {final:.3f}"


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_output_is_deterministic_in_eval_mode(
    name: str, spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    model = build_model(make_config(name), spec)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(sample_features).prediction, model(sample_features).prediction)


def test_unimodal_models_ignore_the_other_modalities(
    spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    model = build_model(make_config("text_only"), spec)
    model.eval()
    with torch.no_grad():
        baseline = model(sample_features).prediction
        perturbed = dict(sample_features)
        perturbed[Modality.AUDIO] = torch.randn_like(perturbed[Modality.AUDIO]) * 10
        perturbed[Modality.VISUAL] = torch.zeros_like(perturbed[Modality.VISUAL])
        assert torch.equal(model(perturbed).prediction, baseline)


def test_late_fusion_exposes_per_modality_decisions(
    spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    model = build_model(make_config("late"), spec)
    out = model(sample_features)
    assert set(out.per_modality) == set(Modality.all())
    weights = out.attention["modality_weights"]
    assert pytest.approx(1.0, abs=1e-5) == float(weights.sum())


def test_mult_returns_all_six_directional_attention_maps(
    spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    model = build_model(make_config("mult"), spec)
    model.eval()  # in train mode the returned weights are post-dropout and do not sum to 1
    with torch.no_grad():
        out = model(sample_features)
    assert len(out.attention) == 6, "three modalities give six ordered pairs"
    assert "text<-audio" in out.attention
    for weights in out.attention.values():
        assert weights.shape[0] == 4
        # Attention rows are distributions over the source sequence.
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4)


def test_tfn_fused_dimension_is_the_outer_product_size(spec: DataSpec) -> None:
    cfg = make_config("tfn")
    model = build_model(cfg, spec)
    features = {m: torch.randn(2, spec.seq_len, d) for m, d in spec.dims.items()}
    out = model(features)
    assert out.fused is not None
    assert out.fused.shape[-1] == (cfg.tensor_dim + 1) ** 3


def test_lmf_has_far_fewer_parameters_than_tfn(spec: DataSpec) -> None:
    """The point of LMF: same interaction structure, a fraction of the parameters."""
    tfn = build_model(ModelConfig(name="tfn", hidden=16, tensor_dim=12), spec)
    lmf = build_model(ModelConfig(name="lmf", hidden=16, rank=4), spec)
    assert lmf.num_parameters < tfn.num_parameters


# ------------------------------------------------------------------ modality dropout


def test_modality_dropout_is_inactive_in_eval_mode(
    spec: DataSpec, sample_features: dict[Modality, torch.Tensor]
) -> None:
    cfg = ModelConfig(name="late", hidden=16, modality_dropout=0.9)
    model = build_model(cfg, spec)
    model.eval()
    assert model.apply_modality_dropout(sample_features) is sample_features


def test_modality_dropout_drops_roughly_the_configured_fraction(spec: DataSpec) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(name="late", hidden=16, modality_dropout=0.5)
    model = build_model(cfg, spec)
    model.train()
    features = {m: torch.ones(512, spec.seq_len, d) for m, d in spec.dims.items()}
    out = model.apply_modality_dropout(features)
    dropped = torch.stack([(out[m].abs().sum(dim=(1, 2)) == 0).float() for m in Modality.all()])
    rate = float(dropped.mean())
    # Slightly below 0.5: rows that lose everything have one modality revived.
    assert 0.35 < rate < 0.5


def test_modality_dropout_never_drops_every_modality(spec: DataSpec) -> None:
    """A sample with no modalities at all is not a training signal, it is noise."""
    torch.manual_seed(1)
    cfg = ModelConfig(name="late", hidden=16, modality_dropout=0.95)
    model = build_model(cfg, spec)
    model.train()
    features = {m: torch.ones(256, spec.seq_len, d) for m, d in spec.dims.items()}
    out = model.apply_modality_dropout(features)
    surviving = torch.stack(
        [(out[m].abs().sum(dim=(1, 2)) > 0).float() for m in Modality.all()]
    ).sum(dim=0)
    assert float(surviving.min()) >= 1.0


def test_mask_mode_substitutes_the_learned_token(spec: DataSpec) -> None:
    torch.manual_seed(2)
    cfg = ModelConfig(name="late", hidden=16, modality_dropout=1.0, modality_dropout_mode="mask")
    model = build_model(cfg, spec)
    with torch.no_grad():
        model.mask_tokens[Modality.AUDIO.value].fill_(7.0)
    model.train()
    features = {m: torch.ones(8, spec.seq_len, d) for m, d in spec.dims.items()}
    out = model.apply_modality_dropout(features)
    audio = out[Modality.AUDIO]
    dropped_rows = (audio == 7.0).all(dim=-1).all(dim=-1)
    assert bool(dropped_rows.any()), "mask mode never substituted the token"


def test_mask_vectors_are_exposed_for_evaluation(spec: DataSpec) -> None:
    cfg = ModelConfig(name="late", hidden=16, modality_dropout=0.3, modality_dropout_mode="mask")
    model = build_model(cfg, spec)
    vectors = model.mask_vectors()
    assert set(vectors) == set(Modality.all())
    assert not vectors[Modality.TEXT].requires_grad


# ------------------------------------------------------------------ registry


def test_registry_and_sophistication_order_agree() -> None:
    assert set(SOPHISTICATION_ORDER) <= set(MODEL_REGISTRY)
    assert SOPHISTICATION_ORDER[-1] == "mult", "MulT is the most sophisticated point"
    assert SOPHISTICATION_ORDER.index("late") < SOPHISTICATION_ORDER.index("mult")


def test_unknown_model_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        build_model(ModelConfig(name="nope"), DataSpec(dims={Modality.TEXT: 4}, seq_len=4))


def test_model_summary_reports_the_registry_fields(spec: DataSpec) -> None:
    model = build_model(make_config("mult"), spec)
    summary = model_summary(model)
    assert summary["name"] == "mult"
    assert summary["parameters"] == model.num_parameters
    assert summary["modalities"] == ["text", "audio", "visual"]


def test_available_models_is_sorted_and_complete() -> None:
    assert available_models() == sorted(MODEL_REGISTRY)


def test_classification_task_emits_class_logits(tiny_bundle: DatasetBundle) -> None:
    spec = DataSpec(
        dims=tiny_bundle.dims, seq_len=tiny_bundle.seq_len, task="classification", num_classes=7
    )
    model = build_model(make_config("late"), spec)
    features = {m: torch.randn(3, spec.seq_len, d) for m, d in spec.dims.items()}
    out = model(features)
    assert out.prediction.shape == (3, 7)
    loss = model.loss(out.prediction, torch.tensor([0, 3, 6]))
    assert torch.isfinite(loss)
