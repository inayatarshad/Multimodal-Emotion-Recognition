"""Significance testing: the statistics that separate a result from a plot."""

from __future__ import annotations

import numpy as np
import pytest

from wfb.evaluation.significance import (
    compare_all,
    holm_bonferroni,
    paired_bootstrap,
    paired_wilcoxon,
    seed_variance_check,
)


def test_identical_systems_are_not_significantly_different() -> None:
    rng = np.random.default_rng(0)
    errors = rng.gamma(2.0, 0.5, size=400)
    result = paired_bootstrap(errors, errors.copy(), n_resamples=2000, seed=1)
    assert result.difference == pytest.approx(0.0)
    assert result.p_value > 0.5
    assert not result.significant


def test_a_clearly_better_system_is_detected() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.gamma(2.0, 0.5, size=500)
    improved = baseline - 0.3  # uniformly lower error on the same samples
    result = paired_bootstrap(improved, baseline, n_resamples=2000, seed=2, name_a="new")
    assert result.difference < 0
    assert result.p_value < 0.01
    assert result.significant
    assert result.ci_high < 0, "the CI should exclude zero"


def test_a_tiny_difference_in_noise_is_not_significant() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, size=200)
    b = a + rng.normal(0.0, 1.0, size=200) * 0.01
    assert paired_bootstrap(a, b, n_resamples=2000, seed=3).p_value > 0.05


def test_pairing_is_enforced() -> None:
    with pytest.raises(ValueError, match="matched samples"):
        paired_bootstrap(np.zeros(10), np.zeros(9))


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=100), rng.normal(size=100)
    first = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    second = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert first.p_value == second.p_value
    assert first.ci_low == second.ci_low


def test_wilcoxon_agrees_with_the_bootstrap_on_a_clear_difference() -> None:
    rng = np.random.default_rng(4)
    baseline = rng.gamma(2.0, 0.5, size=300)
    improved = baseline - 0.25
    bootstrap = paired_bootstrap(improved, baseline, n_resamples=2000, seed=5)
    wilcoxon = paired_wilcoxon(improved, baseline)
    assert bootstrap.p_value < 0.05
    assert wilcoxon.p_value < 0.05


def test_wilcoxon_on_identical_vectors_is_not_significant() -> None:
    errors = np.linspace(0, 1, 50)
    assert paired_wilcoxon(errors, errors.copy()).p_value == pytest.approx(1.0)


# ------------------------------------------------------------------ correction


def test_holm_bonferroni_rejects_only_the_small_p_values() -> None:
    rejected = holm_bonferroni([0.001, 0.02, 0.6, 0.9], alpha=0.05)
    assert rejected[0] is True
    assert rejected[2] is False
    assert rejected[3] is False


def test_holm_bonferroni_is_more_conservative_than_no_correction() -> None:
    p_values = [0.02] * 10  # each individually "significant"
    assert not any(holm_bonferroni(p_values, alpha=0.05))


def test_holm_bonferroni_steps_down() -> None:
    """Once one hypothesis is accepted, every larger p-value is accepted too.

    With four hypotheses the second-smallest is tested against alpha/3 = 0.0167, so 0.02
    fails there and the step-down halts — even though 0.02 would pass the *last*
    threshold of alpha/1. That halting behaviour is the whole point of the procedure.
    """
    rejected = holm_bonferroni([0.001, 0.02, 0.02, 0.02], alpha=0.05)
    assert rejected == [True, False, False, False]


def test_holm_bonferroni_rejects_all_when_every_threshold_is_met() -> None:
    """Holm is *not* plain Bonferroni: the largest p-value is tested against alpha itself."""
    assert holm_bonferroni([0.001, 0.04, 0.005], alpha=0.05) == [True, True, True]


def test_holm_bonferroni_handles_an_empty_family() -> None:
    assert holm_bonferroni([]) == []


def test_compare_all_covers_every_pair_and_corrects() -> None:
    rng = np.random.default_rng(5)
    base = rng.gamma(2.0, 0.5, size=300)
    errors = {"late": base, "mult": base + 0.4, "lmf": base + 0.05}
    comparisons = compare_all(errors, n_resamples=1000, seed=6)
    assert len(comparisons) == 3  # 3 systems -> 3 pairs
    for record in comparisons:
        assert "significant_corrected" in record
        assert record["n"] == 300
    late_vs_mult = next(c for c in comparisons if {c["a"], c["b"]} == {"late", "mult"})
    assert late_vs_mult["significant_corrected"] is True


# ------------------------------------------------------------------ seed hygiene


def test_seed_variance_check_flags_an_unstable_configuration() -> None:
    result = seed_variance_check([0.80, 0.55, 0.91, 0.62, 0.88], label="mult")
    assert result["warning"] == "high seed variance"
    assert result["n"] == 5


def test_seed_variance_check_is_quiet_for_a_stable_configuration() -> None:
    result = seed_variance_check([0.800, 0.805, 0.798, 0.802, 0.799])
    assert result["warning"] == ""


def test_seed_variance_check_warns_about_a_single_run() -> None:
    """A single-run number is exactly what the protocol forbids."""
    assert "fewer than 2 seeds" in seed_variance_check([0.8])["warning"]
