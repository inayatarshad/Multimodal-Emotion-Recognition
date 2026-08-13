"""Degradation metric tests — the arithmetic behind every headline number."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wfb.evaluation.degradation import (
    AxisResult,
    aggregate_seeds,
    audc,
    brittleness_index,
    chance_level_for,
    critical_threshold,
    modality_reliance,
    normalized_reliance,
    pareto_front,
    retention,
    retention_curve,
    subset_retention,
)
from wfb.evaluation.metrics import (
    compute_metrics,
    per_sample_error,
    primary_metric_for,
    regression_metrics,
)

# ------------------------------------------------------------------ retention


def test_retention_is_one_for_an_unchanged_metric() -> None:
    assert retention(0.8, 0.8, "acc2_non0") == pytest.approx(1.0)


def test_retention_is_zero_at_chance() -> None:
    """The correction that makes MRS interpretable: chance means zero skill retained."""
    assert retention(0.5, 0.8, "acc2_non0") == pytest.approx(0.0)


def test_retention_is_a_half_at_the_midpoint_of_the_skill_range() -> None:
    assert retention(0.65, 0.8, "acc2_non0") == pytest.approx(0.5)


def test_uncorrected_retention_is_a_plain_ratio() -> None:
    assert retention(0.4, 0.8, "acc2_non0", chance=0.0) == pytest.approx(0.5)


def test_retention_handles_lower_is_better_metrics() -> None:
    """MAE has no tabulated chance level, so it falls back to the ratio form."""
    assert retention(2.0, 1.0, "mae") == pytest.approx(0.5)
    assert retention(1.0, 1.0, "mae") == pytest.approx(1.0)


def test_retention_with_an_explicit_baseline_for_mae() -> None:
    """Given a mean-predictor baseline MAE, retention becomes skill-based."""
    assert retention(1.4, 0.8, "mae", chance=1.4) == pytest.approx(0.0)
    assert retention(0.8, 0.8, "mae", chance=1.4) == pytest.approx(1.0)


def test_retention_does_not_divide_by_zero() -> None:
    assert math.isfinite(retention(0.0, 0.0, "mae"))
    assert math.isfinite(retention(0.5, 0.5, "acc2_non0"))
    assert math.isfinite(retention(0.3, 0.0, "corr"))


def test_retention_curve_uses_the_first_point_as_the_anchor() -> None:
    curve = retention_curve([0.9, 0.7, 0.5], "acc2_non0")
    assert curve[0] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(0.0)


def test_chance_level_lookup() -> None:
    assert chance_level_for("acc2_non0") == 0.5
    assert chance_level_for("acc7") == pytest.approx(1 / 7)
    assert chance_level_for("mae") is None
    assert chance_level_for("acc", num_classes=7) == pytest.approx(1 / 7)
    assert chance_level_for("acc", class_prior=0.47) == pytest.approx(0.47)


# ------------------------------------------------------------------ AUDC


def test_audc_of_a_perfectly_robust_model_is_one() -> None:
    assert audc([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_audc_of_a_linear_collapse_is_a_half() -> None:
    assert audc([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_audc_of_an_immediate_collapse_is_near_zero() -> None:
    assert audc([0.0, 0.01, 1.0], [1.0, 0.0, 0.0]) < 0.02


def test_audc_is_normalised_by_the_severity_range() -> None:
    """Doubling the x-range with the same curve shape must not change AUDC."""
    a = audc([0.0, 0.5, 1.0], [1.0, 0.6, 0.2])
    b = audc([0.0, 1.0, 2.0], [1.0, 0.6, 0.2])
    assert a == pytest.approx(b)


def test_audc_clips_an_implausible_spike() -> None:
    """One flattering ladder point must not dominate the integral."""
    clipped = audc([0.0, 0.5, 1.0], [1.0, 8.0, 0.0])
    assert clipped <= 1.0


def test_audc_needs_at_least_two_points() -> None:
    assert math.isnan(audc([0.0], [1.0]))


# ------------------------------------------------------------------ critical threshold


def test_critical_threshold_interpolates_between_ladder_points() -> None:
    # Retention crosses 0.9 halfway between severity 0.0 (1.0) and 0.5 (0.8).
    assert critical_threshold([0.0, 0.5, 1.0], [1.0, 0.8, 0.4]) == pytest.approx(0.25)


def test_critical_threshold_is_none_for_a_robust_model() -> None:
    assert critical_threshold([0.0, 0.5, 1.0], [1.0, 0.98, 0.95]) is None


def test_critical_threshold_uses_a_custom_level() -> None:
    assert critical_threshold([0.0, 1.0], [1.0, 0.0], level=0.5) == pytest.approx(0.5)


# ------------------------------------------------------------------ MRS


def make_removal_axis(name: str, final_retention: float) -> AxisResult:
    return AxisResult(
        axis=name,
        metric="acc2_non0",
        severities=[0.0, 1.0],
        values=[0.8, 0.8 * final_retention],
        retention=[1.0, final_retention],
        audc=0.0,
        critical=None,
        group="removal",
    )


def test_mrs_is_one_minus_retention() -> None:
    axes = {
        "remove.T.zero": make_removal_axis("remove.T.zero", 0.2),
        "remove.A.zero": make_removal_axis("remove.A.zero", 0.7),
        "remove.V.zero": make_removal_axis("remove.V.zero", 0.9),
    }
    scores = modality_reliance(axes)
    assert scores["text"] == pytest.approx(0.8)
    assert scores["audio"] == pytest.approx(0.3)
    assert scores["visual"] == pytest.approx(0.1)


def test_normalized_reliance_sums_to_one_and_preserves_order() -> None:
    normalized = normalized_reliance({"text": 0.8, "audio": 0.3, "visual": 0.1})
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["text"] > normalized["audio"] > normalized["visual"]


def test_normalized_reliance_handles_an_all_zero_case() -> None:
    normalized = normalized_reliance({"text": 0.0, "audio": 0.0, "visual": 0.0})
    assert set(normalized.values()) == {0.0}


def test_subset_retention_reads_the_seven_subset_lattice() -> None:
    axes = {
        f"remove.{subset}.zero": make_removal_axis(f"remove.{subset}.zero", value)
        for subset, value in {
            "T": 0.2,
            "A": 0.8,
            "V": 0.9,
            "TA": 0.1,
            "TV": 0.15,
            "AV": 0.7,
            "TAV": 0.0,
        }.items()
    }
    subsets = subset_retention(axes)
    assert len(subsets) == 7
    assert subsets["TAV"] == pytest.approx(0.0)
    assert subsets["AV"] > subsets["T"], "text-dominance: losing A+V hurts less than losing T"


# ------------------------------------------------------------------ brittleness / Pareto


def test_brittleness_index_is_negative_when_h1_holds() -> None:
    """Better clean performance paired with worse robustness — H1's signature."""
    clean = {"late": 0.75, "early": 0.79, "lmf": 0.81, "tfn": 0.82, "mult": 0.86}
    robustness = {"late": 0.90, "early": 0.82, "lmf": 0.76, "tfn": 0.71, "mult": 0.60}
    index = brittleness_index(clean, robustness)
    assert index["pearson"] < -0.8
    assert index["spearman"] == pytest.approx(-1.0)
    assert index["n"] == 5


def test_brittleness_index_is_positive_when_h1_is_disconfirmed() -> None:
    """A clean disconfirmation must read as clearly as a confirmation."""
    clean = {"late": 0.75, "early": 0.79, "mult": 0.86}
    robustness = {"late": 0.60, "early": 0.72, "mult": 0.88}
    assert brittleness_index(clean, robustness)["spearman"] == pytest.approx(1.0)


def test_brittleness_index_needs_at_least_three_models() -> None:
    assert math.isnan(brittleness_index({"a": 1.0}, {"a": 1.0})["pearson"])


def test_pareto_front_keeps_only_non_dominated_points() -> None:
    points = {
        "late": (0.75, 0.90),  # robust, weaker clean
        "mult": (0.86, 0.60),  # strong clean, brittle
        "lmf": (0.80, 0.75),  # in between, non-dominated
        "dominated": (0.70, 0.50),
    }
    front = pareto_front(points)
    assert "dominated" not in front
    assert {"late", "mult", "lmf"} == set(front)


def test_aggregate_seeds_reports_mean_and_std() -> None:
    stats = aggregate_seeds([0.80, 0.82, 0.78, 0.81, 0.79])
    assert stats["mean"] == pytest.approx(0.80)
    assert stats["std"] > 0
    assert stats["n"] == 5


def test_aggregate_seeds_ignores_non_finite_runs() -> None:
    stats = aggregate_seeds([0.8, float("nan"), 0.9])
    assert stats["n"] == 2


# ------------------------------------------------------------------ task metrics


def test_regression_metrics_of_a_perfect_predictor() -> None:
    labels = np.array([-2.0, -1.0, 1.0, 2.5])
    metrics = regression_metrics(labels, labels)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["corr"] == pytest.approx(1.0)
    assert metrics["acc7"] == pytest.approx(1.0)
    assert metrics["acc2_non0"] == pytest.approx(1.0)


def test_acc2_conventions_differ_on_neutral_samples() -> None:
    """has0 counts the neutral sample; non0 excludes it. The gap is the convention gap."""
    labels = np.array([0.0, 1.0, -1.0, 2.0])
    predictions = np.array([-0.5, 1.0, -1.0, 2.0])  # wrong only on the neutral sample
    metrics = regression_metrics(predictions, labels)
    assert metrics["acc2_non0"] == pytest.approx(1.0)
    assert metrics["acc2_has0"] == pytest.approx(0.75)


def test_correlation_is_zero_for_a_constant_prediction() -> None:
    """A collapsed model must yield 0, not NaN — a NaN here poisons every aggregate."""
    metrics = regression_metrics(np.full(8, 0.3), np.arange(8, dtype=float) - 4)
    assert metrics["corr"] == 0.0
    assert all(math.isfinite(v) for v in metrics.values())


def test_classification_metrics_on_a_perfect_predictor() -> None:
    logits = np.eye(4)[[0, 1, 2, 3]]
    metrics = compute_metrics(logits, np.array([0, 1, 2, 3]), task="classification")
    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["f1_weighted"] == pytest.approx(1.0)


def test_per_sample_error_is_one_value_per_sample() -> None:
    errors = per_sample_error(np.array([1.0, -1.0, 0.5]), np.array([1.0, 1.0, 0.0]))
    assert errors.shape == (3,)
    assert errors[0] == pytest.approx(0.0)
    assert errors[1] == pytest.approx(2.0)


def test_primary_metric_defaults_per_task() -> None:
    assert primary_metric_for("regression") == "acc2_non0"
    assert primary_metric_for("classification") == "f1_weighted"


def test_axis_result_round_trips_to_a_dict() -> None:
    result = AxisResult.build(
        axis="audio.gaussian_noise",
        metric="acc2_non0",
        severities=[0.0, 0.5, 1.0],
        per_point_metrics=[{"acc2_non0": v} for v in (0.8, 0.7, 0.6)],
        chance=0.5,
    )
    payload = result.to_dict()
    assert payload["axis"] == "audio.gaussian_noise"
    assert payload["retention"][0] == pytest.approx(1.0)
    assert payload["chance"] == 0.5
    assert 0.0 < payload["audc"] < 1.0
