"""Metrics, degradation summaries, significance testing and the evaluation sweep."""

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
    METRIC_DIRECTION,
    classification_metrics,
    compute_metrics,
    per_sample_error,
    primary_metric_for,
    regression_metrics,
)
from wfb.evaluation.runner import SweepResult, predict_under_plan, run_sweep
from wfb.evaluation.significance import (
    ComparisonResult,
    compare_all,
    holm_bonferroni,
    paired_bootstrap,
    paired_wilcoxon,
    seed_variance_check,
)

__all__ = [
    "METRIC_DIRECTION",
    "AxisResult",
    "ComparisonResult",
    "SweepResult",
    "aggregate_seeds",
    "audc",
    "brittleness_index",
    "chance_level_for",
    "classification_metrics",
    "compare_all",
    "compute_metrics",
    "critical_threshold",
    "holm_bonferroni",
    "modality_reliance",
    "normalized_reliance",
    "paired_bootstrap",
    "paired_wilcoxon",
    "pareto_front",
    "per_sample_error",
    "predict_under_plan",
    "primary_metric_for",
    "regression_metrics",
    "retention",
    "retention_curve",
    "run_sweep",
    "seed_variance_check",
    "subset_retention",
]
