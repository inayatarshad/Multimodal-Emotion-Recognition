"""Degradation metrics — retention curves, AUDC, MRS, critical thresholds, brittleness.

These are the project's actual contribution: the task metrics above are standard, but
the *summaries of how they fall apart* are what let architectures be compared on
robustness with single numbers.

Definitions used throughout:

* **Retention(c)** — how much of the model's *skill* survives corruption ``c``. Skill is
  measured above chance, not above zero:

  ``Retention(c) = (metric(c) − chance) / (metric(clean) − chance)``

  for higher-is-better metrics, and the mirrored form for lower-is-better ones (MAE).
  1.0 means no degradation; 0.0 means reduced to chance.

  **The chance correction is not cosmetic.** Binary accuracy floors at 0.5, so an
  uncorrected retention for a destroyed model bottoms out near ``0.5 / clean`` — about
  0.6 for a typical MOSI model. A text-only model with its text removed would then show
  a Modality Reliance Score of ~0.3 despite being, by construction, completely
  incapacitated. Every reliance and AUDC number would be compressed into the top third
  of its range, and comparisons between architectures with different clean scores would
  be systematically distorted. With the correction, that same model scores an MRS of
  ~1.0, which is the truth.
* **AUDC** — the mean of the retention curve over the severity ladder (trapezoidal, then
  normalised by the severity range). It is an *area under retention*, so **higher is more
  robust**. Reported per (model, axis).
* **Critical threshold** — the severity at which retention first crosses below 0.9,
  linearly interpolated between ladder points. ``None`` means it never does.
* **MRS(m)** — ``1 - Retention(remove m)``. 1.0 means the model is worthless without that
  modality; 0.0 means it never needed it.
* **Brittleness index** — the correlation, *across models*, between clean performance and
  mean AUDC. H1 predicts it is negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wfb.evaluation.metrics import METRIC_DIRECTION
from wfb.types import Modality

CRITICAL_LEVEL = 0.9
_EPS = 1e-9

CHANCE_LEVEL: dict[str, float] = {
    "acc2_has0": 0.5,
    "acc2_non0": 0.5,
    "f1_has0": 0.5,
    "f1_non0": 0.5,
    "acc7": 1.0 / 7.0,
    "acc5": 1.0 / 5.0,
    "corr": 0.0,
}
"""Value a trivially uninformed predictor achieves. Metrics absent from this table
(notably MAE, and the MELD classification metrics whose chance level depends on the class
prior) get no correction unless one is passed explicitly — see
:func:`chance_level_for`."""


def chance_level_for(
    metric: str, num_classes: int = 2, class_prior: float | None = None
) -> float | None:
    """Chance level for ``metric``, or ``None`` when it cannot be determined.

    Args:
        metric: Metric name.
        num_classes: For classification accuracy, the number of classes.
        class_prior: Majority-class frequency. For weighted F1 and accuracy on an
            imbalanced corpus (MELD is very imbalanced — 'neutral' is ~47%), the
            majority-class predictor beats uniform guessing by a wide margin, and it is
            the honest floor.
    """
    if metric in CHANCE_LEVEL:
        return CHANCE_LEVEL[metric]
    if metric in {"acc", "f1_weighted", "f1_macro"}:
        if class_prior is not None:
            return class_prior if metric != "f1_macro" else None
        return 1.0 / max(num_classes, 2)
    return None


def retention(value: float, clean: float, metric: str, chance: float | None = None) -> float:
    """Retention of a single corrupted measurement relative to its clean baseline.

    With ``chance`` supplied, this is the fraction of *skill above chance* retained;
    without it, the plain ratio. Both directions guard against division by zero, which
    genuinely occurs — a clean MAE of 0, or a model whose accuracy collapses entirely.
    """
    direction = METRIC_DIRECTION.get(metric, "higher")
    if chance is None:
        chance = CHANCE_LEVEL.get(metric)

    if direction == "lower":
        if chance is not None:  # e.g. MAE of a mean-predicting baseline
            span = chance - clean
            if abs(span) <= _EPS:
                return 1.0
            return float(max((chance - value) / span, 0.0))
        if value <= _EPS:
            return 1.0 if clean <= _EPS else float("inf")
        return float(clean / value)

    if chance is not None:
        span = clean - chance
        if abs(span) <= _EPS:
            return 1.0
        return float((value - chance) / span)
    if abs(clean) <= _EPS:
        return 1.0
    return float(value / clean)


def retention_curve(
    values: list[float], metric: str, clean: float | None = None, chance: float | None = None
) -> list[float]:
    """Retention at each ladder point. ``clean`` defaults to the first (severity-0) point."""
    if not values:
        return []
    baseline = values[0] if clean is None else clean
    return [retention(v, baseline, metric, chance) for v in values]


def audc(
    severities: list[float] | tuple[float, ...],
    retentions: list[float],
    clip_min: float = 0.0,
    clip_max: float = 1.5,
) -> float:
    """Area under the retention curve, normalised to the severity range.

    Retention is clipped into ``[clip_min, clip_max]`` for the integral only; the raw
    curve is always preserved on :attr:`AxisResult.retention`.

    Both clips earn their place. The floor at 0 treats "at chance" and "below chance" as
    the same amount of useful skill — none — so a model that ends up mildly
    anti-correlated is not scored as *worse than useless* in a way that would swamp the
    integral. The ceiling at 1.5 stops a corruption that happens to flatter one metric
    at one ladder point from dominating the area and making a model look robust.
    """
    if len(severities) < 2:
        return float("nan")
    x = np.asarray(severities, dtype=np.float64)
    y = np.clip(np.asarray(retentions, dtype=np.float64), clip_min, clip_max)
    span = float(x[-1] - x[0])
    if span <= _EPS:
        return float("nan")
    return float(np.trapezoid(y, x) / span)


def critical_threshold(
    severities: list[float] | tuple[float, ...],
    retentions: list[float],
    level: float = CRITICAL_LEVEL,
) -> float | None:
    """Severity at which retention first drops below ``level`` (linearly interpolated)."""
    for i in range(1, len(retentions)):
        if retentions[i] < level <= retentions[i - 1]:
            span = retentions[i - 1] - retentions[i]
            if span <= _EPS:
                return float(severities[i])
            frac = (retentions[i - 1] - level) / span
            return float(severities[i - 1] + frac * (severities[i] - severities[i - 1]))
    if retentions and retentions[0] < level:
        return float(severities[0])
    return None


@dataclass
class AxisResult:
    """One (model, corruption axis) degradation curve and its summaries."""

    axis: str
    metric: str
    severities: list[float]
    values: list[float]
    retention: list[float]
    audc: float
    critical: float | None
    group: str = "graded"
    modality: str | None = None
    kind: str = ""
    chance: float | None = None
    all_metrics: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        axis: str,
        metric: str,
        severities: list[float],
        per_point_metrics: list[dict[str, float]],
        group: str = "graded",
        modality: str | None = None,
        kind: str = "",
        clean_value: float | None = None,
        chance: float | None = None,
    ) -> AxisResult:
        """Compute retention, AUDC and the critical threshold for one axis."""
        values = [m[metric] for m in per_point_metrics]
        curve = retention_curve(values, metric, clean_value, chance)
        return cls(
            axis=axis,
            metric=metric,
            severities=list(severities),
            values=values,
            retention=curve,
            audc=audc(severities, curve),
            critical=critical_threshold(severities, curve),
            group=group,
            modality=modality,
            kind=kind,
            chance=chance,
            all_metrics=per_point_metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "axis": self.axis,
            "metric": self.metric,
            "group": self.group,
            "modality": self.modality,
            "kind": self.kind,
            "chance": self.chance,
            "severities": self.severities,
            "values": self.values,
            "retention": self.retention,
            "audc": self.audc,
            "critical": self.critical,
            "all_metrics": self.all_metrics,
        }


def modality_reliance(
    removal_axes: dict[str, AxisResult], variant: str = "zero"
) -> dict[str, float]:
    """Modality Reliance Score per single modality, from the removal lattice.

    ``MRS(m) = 1 - Retention(remove m at severity 1)``. Values near 1 mean the model
    cannot function without ``m``; near 0 that it never used it. Q2's text-dominance
    claim is exactly ``MRS(text) >> MRS(audio) + MRS(visual)``.
    """
    scores: dict[str, float] = {}
    for modality in Modality.all():
        key = f"remove.{modality.short}.{variant}"
        result = removal_axes.get(key)
        if result is None or not result.retention:
            continue
        scores[modality.value] = float(1.0 - result.retention[-1])
    return scores


def normalized_reliance(scores: dict[str, float]) -> dict[str, float]:
    """MRS renormalised to sum to 1, exposing *relative* asymmetry across modalities."""
    positive = {k: max(v, 0.0) for k, v in scores.items()}
    total = sum(positive.values())
    if total <= _EPS:
        return dict.fromkeys(positive, 0.0)
    return {k: v / total for k, v in positive.items()}


def subset_retention(
    removal_axes: dict[str, AxisResult], variant: str = "zero"
) -> dict[str, float]:
    """Retention for each of the 7 non-empty removal subsets, keyed ``T``/``TA``/``TAV``..."""
    out: dict[str, float] = {}
    for key, result in removal_axes.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "remove" and parts[2] == variant and result.retention:
            out[parts[1]] = float(result.retention[-1])
    return out


def brittleness_index(
    clean_scores: dict[str, float], audc_scores: dict[str, float]
) -> dict[str, float]:
    """Correlation across models between clean performance and mean AUDC.

    H1 predicts a negative value: the models that are best on clean data are the ones
    that fall furthest under corruption. This is the money plot's single number.

    Returns Pearson and Spearman coefficients plus the sample size; with 6–8 architectures
    the Spearman version is the honest one to quote, and neither is worth a p-value.
    """
    models = sorted(set(clean_scores) & set(audc_scores))
    if len(models) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": float(len(models))}
    x = np.array([clean_scores[m] for m in models], dtype=np.float64)
    y = np.array([audc_scores[m] for m in models], dtype=np.float64)
    if x.std() < _EPS or y.std() < _EPS:
        return {"pearson": 0.0, "spearman": 0.0, "n": float(len(models))}
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    spearman_r = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {"pearson": pearson_r, "spearman": spearman_r, "n": float(len(models))}


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Names on the Pareto frontier of (clean performance, AUDC), both maximised.

    Drives the Robustness Pareto view: a point is on the frontier if no other point is at
    least as good on both axes and strictly better on one.
    """
    front: list[str] = []
    for name, (clean, robust) in points.items():
        dominated = any(
            other != name and oc >= clean and orr >= robust and (oc > clean or orr > robust)
            for other, (oc, orr) in points.items()
        )
        if not dominated:
            front.append(name)
    return sorted(front)


def aggregate_seeds(values: list[float]) -> dict[str, float]:
    """Mean, std and n over seeds. Never report a single run — this is the gate."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": float(array.size),
    }
