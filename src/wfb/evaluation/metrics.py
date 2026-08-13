"""Task metrics, in the exact conventions the CMU-MOSI/MOSEI literature uses.

Two conventions for binary accuracy coexist in that literature and papers rarely say
which they used, which is a running source of 1–3 point discrepancies in reproduction
attempts:

* ``acc2_has0`` — every test sample, with ``pred >= 0`` counted as positive;
* ``acc2_non0`` — neutral samples (``label == 0``) excluded first.

``acc2_non0`` runs 1–2 points higher and is what most recent papers report. We compute
both, always, and the reproduction table states which one each published number used.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor

Direction = Literal["higher", "lower"]

METRIC_DIRECTION: dict[str, Direction] = {
    "mae": "lower",
    "corr": "higher",
    "acc7": "higher",
    "acc5": "higher",
    "acc2_has0": "higher",
    "acc2_non0": "higher",
    "f1_has0": "higher",
    "f1_non0": "higher",
    "acc": "higher",
    "f1_weighted": "higher",
    "f1_macro": "higher",
}

PRIMARY_METRIC: dict[str, str] = {
    "regression": "acc2_non0",
    "classification": "f1_weighted",
}


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 for a degenerate (constant) input.

    ``np.corrcoef`` returns NaN when either vector has zero variance, which happens for
    real when a model collapses to a constant prediction under heavy corruption — and a
    NaN there would silently poison every aggregate downstream.
    """
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(
    predictions: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    label_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, float]:
    """MOSI/MOSEI sentiment metrics: MAE, correlation, Acc-7, Acc-2 and F1 (both conventions)."""
    pred = _to_numpy(predictions)
    true = _to_numpy(labels)
    lo, hi = label_range

    mae = float(np.mean(np.abs(pred - true)))
    corr = pearson(pred, true)

    # Acc-7: round to the nearest integer sentiment level within the label range.
    pred_level = np.clip(np.round(pred), lo, hi)
    true_level = np.clip(np.round(true), lo, hi)
    acc7 = float(np.mean(pred_level == true_level))
    acc5 = float(np.mean(np.clip(pred_level, -2, 2) == np.clip(true_level, -2, 2)))

    # has0: neutral folded into the positive class (pred >= 0).
    pred_bin_has0 = (pred >= 0).astype(int)
    true_bin_has0 = (true >= 0).astype(int)
    acc2_has0 = float(np.mean(pred_bin_has0 == true_bin_has0))
    f1_has0 = float(f1_score(true_bin_has0, pred_bin_has0, average="weighted", zero_division=0))

    # non0: neutral samples dropped entirely.
    non_zero = true != 0
    if bool(non_zero.any()):
        pred_bin = (pred[non_zero] > 0).astype(int)
        true_bin = (true[non_zero] > 0).astype(int)
        acc2_non0 = float(np.mean(pred_bin == true_bin))
        f1_non0 = float(f1_score(true_bin, pred_bin, average="weighted", zero_division=0))
    else:
        acc2_non0, f1_non0 = acc2_has0, f1_has0

    return {
        "mae": mae,
        "corr": corr,
        "acc7": acc7,
        "acc5": acc5,
        "acc2_has0": acc2_has0,
        "acc2_non0": acc2_non0,
        "f1_has0": f1_has0,
        "f1_non0": f1_non0,
    }


def classification_metrics(
    logits: Tensor | np.ndarray, labels: Tensor | np.ndarray
) -> dict[str, float]:
    """MELD-style metrics: accuracy plus weighted and macro F1."""
    if isinstance(logits, Tensor):
        pred = logits.detach().cpu()
        pred_class = (pred.argmax(dim=-1) if pred.ndim > 1 else pred).numpy().reshape(-1)
    else:
        array = np.asarray(logits)
        pred_class = (array.argmax(axis=-1) if array.ndim > 1 else array).reshape(-1)
    true = _to_numpy(labels).astype(int)
    pred_class = pred_class.astype(int)
    return {
        "acc": float(np.mean(pred_class == true)),
        "f1_weighted": float(f1_score(true, pred_class, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(true, pred_class, average="macro", zero_division=0)),
    }


def compute_metrics(
    predictions: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    task: str = "regression",
    label_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, float]:
    """Dispatch to the metric set for ``task``."""
    if task == "regression":
        return regression_metrics(predictions, labels, label_range)
    return classification_metrics(predictions, labels)


def per_sample_error(
    predictions: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    task: str = "regression",
    metric: str = "auto",
) -> np.ndarray:
    """Per-sample loss vector, for the paired significance tests.

    Paired tests need one number per *sample*, not per dataset — absolute error for
    regression, 0/1 misclassification for classification.
    """
    true = _to_numpy(labels)
    if task == "regression" and metric in {"auto", "mae"}:
        absolute: np.ndarray = np.abs(_to_numpy(predictions) - true)
        return absolute
    if task == "regression":  # binary correctness of the sign decision
        pred = _to_numpy(predictions)
        wrong_sign: np.ndarray = ((pred > 0).astype(int) != (true > 0).astype(int)).astype(float)
        return wrong_sign
    if isinstance(predictions, Tensor):
        pred_class = predictions.detach().cpu()
        classes = (pred_class.argmax(dim=-1) if pred_class.ndim > 1 else pred_class).numpy()
    else:
        array = np.asarray(predictions)
        classes = array.argmax(axis=-1) if array.ndim > 1 else array
    misclassified: np.ndarray = (classes.reshape(-1).astype(int) != true.astype(int)).astype(float)
    return misclassified


def is_better(metric: str, a: float, b: float) -> bool:
    """True when ``a`` is a better value than ``b`` for ``metric``."""
    return a < b if METRIC_DIRECTION.get(metric, "higher") == "lower" else a > b


def primary_metric_for(task: str) -> str:
    """The metric retention and AUDC are computed on by default."""
    return PRIMARY_METRIC.get(task, "acc2_non0")


@torch.no_grad()
def evaluate_predictions(
    predictions: Tensor,
    labels: Tensor,
    task: str = "regression",
    label_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, float]:
    """Metric dict from raw model outputs (no gradient bookkeeping)."""
    return compute_metrics(predictions, labels, task, label_range)
