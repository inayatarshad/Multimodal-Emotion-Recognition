"""The evaluation sweep: one trained checkpoint, many corruption plans.

This is where the "train once, evaluate exhaustively" asymmetry pays off. Plans are
deduplicated across axes first (every axis shares the same severity-0 anchor), then each
unique plan costs exactly one forward pass over the split.

Per-sample predictions are retained in memory for the clean plan and every removal plan,
because the paired significance tests need them; they are not written to JSON, which
would be ~200 MB per model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wfb.corruption.sweeps import SweepAxis, unique_plans
from wfb.data.datamodule import MultimodalDataModule
from wfb.evaluation.degradation import (
    AxisResult,
    chance_level_for,
    modality_reliance,
    normalized_reliance,
    subset_retention,
)
from wfb.evaluation.metrics import compute_metrics, per_sample_error, primary_metric_for
from wfb.models.base import BaseFusionModel
from wfb.types import CorruptionPlan, SplitData, SplitName


def _majority_class_rate(split: SplitData) -> float:
    """Frequency of the most common label — the majority-class baseline's accuracy."""
    labels = split.labels.reshape(-1)
    if labels.numel() == 0:
        return 0.0
    counts = torch.bincount(labels.long())
    return float(counts.max().item() / labels.numel())


@dataclass
class SweepResult:
    """Everything one (model, seed, dataset) evaluation produces."""

    model: str
    dataset: str
    seed: int
    split: SplitName
    metric: str
    clean_metrics: dict[str, float]
    axes: dict[str, AxisResult]
    mrs: dict[str, float] = field(default_factory=dict)
    mrs_normalized: dict[str, float] = field(default_factory=dict)
    subset_retention: dict[str, float] = field(default_factory=dict)
    parameters: int = 0
    provenance: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    labels: np.ndarray | None = field(default=None, repr=False)

    @property
    def mean_audc(self) -> float:
        """Mean AUDC over the graded axes — the model's single robustness number."""
        values = [a.audc for a in self.axes.values() if a.group == "graded" and np.isfinite(a.audc)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def clean_score(self) -> float:
        """Clean performance on the primary metric."""
        return self.clean_metrics.get(self.metric, float("nan"))

    def errors(self, plan_key: str = "clean", task: str = "regression") -> np.ndarray:
        """Per-sample error vector under ``plan_key``, for the significance tests."""
        if plan_key not in self.predictions or self.labels is None:
            raise KeyError(f"No retained predictions for plan {plan_key!r}")
        return per_sample_error(self.predictions[plan_key], self.labels, task=task)

    def to_dict(self, include_curves: bool = True) -> dict[str, Any]:
        """JSON-serialisable form. Predictions are deliberately excluded."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "seed": self.seed,
            "split": self.split,
            "metric": self.metric,
            "clean_metrics": self.clean_metrics,
            "clean_score": self.clean_score,
            "mean_audc": self.mean_audc,
            "mrs": self.mrs,
            "mrs_normalized": self.mrs_normalized,
            "subset_retention": self.subset_retention,
            "parameters": self.parameters,
            "provenance": self.provenance,
            "config": self.config,
            "timing": self.timing,
            "axes": (
                {k: v.to_dict() for k, v in self.axes.items()}
                if include_curves
                else {k: {"audc": v.audc, "critical": v.critical} for k, v in self.axes.items()}
            ),
        }


@torch.no_grad()
def predict_under_plan(
    model: BaseFusionModel,
    datamodule: MultimodalDataModule,
    plan: CorruptionPlan,
    split: SplitName = "test",
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Run the model over ``split`` with ``plan`` applied. Returns (predictions, labels)."""
    model.eval()
    model.to(device)
    loader = datamodule.corrupted_loader(split, plan)
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in loader:
        features = {m: batch[m.value].to(device) for m in model.active}
        out = model(features)
        predictions.append(out.prediction.detach().cpu())
        labels.append(batch["label"].detach().cpu())
    return torch.cat(predictions), torch.cat(labels)


def run_sweep(
    model: BaseFusionModel,
    datamodule: MultimodalDataModule,
    axes: list[SweepAxis],
    split: SplitName = "test",
    metric: str | None = None,
    device: torch.device | str = "cpu",
    keep_predictions_for: tuple[str, ...] = ("clean",),
    progress: bool = False,
) -> SweepResult:
    """Evaluate one model across every plan in ``axes``.

    Args:
        model: A trained architecture.
        datamodule: Supplies the corrupted loaders (and therefore the shared RNG).
        axes: The corruption grid, from :mod:`wfb.corruption.sweeps`.
        split: Which split to evaluate. Always ``test`` for reported numbers.
        metric: Primary metric for retention/AUDC. Defaults per task.
        device: Torch device.
        keep_predictions_for: Plan-key prefixes whose per-sample predictions are retained
            in memory for the significance tests (``"clean"`` and removal plans by default).
        progress: Print a line per plan.

    Returns:
        A fully populated :class:`SweepResult`.
    """
    bundle = datamodule.bundle
    task = bundle.task
    metric = metric or primary_metric_for(task)

    # Give the model's learned mask tokens to the corruption layer, so the `mask` removal
    # variant uses the token the model was actually trained with.
    mask_vectors = model.mask_vectors()
    started = time.perf_counter()

    plans = unique_plans(axes)
    per_plan_metrics: dict[str, dict[str, float]] = {}
    retained: dict[str, np.ndarray] = {}
    labels_np: np.ndarray | None = None

    for key, plan in plans.items():
        dataset = datamodule.dataset(split, plan)
        dataset.mask_vectors = mask_vectors
        predictions, labels = _predict(model, dataset, datamodule, split, plan, device)
        per_plan_metrics[key] = compute_metrics(
            predictions, labels, task=task, label_range=bundle.label_range
        )
        if labels_np is None:
            labels_np = labels.numpy()
        if any(key.startswith(prefix) for prefix in keep_predictions_for) or key == "clean":
            retained[key] = predictions.numpy()
        if progress:
            print(f"  {key:44s} {metric}={per_plan_metrics[key][metric]:.4f}")

    clean_metrics = per_plan_metrics.get("clean", {})
    clean_value = clean_metrics.get(metric)

    # Retention is measured as skill above chance — see wfb.evaluation.degradation.
    # On an imbalanced classification corpus the majority-class rate, not 1/C, is the
    # honest floor, so it is computed from the split's actual label distribution.
    chance = chance_level_for(
        metric,
        num_classes=bundle.num_classes,
        class_prior=_majority_class_rate(bundle[split]) if task == "classification" else None,
    )

    results: dict[str, AxisResult] = {}
    for axis in axes:
        points = [per_plan_metrics[plan.key()] for plan in axis.plans]
        results[axis.name] = AxisResult.build(
            axis=axis.name,
            metric=metric,
            severities=list(axis.severities),
            per_point_metrics=points,
            group=axis.group,
            modality=axis.modality.value if axis.modality else None,
            kind=axis.kind,
            clean_value=clean_value,
            chance=chance,
        )

    removal_axes = {k: v for k, v in results.items() if v.group == "removal"}
    mrs = modality_reliance(removal_axes)

    return SweepResult(
        model=model.cfg.name,
        dataset=bundle.name,
        seed=datamodule.seed,
        split=split,
        metric=metric,
        clean_metrics=clean_metrics,
        axes=results,
        mrs=mrs,
        mrs_normalized=normalized_reliance(mrs),
        subset_retention=subset_retention(removal_axes),
        parameters=model.num_parameters,
        provenance=bundle.provenance.to_dict(),
        config={"model": dict(model.cfg.__dict__), "n_plans": len(plans)},
        timing={"sweep_seconds": time.perf_counter() - started},
        predictions=retained,
        labels=labels_np,
    )


@torch.no_grad()
def _predict(
    model: BaseFusionModel,
    dataset: Any,
    datamodule: MultimodalDataModule,
    split: SplitName,  # noqa: ARG001 - kept for signature symmetry with the public API
    plan: CorruptionPlan,  # noqa: ARG001
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Forward pass over a prepared dataset (which already carries the plan)."""
    from torch.utils.data import DataLoader

    from wfb.data.datamodule import collate

    model.eval()
    model.to(device)
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=datamodule.loader_cfg.eval_batch_size,
        shuffle=False,
        num_workers=datamodule.loader_cfg.num_workers,
        collate_fn=collate,
    )
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in loader:
        features = {m: batch[m.value].to(device) for m in model.active}
        predictions.append(model(features).prediction.detach().cpu())
        labels.append(batch["label"].detach().cpu())
    return torch.cat(predictions), torch.cat(labels)
