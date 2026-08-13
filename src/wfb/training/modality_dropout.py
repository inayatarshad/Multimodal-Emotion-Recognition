"""Modality-dropout regularisation — the Q3 mitigation arm.

The mechanism itself lives on :meth:`wfb.models.base.BaseFusionModel.apply_modality_dropout`
so that every architecture gets it identically. This module holds the *protocol* around
it: which probabilities to sweep, how to name the variants, and how to pair a
dropout-trained model with its untrained counterpart for the Pareto plot.

The question is not "does modality dropout help robustness" — it does, trivially, by
training on the corrupted distribution. The question is what it *costs* on clean data,
and whether that cost is architecture-dependent. If loosely-coupled models pay almost
nothing and cross-attention models pay a lot, that is a second, independent line of
evidence for H1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from wfb.models.base import ModelConfig

DROPOUT_PROBABILITIES: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5)
"""The protocol's sweep. 0.0 is the untrained-for-robustness control."""


@dataclass(frozen=True)
class MitigationVariant:
    """One point in the mitigation arm."""

    base_model: str
    probability: float
    mode: str = "zero"

    @property
    def tag(self) -> str:
        """Run tag, e.g. ``md0.3`` — appended to the run directory name."""
        if self.probability <= 0:
            return ""
        suffix = "m" if self.mode == "mask" else ""
        return f"md{self.probability:g}{suffix}"

    @property
    def label(self) -> str:
        """Display label for plots and tables."""
        return self.base_model if not self.tag else f"{self.base_model}+{self.tag}"


def apply_variant(cfg: ModelConfig, variant: MitigationVariant) -> ModelConfig:
    """Return a copy of ``cfg`` configured for ``variant``."""
    return replace(cfg, modality_dropout=variant.probability, modality_dropout_mode=variant.mode)


def variants_for(
    model_name: str,
    probabilities: tuple[float, ...] = DROPOUT_PROBABILITIES,
    mode: str = "zero",
) -> list[MitigationVariant]:
    """All mitigation variants for one architecture, control first."""
    return [MitigationVariant(model_name, p, mode) for p in probabilities]


def pair_with_control(records: list[dict[str, Any]], key: str = "label") -> list[dict[str, Any]]:
    """Link each dropout-trained record to its p=0 control.

    The Pareto view draws these as arrows — clean performance lost on the x-axis against
    robustness gained on the y-axis — which is the form a practitioner can actually read
    a decision off.
    """
    controls = {
        r.get("base_model", r[key]): r
        for r in records
        if float(r.get("modality_dropout", 0.0)) == 0.0
    }
    linked: list[dict[str, Any]] = []
    for record in records:
        probability = float(record.get("modality_dropout", 0.0))
        if probability == 0.0:
            continue
        control = controls.get(record.get("base_model", ""))
        if control is None:
            continue
        linked.append(
            {
                "model": record.get("base_model", record[key]),
                "probability": probability,
                "clean_control": control.get("clean_score"),
                "clean_variant": record.get("clean_score"),
                "audc_control": control.get("mean_audc"),
                "audc_variant": record.get("mean_audc"),
                "clean_delta": _delta(record.get("clean_score"), control.get("clean_score")),
                "audc_delta": _delta(record.get("mean_audc"), control.get("mean_audc")),
            }
        )
    return linked


def _delta(variant: float | None, control: float | None) -> float | None:
    if variant is None or control is None:
        return None
    return float(variant) - float(control)
