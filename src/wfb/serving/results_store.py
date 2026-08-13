"""Reads the committed sweep JSON and aggregates it across seeds for the API.

The results endpoints serve *precomputed* numbers — the degradation explorer must not
re-run a 200-plan sweep per page load. Aggregation across seeds happens here so the
frontend receives mean ± std and never has to know how many seeds there were.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wfb.evaluation.degradation import brittleness_index, pareto_front

logger = logging.getLogger(__name__)


@dataclass
class SweepRecord:
    """One sweep JSON file, parsed."""

    model: str
    dataset: str
    seed: int
    metric: str
    clean_score: float
    mean_audc: float
    parameters: int
    provenance: str
    mrs: dict[str, float]
    mrs_normalized: dict[str, float]
    subset_retention: dict[str, float]
    axes: dict[str, dict[str, Any]]
    modality_dropout: float = 0.0
    tag: str = ""

    @property
    def label(self) -> str:
        """Display name, distinguishing mitigation variants from their control."""
        return self.model if not self.tag else f"{self.model}+{self.tag}"

    @classmethod
    def from_json(cls, path: Path) -> SweepRecord | None:
        """Parse a sweep file, returning ``None`` if it is malformed."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable results file %s: %s", path, exc)
            return None
        config = raw.get("config", {}).get("model", {})
        dropout = float(config.get("modality_dropout", 0.0))
        stem_parts = path.stem.split("_")
        tag = ""
        if dropout > 0:
            tag = f"md{dropout:g}"
        elif len(stem_parts) > 3 and stem_parts[-1] == "sweep" and len(stem_parts) > 4:
            tag = stem_parts[-2]
        return cls(
            model=str(raw.get("model", "unknown")),
            dataset=str(raw.get("dataset", "unknown")),
            seed=int(raw.get("seed", 0)),
            metric=str(raw.get("metric", "acc2_non0")),
            clean_score=float(raw.get("clean_score", float("nan"))),
            mean_audc=float(raw.get("mean_audc", float("nan"))),
            parameters=int(raw.get("parameters", 0)),
            provenance=str(raw.get("provenance", {}).get("source", "unknown")),
            mrs={k: float(v) for k, v in (raw.get("mrs") or {}).items()},
            mrs_normalized={k: float(v) for k, v in (raw.get("mrs_normalized") or {}).items()},
            subset_retention={k: float(v) for k, v in (raw.get("subset_retention") or {}).items()},
            axes=raw.get("axes") or {},
            modality_dropout=dropout,
            tag=tag,
        )


@dataclass
class ResultsStore:
    """All sweep records found on disk, grouped for the results endpoints."""

    directory: Path
    records: list[SweepRecord] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str | Path) -> ResultsStore:
        """Scan ``directory`` for ``*_sweep.json`` files."""
        path = Path(directory)
        records: list[SweepRecord] = []
        if path.exists():
            for file in sorted(path.glob("*_sweep.json")):
                record = SweepRecord.from_json(file)
                if record is not None:
                    records.append(record)
        logger.info("Loaded %d sweep records from %s", len(records), path)
        return cls(directory=path, records=records)

    @property
    def is_empty(self) -> bool:
        """True when no sweep has been run yet."""
        return not self.records

    def for_dataset(self, dataset: str | None) -> list[SweepRecord]:
        """Records for one dataset (or all of them)."""
        if dataset is None:
            return self.records
        return [r for r in self.records if r.dataset == dataset]

    def datasets(self) -> list[str]:
        """Every dataset present."""
        return sorted({r.dataset for r in self.records})

    def provenance(self, dataset: str | None = None) -> str:
        """Provenance of the underlying features, surfaced to the UI."""
        sources = {r.provenance for r in self.for_dataset(dataset)}
        return "+".join(sorted(sources)) if sources else "none"

    # ------------------------------------------------------------------ aggregation

    def degradation_curves(self, dataset: str | None = None) -> list[dict[str, Any]]:
        """Retention curves per (label, axis), averaged over seeds with std bands."""
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.for_dataset(dataset):
            for axis_name, axis in record.axes.items():
                grouped[(record.label, axis_name)].append(axis)

        curves: list[dict[str, Any]] = []
        for (label, axis_name), axes in sorted(grouped.items()):
            retentions = np.array([a["retention"] for a in axes], dtype=np.float64)
            values = np.array([a["values"] for a in axes], dtype=np.float64)
            audcs = np.array([a["audc"] for a in axes], dtype=np.float64)
            criticals = [a["critical"] for a in axes if a.get("critical") is not None]
            curves.append(
                {
                    "model": label,
                    "axis": axis_name,
                    "metric": axes[0].get("metric", "acc2_non0"),
                    "severities": axes[0]["severities"],
                    "retention": retentions.mean(axis=0).tolist(),
                    "retention_std": (
                        retentions.std(axis=0, ddof=1).tolist()
                        if len(axes) > 1
                        else [0.0] * retentions.shape[1]
                    ),
                    "values": values.mean(axis=0).tolist(),
                    "audc": float(np.nanmean(audcs)),
                    "audc_std": float(np.nanstd(audcs, ddof=1)) if len(axes) > 1 else 0.0,
                    "critical": float(np.mean(criticals)) if criticals else None,
                    "seeds": len(axes),
                }
            )
        return curves

    def reliance(self, dataset: str | None = None) -> list[dict[str, Any]]:
        """Modality Reliance Scores per model, averaged over seeds."""
        grouped: dict[str, list[SweepRecord]] = defaultdict(list)
        for record in self.for_dataset(dataset):
            grouped[record.label].append(record)

        entries: list[dict[str, Any]] = []
        for label, records in sorted(grouped.items()):
            entries.append(
                {
                    "model": label,
                    "mrs": _mean_dicts([r.mrs for r in records]),
                    "mrs_normalized": _mean_dicts([r.mrs_normalized for r in records]),
                    "subset_retention": _mean_dicts([r.subset_retention for r in records]),
                }
            )
        return entries

    def pareto(self, dataset: str | None = None) -> list[dict[str, Any]]:
        """Clean performance against robustness, with the frontier marked."""
        grouped: dict[str, list[SweepRecord]] = defaultdict(list)
        for record in self.for_dataset(dataset):
            grouped[record.label].append(record)

        points: dict[str, tuple[float, float]] = {}
        rows: list[dict[str, Any]] = []
        for label, records in sorted(grouped.items()):
            clean = float(np.nanmean([r.clean_score for r in records]))
            robust = float(np.nanmean([r.mean_audc for r in records]))
            points[label] = (clean, robust)
            rows.append(
                {
                    "label": label,
                    "base_model": records[0].model,
                    "modality_dropout": records[0].modality_dropout,
                    "clean_score": clean,
                    "mean_audc": robust,
                    "parameters": records[0].parameters,
                    "on_frontier": False,
                }
            )
        frontier = set(pareto_front(points))
        for row in rows:
            row["on_frontier"] = row["label"] in frontier
        return rows

    def brittleness(self, dataset: str | None = None) -> dict[str, float]:
        """The H1 correlation across models: clean performance vs mean AUDC."""
        grouped: dict[str, list[SweepRecord]] = defaultdict(list)
        for record in self.for_dataset(dataset):
            if record.modality_dropout == 0.0:  # controls only — the mitigation arm would confound
                grouped[record.model].append(record)
        clean = {k: float(np.nanmean([r.clean_score for r in v])) for k, v in grouped.items()}
        audc = {k: float(np.nanmean([r.mean_audc for r in v])) for k, v in grouped.items()}
        return brittleness_index(clean, audc)

    def metric(self, dataset: str | None = None) -> str:
        """The primary metric used by the stored records."""
        records = self.for_dataset(dataset)
        return records[0].metric if records else "acc2_non0"


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Element-wise mean over a list of dicts with the same keys."""
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d})
    return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}
