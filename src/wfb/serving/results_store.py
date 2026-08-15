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
        model = str(raw.get("model", "unknown"))
        dataset = str(raw.get("dataset", "unknown"))
        seed = int(raw.get("seed", 0))
        tag = _parse_tag(path, dataset, model, seed, dropout, raw.get("tag"))
        return cls(
            model=model,
            dataset=dataset,
            seed=seed,
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
        """Retention curves per (label, axis), averaged over seeds with std bands.

        Only records sharing a severity ladder are averaged together. Results computed
        under different grids are not seeds of the same measurement; mixing them would
        either crash on ragged arrays or — worse, when the ladders happen to be the same
        length — silently average incomparable curves into a confidence band. The
        majority ladder wins and the rest are dropped with a warning naming what was
        excluded.
        """
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.for_dataset(dataset):
            for axis_name, axis in record.axes.items():
                grouped[(record.label, axis_name)].append(axis)

        curves: list[dict[str, Any]] = []
        for (label, axis_name), all_axes in sorted(grouped.items()):
            axes, dropped = _majority_ladder(all_axes)
            if dropped:
                logger.warning(
                    "%s / %s: ignoring %d result(s) whose severity ladder differs from "
                    "the majority %s (found %s). Re-run them with the same preset, or "
                    "pass --force.",
                    label,
                    axis_name,
                    len(dropped),
                    axes[0].get("severities"),
                    [d.get("severities") for d in dropped],
                )
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


def _parse_tag(
    path: Path,
    dataset: str,
    model: str,
    seed: int,
    dropout: float,
    explicit: object = None,
) -> str:
    """Recover a run's variant tag (e.g. ``md0.3``) from its results file.

    Prefers a tag recorded inside the file. Otherwise it strips the *known* prefix
    ``{dataset}_{model}_s{seed}`` and the ``_sweep`` suffix from the filename, rather
    than splitting on underscores and taking a fixed position.

    That distinction is the fix for a real bug: positional splitting silently mistook the
    seed for a tag on every model whose name contains an underscore — ``text_only``,
    ``audio_only``, ``visual_only`` — so each of their seeds became a separate
    "architecture". They never aggregated, and the headline table carried a duplicate row
    per seed for a third of the models.
    """
    if isinstance(explicit, str) and explicit:
        return explicit
    if dropout > 0:
        return f"md{dropout:g}"

    stem = path.stem
    prefix = f"{dataset}_{model}_s{seed}"
    suffix = "_sweep"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)].lstrip("_")
    return ""


def _majority_ladder(
    axes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``axes`` into (those on the most common severity ladder, the rest).

    Ties are broken toward the longer ladder — a finer sweep is the more informative
    measurement, so when two grids are equally represented, keep that one.
    """
    if len(axes) <= 1:
        return axes, []
    by_ladder: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for axis in axes:
        by_ladder[tuple(float(s) for s in axis.get("severities", ()))].append(axis)
    if len(by_ladder) == 1:
        return axes, []
    winner = max(by_ladder, key=lambda ladder: (len(by_ladder[ladder]), len(ladder)))
    kept = by_ladder[winner]
    dropped = [a for ladder, group in by_ladder.items() if ladder != winner for a in group]
    return kept, dropped


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Element-wise mean over a list of dicts with the same keys."""
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d})
    return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}
