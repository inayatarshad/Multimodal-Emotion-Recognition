"""Aggregation of committed sweep JSON across seeds.

Both mismatched-grid tests are regressions. A `dev` run reused two cached results that a
previous `smoke` run had written under a coarser corruption grid; the resume check only
looked at (model, dataset, seed, tag), so incomparable runs were accepted as seeds of the
same measurement. It surfaced as an opaque numpy shape error, but the crash was luck —
with equal-length ladders it would have averaged incomparable curves into a confidence
band and reported it as a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wfb.corruption.sweeps import grid_signature, smoke_grid, standard_grid
from wfb.serving.results_store import ResultsStore


def write_sweep(
    directory: Path,
    model: str,
    seed: int,
    severities: list[float],
    clean: float = 0.8,
    audc: float = 0.9,
    dataset: str = "mosi",
) -> Path:
    """Write a minimal but structurally faithful sweep file."""
    n = len(severities)
    retention = [1.0 - (i / max(n - 1, 1)) * 0.4 for i in range(n)]
    payload: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "metric": "acc2_non0",
        "clean_score": clean,
        "mean_audc": audc,
        "parameters": 12345,
        "provenance": {"source": "synthetic"},
        "mrs": {"text": 0.5, "audio": 0.3, "visual": 0.1},
        "mrs_normalized": {"text": 0.56, "audio": 0.33, "visual": 0.11},
        "subset_retention": {"T": 0.5, "A": 0.7, "V": 0.9, "TAV": 0.0},
        "config": {"model": {"modality_dropout": 0.0}},
        "axes": {
            "audio.gaussian_noise": {
                "axis": "audio.gaussian_noise",
                "metric": "acc2_non0",
                "group": "graded",
                "severities": severities,
                "values": [clean * r for r in retention],
                "retention": retention,
                "audc": audc,
                "critical": None,
            }
        },
    }
    path = directory / f"{dataset}_{model}_s{seed}_sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seeds_on_the_same_ladder_are_averaged(tmp_path: Path) -> None:
    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0], audc=0.90)
    write_sweep(tmp_path, "late", 1, [0.0, 0.5, 1.0], audc=0.80)

    curves = ResultsStore.load(tmp_path).degradation_curves("mosi")
    assert len(curves) == 1
    assert curves[0]["seeds"] == 2
    assert curves[0]["audc"] == pytest.approx(0.85)
    assert curves[0]["audc_std"] > 0


def test_mismatched_ladders_do_not_crash(tmp_path: Path) -> None:
    """The original failure: ragged retention lists reached numpy and blew up."""
    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0])
    write_sweep(tmp_path, "late", 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    curves = ResultsStore.load(tmp_path).degradation_curves("mosi")
    assert len(curves) == 1
    assert len(curves[0]["retention"]) == len(curves[0]["severities"])


def test_the_majority_ladder_wins(tmp_path: Path) -> None:
    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0])
    write_sweep(tmp_path, "late", 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    write_sweep(tmp_path, "late", 2, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    curves = ResultsStore.load(tmp_path).degradation_curves("mosi")
    assert curves[0]["severities"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert curves[0]["seeds"] == 2, "the odd run out must not be counted as a seed"


def test_a_tie_prefers_the_finer_ladder(tmp_path: Path) -> None:
    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0])
    write_sweep(tmp_path, "late", 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    curves = ResultsStore.load(tmp_path).degradation_curves("mosi")
    assert len(curves[0]["severities"]) == 6


def test_incomparable_runs_are_never_averaged_together(tmp_path: Path) -> None:
    """The dangerous case: equal-length ladders over *different* severities.

    This would not crash. Without the ladder check it would silently produce a seed band
    over two measurements that are not the same measurement.
    """
    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0], audc=0.9)
    write_sweep(tmp_path, "late", 1, [0.0, 0.1, 0.2], audc=0.1)

    curves = ResultsStore.load(tmp_path).degradation_curves("mosi")
    assert curves[0]["seeds"] == 1
    assert curves[0]["audc_std"] == 0.0
    assert curves[0]["audc"] in (pytest.approx(0.9), pytest.approx(0.1))


def test_reporting_survives_a_mixed_results_directory(tmp_path: Path) -> None:
    """End to end: the tables must render rather than raise."""
    from wfb.reporting.tables import full_report

    write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0], clean=0.78, audc=0.95)
    write_sweep(tmp_path, "late", 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], clean=0.80, audc=0.93)
    write_sweep(tmp_path, "mult", 0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], clean=0.86, audc=0.71)
    write_sweep(tmp_path, "tfn", 0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], clean=0.83, audc=0.80)

    report = full_report(ResultsStore.load(tmp_path), "mosi")
    assert "SYNTHETIC" in report, "synthetic provenance must be called out"
    assert "late" in report and "mult" in report


# ------------------------------------------------------------------ run labelling


@pytest.mark.parametrize("model", ["text_only", "audio_only", "visual_only"])
def test_seeds_aggregate_for_models_whose_name_contains_an_underscore(
    model: str, tmp_path: Path
) -> None:
    """Regression: positional filename parsing mistook the seed for a variant tag.

    ``mosi_text_only_s0_sweep`` split into 5 parts, and the tag was read from a fixed
    position, yielding ``text_only+s0``. Every unimodal baseline therefore appeared once
    per seed as a separate architecture and never aggregated.
    """
    write_sweep(tmp_path, model, 0, [0.0, 0.5, 1.0], audc=0.9)
    write_sweep(tmp_path, model, 1, [0.0, 0.5, 1.0], audc=0.8)

    store = ResultsStore.load(tmp_path)
    labels = {r.label for r in store.records}
    assert labels == {model}, f"expected one label, got {labels}"

    curves = store.degradation_curves("mosi")
    assert len(curves) == 1
    assert curves[0]["seeds"] == 2
    assert curves[0]["audc"] == pytest.approx(0.85)


def test_mitigation_variants_stay_separate_from_their_control(tmp_path: Path) -> None:
    """The tag mechanism must still do its actual job."""
    control = write_sweep(tmp_path, "late", 0, [0.0, 0.5, 1.0])

    # Same model and seed, different training regime -> its own file and its own label.
    payload = json.loads(control.read_text(encoding="utf-8"))
    payload["config"]["model"]["modality_dropout"] = 0.3
    (tmp_path / "mosi_late_s0_md0.3_sweep.json").write_text(json.dumps(payload), encoding="utf-8")

    labels = {r.label for r in ResultsStore.load(tmp_path).records}
    assert labels == {"late", "late+md0.3"}


def test_empty_results_directory_is_not_an_error(tmp_path: Path) -> None:
    store = ResultsStore.load(tmp_path)
    assert store.is_empty
    assert store.degradation_curves() == []
    assert store.pareto() == []


# ------------------------------------------------------------------ grid signature


def test_grid_signature_is_stable_and_order_independent() -> None:
    axes = standard_grid()
    assert grid_signature(axes) == grid_signature(list(reversed(axes)))


def test_grid_signature_distinguishes_different_severity_ladders() -> None:
    coarse = standard_grid(severities=(0.0, 0.5, 1.0))
    fine = standard_grid(severities=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert grid_signature(coarse) != grid_signature(fine)


def test_grid_signature_distinguishes_different_axis_sets() -> None:
    """The exact confusion that caused the bug: smoke grid vs standard grid."""
    assert grid_signature(smoke_grid()) != grid_signature(standard_grid())
