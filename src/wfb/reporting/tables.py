"""Markdown table generation from the committed sweep JSON.

Reviewers spend ninety seconds on a repository. The headline table has to be above the
fold in the README and it has to be generated from the results files, never hand-typed —
a table that can drift from the JSON it summarises is a table nobody should trust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wfb.models import SOPHISTICATION_ORDER
from wfb.serving.results_store import ResultsStore
from wfb.types import Modality

README_START = "<!-- RESULTS_TABLE_START -->"
README_END = "<!-- RESULTS_TABLE_END -->"


def _order_key(label: str) -> tuple[int, str]:
    base = label.split("+")[0]
    try:
        return (SOPHISTICATION_ORDER.index(base), label)
    except ValueError:
        return (len(SOPHISTICATION_ORDER), label)


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _pm(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def headline_table(store: ResultsStore, dataset: str | None = None) -> str:
    """Clean performance, mean AUDC and per-modality reliance, one row per architecture.

    This is the table that answers H1 and Q2 at a glance.
    """
    pareto = {row["label"]: row for row in store.pareto(dataset)}
    reliance = {row["model"]: row for row in store.reliance(dataset)}
    curves = store.degradation_curves(dataset)
    metric = store.metric(dataset)

    audc_std: dict[str, float] = {}
    seeds: dict[str, int] = {}
    for curve in curves:
        if curve["axis"].startswith("remove."):
            continue
        audc_std.setdefault(curve["model"], 0.0)
        audc_std[curve["model"]] = max(audc_std[curve["model"]], float(curve["audc_std"]))
        seeds[curve["model"]] = max(seeds.get(curve["model"], 0), int(curve["seeds"]))

    if not pareto:
        return "_No sweep results yet — run `make experiments`._"

    lines: list[str] = []
    provenance = store.provenance(dataset)
    if "synthetic" in provenance:
        # Loud, unmissable, and at the top. A synthetic number in a table that reads as
        # real is the single worst failure mode this repository could have.
        lines += [
            "> **WARNING - these numbers come from SYNTHETIC data, not a real corpus.** They "
            "demonstrate that the pipeline runs end to end; they are not results. See "
            "[docs/DATA.md](docs/DATA.md) for how to obtain CMU-MOSI/MOSEI, and "
            "[docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the gate that must pass "
            "before any number here is reportable.",
            "",
        ]

    lines += [
        f"| Architecture | Params | Clean {metric} | Mean AUDC "
        "| MRS(T) | MRS(A) | MRS(V) | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(pareto, key=_order_key):
        row = pareto[label]
        mrs = reliance.get(label, {}).get("mrs", {})
        frontier = " (*)" if row["on_frontier"] else ""
        lines.append(
            f"| **{label}**{frontier} | {row['parameters']:,} | "
            f"{_fmt(row['clean_score'])} | "
            f"{_pm(row['mean_audc'], audc_std.get(label, 0.0))} | "
            f"{_fmt(mrs.get(Modality.TEXT.value, float('nan')), 2)} | "
            f"{_fmt(mrs.get(Modality.AUDIO.value, float('nan')), 2)} | "
            f"{_fmt(mrs.get(Modality.VISUAL.value, float('nan')), 2)} | "
            f"{seeds.get(label, 1)} |"
        )

    brittleness = store.brittleness(dataset)
    lines.append("")
    lines.append(
        f"**Brittleness index** (clean vs AUDC across architectures): "
        f"Pearson {_fmt(brittleness.get('pearson', float('nan')), 2)}, "
        f"Spearman {_fmt(brittleness.get('spearman', float('nan')), 2)} "
        f"over n={int(brittleness.get('n', 0))} architectures. "
        f"{_h1_verdict(brittleness, provenance)}"
    )
    lines.append("")
    lines.append(
        f"(*) = on the robustness Pareto frontier. AUDC is the area under the "
        f"chance-corrected retention curve — **higher is more robust**. "
        f"Feature provenance: `{store.provenance(dataset)}`."
    )
    return "\n".join(lines)


def _h1_verdict(brittleness: dict[str, float], provenance: str = "") -> str:
    """State plainly what the number says about H1, including a disconfirmation.

    On synthetic features the verdict is deliberately refused rather than softened. The
    generator plants a text x audio interaction that only multiplicative and attention
    fusion can capture, so those architectures score higher on clean data *and* lose more
    when corruption destroys that interaction. A negative brittleness index is therefore
    a property of the generator, not evidence about real fusion mechanisms — and reading
    it as support for H1 would be circular. A near-perfect coefficient is the tell.
    """
    spearman = brittleness.get("spearman", float("nan"))
    if not np.isfinite(spearman) or brittleness.get("n", 0) < 3:
        return "Too few architectures to read a trend."
    if "synthetic" in provenance:
        return (
            "**This says nothing about H1.** On synthetic features the trend is circular: "
            "the generator plants a text x audio interaction that the sophisticated "
            "architectures exploit for their clean-data advantage, and that same "
            "interaction is what corruption removes first. The coefficient confirms the "
            "measurement chain works; it is not evidence."
        )
    if spearman <= -0.5:
        return "Consistent with H1: stronger clean performance goes with faster degradation."
    if spearman >= 0.5:
        return (
            "**H1 is disconfirmed** on this data: the strongest architectures are also "
            "the most robust."
        )
    return "No clear monotone relationship — H1 is not supported either way."


def reliance_matrix(store: ResultsStore, dataset: str | None = None) -> str:
    """The 7-subset removal grid: retention when each modality subset is removed."""
    entries = store.reliance(dataset)
    if not entries:
        return "_No reliance results yet._"

    subsets = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
    header = " | ".join(f"−{s}" for s in subsets)
    lines = [f"| Architecture | {header} |", "|---" * (len(subsets) + 1) + "|"]
    for entry in sorted(entries, key=lambda e: _order_key(e["model"])):
        cells = " | ".join(_fmt(entry["subset_retention"].get(s, float("nan")), 2) for s in subsets)
        lines.append(f"| **{entry['model']}** | {cells} |")
    lines.append("")
    lines.append(
        "Values are retention of clean skill above chance. `−T` means text was removed. "
        "If `−AV` > `−T`, the model is text-dominated (Q2)."
    )
    return "\n".join(lines)


def axis_table(store: ResultsStore, dataset: str | None = None) -> str:
    """AUDC per (architecture, corruption axis) — the full degradation matrix."""
    curves = [c for c in store.degradation_curves(dataset) if not c["axis"].startswith("remove.")]
    if not curves:
        return "_No graded sweep results yet._"

    models = sorted({c["model"] for c in curves}, key=_order_key)
    axes = sorted({c["axis"] for c in curves})
    lookup = {(c["model"], c["axis"]): c for c in curves}

    lines = ["| Corruption axis | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for axis in axes:
        cells = []
        for model in models:
            curve = lookup.get((model, axis))
            cells.append("—" if curve is None else _pm(curve["audc"], curve["audc_std"], 2))
        lines.append(f"| `{axis}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mitigation_table(store: ResultsStore, dataset: str | None = None) -> str:
    """Modality-dropout trade-off: clean cost against robustness gained (Q3)."""
    rows = store.pareto(dataset)
    controls = {r["base_model"]: r for r in rows if r["modality_dropout"] == 0.0}
    variants = [r for r in rows if r["modality_dropout"] > 0.0]
    if not variants:
        return "_Mitigation arm not run yet._"

    lines = [
        "| Architecture | p(drop) | Clean Δ | AUDC Δ | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for row in sorted(variants, key=lambda r: (_order_key(r["base_model"]), r["modality_dropout"])):
        control = controls.get(row["base_model"])
        if control is None:
            continue
        clean_delta = row["clean_score"] - control["clean_score"]
        audc_delta = row["mean_audc"] - control["mean_audc"]
        verdict = (
            "robustness bought cheaply"
            if audc_delta > 0 and clean_delta > -0.01
            else "robustness bought with accuracy"
            if audc_delta > 0
            else "no gain"
        )
        lines.append(
            f"| {row['base_model']} | {row['modality_dropout']:.1f} | "
            f"{clean_delta:+.3f} | {audc_delta:+.3f} | {verdict} |"
        )
    return "\n".join(lines)


def full_report(store: ResultsStore, dataset: str | None = None) -> str:
    """Every table, in the order the paper uses them."""
    sections = [
        "## Headline results\n",
        headline_table(store, dataset),
        "\n\n## Modality reliance — the 7-subset removal grid\n",
        reliance_matrix(store, dataset),
        "\n\n## AUDC by corruption axis\n",
        axis_table(store, dataset),
        "\n\n## Mitigation: modality dropout\n",
        mitigation_table(store, dataset),
        "",
    ]
    return "\n".join(sections)


def update_readme(readme: Path, table: str) -> bool:
    """Replace the marked region of the README with ``table``.

    Returns True when the file changed. Refuses to guess if the markers are missing —
    silently appending would be worse than failing.
    """
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(f"{readme} is missing the {README_START} / {README_END} markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    updated = f"{head}{README_START}\n{table}\n{README_END}{tail}"
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True
