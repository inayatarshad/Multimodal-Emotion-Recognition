"""Paper figures, generated from the committed sweep JSON.

Every figure regenerates from `experiments/results/`, so a figure can never disagree with
the numbers it is supposed to depict. Matplotlib only — no seaborn styling — because the
figures go into a LaTeX document where exact sizing matters.

The palette is colourblind-safe and every series is also distinguished by marker and dash
pattern, so all four figures survive greyscale printing. That is a submission requirement
at most venues and an accessibility requirement everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wfb.models import SOPHISTICATION_ORDER
from wfb.serving.results_store import ResultsStore

# Okabe–Ito: the standard colourblind-safe qualitative palette.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
DASHES: tuple[Any, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), "-", "--")


def _style(index: int) -> dict[str, Any]:
    return {
        "color": PALETTE[index % len(PALETTE)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": DASHES[index % len(DASHES)],
        "linewidth": 1.8,
        "markersize": 5,
    }


def _model_order(labels: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, str]:
        base = label.split("+")[0]
        try:
            return (SOPHISTICATION_ORDER.index(base), label)
        except ValueError:
            return (len(SOPHISTICATION_ORDER), label)

    return sorted(labels, key=key)


def _setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def degradation_curves_figure(
    store: ResultsStore, output: Path, dataset: str | None = None, max_axes: int = 6
) -> Path | None:
    """Faceted retention curves with seed bands — one panel per corruption axis."""
    plt = _setup()
    curves = [c for c in store.degradation_curves(dataset) if not c["axis"].startswith("remove.")]
    if not curves:
        return None

    axes_names = sorted({c["axis"] for c in curves})[:max_axes]
    models = _model_order(sorted({c["model"] for c in curves}))
    columns = min(3, len(axes_names))
    rows = int(np.ceil(len(axes_names) / columns))

    figure, grid = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), squeeze=False)
    for panel_index, axis_name in enumerate(axes_names):
        ax = grid[panel_index // columns][panel_index % columns]
        for model_index, model in enumerate(models):
            curve = next(
                (c for c in curves if c["model"] == model and c["axis"] == axis_name), None
            )
            if curve is None:
                continue
            x = np.asarray(curve["severities"])
            y = np.asarray(curve["retention"])
            err = np.asarray(curve["retention_std"])
            style = _style(model_index)
            ax.plot(x, y, label=model, **style)
            if err.any():
                ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.15, linewidth=0)
        ax.axhline(0.9, color="grey", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_title(axis_name, fontsize=9)
        ax.set_xlabel("severity")
        ax.set_ylabel("retention")
        ax.set_ylim(-0.05, 1.15)

    for empty in range(len(axes_names), rows * columns):
        grid[empty // columns][empty % columns].axis("off")

    handles, labels = grid[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle("Retention under graded corruption (dotted line: 0.9 critical threshold)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def reliance_heatmap_figure(
    store: ResultsStore, output: Path, dataset: str | None = None
) -> Path | None:
    """Modality Reliance Score heatmap — where Q2's text-dominance finding lands."""
    plt = _setup()
    entries = store.reliance(dataset)
    if not entries:
        return None

    models = _model_order([e["model"] for e in entries])
    lookup = {e["model"]: e for e in entries}
    modalities = ["text", "audio", "visual"]
    matrix = np.array(
        [[lookup[m]["mrs"].get(mod, np.nan) for mod in modalities] for m in models],
        dtype=np.float64,
    )

    figure, ax = plt.subplots(figsize=(4.2, 0.45 * len(models) + 1.8))
    # Sequential, perceptually uniform, and readable in greyscale.
    image = ax.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(modalities)), [m.capitalize() for m in modalities])
    ax.set_yticks(range(len(models)), models)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] < 0.55 else "black",
                )
    figure.colorbar(image, ax=ax, label="MRS (1 = useless without it)")
    ax.set_title("Modality Reliance Score")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def pareto_figure(store: ResultsStore, output: Path, dataset: str | None = None) -> Path | None:
    """Clean performance against robustness, with mitigation variants drawn as arrows."""
    plt = _setup()
    points = store.pareto(dataset)
    if not points:
        return None

    figure, ax = plt.subplots(figsize=(5.2, 4.0))
    sizes = np.log10(np.array([max(p["parameters"], 1) for p in points], dtype=np.float64))
    scaled = 30 + 220 * (sizes - sizes.min()) / max(float(np.ptp(sizes)), 1e-9)

    controls = {p["base_model"]: p for p in points if p["modality_dropout"] == 0.0}
    for index, point in enumerate(points):
        style = _style(index)
        ax.scatter(
            point["clean_score"],
            point["mean_audc"],
            s=scaled[index],
            color=style["color"],
            marker=style["marker"],
            edgecolor="black" if point["on_frontier"] else "none",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            point["label"],
            (point["clean_score"], point["mean_audc"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
        control = controls.get(point["base_model"])
        if control is not None and point["modality_dropout"] > 0:
            ax.annotate(
                "",
                xy=(point["clean_score"], point["mean_audc"]),
                xytext=(control["clean_score"], control["mean_audc"]),
                arrowprops={"arrowstyle": "->", "color": "grey", "linewidth": 1.0, "alpha": 0.8},
            )

    frontier = sorted(
        ((p["clean_score"], p["mean_audc"]) for p in points if p["on_frontier"]),
        key=lambda xy: xy[0],
    )
    if len(frontier) > 1:
        ax.plot(*zip(*frontier, strict=True), color="grey", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xlabel(f"clean {store.metric(dataset)}")
    ax.set_ylabel("mean AUDC (robustness)")
    ax.set_title("Robustness Pareto frontier\n(point size: parameters; arrows: modality dropout)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def brittleness_figure(
    store: ResultsStore, output: Path, dataset: str | None = None
) -> Path | None:
    """The money plot: clean performance against AUDC, with the H1 trend line."""
    plt = _setup()
    points = [p for p in store.pareto(dataset) if p["modality_dropout"] == 0.0]
    if len(points) < 3:
        return None

    x = np.array([p["clean_score"] for p in points])
    y = np.array([p["mean_audc"] for p in points])
    brittleness = store.brittleness(dataset)

    figure, ax = plt.subplots(figsize=(5.0, 4.0))
    for index, point in enumerate(points):
        style = _style(index)
        ax.scatter(x[index], y[index], s=90, color=style["color"], marker=style["marker"], zorder=3)
        ax.annotate(
            point["label"],
            (x[index], y[index]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )
    if x.std() > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 50)
        ax.plot(grid, slope * grid + intercept, color="grey", linestyle="--", linewidth=1.2)

    ax.set_xlabel(f"clean {store.metric(dataset)}")
    ax.set_ylabel("mean AUDC (robustness)")
    ax.set_title(
        f"Brittleness index: Spearman {brittleness.get('spearman', float('nan')):.2f} "
        f"(n={int(brittleness.get('n', 0))})\n"
        "H1 predicts a negative slope"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def generate_all(
    results_dir: str | Path = "experiments/results",
    output_dir: str | Path = "paper/figures",
    dataset: str | None = None,
) -> list[Path]:
    """Regenerate every figure. Returns the paths actually written."""
    store = ResultsStore.load(results_dir)
    out = Path(output_dir)
    produced = [
        degradation_curves_figure(store, out / "fig1_degradation_curves.png", dataset),
        reliance_heatmap_figure(store, out / "fig2_modality_reliance.png", dataset),
        brittleness_figure(store, out / "fig3_brittleness.png", dataset),
        pareto_figure(store, out / "fig4_pareto.png", dataset),
    ]
    return [p for p in produced if p is not None]
