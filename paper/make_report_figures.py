"""Schematic figures for the technical report.

These are diagrams of the method, not plots of results, so they are safe to generate
independently of any experiment run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1b2430"
ACCENT = "#0f7b6c"
MUTED = "#6b7684"
LIGHT = "#eef2f5"
OUT = Path(__file__).parent / "figures"


def _box(ax, x, y, w, h, label, sub="", fc=LIGHT, ec=INK, fs=8.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012", facecolor=fc, edgecolor=ec, linewidth=1.1
        )
    )
    ax.text(
        x + w / 2,
        y + h * (0.60 if sub else 0.5),
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        weight="bold",
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=7, color=MUTED)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=MUTED,
            shrinkA=2,
            shrinkB=2,
        )
    )


def pipeline_figure(path: Path) -> Path:
    """Figure 1: how a single evaluation point is produced."""
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    _box(ax, 0.05, 2.30, 1.55, 0.62, "Text", "BERT, 768-d")
    _box(ax, 0.05, 1.44, 1.55, 0.62, "Audio", "COVAREP, 74-d")
    _box(ax, 0.05, 0.58, 1.55, 0.62, "Visual", "Facet, 35-d")

    _box(ax, 2.15, 1.30, 1.85, 1.30, "Corruption\noperator", "severity s in [0,1]", fc="#fdf1e3")
    for y in (2.61, 1.75, 0.89):
        _arrow(ax, 1.60, y, 2.15, 1.95)

    _box(ax, 4.55, 1.30, 1.95, 1.30, "Six fusion\narchitectures", "identical corrupted input")
    _arrow(ax, 4.00, 1.95, 4.55, 1.95)

    _box(ax, 7.05, 1.72, 2.85, 0.78, "Retention(s)", "skill above chance, relative to clean")
    _box(ax, 7.05, 0.72, 2.85, 0.78, "AUDC / MRS", "one robustness number per pair")
    _arrow(ax, 6.50, 2.05, 7.05, 2.11)
    _arrow(ax, 6.50, 1.85, 7.05, 1.11)

    ax.text(
        5.0,
        0.18,
        "Severity 0 is a bitwise identity; the RNG is derived from (plan, sample) "
        "so every architecture sees the same corrupted tensor.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
        style="italic",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def concept_figure(path: Path) -> Path:
    """Figure 2: what the hypothesis predicts, drawn as an illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))

    s = np.linspace(0, 1, 60)
    robust = 1 - 0.30 * s**1.6
    brittle = 1 - 0.92 * s**1.7

    ax = axes[0]
    ax.plot(s, robust, color=ACCENT, lw=2.2, label="loosely coupled fusion")
    ax.plot(s, brittle, color="#c1462f", lw=2.2, ls="--", label="tightly coupled fusion")
    ax.axhline(0.9, color=MUTED, lw=0.9, ls=":")
    ax.text(0.02, 0.905, "critical threshold", fontsize=7, color=MUTED)
    ax.fill_between(s, robust, brittle, color="#c1462f", alpha=0.07)
    ax.set_xlabel("corruption severity", fontsize=9)
    ax.set_ylabel("retention of clean skill", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title("H1: sophistication buys accuracy, not robustness", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    clean = np.array([0.74, 0.80, 0.83, 0.84, 0.86, 0.88])
    audc = np.array([0.97, 0.95, 0.92, 0.90, 0.87, 0.83])
    ax.scatter(clean, audc, s=70, color=ACCENT, zorder=3)
    fit = np.polyfit(clean, audc, 1)
    ax.plot(clean, np.polyval(fit, clean), color=MUTED, ls="--", lw=1.2)
    ax.set_xlabel("clean accuracy", fontsize=9)
    ax.set_ylabel("mean AUDC (robustness)", fontsize=9)
    ax.set_title("Brittleness index: the trade-off, if it exists", fontsize=9.5, color=INK)

    for a in axes:
        a.grid(alpha=0.22)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)

    fig.text(
        0.5,
        -0.04,
        "Illustrative shapes showing what the protocol measures; not experimental results.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
        style="italic",
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(pipeline_figure(OUT / "report_fig1_pipeline.png"))
    print(concept_figure(OUT / "report_fig2_concept.png"))
