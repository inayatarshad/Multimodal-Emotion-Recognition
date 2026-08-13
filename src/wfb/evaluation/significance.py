"""Paired significance testing for architecture comparisons.

Two architectures are compared **on the same corrupted samples** — that is what
:func:`wfb.corruption.registry.plan_generator` guarantees — so the paired tests here are
valid and far more powerful than the unpaired alternative.

The default is a paired bootstrap over per-sample errors rather than a t-test: absolute
errors are strongly right-skewed and 0/1 misclassification indicators are not remotely
normal, so a t-test's assumptions fail exactly where the differences are most interesting.
The bootstrap makes no distributional assumption and is cheap at these sample sizes.

Multiple comparisons are corrected with Holm–Bonferroni, which is uniformly more powerful
than plain Bonferroni and needs no independence assumption (unlike Benjamini–Hochberg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    """The outcome of comparing two systems on the same samples."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    difference: float
    p_value: float
    ci_low: float
    ci_high: float
    effect_size: float
    n: int
    test: str

    @property
    def significant(self) -> bool:
        """Significant at the conventional 5% level, before correction."""
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "a": self.name_a,
            "b": self.name_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference,
            "p_value": self.p_value,
            "ci": [self.ci_low, self.ci_high],
            "effect_size": self.effect_size,
            "n": self.n,
            "test": self.test,
        }


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    name_a: str = "a",
    name_b: str = "b",
) -> ComparisonResult:
    """Two-sided paired bootstrap on the per-sample error difference.

    Args:
        errors_a: Per-sample errors for system A, ``(N,)``. Lower is better.
        errors_b: Per-sample errors for system B, same samples in the same order.
        n_resamples: Bootstrap resamples.
        seed: RNG seed — the test is reproducible.
        name_a: Label for A.
        name_b: Label for B.

    Returns:
        The observed difference (A − B), a 95% percentile CI, a two-sided p-value from
        the sign of the resampled differences, and a paired Cohen's d.
    """
    a = np.asarray(errors_a, dtype=np.float64).reshape(-1)
    b = np.asarray(errors_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"paired test needs matched samples, got {a.shape} and {b.shape}. "
            "Both systems must be evaluated on the same corrupted inputs."
        )
    n = a.size
    if n == 0:
        raise ValueError("no samples to compare")

    diff = a - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diff[idx].mean(axis=1)

    ci_low, ci_high = (float(v) for v in np.percentile(resampled, [2.5, 97.5]))
    # Two-sided p: how often the resampled mean sits on the other side of zero.
    tail = float(np.mean(resampled >= 0.0)) if observed < 0 else float(np.mean(resampled <= 0.0))
    p_value = min(1.0, 2.0 * tail)

    spread = float(diff.std(ddof=1)) if n > 1 else 0.0
    effect = observed / spread if spread > 1e-12 else 0.0

    return ComparisonResult(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        difference=observed,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=effect,
        n=n,
        test="paired_bootstrap",
    )


def paired_wilcoxon(
    errors_a: np.ndarray, errors_b: np.ndarray, name_a: str = "a", name_b: str = "b"
) -> ComparisonResult:
    """Wilcoxon signed-rank test — a rank-based cross-check on the bootstrap.

    Reported alongside the bootstrap in the paper's appendix: when a non-parametric rank
    test and a resampling test disagree, the difference is not robust and should not be
    claimed.
    """
    a = np.asarray(errors_a, dtype=np.float64).reshape(-1)
    b = np.asarray(errors_b, dtype=np.float64).reshape(-1)
    diff = a - b
    if np.allclose(diff, 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    spread = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return ComparisonResult(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        difference=float(diff.mean()),
        p_value=p_value,
        ci_low=float("nan"),
        ci_high=float("nan"),
        effect_size=float(diff.mean() / spread) if spread > 1e-12 else 0.0,
        n=int(a.size),
        test="wilcoxon",
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down correction.

    Returns one boolean per input p-value, in the input order: True means the null is
    rejected at family-wise error rate ``alpha``.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (m - rank):
            rejected[index] = True
        else:
            break  # step-down: everything larger stays accepted
    return rejected


def compare_all(
    errors: dict[str, np.ndarray], n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Pairwise paired-bootstrap comparisons across systems, Holm-corrected.

    Args:
        errors: ``{system_name: per-sample error vector}``, all on the same samples.
        n_resamples: Bootstrap resamples per pair.
        seed: Base RNG seed.
        alpha: Family-wise error rate.

    Returns:
        One record per pair, each with the raw p-value and a ``significant_corrected`` flag.
    """
    names = sorted(errors)
    results: list[ComparisonResult] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            results.append(
                paired_bootstrap(
                    errors[name_a],
                    errors[name_b],
                    n_resamples=n_resamples,
                    seed=seed,
                    name_a=name_a,
                    name_b=name_b,
                )
            )
    corrected = holm_bonferroni([r.p_value for r in results], alpha=alpha)
    return [
        {**r.to_dict(), "significant_corrected": flag}
        for r, flag in zip(results, corrected, strict=True)
    ]


def seed_variance_check(values: list[float], label: str = "") -> dict[str, Any]:
    """Flag configurations whose seed-to-seed spread swamps the effects being claimed.

    A coefficient of variation above 10% means the multi-seed protocol is doing real work
    and any single-run number would have been noise.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return {"label": label, "n": int(array.size), "warning": "fewer than 2 seeds"}
    mean, std = float(array.mean()), float(array.std(ddof=1))
    cv = std / abs(mean) if abs(mean) > 1e-12 else float("inf")
    return {
        "label": label,
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "cv": cv,
        "warning": "high seed variance" if cv > 0.10 else "",
    }
