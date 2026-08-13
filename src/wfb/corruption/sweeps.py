"""Standard evaluation grids: the removal lattice and the graded severity sweeps.

An *axis* is one (modality, operator) pair evaluated at a monotone ladder of severities.
AUDC is defined per axis; retention curves are plotted per axis. Keeping the axis as an
explicit object — rather than a loose list of plans — is what makes the downstream
degradation metrics unambiguous about what they are integrating over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from wfb.types import CorruptionPlan, CorruptionSpec, Modality

DEFAULT_SEVERITIES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Six levels including the clean anchor, per the protocol's '~6 severity levels'."""

REMOVAL_VARIANTS: tuple[str, ...] = ("zero", "mean", "mask")

# (modality, operator) pairs that make up the graded protocol.
GRADED_AXES: tuple[tuple[Modality, str], ...] = (
    (Modality.AUDIO, "gaussian_noise"),
    (Modality.AUDIO, "frame_dropout"),
    (Modality.AUDIO, "burst_dropout"),
    (Modality.AUDIO, "clipping"),
    (Modality.TEXT, "asr_error"),
    (Modality.TEXT, "token_dropout"),
    (Modality.TEXT, "word_shuffle"),
    (Modality.VISUAL, "frame_dropout"),
    (Modality.VISUAL, "occlusion"),
    (Modality.VISUAL, "blur"),
)


@dataclass(frozen=True)
class SweepAxis:
    """One corruption family swept over severities.

    Attributes:
        name: Stable identifier, e.g. ``audio.gaussian_noise``.
        modality: The corrupted modality, or ``None`` for all-modality axes.
        kind: Registered operator name.
        severities: Monotone ladder, always starting at 0 (the clean anchor).
        plans: One plan per severity, aligned index-wise with ``severities``.
    """

    name: str
    modality: Modality | None
    kind: str
    severities: tuple[float, ...]
    plans: tuple[CorruptionPlan, ...]
    group: str = "graded"
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.severities) != len(self.plans):
            raise ValueError("severities and plans must be the same length")
        if self.severities[0] != 0.0:
            raise ValueError(
                f"axis {self.name}: the first severity must be 0 — AUDC and retention are "
                "defined relative to the clean anchor at the head of the axis"
            )
        if list(self.severities) != sorted(self.severities):
            raise ValueError(f"axis {self.name}: severities must be non-decreasing")


def graded_axis(
    modality: Modality,
    kind: str,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    **params: object,
) -> SweepAxis:
    """Build a single-modality graded axis."""
    plans = tuple(CorruptionPlan((CorruptionSpec(modality, kind, s, params),)) for s in severities)
    return SweepAxis(
        name=f"{modality.value}.{kind}",
        modality=modality,
        kind=kind,
        severities=severities,
        plans=plans,
        group="graded",
        params=dict(params),
    )


def misalignment_axis(
    severities: tuple[float, ...] = DEFAULT_SEVERITIES, max_shift: int = 10
) -> SweepAxis:
    """Temporal misalignment applied to audio and visual, with text held as reference.

    Shifting every stream equally would be a no-op for any architecture that is
    time-invariant, so the reference stream must stay put — the corruption is *relative*
    misalignment, which is what actually happens when streams are buffered separately.
    """
    plans = tuple(
        CorruptionPlan(
            (
                CorruptionSpec(Modality.AUDIO, "shift", s, {"max_shift": max_shift}),
                CorruptionSpec(Modality.VISUAL, "shift", s, {"max_shift": max_shift}),
            )
        )
        for s in severities
    )
    return SweepAxis(
        name="all.misalign",
        modality=None,
        kind="shift",
        severities=severities,
        plans=plans,
        group="graded",
        params={"max_shift": max_shift},
    )


def removal_axis(
    modalities: tuple[Modality, ...],
    variant: str = "zero",
    severities: tuple[float, ...] = (0.0, 1.0),
) -> SweepAxis:
    """Axis that removes a *set* of modalities together, faded in over ``severities``."""
    subset = "".join(m.short for m in modalities)
    plans = tuple(
        CorruptionPlan(tuple(CorruptionSpec(m, variant, s) for m in modalities)) for s in severities
    )
    return SweepAxis(
        name=f"remove.{subset}.{variant}",
        modality=modalities[0] if len(modalities) == 1 else None,
        kind=variant,
        severities=severities,
        plans=plans,
        group="removal",
        params={"subset": subset, "variant": variant},
    )


def removal_grid(variants: tuple[str, ...] = REMOVAL_VARIANTS) -> list[SweepAxis]:
    """All 7 non-empty subsets of {T, A, V}, for each removal variant.

    The 7-subset table is the backbone of Q2: comparing "remove text" against
    "remove audio + visual" is the direct measurement of reliance asymmetry.
    """
    axes: list[SweepAxis] = []
    modalities = Modality.all()
    for variant in variants:
        for size in (1, 2, 3):
            for subset in combinations(modalities, size):
                axes.append(removal_axis(subset, variant))
    return axes


def standard_grid(
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    removal_variants: tuple[str, ...] = REMOVAL_VARIANTS,
    include_graded: bool = True,
    include_misalign: bool = True,
) -> list[SweepAxis]:
    """The full protocol: graded axes + misalignment + the removal lattice."""
    axes: list[SweepAxis] = []
    if include_graded:
        axes.extend(graded_axis(m, k, severities) for m, k in GRADED_AXES)
    if include_misalign:
        axes.append(misalignment_axis(severities))
    axes.extend(removal_grid(removal_variants))
    return axes


def unique_plans(axes: list[SweepAxis]) -> dict[str, CorruptionPlan]:
    """Deduplicate plans across axes, keyed by :meth:`CorruptionPlan.key`.

    Every axis contains a severity-0 plan, and all of them are the same clean plan; the
    evaluation loop should run it once, not once per axis. On the full grid this removes
    ~30 redundant passes over the test set.
    """
    out: dict[str, CorruptionPlan] = {}
    for axis in axes:
        for plan in axis.plans:
            out.setdefault(plan.key(), plan)
    return out


def smoke_grid() -> list[SweepAxis]:
    """A tiny grid for tests and CI: three axes, three severities."""
    severities = (0.0, 0.5, 1.0)
    return [
        graded_axis(Modality.AUDIO, "gaussian_noise", severities),
        graded_axis(Modality.TEXT, "asr_error", severities),
        *[removal_axis((m,), "zero") for m in Modality.all()],
    ]
