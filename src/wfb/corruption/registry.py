"""Name -> operator registry, and pipeline application.

Every operator registers itself with :func:`register`; configs and the HTTP API refer to
operators by name only, so adding a corruption family requires no changes anywhere else.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from wfb.types import (
    CorruptionPlan,
    CorruptionSpec,
    FeatureStats,
    Modality,
    ModalityDict,
)

if TYPE_CHECKING:  # pragma: no cover
    from wfb.corruption.base import Corruption

_REGISTRY: dict[str, type[Corruption]] = {}
_LOADED = False


def register(cls: type[Corruption]) -> type[Corruption]:
    """Class decorator: add ``cls`` to the registry under ``cls.name``."""
    name = cls.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Corruption name {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def _ensure_loaded() -> None:
    """Import the operator modules so their decorators have run."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for module in ("base", "common", "audio", "text", "visual"):
        importlib.import_module(f"wfb.corruption.{module}")


def get(name: str) -> type[Corruption]:
    """Look up an operator class by name."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown corruption {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    """All registered operator names, sorted."""
    _ensure_loaded()
    return sorted(_REGISTRY)


def catalogue() -> list[dict[str, object]]:
    """Registry description for the API's ``/api/corruptions`` endpoint and the docs."""
    _ensure_loaded()
    return [
        {
            "name": name,
            "applies_to": [m.value for m in cls.applies_to],
            "unit": cls.physical_unit,
            "doc": (cls.__doc__ or "").strip().split("\n")[0],
        }
        for name, cls in sorted(_REGISTRY.items())
    ]


def build(spec: CorruptionSpec) -> Corruption:
    """Instantiate the operator described by ``spec``, applying its params."""
    cls = get(spec.kind)
    if spec.modality not in cls.applies_to:
        raise ValueError(
            f"Corruption {spec.kind!r} does not apply to {spec.modality.value} "
            f"(only {[m.value for m in cls.applies_to]})"
        )
    return cls(**dict(spec.params))


def plan_generator(plan: CorruptionPlan, index: int, base_seed: int = 0) -> torch.Generator:
    """Deterministic RNG for one (plan, sample) pair.

    Every architecture evaluated under the same plan therefore sees *bit-identical*
    corrupted inputs. Without this, differences between architectures would be partly
    noise from independent corruption draws, and the paired significance tests in
    :mod:`wfb.evaluation.significance` would be invalid.
    """
    digest = int(plan.hash(), 16) if plan.specs else 0
    seed = (digest ^ (index * 0x9E3779B1) ^ base_seed) % (2**31 - 1)
    return torch.Generator().manual_seed(seed)


def apply_plan(
    features: ModalityDict,
    plan: CorruptionPlan,
    stats: dict[Modality, FeatureStats] | None = None,
    generator: torch.Generator | None = None,
    mask_vectors: dict[Modality, Tensor] | None = None,
) -> ModalityDict:
    """Apply every spec in ``plan`` to ``features``.

    Specs are applied in order, so a plan can compose (e.g. noise then frame dropout).
    Modalities not mentioned in the plan are passed through untouched — the same tensor
    object, not a copy, since operators never mutate in place.
    """
    from wfb.corruption.base import CorruptionContext

    if plan.is_clean:
        return dict(features)

    out = dict(features)
    for spec in plan.specs:
        if spec.is_identity:
            continue
        if spec.modality not in out:
            continue
        operator = build(spec)
        ctx = CorruptionContext(
            modality=spec.modality,
            stats=None if stats is None else stats.get(spec.modality),
            generator=generator,
            mask_vectors=mask_vectors or {},
        )
        out[spec.modality] = operator(out[spec.modality], spec.severity, ctx)
    return out


def describe_plan(plan: CorruptionPlan) -> str:
    """Human-readable one-liner, e.g. ``audio:SNR 8.0 dB + text:WER 20%``."""
    if plan.is_clean:
        return "clean"
    parts: list[str] = []
    for spec in plan.specs:
        if spec.is_identity:
            continue
        operator = build(spec)
        parts.append(f"{spec.modality.value}:{operator.describe(spec.severity)}")
    return " + ".join(parts) if parts else "clean"


__all__ = [
    "apply_plan",
    "available",
    "build",
    "catalogue",
    "describe_plan",
    "get",
    "plan_generator",
    "register",
]
