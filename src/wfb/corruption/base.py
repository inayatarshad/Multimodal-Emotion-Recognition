"""Corruption operator interface.

Design rules, all of them load-bearing:

* **Severity is normalised.** Every operator takes ``severity in [0, 1]`` and maps it
  onto its own physical parameter internally. This lets a single sweep configuration
  drive every corruption family, and makes AUDC comparable across families.
* **Severity 0 is an exact identity.** Not "approximately" — bitwise. Each operator is
  responsible for this and each is unit-tested for it. The base class deliberately does
  *not* short-circuit, because then the test would be testing the base class rather than
  the operator, and the bug it is meant to catch (an operator that quietly perturbs its
  clean baseline) would sail through.
* **Randomness is explicit.** Operators never touch global RNG state; they draw from a
  :class:`torch.Generator` supplied in the context. Evaluation seeds that generator from
  ``(plan_hash, sample_index)``, so every architecture sees *bit-identical* corrupted
  inputs — which is what makes the paired significance tests valid.
* **Operators are shape-preserving.** ``(..., T, D)`` in, same shape out. Operators that
  conceptually change length (ASR insertion/deletion) re-pad to ``T``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import Tensor

from wfb.types import FeatureStats, Modality


@dataclass
class CorruptionContext:
    """Everything an operator may need beyond the tensor itself."""

    modality: Modality
    stats: FeatureStats | None = None
    """Train-set statistics, for SNR calibration and mean-fill."""
    generator: torch.Generator | None = None
    mask_vectors: dict[Modality, Tensor] = field(default_factory=dict)
    """Learned mask tokens supplied by a model that was trained with them."""

    def randn(self, *shape: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Draw standard-normal noise from the context generator."""
        return torch.randn(*shape, generator=self.generator, device=device, dtype=dtype)

    def rand(self, *shape: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Draw uniform ``[0, 1)`` noise from the context generator."""
        return torch.rand(*shape, generator=self.generator, device=device, dtype=dtype)

    def randint(self, high: int, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Draw integers in ``[0, high)`` from the context generator."""
        return torch.randint(
            low=0, high=max(high, 1), size=shape, generator=self.generator, device=device
        )


class Corruption(ABC):
    """Base class for all corruption operators."""

    name: ClassVar[str] = "abstract"
    applies_to: ClassVar[tuple[Modality, ...]] = Modality.all()
    physical_unit: ClassVar[str] = ""
    """Human-readable name of what severity maps onto (e.g. ``"SNR (dB)"``)."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def __call__(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Apply the operator.

        Args:
            x: Features, ``(..., T, D)``.
            severity: Normalised severity in ``[0, 1]``.
            ctx: Statistics, RNG and optional mask tokens.

        Returns:
            A corrupted tensor of the same shape. The input is never modified in place.
        """
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"{self.name}: severity must be in [0, 1], got {severity}")
        if x.ndim < 2:
            raise ValueError(f"{self.name}: expected (..., T, D), got {tuple(x.shape)}")
        out = self.apply(x, float(severity), ctx)
        if out.shape != x.shape:
            raise RuntimeError(
                f"{self.name} changed shape {tuple(x.shape)} -> {tuple(out.shape)}; "
                "operators must be shape-preserving"
            )
        return out

    @abstractmethod
    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Operator body. Must be an exact identity when ``severity == 0``."""

    def describe(self, severity: float) -> str:
        """Physical interpretation of ``severity``, for axis labels and the demo UI."""
        return f"severity={severity:.2f}"

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({extra})"


class Identity(Corruption):
    """The null operator. Present so 'no corruption' is a first-class plan element."""

    name = "none"
    physical_unit = "none"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:  # noqa: ARG002
        """Return the input unchanged."""
        return x

    def describe(self, severity: float) -> str:  # noqa: ARG002
        """Always 'clean'."""
        return "clean"


def frame_mask(x: Tensor, keep: Tensor, fill: Tensor | float = 0.0) -> Tensor:
    """Replace frames where ``keep`` is False with ``fill``.

    Args:
        x: ``(..., T, D)``.
        keep: Boolean ``(..., T)`` — True means the frame survives.
        fill: Scalar or ``(D,)`` replacement value.

    Returns:
        A new tensor with dropped frames replaced.
    """
    keep_expanded = keep.unsqueeze(-1)
    if isinstance(fill, Tensor):
        fill_expanded = fill.to(x.device, x.dtype).expand_as(x)
    else:
        fill_expanded = torch.full_like(x, float(fill))
    return torch.where(keep_expanded, x, fill_expanded)
