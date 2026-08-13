"""Modality-agnostic corruption operators: complete removal and temporal misalignment.

The three removal variants (``zero`` / ``mean`` / ``mask``) are not interchangeable, and
comparing them is itself a small result: zero-filling puts the model at a point that may
be far outside its input distribution, mean-filling puts it at the distribution centre,
and a learned mask token lets the model *know* the modality is gone. Papers pick one and
rarely say why; we report all three.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from wfb.corruption.base import Corruption, CorruptionContext
from wfb.corruption.registry import register


@register
class ZeroOut(Corruption):
    """Complete removal by zero-filling, faded in linearly with severity.

    At ``severity == 1`` the modality is gone. Intermediate severities scale the features
    towards zero, which is the natural graded interpolation of "removal" and keeps the
    operator continuous — an AUDC over a discontinuous operator is not meaningful.
    """

    name = "zero"
    physical_unit = "fraction removed"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:  # noqa: ARG002
        """Scale features towards zero."""
        return x * (1.0 - severity)

    def describe(self, severity: float) -> str:
        """E.g. ``40% zeroed``."""
        return f"{severity * 100:.0f}% zeroed"


@register
class MeanFill(Corruption):
    """Complete removal by replacing features with the training-set mean.

    Falls back to zero-fill when no statistics are available (which, on z-scored
    features, is the same thing — a fact worth stating in the paper rather than hiding).
    """

    name = "mean"
    physical_unit = "fraction replaced"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Interpolate towards the train mean."""
        if ctx.stats is None:
            return x * (1.0 - severity)
        mean = ctx.stats.mean.to(x.device, x.dtype)
        return x * (1.0 - severity) + mean * severity

    def describe(self, severity: float) -> str:
        """E.g. ``40% mean-filled``."""
        return f"{severity * 100:.0f}% mean-filled"


@register
class MaskToken(Corruption):
    """Complete removal by substituting a learned mask token.

    Requires a model trained with mask-token support; ``ctx.mask_vectors`` carries the
    learned embedding. Without one this degenerates to :class:`MeanFill`, and the
    evaluation record says so.
    """

    name = "mask"
    physical_unit = "fraction masked"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Interpolate towards the learned mask vector."""
        vector = ctx.mask_vectors.get(ctx.modality)
        if vector is None:
            return MeanFill().apply(x, severity, ctx)
        return x * (1.0 - severity) + vector.to(x.device, x.dtype) * severity

    def describe(self, severity: float) -> str:
        """E.g. ``40% masked``."""
        return f"{severity * 100:.0f}% masked"


@register
class TemporalShift(Corruption):
    """Temporal misalignment: roll the sequence by ``k`` frames.

    Real pipelines misalign streams all the time — buffering, variable ASR latency, a
    dropped video frame. Aligned benchmarks make this invisible. ``max_shift`` defaults
    to 10 frames (20% of the standard 50-frame window).

    The shift is by whole frames, so severity is quantised; ``describe`` reports the
    frame count actually used so plots can be labelled honestly.
    """

    name = "shift"
    physical_unit = "frames"

    def __init__(self, max_shift: int = 10, circular: bool = False) -> None:
        super().__init__(max_shift=max_shift, circular=circular)
        self.max_shift = max_shift
        self.circular = circular

    def _frames(self, severity: float, seq_len: int) -> int:
        return min(round(severity * self.max_shift), seq_len)

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:  # noqa: ARG002
        """Roll along the time axis, padding with zeros unless ``circular``."""
        shift = self._frames(severity, x.shape[-2])
        if shift == 0:
            return x
        rolled = torch.roll(x, shifts=shift, dims=-2)
        if self.circular:
            return rolled
        rolled[..., :shift, :] = 0.0
        return rolled

    def describe(self, severity: float) -> str:
        """E.g. ``shift 4 frames``."""
        return f"shift {self._frames(severity, 10**6)} frames"


@register
class GaussianNoise(Corruption):
    """Additive Gaussian noise at a severity-controlled signal-to-noise ratio.

    Severity maps to noise amplitude *linearly*, so
    ``SNR(dB) = -20 * log10(severity)``:

    | severity | 0 | 0.1 | 0.2 | 0.4 | 0.7 | 1.0 |
    |---|---|---|---|---|---|---|
    | SNR (dB) | inf | 20 | 14 | 8 | 3.1 | 0 |

    A linear-in-amplitude map is what makes ``severity == 0`` an exact identity; an
    SNR-linear map would have to special-case zero, and special cases are where the bugs
    live. The sweep still spans the 20 -> 0 dB range the protocol asks for.

    Noise is scaled per feature dimension by the training-set RMS, so a feature with a
    naturally large range is not preferentially destroyed.
    """

    name = "gaussian_noise"
    physical_unit = "SNR (dB)"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Add per-feature-calibrated Gaussian noise."""
        if severity == 0.0:
            return x
        scale = (
            ctx.stats.rms.to(x.device, x.dtype)
            if ctx.stats is not None
            else x.reshape(-1, x.shape[-1]).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        )
        noise = ctx.randn(*x.shape, device=x.device, dtype=x.dtype)
        return x + noise * scale * severity

    def describe(self, severity: float) -> str:
        """Report the effective SNR."""
        if severity <= 0:
            return "SNR inf dB"
        snr = -20 * math.log10(severity)
        return f"SNR {abs(snr) if abs(snr) < 0.05 else snr:.1f} dB"
