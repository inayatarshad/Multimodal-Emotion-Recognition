"""Audio-specific corruption operators.

These act on COVAREP-style frame features rather than waveforms — see PLAN.md D3 for why
(three orders of magnitude cheaper, and every architecture provably sees the same
tensor). Each operator is designed to imitate a failure mode that actually occurs:
packet loss, microphone clipping, a noisy room, a dead channel mid-utterance.
"""

from __future__ import annotations

import torch
from torch import Tensor

from wfb.corruption.base import Corruption, CorruptionContext, frame_mask
from wfb.corruption.registry import register
from wfb.types import Modality


@register
class FrameDropout(Corruption):
    """Independently drop a fraction ``severity`` of audio frames (packet loss).

    Dropped frames are zeroed rather than removed, keeping the sequence rectangular and
    the timing of surviving frames intact — which is the point: the model still sees
    *where* the gaps are.
    """

    name = "frame_dropout"
    applies_to = (Modality.AUDIO, Modality.VISUAL, Modality.TEXT)
    physical_unit = "% frames dropped"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Zero a Bernoulli(severity) subset of frames."""
        if severity == 0.0:
            return x
        keep = ctx.rand(*x.shape[:-1], device=x.device, dtype=torch.float32) >= severity
        return frame_mask(x, keep, 0.0)

    def describe(self, severity: float) -> str:
        """E.g. ``30% frames dropped``."""
        return f"{severity * 100:.0f}% frames dropped"


@register
class Clipping(Corruption):
    """Microphone clipping: hard-limit feature magnitudes to a shrinking envelope.

    Severity 1 clips at 10% of the per-feature RMS, which flattens almost all dynamics.
    Unlike additive noise this is a *deterministic, information-destroying* corruption —
    a useful contrast, because a model cannot average it away.
    """

    name = "clipping"
    applies_to = (Modality.AUDIO,)
    physical_unit = "clip threshold (x RMS)"

    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 4.0) -> None:
        super().__init__(min_ratio=min_ratio, max_ratio=max_ratio)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _threshold(self, severity: float) -> float:
        return float(self.max_ratio * (self.min_ratio / self.max_ratio) ** severity)

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Clamp to +-threshold x RMS."""
        if severity == 0.0:
            return x
        scale = (
            ctx.stats.rms.to(x.device, x.dtype)
            if ctx.stats is not None
            else x.reshape(-1, x.shape[-1]).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        )
        limit = scale * self._threshold(severity)
        return torch.maximum(torch.minimum(x, limit), -limit)

    def describe(self, severity: float) -> str:
        """E.g. ``clip at 0.63x RMS``."""
        return f"clip at {self._threshold(severity):.2f}x RMS"


@register
class ChannelDropout(Corruption):
    """Kill a contiguous block of audio frames — a dead microphone for part of the clip.

    Block length is ``severity * T``, positioned uniformly at random. Bursty loss is
    strictly harder than the same *amount* of independent loss, because interpolation
    across a long gap is impossible; reporting both isolates whether a model's robustness
    comes from redundancy or from smoothing.
    """

    name = "burst_dropout"
    applies_to = (Modality.AUDIO, Modality.VISUAL)
    physical_unit = "% contiguous frames lost"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Zero one contiguous run of frames."""
        if severity == 0.0:
            return x
        seq_len = x.shape[-2]
        length = round(severity * seq_len)
        if length <= 0:
            return x
        leading = x.shape[:-2]
        start = ctx.randint(max(seq_len - length + 1, 1), leading, device=x.device)
        positions = torch.arange(seq_len, device=x.device).expand(*leading, seq_len)
        lost = (positions >= start.unsqueeze(-1)) & (positions < (start + length).unsqueeze(-1))
        return frame_mask(x, ~lost, 0.0)

    def describe(self, severity: float) -> str:
        """E.g. ``30% contiguous loss``."""
        return f"{severity * 100:.0f}% contiguous loss"
