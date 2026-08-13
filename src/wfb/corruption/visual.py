"""Visual corruption operators for facial action-unit / Facet feature sequences.

A camera failure does not look like Gaussian noise. It looks like the face leaving the
frame (occlusion), the tracker losing lock (frame dropout, already covered by the shared
operator), or a low-quality stream smearing the fine-grained AU estimates (blur).
"""

from __future__ import annotations

import torch
from torch import Tensor

from wfb.corruption.base import Corruption, CorruptionContext, frame_mask
from wfb.corruption.registry import register
from wfb.types import Modality


@register
class Occlusion(Corruption):
    """Occlude a contiguous stretch of frames — the face leaves shot or is covered.

    The occluded region is mean-filled rather than zeroed when statistics are available:
    a face tracker that loses the face emits its default/neutral estimate, not zeros.
    """

    name = "occlusion"
    applies_to = (Modality.VISUAL,)
    physical_unit = "% frames occluded"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Mean-fill one contiguous run of frames."""
        if severity == 0.0:
            return x
        seq_len = x.shape[-2]
        length = round(severity * seq_len)
        if length <= 0:
            return x
        leading = x.shape[:-2]
        start = ctx.randint(max(seq_len - length + 1, 1), leading, device=x.device)
        positions = torch.arange(seq_len, device=x.device).expand(*leading, seq_len)
        occluded = (positions >= start.unsqueeze(-1)) & (positions < (start + length).unsqueeze(-1))
        fill: Tensor | float = (
            ctx.stats.mean.to(x.device, x.dtype) if ctx.stats is not None else 0.0
        )
        return frame_mask(x, ~occluded, fill)

    def describe(self, severity: float) -> str:
        """E.g. ``40% occluded``."""
        return f"{severity * 100:.0f}% occluded"


@register
class TemporalBlur(Corruption):
    """Gaussian smoothing along time: a low-frame-rate or motion-blurred stream.

    Implemented as a depthwise 1-D convolution with a Gaussian kernel whose sigma grows
    with severity. Expression *dynamics* are destroyed while the mean expression
    survives, which is the interesting failure mode for emotion recognition: models that
    read affect from micro-dynamics should suffer far more than models reading a static
    average face.
    """

    name = "blur"
    applies_to = (Modality.VISUAL, Modality.AUDIO)
    physical_unit = "sigma (frames)"

    def __init__(self, max_sigma: float = 6.0) -> None:
        super().__init__(max_sigma=max_sigma)
        self.max_sigma = max_sigma

    def _sigma(self, severity: float) -> float:
        return severity * self.max_sigma

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:  # noqa: ARG002
        """Convolve along the time axis with a Gaussian kernel."""
        sigma = self._sigma(severity)
        if sigma <= 0.0:
            return x
        radius = max(round(3 * sigma), 1)
        offsets = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()

        leading = x.shape[:-2]
        seq_len, dim = x.shape[-2], x.shape[-1]
        flat = x.reshape(-1, seq_len, dim).permute(0, 2, 1)  # (N, D, T)
        weight = kernel.view(1, 1, -1).expand(dim, 1, kernel.numel())
        padded = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
        smoothed = torch.nn.functional.conv1d(padded, weight, groups=dim)
        return smoothed.permute(0, 2, 1).reshape(*leading, seq_len, dim)

    def describe(self, severity: float) -> str:
        """E.g. ``blur sigma=2.4 frames``."""
        return f"blur sigma={self._sigma(severity):.1f} frames"


@register
class FeatureNoise(Corruption):
    """Independent per-feature jitter — tracker estimation error on AU intensities.

    Distinct from :class:`~wfb.corruption.common.GaussianNoise` in that the perturbation
    is scaled by each feature's train **std** rather than its RMS, i.e. it corrupts the
    *variation* rather than the absolute level. On z-scored features the two coincide;
    on raw features they do not, and the visual stream is usually consumed raw.
    """

    name = "feature_noise"
    applies_to = (Modality.VISUAL, Modality.AUDIO, Modality.TEXT)
    physical_unit = "noise (x std)"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Add noise scaled by the per-feature training standard deviation."""
        if severity == 0.0:
            return x
        scale = (
            ctx.stats.std.to(x.device, x.dtype)
            if ctx.stats is not None
            else x.reshape(-1, x.shape[-1]).std(dim=0).clamp_min(1e-6)
        )
        return x + ctx.randn(*x.shape, device=x.device, dtype=x.dtype) * scale * severity

    def describe(self, severity: float) -> str:
        """E.g. ``noise 0.4x std``."""
        return f"noise {severity:.2f}x std"
