"""Deterministic synthetic multimodal corpus with a *known* ground truth.

Why this exists
---------------
CMU-MultimodalSDK downloads are slow and frequently broken, and the corpora cannot be
committed. Without a fallback, nothing above the loader could be tested, developed, or
demoed. So the loader chain ends here.

But this is not noise. The generator plants a specific, documented structure:

* the label is **text-dominant** — text carries ~60% of the linear signal variance,
  audio ~25%, visual ~15% (matching the asymmetry Q2 asks about on real data);
* a genuine **text x audio interaction term** carries the remainder — it is recoverable
  *only* by a model that multiplies or attends across those two streams, so tensor and
  cross-attention fusion have a real advantage over late fusion here;
* features are temporally smooth, so temporal-misalignment corruption actually bites.

The consequence is that the degradation pipeline can be validated against an answer we
already know: if the Modality Reliance Score for text does not come out largest, or if a
severity-0 sweep point differs from the clean baseline, the bug is in our code, not in
the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from wfb.types import (
    DatasetBundle,
    FeatureStats,
    Modality,
    Provenance,
    SplitData,
    SplitName,
    TaskType,
)

# Linear signal budget. Squared coefficients are the variance shares.
_VARIANCE_SHARE: dict[Modality, float] = {
    Modality.TEXT: 0.60,
    Modality.AUDIO: 0.25,
    Modality.VISUAL: 0.15,
}
# Share of total label variance carried by the text x audio interaction.
_INTERACTION_SHARE = 0.18


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for the synthetic corpus. Defaults mirror CMU-MOSI's size and shapes."""

    name: str = "synthetic"
    task: TaskType = "regression"
    n_train: int = 1284
    n_val: int = 229
    n_test: int = 686
    seq_len: int = 50
    dims: tuple[int, int, int] = (300, 5, 20)  # text, audio, visual — MOSI convention
    num_classes: int = 1
    noise_scale: float = 0.9
    seed: int = 20260813

    @property
    def dim_map(self) -> dict[Modality, int]:
        """Feature dimensionality per modality."""
        return dict(zip(Modality.all(), self.dims, strict=True))


_MODULATION_DEPTH = 0.5


def _temporal_profile(n: int, seq_len: int, generator: torch.Generator) -> Tensor:
    """Smooth, strictly positive per-sample temporal envelope in ``(n, seq_len)``.

    Three low-frequency sinusoids with random phase give frame-to-frame correlation, so
    frame dropout, blur and misalignment all have something real to destroy.

    The **DC offset of 1.0 is essential**: with a zero-mean envelope the planted latent
    factor cancels under any pooling, and — because the phase is random per sample — it
    is not recoverable by a fixed linear map over the flattened sequence either. The
    corpus would then contain a signal that no architecture can extract, and every
    baseline would sit at chance while looking like a training problem.
    """
    t = torch.linspace(0.0, 1.0, seq_len).unsqueeze(0)  # (1, T)
    oscillation = torch.zeros(n, seq_len)
    for harmonic in (1.0, 2.0, 3.5):
        phase = torch.rand(n, 1, generator=generator) * 2 * torch.pi
        weight = torch.randn(n, 1, generator=generator) * (1.0 / harmonic)
        oscillation = oscillation + weight * torch.sin(2 * torch.pi * harmonic * t + phase)
    oscillation = oscillation / oscillation.std(dim=1, keepdim=True).clamp_min(1e-6)
    return 1.0 + _MODULATION_DEPTH * oscillation


def _modality_features(
    factor: Tensor,
    dim: int,
    seq_len: int,
    projection: Tensor,
    noise_scale: float,
    generator: torch.Generator,
) -> Tensor:
    """Render a latent scalar ``factor`` (n,) into ``(n, seq_len, dim)`` features."""
    n = factor.shape[0]
    envelope = _temporal_profile(n, seq_len, generator)  # (n, T)
    signal = factor.view(n, 1, 1) * envelope.unsqueeze(-1) * projection.view(1, 1, dim)
    noise = torch.randn(n, seq_len, dim, generator=generator) * noise_scale
    # Smooth the noise along time so the features are not white — otherwise temporal
    # corruptions would be trivially destructive.
    kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
    smoothed = (
        torch.nn.functional.conv1d(
            noise.permute(0, 2, 1).reshape(n * dim, 1, seq_len), kernel, padding=1
        )
        .reshape(n, dim, seq_len)
        .permute(0, 2, 1)
    )
    return signal + smoothed


def _make_split(
    n: int,
    cfg: SyntheticConfig,
    projections: dict[Modality, Tensor],
    generator: torch.Generator,
    prefix: str,
) -> SplitData:
    """Generate one split."""
    linear_scale = (1.0 - _INTERACTION_SHARE) ** 0.5
    factors = {m: torch.randn(n, generator=generator) for m in Modality.all()}

    signal = torch.zeros(n)
    for modality, share in _VARIANCE_SHARE.items():
        signal = signal + linear_scale * (share**0.5) * factors[modality]
    interaction = factors[Modality.TEXT] * factors[Modality.AUDIO]
    signal = signal + (_INTERACTION_SHARE**0.5) * interaction
    signal = signal + 0.15 * torch.randn(n, generator=generator)

    # MOSI-like label distribution: continuous in [-3, 3], mass away from zero.
    labels = (1.6 * signal).clamp(-3.0, 3.0)

    features = {
        m: _modality_features(
            factors[m], cfg.dim_map[m], cfg.seq_len, projections[m], cfg.noise_scale, generator
        )
        for m in Modality.all()
    }

    if cfg.task == "classification":
        edges = torch.linspace(-3.0, 3.0, cfg.num_classes + 1)[1:-1]
        class_labels = torch.bucketize(labels, edges)
        return SplitData(
            ids=[f"{prefix}_{i:05d}" for i in range(n)],
            features=features,
            labels=class_labels.long(),
        )

    return SplitData(
        ids=[f"{prefix}_{i:05d}" for i in range(n)],
        features=features,
        labels=labels.float(),
    )


def compute_stats(train: SplitData, chunk: int = 1024) -> dict[Modality, FeatureStats]:
    """Per-feature train statistics used to calibrate corruption operators.

    Computed over the flattened ``(N * T)`` frame axis, on the **train split only**.

    Accumulated in float64 over chunks rather than materialising
    ``tensor.reshape(-1, D).float()``. That one-liner doubles a float32 corpus and
    quadruples a float16 one — 8 GB for CMU-MOSEI — and the float64 running sums are
    also numerically better than a single-pass float32 mean over a million frames.
    """
    stats: dict[Modality, FeatureStats] = {}
    for modality, tensor in train.features.items():
        dim = tensor.shape[-1]
        total = torch.zeros(dim, dtype=torch.float64)
        total_sq = torch.zeros(dim, dtype=torch.float64)
        frames = 0
        for start in range(0, tensor.shape[0], chunk):
            block = tensor[start : start + chunk].reshape(-1, dim).to(torch.float64)
            total += block.sum(dim=0)
            total_sq += block.pow(2).sum(dim=0)
            frames += block.shape[0]
            del block

        count = max(frames, 1)
        mean = total / count
        mean_square = total_sq / count
        # Var = E[x^2] - E[x]^2, clamped because catastrophic cancellation can push a
        # near-constant feature marginally negative.
        variance = (mean_square - mean.pow(2)).clamp_min(0.0)
        stats[modality] = FeatureStats(
            mean=mean.to(torch.float32),
            std=variance.sqrt().to(torch.float32).clamp_min(1e-6),
            rms=mean_square.sqrt().to(torch.float32).clamp_min(1e-6),
        )
    return stats


def make_synthetic_bundle(cfg: SyntheticConfig | None = None) -> DatasetBundle:
    """Build the full synthetic :class:`~wfb.types.DatasetBundle`, deterministically."""
    cfg = cfg or SyntheticConfig()
    generator = torch.Generator().manual_seed(cfg.seed)

    # One shared projection per modality across splits — otherwise train and test would
    # live in different feature spaces and nothing would generalise.
    projections = {
        m: torch.randn(cfg.dim_map[m], generator=generator) / (cfg.dim_map[m] ** 0.25)
        for m in Modality.all()
    }

    sizes: dict[SplitName, int] = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test": cfg.n_test,
    }
    splits = {
        name: _make_split(size, cfg, projections, generator, prefix=name)
        for name, size in sizes.items()
    }

    class_names = (
        [f"class_{i}" for i in range(cfg.num_classes)] if cfg.task == "classification" else []
    )
    return DatasetBundle(
        name=cfg.name,
        task=cfg.task,
        splits=splits,
        stats=compute_stats(splits["train"]),
        provenance=Provenance(
            source="synthetic",
            detail=(
                f"seed={cfg.seed} shares=T{_VARIANCE_SHARE[Modality.TEXT]:.2f}/"
                f"A{_VARIANCE_SHARE[Modality.AUDIO]:.2f}/V{_VARIANCE_SHARE[Modality.VISUAL]:.2f} "
                f"interaction={_INTERACTION_SHARE:.2f}"
            ),
        ),
        num_classes=cfg.num_classes,
        label_range=(-3.0, 3.0),
        class_names=class_names,
    )


def expected_reliance_order() -> tuple[Modality, ...]:
    """Ground-truth modality importance ordering planted by the generator.

    Used by ``tests/test_degradation.py`` to check the Modality Reliance Score machinery
    against an answer that is known in advance.
    """
    return tuple(sorted(_VARIANCE_SHARE, key=lambda m: _VARIANCE_SHARE[m], reverse=True))
