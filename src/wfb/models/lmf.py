"""Low-rank Multimodal Fusion (Liu et al., ACL 2018).

LMF computes the same trimodal outer product as TFN but never materialises it. The output
weight tensor is decomposed into ``rank`` modality-specific factors, so fusion becomes:
project each augmented embedding with its factor, multiply the results elementwise, sum
over the rank axis. Cost drops from exponential to linear in the number of modalities.

For H1, LMF is the interesting middle of the axis: it has TFN's multiplicative structure —
so it should share TFN's collapse-under-removal behaviour — but with far fewer parameters.
If brittleness tracks *parameter count*, LMF and TFN will diverge; if it tracks the
*form of the interaction*, they will not. That contrast is what the pair is for.
"""

from __future__ import annotations

import torch
from torch import nn

from wfb.models.base import BaseFusionModel, ModelOutput, squeeze_regression
from wfb.models.encoders import PredictionHead, TemporalEncoder
from wfb.types import ModalityDict


class LowRankFusionModel(BaseFusionModel):
    """Rank-decomposed tensor fusion."""

    def _build(self) -> None:
        self.encoders = nn.ModuleDict(
            {
                m.value: TemporalEncoder(
                    input_dim=self.spec.dims[m],
                    hidden=self.cfg.hidden,
                    kind=self.cfg.encoder,
                    layers=self.cfg.layers,
                    heads=self.cfg.heads,
                    dropout=self.cfg.dropout,
                )
                for m in self.active
            }
        )
        rank, out_dim = self.cfg.rank, self.cfg.post_fusion_dim
        # One factor per modality: (rank, hidden + 1, post_fusion_dim).
        self.factors = nn.ParameterDict(
            {
                m.value: nn.Parameter(torch.empty(rank, self.cfg.hidden + 1, out_dim))
                for m in self.active
            }
        )
        for param in self.factors.values():
            nn.init.xavier_normal_(param)
        self.fusion_weights = nn.Parameter(torch.full((1, rank), 1.0 / rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, out_dim))
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.head = PredictionHead(
            out_dim, self.cfg.fusion_hidden, self.spec.output_dim, self.cfg.dropout
        )

    def forward(self, features: ModalityDict) -> ModelOutput:
        """Factorised outer product: project, multiply elementwise, sum over rank."""
        fused: torch.Tensor | None = None
        for modality in self.active:
            _, pooled = self.encoders[modality.value](features[modality])
            ones = pooled.new_ones(pooled.shape[0], 1)
            augmented = torch.cat([ones, pooled], dim=-1)  # (B, H+1)
            # (B, H+1) x (rank, H+1, out) -> (B, rank, out)
            projected = torch.einsum("bh,rho->bro", augmented, self.factors[modality.value])
            fused = projected if fused is None else fused * projected

        assert fused is not None
        combined = torch.einsum("bro,zr->bo", fused, self.fusion_weights) + self.fusion_bias
        combined = self.dropout(combined)
        return ModelOutput(
            prediction=squeeze_regression(self.head(combined), self.spec.task), fused=combined
        )
