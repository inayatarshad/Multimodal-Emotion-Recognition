"""Tensor Fusion Network (Zadeh et al., EMNLP 2017).

Every modality's pooled embedding is augmented with a constant 1 and the three vectors
are combined by outer product. The augmenting 1 is the whole trick: the resulting tensor
contains all unimodal, bimodal and trimodal interaction terms simultaneously, so a single
linear layer on top can read off any of them.

That is also precisely why TFN is a good H1 subject. Its representation is *multiplicative*:
when one modality is zeroed, every interaction term containing it collapses to zero at
once, and the surviving unimodal terms are the only signal left. It should degrade
sharply and predictably — and if it does not, H1 is in trouble in an interesting way.
"""

from __future__ import annotations

import torch
from torch import nn

from wfb.models.base import BaseFusionModel, ModelOutput, squeeze_regression
from wfb.models.encoders import PredictionHead, TemporalEncoder
from wfb.types import ModalityDict


class TensorFusionModel(BaseFusionModel):
    """Outer-product tensor fusion over pooled per-modality embeddings."""

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
        # Project into a small subspace first: the tensor is (tensor_dim + 1)^k, so this
        # is the difference between a 15k-parameter head and a 17-million-parameter one.
        self.project = nn.ModuleDict(
            {
                m.value: nn.Sequential(
                    nn.Linear(self.cfg.hidden, self.cfg.tensor_dim),
                    nn.Tanh(),
                    nn.Dropout(self.cfg.dropout),
                )
                for m in self.active
            }
        )
        fused_dim = (self.cfg.tensor_dim + 1) ** len(self.active)
        self.head = PredictionHead(
            fused_dim, self.cfg.post_fusion_dim, self.spec.output_dim, self.cfg.dropout
        )

    def forward(self, features: ModalityDict) -> ModelOutput:
        """Outer-product the augmented embeddings and read out with an MLP."""
        augmented: list[torch.Tensor] = []
        for modality in self.active:
            _, pooled = self.encoders[modality.value](features[modality])
            projected = self.project[modality.value](pooled)
            ones = projected.new_ones(projected.shape[0], 1)
            augmented.append(torch.cat([ones, projected], dim=-1))

        fused = augmented[0]
        for nxt in augmented[1:]:
            # (B, P) x (B, Q) -> (B, P*Q), accumulating the full interaction lattice.
            fused = torch.bmm(fused.unsqueeze(2), nxt.unsqueeze(1)).flatten(start_dim=1)

        return ModelOutput(
            prediction=squeeze_regression(self.head(fused), self.spec.task), fused=fused
        )
