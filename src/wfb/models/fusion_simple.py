"""The two anchors of the fusion-sophistication axis: early and late fusion.

Late fusion is the *loose coupling* end — modalities never interact until their decisions
are averaged, so removing one leaves the others' computation entirely intact. Under H1 it
should be the most robust architecture in the study, and it is the control against which
every sophisticated model's brittleness is measured.

Early fusion is the cheapest form of tight coupling: concatenate at the frame level and
let a single encoder do whatever it likes. It shares late fusion's simplicity but not its
independence, which makes the pair a clean two-point test of the hypothesis before the
tensor and attention models are involved.
"""

from __future__ import annotations

import torch
from torch import nn

from wfb.models.base import BaseFusionModel, ModelOutput, squeeze_regression
from wfb.models.encoders import PredictionHead, TemporalEncoder
from wfb.types import Modality, ModalityDict


class EarlyFusionModel(BaseFusionModel):
    """Frame-level feature concatenation into a single shared encoder."""

    def _build(self) -> None:
        total_dim = sum(self.spec.dims[m] for m in self.active)
        self.encoder = TemporalEncoder(
            input_dim=total_dim,
            hidden=self.cfg.hidden,
            kind=self.cfg.encoder,
            layers=self.cfg.layers,
            heads=self.cfg.heads,
            dropout=self.cfg.dropout,
        )
        self.head = PredictionHead(
            self.cfg.hidden, self.cfg.fusion_hidden, self.spec.output_dim, self.cfg.dropout
        )

    def forward(self, features: ModalityDict) -> ModelOutput:
        """Concatenate along the feature axis, then encode."""
        stacked = torch.cat([features[m] for m in self.active], dim=-1)
        _, pooled = self.encoder(stacked)
        return ModelOutput(
            prediction=squeeze_regression(self.head(pooled), self.spec.task), fused=pooled
        )


class LateFusionModel(BaseFusionModel):
    """Independent per-modality encoders and heads, combined at the decision level.

    Combination is a *learned* weighted average (a softmax over one scalar per modality)
    rather than a plain mean. A plain mean would make the model gratuitously bad at
    exploiting the strong text signal, and would then look robust for the wrong reason —
    it would have less to lose. The weights are learned on clean data and frozen at eval
    time, so no test-time adaptation sneaks in.
    """

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
        self.heads = nn.ModuleDict(
            {
                m.value: PredictionHead(
                    self.cfg.hidden,
                    self.cfg.fusion_hidden,
                    self.spec.output_dim,
                    self.cfg.dropout,
                )
                for m in self.active
            }
        )
        self.combination_logits = nn.Parameter(torch.zeros(len(self.active)))

    def forward(self, features: ModalityDict) -> ModelOutput:
        """Predict per modality, then average the decisions with learned weights."""
        weights = torch.softmax(self.combination_logits, dim=0)
        per_modality: dict[Modality, torch.Tensor] = {}
        pooled_all: list[torch.Tensor] = []
        prediction: torch.Tensor | None = None

        for i, modality in enumerate(self.active):
            _, pooled = self.encoders[modality.value](features[modality])
            logits = squeeze_regression(self.heads[modality.value](pooled), self.spec.task)
            per_modality[modality] = logits
            pooled_all.append(pooled)
            weighted = logits * weights[i]
            prediction = weighted if prediction is None else prediction + weighted

        assert prediction is not None
        return ModelOutput(
            prediction=prediction,
            fused=torch.cat(pooled_all, dim=-1),
            per_modality=per_modality,
            attention={"modality_weights": weights.detach().unsqueeze(0)},
        )
