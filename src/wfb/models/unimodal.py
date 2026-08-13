"""Unimodal baselines — the floor every fusion architecture must beat.

They are also the reference point for the Modality Reliance Score: a fusion model whose
performance under "text removed" collapses to the *audio+visual* unimodal level has
learned nothing that survives without text.
"""

from __future__ import annotations

from wfb.models.base import BaseFusionModel, ModelOutput, squeeze_regression
from wfb.models.encoders import PredictionHead, TemporalEncoder
from wfb.types import ModalityDict


class UnimodalModel(BaseFusionModel):
    """Single-modality encoder plus a prediction head.

    ``cfg.modalities`` selects the modality; the three unimodal baselines are the same
    class with different configs, which guarantees they differ in nothing else.
    """

    def _build(self) -> None:
        if len(self.active) != 1:
            raise ValueError(
                f"UnimodalModel takes exactly one modality, got {[m.value for m in self.active]}"
            )
        self.modality = self.active[0]
        self.encoder = TemporalEncoder(
            input_dim=self.spec.dims[self.modality],
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
        """Encode the single active modality and predict."""
        _, pooled = self.encoder(features[self.modality])
        prediction = squeeze_regression(self.head(pooled), self.spec.task)
        return ModelOutput(
            prediction=prediction, fused=pooled, per_modality={self.modality: prediction}
        )
