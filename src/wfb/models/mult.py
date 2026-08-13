"""Multimodal Transformer — MulT (Tsai et al., ACL 2019).

The sophisticated end of the axis, and the model H1 predicts will break first.

Architecture, faithfully: each modality is projected by a temporal 1-D convolution, then
for every ordered pair ``(target, source)`` a **directional crossmodal transformer** lets
the target attend to the source — six of them for three modalities. The two crossmodal
streams for a given target are concatenated and passed through a self-attention
transformer; the final positions of all three targets are concatenated for the head.

Why this is the brittleness candidate: the target streams are *queries onto other
modalities*. When a source is zeroed, the corresponding attention output does not merely
lose information — it becomes attention over a constant, whose value the downstream
layers have never seen during training. Nothing in the architecture degrades gracefully
by construction; robustness would have to be learned. Whether it is, is the experiment.

Attention weights are returned from every crossmodal block, which is what the demo's
explanation panel draws.
"""

from __future__ import annotations

from itertools import permutations

import torch
from torch import Tensor, nn

from wfb.models.base import BaseFusionModel, ModelOutput, squeeze_regression
from wfb.models.encoders import PositionalEncoding, PredictionHead
from wfb.types import Modality, ModalityDict


class CrossmodalAttentionLayer(nn.Module):
    """One pre-norm crossmodal attention block: target queries, source keys/values."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: Tensor, source: Tensor) -> tuple[Tensor, Tensor]:
        """Attend from ``target`` to ``source``; return the update and its weights."""
        query = self.norm_q(target)
        key = self.norm_kv(source)
        attended, weights = self.attention(query, key, key, need_weights=True)
        hidden = target + self.dropout(attended)
        hidden = hidden + self.dropout(self.feedforward(self.norm_ff(hidden)))
        return hidden, weights


class CrossmodalTransformer(nn.Module):
    """A stack of :class:`CrossmodalAttentionLayer` sharing one source sequence."""

    def __init__(self, hidden: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossmodalAttentionLayer(hidden, heads, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, target: Tensor, source: Tensor) -> tuple[Tensor, Tensor]:
        """Run the stack; the returned weights come from the final layer."""
        weights = target.new_zeros(target.shape[0], target.shape[1], source.shape[1])
        for layer in self.layers:
            target, weights = layer(target, source)
        return self.norm(target), weights


class MultimodalTransformerModel(BaseFusionModel):
    """Directional crossmodal attention across all ordered modality pairs."""

    def _build(self) -> None:
        hidden = self.cfg.hidden
        self.temporal_conv = nn.ModuleDict(
            {
                m.value: nn.Conv1d(self.spec.dims[m], hidden, kernel_size=3, padding=1, bias=False)
                for m in self.active
            }
        )
        self.positional = PositionalEncoding(hidden, dropout=self.cfg.dropout)

        self.pairs = [(t, s) for t, s in permutations(self.active, 2)]
        self.crossmodal = nn.ModuleDict(
            {
                _pair_key(target, source): CrossmodalTransformer(
                    hidden, self.cfg.heads, self.cfg.layers, self.cfg.dropout
                )
                for target, source in self.pairs
            }
        )

        # Per-target self-attention over the concatenated crossmodal streams.
        sources_per_target = max(len(self.active) - 1, 1)
        self.stream_dim = hidden * sources_per_target
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.stream_dim,
            nhead=self.cfg.heads,
            dim_feedforward=self.stream_dim * 4,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.self_attention = nn.ModuleDict(
            {
                m.value: nn.TransformerEncoder(
                    encoder_layer, num_layers=self.cfg.layers, enable_nested_tensor=False
                )
                for m in self.active
            }
        )
        self.head = PredictionHead(
            self.stream_dim * len(self.active),
            self.cfg.fusion_hidden,
            self.spec.output_dim,
            self.cfg.dropout,
        )

    def _project(self, features: ModalityDict) -> dict[Modality, Tensor]:
        projected: dict[Modality, Tensor] = {}
        for modality in self.active:
            x = features[modality].transpose(1, 2)  # (B, D, T) for Conv1d
            convolved = self.temporal_conv[modality.value](x).transpose(1, 2)
            projected[modality] = self.positional(convolved)
        return projected

    def forward(self, features: ModalityDict) -> ModelOutput:
        """Six directional attention passes, per-target self-attention, then readout."""
        projected = self._project(features)
        attention_maps: dict[str, Tensor] = {}
        target_states: list[Tensor] = []

        for target in self.active:
            streams: list[Tensor] = []
            for source in self.active:
                if source is target:
                    continue
                key = _pair_key(target, source)
                attended, weights = self.crossmodal[key](projected[target], projected[source])
                streams.append(attended)
                attention_maps[key] = weights.detach()
            if not streams:  # single-modality degenerate case
                streams = [projected[target]]
            combined = torch.cat(streams, dim=-1)
            encoded = self.self_attention[target.value](combined)
            target_states.append(encoded[:, -1, :])  # final position, as in the paper

        fused = torch.cat(target_states, dim=-1)
        return ModelOutput(
            prediction=squeeze_regression(self.head(fused), self.spec.task),
            fused=fused,
            attention=attention_maps,
        )


def _pair_key(target: Modality, source: Modality) -> str:
    """``text<-audio`` style key for the crossmodal module dict and the demo payload."""
    return f"{target.value}<-{source.value}"
