"""Per-modality temporal encoders shared by every architecture.

Holding the encoder family fixed across architectures is what makes the comparison in H1
a comparison of *fusion*, not of encoders. Only the fusion mechanism is allowed to vary.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to a ``(B, T, H)`` sequence."""

    def __init__(self, hidden: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_len, hidden)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to ``(B, T, H)``."""
        pe = self.pe
        assert isinstance(pe, Tensor)
        out: Tensor = self.dropout(x + pe[:, : x.shape[1], :])
        return out


class TemporalEncoder(nn.Module):
    """Project one modality's features and encode them over time.

    Returns both the full sequence ``(B, T, H)`` and a pooled summary ``(B, H)``; fusion
    mechanisms take whichever they need. ``kind``:

    * ``lstm`` — 1-layer LSTM, the CMU-MOSI/MOSEI convention (TFN, LMF, late fusion all
      use one). Pooled = final hidden state.
    * ``transformer`` — pre-norm encoder stack with sinusoidal positions. Pooled = mean.
    * ``mean`` — projection then mean-pool. A deliberately weak encoder, useful as an
      ablation to check that results are not an artefact of encoder capacity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        kind: str = "lstm",
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        self.input_proj = nn.Linear(input_dim, hidden)
        self.dropout = nn.Dropout(dropout)

        if kind == "lstm":
            self.rnn = nn.LSTM(
                hidden,
                hidden // 2 if bidirectional else hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if layers > 1 else 0.0,
            )
        elif kind == "transformer":
            self.pos = PositionalEncoding(hidden, dropout=dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )
        elif kind != "mean":
            raise ValueError(f"Unknown encoder kind {kind!r}")

        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``(B, T, D)`` into ``((B, T, H), (B, H))``."""
        h = self.dropout(self.input_proj(x))
        if self.kind == "lstm":
            seq, (hn, _) = self.rnn(h)
            pooled = hn[-1] if hn.shape[0] == 1 else torch.cat([hn[-2], hn[-1]], dim=-1)
        elif self.kind == "transformer":
            seq = self.transformer(self.pos(h))
            pooled = seq.mean(dim=1)
        else:
            seq = h
            pooled = h.mean(dim=1)
        return self.norm(seq), self.norm(pooled)


class PredictionHead(nn.Module):
    """Two-layer MLP head producing regression scores or class logits."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map a fused representation to ``(B, output_dim)``."""
        out: Tensor = self.net(x)
        return out
