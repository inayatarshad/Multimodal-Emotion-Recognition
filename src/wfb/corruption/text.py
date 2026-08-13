"""Text-specific corruption operators, chiefly a simulated ASR error channel.

The realistic deployment failure for the text stream is not "text is missing" — it is
"text arrived, via ASR, wrong". So the headline text operator reproduces the three ASR
error types with their usual proportions, at a controllable word error rate.

Everything here works on the *feature* sequence (GloVe vectors, one per aligned word),
because that is what the cached corpora contain. A substitution therefore swaps in
another real word vector drawn from the same utterance rather than an arbitrary random
vector — that keeps the corrupted input on the data manifold, which matters: a model that
merely detects out-of-distribution garbage would look spuriously robust.
"""

from __future__ import annotations

import torch
from torch import Tensor

from wfb.corruption.base import Corruption, CorruptionContext, frame_mask
from wfb.corruption.registry import register
from wfb.types import Modality

# Empirical ASR error-type proportions (substitution-dominated).
_P_SUBSTITUTE = 0.60
_P_DELETE = 0.25  # remaining 0.15 is insertion


@register
class AsrError(Corruption):
    """Simulated ASR transcription errors at a controlled word error rate.

    Severity maps linearly to WER over ``[0, max_wer]`` (default 40%, the protocol's
    range). Each affected token becomes a substitution (60%), a deletion (25%) or an
    insertion (15%). Deletions shift the sequence left and insertions duplicate a token
    and shift right, so the *alignment* between text and the other streams degrades too —
    which is exactly the coupling that cross-modal attention depends on.

    Implemented as an index permutation followed by a single gather, so it is fully
    vectorised over the batch.
    """

    name = "asr_error"
    applies_to = (Modality.TEXT,)
    physical_unit = "WER"

    def __init__(self, max_wer: float = 0.4) -> None:
        super().__init__(max_wer=max_wer)
        self.max_wer = max_wer

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Rewrite the token sequence under a simulated ASR error channel."""
        wer = severity * self.max_wer
        if wer <= 0.0:
            return x

        leading = x.shape[:-2]
        seq_len = x.shape[-2]
        flat = x.reshape(-1, seq_len, x.shape[-1])
        n = flat.shape[0]
        device = flat.device

        affected = ctx.rand(n, seq_len, device=device, dtype=torch.float32) < wer
        kind = ctx.rand(n, seq_len, device=device, dtype=torch.float32)
        is_sub = affected & (kind < _P_SUBSTITUTE)
        is_del = affected & (kind >= _P_SUBSTITUTE) & (kind < _P_SUBSTITUTE + _P_DELETE)
        is_ins = affected & (kind >= _P_SUBSTITUTE + _P_DELETE)

        # Source index for each output copy: substitutions point at a different token of
        # the same utterance (a plausible confusion), everything else points at itself.
        base = torch.arange(seq_len, device=device).expand(n, seq_len).clone()
        replacement = ctx.randint(seq_len, (n, seq_len), device=device)
        source = torch.where(is_sub, replacement, base)

        # How many output slots each input token occupies: 0 deleted, 2 inserted, else 1.
        counts = torch.ones(n, seq_len, dtype=torch.long, device=device)
        counts = counts.masked_fill(is_del, 0).masked_fill(is_ins, 2)
        starts = counts.cumsum(dim=1) - counts

        out_index = torch.full((n, seq_len), -1, dtype=torch.long, device=device)
        for copy in range(2):
            has_copy = counts > copy
            target = starts + copy
            valid = has_copy & (target < seq_len)
            if not bool(valid.any()):
                continue
            rows, cols = valid.nonzero(as_tuple=True)
            out_index[rows, target[rows, cols]] = source[rows, cols]

        gathered = torch.gather(
            flat, 1, out_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        )
        gathered = frame_mask(gathered, out_index >= 0, 0.0)
        return gathered.reshape(*leading, seq_len, x.shape[-1])

    def describe(self, severity: float) -> str:
        """E.g. ``WER 24%``."""
        return f"WER {severity * self.max_wer * 100:.0f}%"


@register
class TokenDropout(Corruption):
    """Drop individual word vectors, keeping their slots (an unrecognised word).

    Contrast with :class:`AsrError` deletions, which close the gap: here the timing is
    preserved and only the content is lost, which isolates *content* loss from
    *alignment* loss.
    """

    name = "token_dropout"
    applies_to = (Modality.TEXT,)
    physical_unit = "% tokens dropped"

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Zero a Bernoulli(severity) subset of tokens."""
        if severity == 0.0:
            return x
        keep = ctx.rand(*x.shape[:-1], device=x.device, dtype=torch.float32) >= severity
        return frame_mask(x, keep, 0.0)

    def describe(self, severity: float) -> str:
        """E.g. ``30% tokens dropped``."""
        return f"{severity * 100:.0f}% tokens dropped"


@register
class WordShuffle(Corruption):
    """Locally permute word order within a sliding window.

    Destroys syntax while preserving the bag of words. If a "sophisticated" model is
    really just doing bag-of-words over text, this operator will barely hurt it — a
    cheap, sharp probe of whether the text encoder uses order at all.
    """

    name = "word_shuffle"
    applies_to = (Modality.TEXT,)
    physical_unit = "window (frames)"

    def __init__(self, max_window: int = 12) -> None:
        super().__init__(max_window=max_window)
        self.max_window = max_window

    def _window(self, severity: float) -> int:
        return round(severity * self.max_window)

    def apply(self, x: Tensor, severity: float, ctx: CorruptionContext) -> Tensor:
        """Shuffle tokens within non-overlapping windows of the severity-derived size."""
        window = self._window(severity)
        if window < 2:
            return x
        leading = x.shape[:-2]
        seq_len = x.shape[-2]
        flat = x.reshape(-1, seq_len, x.shape[-1])
        n = flat.shape[0]

        # Random key per token; sorting the keys within each window yields a permutation.
        keys = ctx.rand(n, seq_len, device=x.device, dtype=torch.float32)
        block = torch.arange(seq_len, device=x.device) // window
        order = torch.argsort(block.expand(n, seq_len) * (seq_len + 1.0) + keys, dim=1)
        gathered = torch.gather(flat, 1, order.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
        return gathered.reshape(*leading, seq_len, x.shape[-1])

    def describe(self, severity: float) -> str:
        """E.g. ``shuffle within 6 words``."""
        window = self._window(severity)
        return "no shuffle" if window < 2 else f"shuffle within {window} words"
