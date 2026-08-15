# Data

## What the code expects

Three aligned feature streams per utterance, each `(T, D)` with a shared `T` (50 by
convention), plus a scalar sentiment label or an integer emotion class.

| Dataset | Task | Text | Audio | Visual | Size |
|---|---|---|---|---|---|
| CMU-MOSI | sentiment, [-3, 3] | GloVe 300 | COVAREP 5 | Facet 20 | 2,199 segments |
| CMU-MOSEI | sentiment, [-3, 3] | GloVe 300 | COVAREP 74 | Facet 35 | ~23,500 segments |
| MELD | 7 emotion classes | — | — | — | ~13,700 utterances |
| synthetic | matches whichever it stands in for | configurable | configurable | configurable | configurable |

Feature dimensionality is **read from the tensors**, never hardcoded in the models — the
published dims above are used only to warn when a local archive does not look like what
it claims to be.

## Getting the real corpora

### Option 1 — the aligned pickles (recommended)

The community has converged on redistributing preprocessed, aligned archives rather than
running CMU-MultimodalSDK. They are what the MulT and MMSA codebases consume, and they
are what everyone's published numbers were actually computed on.

**Where to get it.** The MMSA project redistributes both datasets, preprocessed and
aligned, via [Google Drive][mmsa-drive] (or BaiduYun, code `qq0b`) — see the *Datasets*
section of the [MMSA README][mmsa]. The file this pipeline wants is:

```
MOSI/Processed/aligned_50.pkl     sha256 d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9
MOSEI/Processed/aligned_50.pkl    sha256 45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791
```

Take the **`aligned_50`** variant, not `unaligned_50` — the whole corruption protocol
assumes word-aligned streams, and the published numbers we reproduce against are the
aligned ones. Verify before using:

```bash
sha256sum data/raw/mosi_data.pkl
```

Drop the file into `data/raw/` and the loader finds it. Any of these names work:

```
data/raw/mosi_data.pkl      data/raw/mosi.pkl      data/raw/aligned_mosi.pkl
```

Expected structure — a dict of three splits, each with `text` / `audio` / `vision` /
`labels` arrays and optionally `id`:

```python
{"train": {"text": (N, 50, D_t), "audio": (N, 50, 5),
           "vision": (N, 50, 20), "labels": (N, 1), "id": [...]},
 "valid": {...}, "test": {...}}
```

Key names are matched flexibly (`vision`/`visual`/`video`, `valid`/`val`/`dev`, …), so
most variants of these archives load without edits.

> **`D_t` is 768, not 300, in the MMSA archives.** Their text features are BERT, whereas
> the original SDK distribution — and the TFN/LMF/MulT papers we reproduce against — used
> 300-d GloVe. Nothing breaks (dims are read from the tensors, and the loader logs a
> warning), but it is **not** a like-for-like reproduction: BERT features alone are worth
> a couple of points on MOSI. Record which you used. See
> [REPRODUCTION.md](REPRODUCTION.md).

[mmsa]: https://github.com/thuiar/MMSA
[mmsa-drive]: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

### Option 2 — CMU-MultimodalSDK (currently broken upstream)

```bash
uv pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
uv run wfb-data --dataset mosi
```

This would download the computational sequences, align features to word boundaries, fold
by the standard video-level splits and write the tensor cache.

**As of 2026-08-14 this path does not work, for two independent reasons**, both verified
rather than assumed:

1. The SDK **moved**. `github.com/A2Zadeh/CMU-MultimodalSDK` (the URL in PROJECT_SPEC §3.1
   and in most papers) now 404s; it lives at `CMU-MultiComp-Lab/CMU-MultimodalSDK`.
2. The **feature host is down**. Every `.csd` URL points at
   `immortal.multicomp.cs.cmu.edu`, which resolves to `128.2.211.216` but refuses TCP
   connections (10 s timeout, while GitHub answered in 0.66 s from the same machine).

So there is no way to size this download, because it cannot start. Use Option 1. This is
exactly the fragility the loader's fallback chain was designed for, and it is why
`mmsdk_recipes.py` resolves sequence key names by substring match rather than hardcoding
them — if the host ever returns, drifted key names should not be a second failure.

### Option 3 — nothing (the default)

With no corpus present, `load_dataset` falls back to the synthetic generator and logs a
warning. This is deliberate and is what CI uses.

### Option 3 — nothing (the default)

With no corpus present, `load_dataset` falls back to the synthetic generator and logs a
warning. This is deliberate and is what CI uses.

## The synthetic corpus

`wfb/data/synthetic.py` generates a deterministic dataset with a **known ground truth**:

* label variance is **text-dominant** — text ~60%, audio ~25%, visual ~15% of the linear
  signal, mirroring the asymmetry Q2 asks about on real data;
* a genuine **text × audio interaction** carries a further 18%, recoverable only by a
  model that multiplies or attends across those streams — so tensor and cross-attention
  fusion have a real advantage to find;
* features are temporally smooth, so blur, frame dropout and misalignment have structure
  to destroy.

That known ground truth is what makes the corpus useful beyond "something to run": the
degradation pipeline can be validated against an answer already known. `tests/test_data.py`
asserts that a linear probe recovers the modalities in the planted order, and the MRS
machinery is checked against the same ordering. A broken corruption operator therefore
shows up as a wrong answer to a question with a known right answer, rather than as a
plausible-looking plot.

The envelope carries a DC offset of 1.0. This matters: an earlier zero-mean version made
the planted factor cancel under pooling, so no architecture could extract it and every
baseline sat at chance — which looks exactly like a training bug. `_temporal_profile`
documents this.

**Synthetic numbers are never reportable.** `bundle.provenance.source == "synthetic"`
propagates into every results JSON, every API `/health` response, and the README's
generated table.

## Frozen splits

`wfb/data/splits.py` writes `src/wfb/data/splits/{dataset}.json` — ordered sample ids per
split plus a checksum — and verifies every freshly loaded bundle against it. A mismatch
raises rather than warns: every number in the results tables is comparable only because
the split is identical across runs. Re-freezing a *different* split requires
`overwrite=True`, because doing so silently would invalidate every previously recorded
result.

## Preprocessing notes

* **NaN and infinity are real.** CMU-MOSEI's COVAREP features genuinely contain NaNs and
  occasional ±1e30 sentinels for unvoiced frames. `sanitize()` replaces and clips them.
  Left alone they poison the first backward pass and every statistic computed from them,
  which is one of the more common ways a MOSEI reproduction quietly fails.
* **Padding keeps the tail.** Sequences longer than `T` are truncated from the front and
  shorter ones are zero-padded at the front, matching the convention in this literature
  (the informative part of an utterance tends to be late).
* **Z-scoring uses train statistics only.** COVAREP feature scales differ by orders of
  magnitude; without normalisation the first few dimensions dominate every gradient.
  Using val/test statistics would leak.

## Licences

The code is MIT. The corpora are not: CMU-MOSI and CMU-MOSEI are released under their own
terms via CMU-MultimodalSDK, and MELD under its own licence. Nothing in `data/` is
committed. IEMOCAP is deliberately excluded — its licence request has a turnaround time
outside our control (see PROJECT_SPEC §3.4).
