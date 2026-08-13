# Build Plan — `when-fusion-breaks`

Derived from [PROJECT_SPEC.md](PROJECT_SPEC.md). This file is the execution plan;
[docs/PROGRESS.md](docs/PROGRESS.md) is the running log of what is actually done.

## Guiding constraints

1. **Train once, evaluate exhaustively.** Corruption is applied at eval time to cached
   feature tensors. A single trained checkpoint feeds hundreds of evaluation rows.
2. **Nothing in the pipeline may require the network to be testable.** CMU-MultimodalSDK
   is fragile and dataset mirrors rot. Every layer above `data/loaders.py` is written
   against a shape contract, and a deterministic synthetic dataset satisfying that
   contract ships in-repo so CI, tests, and the demo run offline.
3. **Severity is normalised to `[0, 1]` for every corruption operator**, and every operator
   is an exact identity at `severity == 0`. This is unit-tested for all operators —
   it kills a whole class of silent bugs where a "clean" baseline is quietly corrupted.
4. **No hyperparameter lives in code.** Hydra configs only.
5. **Five seeds minimum**, mean ± std everywhere, paired tests when comparing architectures.

## Phases

| Phase | Content | Status gate |
|---|---|---|
| **P1** | Repo scaffold: `pyproject.toml` (uv), ruff + mypy strict, pre-commit, GitHub Actions CI, Makefile + `tasks.ps1` | `uv run pytest` green, `ruff check` + `mypy --strict src` clean |
| **P2** | Data layer: `loaders.py` (real MOSI/MOSEI/MELD + synthetic fallback), frozen splits committed, `datamodule.py` | `make data` end-to-end; shape + split unit tests pass |
| **P3** | Corruption framework: ABC, registry, audio/text/visual/temporal ops, composable pipelines | identity-at-0 test for every registered op |
| **P4** | Models: unimodal ×3, early, late, TFN, LMF, MulT — all `LightningModule` | forward/shape tests + 1-step overfit test per arch |
| **P5** | Training + evaluation: trainer, modality dropout, metrics, degradation (AUDC / MRS / retention / critical threshold), significance | `make train` and `make eval` produce JSON in `experiments/results/` |
| **P6** | Experiment orchestration: `experiments/run_all.py` — the full grid, resumable, per-run config hash | full sweep runs on synthetic data end-to-end |
| **P7** | Backend: FastAPI, model registry warm at startup, threadpool inference, Redis-optional cache, WS live endpoint | `pytest tests/test_api.py` green; `/docs` serves |
| **P8** | Frontend: React 18 + TS + Vite + Tailwind + Recharts — 4 views | `npm run build` clean, typecheck clean |
| **P9** | Paper skeleton + figure generation notebooks + README results table | figures regenerate from committed JSON |

## Key design decisions (and why)

### D1 — Data source strategy is a chain of fallbacks
`loaders.py` resolves a dataset in this order:

1. **Cache hit** — `data/processed/{name}.pt` already exists → load it.
2. **Local raw** — a `.pkl`/`.h5`/`.csd` in `data/raw/` (the MMSA/MulT-format aligned
   pickles are the de-facto standard artefact and what everyone actually uses).
3. **CMU-MultimodalSDK** — `mmsdk` computational sequences, aligned to labels.
4. **Synthetic** — deterministic, seeded, correct shapes and a genuine (learnable)
   multimodal signal so the whole stack is exercisable without any download.

The loader emits an identical `DatasetBundle` regardless of source, so nothing downstream
knows or cares which path was taken. `bundle.provenance` records which one was used, and
every results JSON carries it — so a synthetic-data number can never be mistaken for a real one.

### D2 — Synthetic data is *designed*, not noise
The synthetic generator plants a known ground truth: a text-dominant signal (~60% of label
variance), audio (~25%), visual (~15%), plus a genuine T×A interaction term that only
tensor/attention fusion can capture. This means the *sanity* of the whole degradation
pipeline is checkable — we know a priori what MRS should look like, so a broken corruption
operator shows up as a wrong answer to a question we already know the answer to.

### D3 — Corruption operates on cached features, not raw media
The spec's features are precomputed (GloVe / COVAREP / Facet). So "additive Gaussian noise
at SNR X" is applied in *feature space*, calibrated per-feature against the training-set
variance. This is documented explicitly in the paper as a limitation and a deliberate
choice: it makes the sweep three orders of magnitude cheaper and keeps every architecture
seeing exactly the same corrupted tensor.

### D4 — Metrics honour the MOSI/MOSEI convention split
Acc-2 is reported both ways (`has0` including neutral, `non0` excluding it) because the
literature is inconsistent and the reproduction gate depends on comparing like with like.

## Cut order if time runs short
Pareto arm → MELD → frontend views 3–4 → deployment. Never the reproduction gate,
never the multi-seed protocol.
