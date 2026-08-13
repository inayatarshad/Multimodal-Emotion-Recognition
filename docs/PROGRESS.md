# Progress log

Running record of what is built and what is next. Timeline reference: PROJECT_SPEC §9.

## 2026-08-13 — Session 1: scaffold through full pipeline

Delivered the whole Weeks 1–2 goal (`make data` clean end to end, CI green) plus most of
the infrastructure the later weeks depend on.

**Built**

- Repo scaffold: `uv` + `pyproject.toml`, ruff, `mypy --strict`, pre-commit, GitHub
  Actions CI (lint / typecheck / 4-way test matrix / end-to-end smoke / web build),
  `Makefile` and a `tasks.ps1` mirror for Windows.
- Data layer: fallback chain (cache → local archive → CMU-MultimodalSDK → synthetic),
  frozen split manifests with checksum verification, atomic tensor cache, NaN/Inf
  sanitisation, Lightning datamodule.
- Corruption framework: 14 operators across audio/text/visual/temporal + the three
  removal variants, a name registry, composable plans, and the standard grid (32 axes,
  77 unique plans).
- All six architectures: unimodal ×3, early, late, TFN, LMF, MulT.
- Training: Lightning trainer, modality-dropout regularisation with learned mask tokens.
- Evaluation: MOSI/MOSEI + MELD metrics, retention/AUDC/MRS/critical-threshold/
  brittleness, paired bootstrap + Wilcoxon + Holm–Bonferroni, the sweep runner.
- Serving: FastAPI with warm registry, threadpool inference, Redis-or-memory cache,
  WebSocket live endpoint, full OpenAPI.
- Reporting: markdown tables generated from the results JSON (auto-injected into the
  README) and four colourblind-safe paper figures.
- Orchestration: `experiments/run_all.py` with five presets, resumable.
- 310 tests passing (1 skipped: the `shift` operator has no continuous severity response,
  so the monotone-damage check does not apply to it); ruff and `mypy --strict` clean;
  `npm run build` and `tsc --noEmit` clean.

**Bugs found and fixed during the session** (each one is worth remembering)

1. *The synthetic corpus was unlearnable.* Its temporal envelope was zero-mean with a
   random per-sample phase, so the planted latent factor cancelled under any pooling and
   was not recoverable by a fixed linear map either. Every architecture sat at chance,
   which looks exactly like a training bug. Fixed by giving the envelope a DC offset;
   a linear probe now recovers text 0.65 / audio 0.43 / visual 0.34, matching the planted
   order. `tests/test_data.py` now asserts this, so it cannot regress.
2. *Retention was not chance-corrected.* A text-only model with text removed scored
   MRS 0.23 — it should be ~1.0. Binary accuracy floors at 0.5, so uncorrected retention
   bottoms out around `0.5/clean`. Every reliance and AUDC number was compressed into the
   top third of its range. Now measured as skill above chance, with the majority-class
   rate used for imbalanced classification corpora.
3. *NumPy 2 removed `ndarray.ptp()`* — the Pareto figure crashed on it.
4. Two of my own test assertions were wrong rather than the code: Holm–Bonferroni's
   largest p-value is tested against α itself (not α/m), and MulT's attention weights do
   not sum to 1 in train mode because dropout is applied to them.

**Verified end to end on synthetic data** — models order as expected
(unimodal < late < TFN/LMF < MulT/early), and the recovered modality reliance
(T .62 / A .29 / V .09) matches the planted ground truth (.60/.25/.15).

## Next

**Immediate (Weeks 3–6 in the spec's plan)**

1. Obtain the real MOSI archive and drop it in `data/raw/`; run
   `uv run wfb-data --dataset mosi --no-synthetic`.
2. Run `--preset main` and fill in [REPRODUCTION.md](REPRODUCTION.md). **Nothing
   downstream is reportable until that gate passes.**
3. Then MOSEI, then MELD for cross-dataset generalisation.

**After the gate**

4. Full degradation sweep, 5 seeds — the headline table and the money plot.
5. Mitigation arm (`--preset mitigation`) and the Pareto analysis.
6. Paper draft — `paper/` skeleton is in place and figures regenerate from the JSON.

**Not yet started**

- Emailing the three researchers (spec §12 — it says August, not later).
- arXiv endorsement request for cs.CL.

## Standing constraints

- Never report a single-seed number.
- Never let a synthetic-provenance number reach a table that reads as real.
- The reproduction gate precedes every degradation claim.
- If H1 turns out false, that is reported plainly — a clean disconfirmation is a more
  interesting paper than a mild confirmation, and the reporting code already states the
  verdict either way.
