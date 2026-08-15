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

## 2026-08-14 — Session 2: first multi-seed run, and the data route

**The corpus is not obtainable by the route the spec assumes.** Two independent findings,
both verified rather than inferred:

1. CMU-MultimodalSDK **moved**: `github.com/A2Zadeh/CMU-MultimodalSDK` (the URL in
   PROJECT_SPEC §3.1, in this repo's docs, and in most papers) now 404s. It lives at
   `CMU-MultiComp-Lab/CMU-MultimodalSDK`.
2. The feature host is **down**. Every `.csd` URL points at
   `immortal.multicomp.cs.cmu.edu`, which resolves (`128.2.211.216`) but refuses TCP
   connections — 10 s timeout, while GitHub answered in 0.66 s from the same machine.

So the SDK path cannot download anything, and there is no download size to quote for it.
The working route is MMSA's redistributed `aligned_50.pkl` via Google Drive, now
documented with checksums in [DATA.md](DATA.md).

**Consequence for the gate:** those archives ship **BERT** text features (768-d), not the
**GloVe** 300-d that TFN/LMF/MulT used. That is worth 1–3 Acc-2 points on MOSI — the
entire width of the reproduction gate — and it would distort the comparison invisibly,
since both are "MOSI Acc-2". [REPRODUCTION.md](REPRODUCTION.md) now carries this trap and
two honest ways around it.

**Ran the `dev` preset**: 6 architectures × 2 seeds, full 32-axis grid, 12/12 cells in
48 min on synthetic data. It found three bugs that a single-model, single-seed run
structurally could not:

1. **The resume check ignored the corruption grid.** `run_cell` skipped on
   `(model, dataset, seed, tag)`, so two cached *smoke* results (5 axes, 3-point ladder)
   were silently accepted as `dev` seed-0 results (32 axes, 6-point ladder). It surfaced
   as an opaque numpy ragged-array crash, but **the crash was luck** — with equal-length
   ladders it would have averaged incomparable curves into a seed band and reported it as
   a result. Runs now record a `grid_signature`, and resuming requires it to match.
2. **Aggregation crashed instead of reporting.** `degradation_curves` now groups by
   severity ladder, keeps the majority, and warns naming what it excluded.
3. **Seeds never aggregated for any model with an underscore in its name.** The run label
   was parsed positionally from the filename, so `mosi_text_only_s0_sweep` yielded the tag
   `s0` and the label `text_only+s0`. All three unimodal baselines appeared once per seed
   as separate architectures, with a duplicate headline row each. Tags are now recorded
   explicitly and, for older files, recovered by stripping the known prefix.

Each bug has regression tests (`tests/test_results_store.py`, 14 tests). Suite is now 324
passing, ruff and `mypy --strict` clean.

After the fixes, a clean re-run gave 12/12 cells on one grid, 2 seeds per architecture,
and all four figures (fig3 needs ≥3 architectures, so it had never rendered before).

**The headline number is circular, and the code now says so.** Brittleness came out at
Spearman **−1.00**, Pearson −0.93 — suspiciously perfect, and that is the tell. The
synthetic generator plants a text×audio interaction worth 18% of label variance that only
multiplicative and attention fusion can capture; those architectures therefore lead on
clean data *and* lose the most when corruption destroys that interaction. The negative
index is a property of the generator, not evidence about fusion. `_h1_verdict` is now
provenance-aware and refuses the verdict outright on synthetic features rather than
printing "consistent with H1" under it, and every figure carries a red
`SYNTHETIC DATA — not a result` stamp, because figures get pasted into slides where the
README's banner does not follow them.

Two further findings from reading the figures:

- **`audio.gaussian_noise` barely bites** — retention stays near 1.0 at SNR 0 dB. Not an
  operator bug: independent noise across 50 frames × 300 dims averages out before it
  reaches the classifier. It matters because mean AUDC averages over axes, so a dead axis
  contributes ≈1.0 for everyone and dilutes the between-model signal. Documented in
  [CORRUPTION.md](CORRUPTION.md); check it on real features before reporting pooled AUDC.
- Every architecture landed on the Pareto frontier, because the clean/AUDC trade-off came
  out perfectly monotone. Correct behaviour, useless marker — expect this to resolve on
  real data.

Also fixed two figure-layout bugs found by actually looking at the output: the provenance
stamp collided with the x-axis label, and in fig1 the top row's axis labels overlapped the
bottom row's panel titles (`bbox="tight"` trims the outer margin but does nothing between
subplots).

### Session 2b — the demo, driven for real

The web demo had been built, type-checked and unit-tested, but **never run against a live
API with trained checkpoints**. Doing that surfaced four more bugs, none of which any test
would have caught:

1. **Lightning silently versioned checkpoints.** `ModelCheckpoint(filename="best")` wrote
   `best-v1.ckpt` rather than overwriting, so a *stale smoke* checkpoint (`hidden=24`)
   kept the canonical `best.ckpt` name while the real dev checkpoint (`hidden=32`) sat
   beside it. The serving registry loaded the stale one and failed on a shape mismatch —
   2 of 6 architectures showed as untrained. Fixed with `enable_version_counter=False`,
   and the registry now trusts the path recorded in `train_result.json` instead of
   guessing a filename.
2. **The disk cache laundered provenance.** `load_cache` overwrote `source` with
   `"cache"`, so synthetic features that had passed through the cache came back claiming
   otherwise: `is_synthetic` went False, the UI's synthetic badge never rendered, and any
   results written from a cached synthetic corpus would have **lost the warning banner
   entirely**. This is a hole straight through the one guarantee the provenance system
   exists to provide. The original source now survives; the cache is recorded in `detail`.
3. **The WebSocket status indicator lied.** The first request races the handshake, takes
   the REST fallback, and its `setStatus('polling')` resolved *after* `onopen` had set
   `'live'` — so the badge claimed polling forever while traffic actually went over the
   socket. Also stopped closing a `CONNECTING` socket during StrictMode's double mount.
4. **The frontend carried a duplicate H1 verdict that missed the guard.** `_h1_verdict`
   had been made provenance-aware in Python, but `ParetoView.tsx` had its own copy, so
   the demo still announced "Consistent with H1" over synthetic data — in the single most
   looked-at artifact of the project. Now mirrors the Python rule, with a comment saying
   so.

Verified live in a browser: all 6 architectures trained, WebSocket connected, corruption
rendering in physical units (`text:WER 24% + audio:SNR 4.4 dB + visual:60% occluded`),
all six models diverging under identical input (deltas −0.12 to −0.75, 18–163 ms), and
all four views rendering with no console errors.

The reliance view validates itself nicely: `text_only` shows retention **1.00** when audio
or visual is removed (it genuinely ignores them) and **−0.20** when text goes; late fusion
shows `−AV` (0.76) hurting *less* than `−T` (0.63), which is Q2's text-dominance signature
appearing unprompted.

## Next

**Immediate (Weeks 3–6 in the spec's plan)**

1. Download `MOSI/Processed/aligned_50.pkl` from the MMSA Google Drive (see
   [DATA.md](DATA.md) for the link and checksum), save it as `data/raw/mosi_data.pkl`,
   then run `uv run wfb-data --dataset mosi --no-synthetic`. **Not** the SDK route —
   its host is unreachable.
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
