# When Fusion Breaks: Graceful Degradation in Multimodal Emotion Recognition

**Project specification & build plan**
Author: Inayat · Started: August 2026 · Target: Erasmus Mundus applications Dec 2026 – Mar 2027
Repo name suggestion: `when-fusion-breaks`

---
You are fable 5 the sota model for anykind of resaerch and coding ! you have to give your best 
## 0. Read this first — the priority order

This document specifies a research project with a production-quality demo. Build it in this order, and if you run out of time, cut from the bottom:

| Priority | Component | Why |
|---|---|---|
| **P0** | Reproducible baselines + the degradation study + results tables | This *is* the project. Everything else is packaging. |
| **P0** | Written report / preprint | The single highest-value artifact for an academic panel. |
| **P1** | Clean repo: tests, CI, configs, README | Evidence of engineering maturity. Reviewers skim this. |
| **P2** | Interactive web demo (backend + frontend) | Makes the work *legible* in 30 seconds. Strong differentiator. |
| **P3** | Live public deployment | Nice. Not decisive for this audience. |

A beautiful frontend with no result is a portfolio piece. A strong result with an ugly notebook is a research contribution. You want both, but never sacrifice the first for the second. The UI in this spec is designed so that it *visualizes the research question itself* — it is not decoration.

---

## 1. The research question

Multimodal models fuse text, audio and video and beat unimodal models on benchmarks. But benchmark evaluation assumes all modalities are present and clean. Real deployments do not get that: video drops, microphones clip, ASR transcripts arrive with 15% word error rate.

**Central hypothesis (H1):** *More sophisticated fusion architectures are more brittle.* Models that learn rich cross-modal dependencies (cross-attention transformers) should degrade faster under modality corruption than models that combine modalities loosely (late fusion), because they have learned to depend on interactions that no longer exist.

This is a good hypothesis for three reasons: it is falsifiable, it is interesting whichever way it resolves, and answering it does not require beating anyone's state-of-the-art number.

**Secondary questions:**

- **Q2 — Reliance asymmetry.** Multimodal models on these benchmarks are widely suspected of being text-dominated. Quantify it: how much of the performance survives when text alone is removed vs. audio and video together?
- **Q3 — Mitigation.** Does modality dropout during training buy test-time robustness, and at what cost to clean-data accuracy? Is the trade-off architecture-dependent?
- **Q4 — Graded vs. binary.** Most missing-modality work studies complete absence. Does *degraded* input (noisy audio, ASR errors, occluded faces) behave like partial absence, or differently?

**Deliverable claim shape:** "We evaluate N fusion architectures across M corruption regimes on 2 datasets and show that [architecture class] retains X% of clean performance at Y corruption while [other class] retains Z%."

That sentence is your abstract. Everything below exists to fill in the letters.

---

## 2. Why this project, for these erasmus mundus programmes that i am targetting for scholarships

| Programme | What it sees |
|---|---|
| **MULTICOM** | Human multimodal communication data science — the core domain. Speech, face, language jointly. |
| **EDISS** | Robustness engineering of data-intensive intelligent systems; a serving stack with real constraints. |
| **DEAI** | Data pipeline over heterogeneous modalities; systematic experimentation infrastructure. |
| **EMLDS** | Representation learning, evaluation design, empirical rigour. |
| **EMAI** | Model architecture comparison and analysis of learned cross-modal structure. |

One project, five defensible motivation letters. Write a different final paragraph for each — the one that says why *that consortium's* curriculum is the next step.

---

## 3. Data

### 3.1 Primary: CMU-MOSEI

- ~23,500 annotated video segments, 1,000+ distinct YouTube speakers, 250+ topics
- Labels: sentiment on [-3, +3] continuous, plus 6 emotion categories
- Access via the **CMU-MultimodalSDK** (`github.com/A2Zadeh/CMU-MultimodalSDK`)
- Precomputed aligned features ship with it:
  - **Text**: GloVe 300-d word vectors
  - **Acoustic**: COVAREP, 74-d (pitch, MFCCs, glottal features, voicing)
  - **Visual**: Facet / OpenFace facial action units, ~35-d

**Use the precomputed features.** At 10–20 hrs/week you cannot afford to spend six weeks on feature extraction. Optional upgrade later: swap GloVe for RoBERTa embeddings, which is what recent papers do and is a cheap accuracy win.

### 3.2 Secondary: MELD

- ~13,700 utterances across ~1,400 dialogues from *Friends*
- 7 emotion classes + 3 sentiment classes, multiparty conversational context
- Easier access, raw video available
- **Role**: cross-dataset generalization. If your degradation findings hold on both, the claim is much stronger. This is what elevates the work from "an experiment" to "a finding."

### 3.3 Optional: CMU-MOSI

2,199 segments. Small and fast. Use it during development for rapid iteration so you are not waiting on MOSEI for every debug cycle.

### 3.4 Deliberately excluded

IEMOCAP requires a license request with turnaround time you cannot control. Skip it. Mention in Future Work.

---

## 4. Models to implement

Five architectures spanning the fusion-sophistication axis. This axis *is* the independent variable for H1.

| # | Model | Fusion type | Reference |
|---|---|---|---|
| 1 | Unimodal (×3) | None — text-only, audio-only, video-only | — |
| 2 | Early fusion | Feature concatenation → Transformer encoder | — |
| 3 | Late fusion | Per-modality encoders → decision averaging | — |
| 4 | **TFN** | Outer-product tensor fusion | Zadeh et al., EMNLP 2017 |
| 5 | **LMF** | Low-rank tensor fusion | Liu et al., ACL 2018 |
| 6 | **MulT** | Directional cross-modal attention | Tsai et al., ACL 2019 |

MulT is the strong baseline and the most "sophisticated" point on the axis. TFN/LMF sit in the middle. Late fusion is the loose-coupling anchor.

**Stretch (only if ahead of schedule):** MISA (Hazarika et al., ACM MM 2020) or Self-MM (Yu et al., AAAI 2021). Do not start these before the degradation study is done.

**Reproduction gate:** before running any degradation experiments, your clean-data numbers must land within ~1–2 points of published results for at least MulT and LMF. If they do not, your degradation curves measure your bugs, not the architectures. Do not skip this gate. Log the comparison in the README as a table.

---

## 5. The degradation protocol — the actual contribution

This is the part nobody else has systematized. Build it as a clean, composable corruption API.

### 5.1 Corruption operators

**Complete modality removal** — three variants, because the choice is not neutral and comparing them is itself a small result:

- `zero`: replace features with zeros
- `mean`: replace with training-set mean
- `mask`: learned mask token (requires training-time support)

**Graded corruption** — sweep each over ~6 severity levels:

| Modality | Corruption | Sweep |
|---|---|---|
| Audio | Additive Gaussian noise | SNR 20 → 0 dB |
| Audio | Feature dropout | 0 → 100% of frames |
| Text | Simulated ASR error (substitute/delete/insert) | WER 0 → 40% |
| Text | Token dropout | 0 → 100% |
| Visual | Frame dropout / occlusion | 0 → 100% |
| Visual | Gaussian blur on AU features | increasing σ |
| All | Temporal misalignment | shift 0 → k frames |

**Combinatorial**: all 7 non-empty subsets of {T, A, V} removed, for every architecture.

### 5.2 Metrics

Standard task metrics first — MOSEI: MAE, Pearson correlation, Acc-2 (binary), Acc-7, weighted F1. MELD: weighted F1, accuracy.

Then define the degradation metrics that are your contribution:

- **Retention(c)** = `metric_corrupted(c) / metric_clean` — the degradation curve
- **AUDC** — area under the retention curve over the severity sweep. One number per (model, corruption) pair. This is your headline table.
- **Critical threshold** — severity at which retention drops below 0.9
- **Modality Reliance Score (MRS)** — for modality *m*: `1 − Retention(remove m)`. Normalize across modalities to expose asymmetry (Q2).
- **Brittleness index** — correlation between a model's clean performance and its AUDC across all corruptions. If H1 is right, this is negative and that is the paper's money plot.

### 5.3 Mitigation arm (Q3)

Retrain every architecture with **modality dropout regularization** — randomly zero an entire modality with p ∈ {0.1, 0.3, 0.5} during training. Report the 2D trade-off: clean performance vs. AUDC. Produce a Pareto plot. Anyone who has deployed a model will find this immediately useful.

### 5.4 Statistical hygiene

Non-negotiable, and it is what separates an A-tier project from a student project:

- **5 random seeds minimum** for every configuration. Report mean ± std. Never a single run.
- Paired significance tests when comparing architectures on the same corrupted samples
- Fixed evaluation splits, committed to the repo
- All hyperparameters in version-controlled config, never in code
- Log every run to W&B or MLflow with the config hash

Compute budget: 6 architectures × 2 datasets × 5 seeds × 4 training regimes ≈ 240 training runs, plus inference-only corruption sweeps (cheap — corruption is applied at eval time, so you train once and evaluate many times). Design your code around that asymmetry: **train once, evaluate exhaustively.** On Colab Pro or a single rented GPU this is very manageable if you cache features to disk as tensors.

---

## 6. Repository architecture

```
when-fusion-breaks/
├── README.md                    # results table above the fold
├── pyproject.toml               # uv or poetry
├── Makefile                     # make data / train / eval / paper / serve
├── .github/workflows/ci.yml     # ruff + mypy + pytest on every push
├── .pre-commit-config.yaml
├── docker-compose.yml
│
├── configs/                     # Hydra
│   ├── config.yaml
│   ├── model/{unimodal,early,late,tfn,lmf,mult}.yaml
│   ├── data/{mosei,mosi,meld}.yaml
│   ├── corruption/{noise,dropout,asr,occlusion,misalign}.yaml
│   └── experiment/               # composed sweeps
│
├── src/wfb/
│   ├── data/
│   │   ├── loaders.py           # SDK wrappers, caching to .pt
│   │   ├── datamodule.py
│   │   └── splits.py            # frozen, committed
│   ├── corruption/
│   │   ├── base.py              # Corruption ABC: apply(features, severity)
│   │   ├── audio.py
│   │   ├── text.py
│   │   ├── visual.py
│   │   └── registry.py          # name -> class, composable pipelines
│   ├── models/
│   │   ├── base.py              # LightningModule interface
│   │   ├── unimodal.py
│   │   ├── fusion_simple.py     # early, late
│   │   ├── tfn.py
│   │   ├── lmf.py
│   │   └── mult.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── modality_dropout.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── degradation.py       # AUDC, MRS, retention curves
│   │   └── significance.py
│   └── serving/
│       ├── app.py               # FastAPI
│       ├── inference.py         # model registry, warm cache
│       └── schemas.py           # Pydantic
│
├── experiments/
│   ├── run_all.py
│   └── results/                 # committed JSON — reviewers can verify
│
├── tests/                       # pytest; corruption ops especially
├── notebooks/                   # analysis + figure generation only
├── paper/                       # LaTeX + generated figures
└── web/                         # frontend (Section 8)
```

**Rules for Claude Code:** PyTorch Lightning for training loops. Hydra for config. Type hints everywhere, `mypy --strict` on `src/`. Every corruption operator gets a unit test asserting the severity-0 case is an identity transform — this catches an entire class of silent bug.

---

## 7. Backend

**Stack:** FastAPI + Pydantic v2 + PyTorch + uvicorn, Docker, optional Redis for result caching.

### Endpoints

```
GET  /health
GET  /api/models                     # registry: name, arch, params, clean metrics
GET  /api/samples?dataset=&limit=    # curated demo clips with media URLs
POST /api/predict                    # single inference under a corruption config
POST /api/compare                    # same input, all models, side by side
GET  /api/results/degradation        # precomputed curves for the explorer
GET  /api/results/reliance           # MRS matrices
WS   /ws/live                        # streaming updates as sliders move
```

`POST /api/predict` request:

```jsonc
{
  "sample_id": "mosei_a1b2c3",
  "model": "mult",
  "corruption": {
    "text":   { "type": "asr_error",     "severity": 0.15 },
    "audio":  { "type": "gaussian_noise","severity": 0.40 },
    "visual": { "type": "none",          "severity": 0.0  }
  },
  "return_attention": true
}
```

Response carries prediction, per-class confidences, delta vs. clean prediction, per-modality contribution estimate, cross-modal attention maps, and latency.

### Engineering requirements

- Models loaded once at startup into a registry; never per-request
- Async endpoints; run inference in a thread pool so the event loop stays free
- Redis-cached responses keyed by `(sample_id, model, corruption_hash)` — the demo is highly repetitive, so cache hit rates will be excellent
- Pydantic validation on every boundary, structured JSON logging, request IDs
- Rate limiting, CORS locked to the frontend origin
- OpenAPI docs auto-served at `/docs` — link this from the README

---

## 8. Frontend

**Stack:** React 18 + TypeScript + Vite, TailwindCSS, shadcn/ui, Recharts (or visx) for plots, Framer Motion for transitions, TanStack Query for server state.

**Design direction:** dark, restrained, scientific. Think a well-made research lab tool, not a SaaS landing page. One accent colour. Generous whitespace. Typography does the work — Inter for UI, JetBrains Mono for numbers. Every number that can move should animate when it changes; that motion is what makes degradation *felt* rather than read.

### The four views

**1. Live Degradation Explorer — the hero.**

Video clip on the left. Three corruption sliders on the right, one per modality, each labelled with its corruption type. As you drag, the prediction updates live over the WebSocket. Alongside: a persistent ghost marker showing the clean prediction so the drift is visible, and a running "confidence collapse" readout.

Below, a small multiple of all six architectures responding to the *same* corruption simultaneously. Watching MulT fall off a cliff while late fusion holds steady — if H1 is right — is the entire paper in one interaction. Build this view first; it is the thing a professor will remember.

**2. Results Dashboard.** The degradation curves, faceted by corruption type, with confidence bands from the 5 seeds. Toggle architectures on and off. Hover for exact values. Export any panel to SVG for the paper.

**3. Modality Reliance Matrix.** Heatmap of MRS by (architecture × modality), with the 7-subset removal grid. This is where Q2's text-dominance finding lands.

**4. Robustness Pareto.** Clean performance vs. AUDC scatter, points sized by parameter count, with the Pareto frontier drawn. Modality-dropout-trained variants shown as linked pairs with their untrained counterparts, so the trade-off is a visible arrow.

### Non-negotiables

- Responsive down to tablet; degrade the hero view gracefully on mobile
- Keyboard accessible, ARIA labels on all controls, respects `prefers-reduced-motion`
- Skeleton loaders — never a spinner on a blank page
- Every chart readable in greyscale (colourblind-safe palette; do not encode meaning in hue alone)
- Lighthouse ≥ 90 on performance and accessibility


---

## 10. Outputs

1. **Preprint** — arXiv (cs.CL / cs.LG). Note: first-time arXiv submitters need endorsement in cs categories. Start asking early. Fallback: a well-typeset PDF technical report in the repo, linked from the README. It retains most of the CV value.
2. **Target venue** — a workshop at **ICMI** (International Conference on Multimodal Interaction), the natural home for this work. Naming ICMI in a MULTICOM motivation letter is a strong signal that you know the field.
3. **Repo** — README opens with the headline results table and a demo GIF. Reviewers spend 90 seconds. Spend them well.
4. **Demo** — deploy to Hugging Face Spaces (free, and the ML-native choice) or Fly.io.
5. **Two-page project summary** — a PDF you attach to applications. Not the paper; the executive version, with the money plot.

---

## 11. What makes this A-tier rather than good

Most student ML projects fail on the same five things. Get these right and you are in a different bracket:

1. **A falsifiable hypothesis, stated before the experiments.** Not "I explored multimodal fusion." H1 is a claim that could turn out false.
2. **Reproduction of published baselines before novel claims.** This single practice is what most portfolio projects skip and what every reviewer checks for.
3. **Multiple seeds with error bars.** A single-run bar chart signals inexperience faster than anything else in ML.
4. **Cross-dataset validation.** A finding on one benchmark is an observation. On two, it starts being a result.
5. **Negative results reported honestly.** If H1 is false — if sophisticated fusion turns out *more* robust — say so clearly and explain why. Panels select for scientific honesty, and a clean disconfirmation is a genuinely more interesting paper than a mild confirmation.

---

## 12. Working solo — mitigations

No supervisor is your weakest flank. Three things to do in August, not later:

- **Email 3 professors** whose papers you build on (Tsai, Zadeh, Hazarika, Poria and their students are all reachable). Not asking for supervision — asking one specific, well-posed question about their method. A researcher who has replied to you twice writes a materially different reference letter than a lecturer who graded you.
- **Post progress publicly.** A short write-up at the halfway point invites feedback and creates a timestamped record of the work.
- **Get one code review.** Post the repo somewhere a stranger will critique it before you submit anywhere.

And keep the proportions honest: your transcript carries roughly 60 of the ~100 available points; CV and experience roughly 20; the motivation letter roughly 15. This project is competing for space inside that 20. It is worth doing well, and it is not worth doing at the expense of grades or of five genuinely tailored letters.

---

## 13. First session in Claude Code

Paste this to start:

> Read `PROJECT_SPEC.md`. Scaffold the repository exactly as specified in Section 6, using `uv` for dependency management. Set up: pyproject.toml with PyTorch, Lightning, Hydra, and dev dependencies; pre-commit with ruff and mypy; a GitHub Actions CI workflow running lint, typecheck and pytest; and a Makefile with `data`, `train`, `eval`, `serve` targets. Then implement `src/wfb/data/loaders.py` to download CMU-MOSI via CMU-MultimodalSDK and cache aligned features to disk as tensors, with a unit test verifying shapes and the frozen split. Do not implement models yet. Show me the plan before writing files.

Start with MOSI, not MOSEI. It is ten times smaller and you will find every bug in a tenth of the time.
