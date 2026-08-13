# Architecture

How the code fits together, and why it is shaped this way.

## The one-way dependency chain

```
types.py            ← the shape contract; imports nothing from wfb
   ↑
data/               ← loaders, synthetic corpus, frozen splits, datamodule
   ↑
corruption/         ← operators + registry + sweep grids (depends only on types)
   ↑
models/             ← six architectures behind one LightningModule interface
   ↑
training/           ← Lightning trainer + modality-dropout protocol
   ↑
evaluation/         ← metrics, degradation summaries, significance, sweep runner
   ↑
reporting/  serving/  experiments/   ← consumers; nothing imports back down
```

Two rules keep this honest:

1. **`src/wfb` never imports Hydra or OmegaConf.** Config parsing lives entirely in
   `cli/config.py`, which converts `DictConfig` into plain dataclasses. Everything below
   is usable from a notebook, a test, or FastAPI with no config framework in the way —
   and `mypy --strict` can actually check it.
2. **`corruption/` does not import `models/` or `data/`.** It operates on bare tensors
   plus a `CorruptionContext`. That is what allows the same operator to run inside a
   `Dataset.__getitem__`, inside an HTTP request handler, and inside a unit test with a
   hand-built tensor.

## The central objects

| Object | Defined in | Role |
|---|---|---|
| `DatasetBundle` | `types.py` | Three splits, train statistics, provenance. The single thing a loader returns, whatever its source. |
| `CorruptionPlan` | `types.py` | An ordered tuple of `CorruptionSpec`. Hashable, JSON-serialisable, usable as a cache key. |
| `SweepAxis` | `corruption/sweeps.py` | One (modality, operator) pair over a severity ladder. AUDC is defined *per axis*. |
| `BaseFusionModel` | `models/base.py` | LightningModule; subclasses implement only `_build` and `forward`. |
| `SweepResult` | `evaluation/runner.py` | Everything one (model, seed, dataset) evaluation produces. |
| `ResultsStore` | `serving/results_store.py` | Reads the committed sweep JSON and aggregates across seeds. |

## Three decisions that shape everything else

### Corruption is applied per-sample with a derived seed

`plan_generator(plan, sample_index, base_seed)` hashes the plan and mixes in the sample
index. Consequences:

* **Every architecture sees bit-identical corrupted inputs** under a given plan. This is
  the precondition for the paired significance tests — without it, part of the measured
  difference between two models would be independent corruption noise.
* Corrupted tensors are never materialised or cached. A 200-plan sweep costs one forward
  pass per plan and no extra memory.
* Reruns reproduce exactly, including on a different machine.

### Train once, evaluate exhaustively

Training is expensive; corruption is applied at evaluation time. The full protocol is
~240 training runs but tens of thousands of evaluation rows, and the code is organised
around that asymmetry: `experiments/run_all.py` trains a cell, then hands the checkpoint
to `run_sweep`, which deduplicates plans across axes (every axis shares the same
severity-0 anchor) and evaluates each unique plan once.

### The loader ends in a synthetic fallback

`load_dataset` tries cache → local archive → CMU-MultimodalSDK → synthetic. Every path
returns the same `DatasetBundle`, and `bundle.provenance.source` records which one ran.
This is what makes CI, the test suite, and the demo runnable on a fresh clone with no
downloads — and `provenance` propagates into every results JSON so a synthetic number can
never be mistaken for a real one. See [DATA.md](DATA.md).

## Adding things

**A corruption operator** — subclass `Corruption`, decorate with `@register`, set
`name`/`applies_to`/`physical_unit`, implement `apply` (exact identity at severity 0). It
is then automatically available to configs, the sweep grid, the HTTP API and
`GET /api/corruptions`. Its identity, shape, determinism and monotonicity tests come for
free through the parametrised suite in `tests/test_corruption.py`.

**An architecture** — subclass `BaseFusionModel`, implement `_build` and `forward`, add
it to `MODEL_REGISTRY` and to `SOPHISTICATION_ORDER` (its position on the fusion axis is
the independent variable for H1), and add a `configs/model/*.yaml`. The parametrised
tests in `tests/test_models.py` will then cover it.

**A dataset** — add `configs/data/*.yaml` and, if it needs a bespoke reader, a branch in
`loaders.py`. Nothing else changes: shapes are read from the tensors, not hardcoded.

## Serving

`InferenceRegistry` loads the dataset and every discoverable checkpoint once at startup.
Requests are cache-checked on `(sample_id, model, corruption_hash)`, then run in a thread
pool so a forward pass cannot block the event loop serving `/ws/live`. Architectures
without a checkpoint are still served, flagged `trained: false` — a demo that 500s on a
fresh clone is worse than one that is honest about what it is showing.
