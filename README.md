# When Fusion Breaks

**Graceful degradation in multimodal emotion recognition**

> **H1 — More sophisticated fusion architectures are more brittle.** Models that learn rich
> cross-modal dependencies should degrade faster under modality corruption than models that
> combine modalities loosely, because they depend on interactions that no longer exist.

Six fusion architectures × seven corruption families × graded severity sweeps × five seeds,
evaluated on CMU-MOSI, CMU-MOSEI, and MELD.

## Headline results

<!-- RESULTS_TABLE_START -->
> **WARNING - these numbers come from SYNTHETIC data, not a real corpus.** They demonstrate that the pipeline runs end to end; they are not results. See [docs/DATA.md](docs/DATA.md) for how to obtain CMU-MOSI/MOSEI, and [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the gate that must pass before any number here is reportable.

| Architecture | Params | Clean acc2_non0 | Mean AUDC | MRS(T) | MRS(A) | MRS(V) | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| **text_only** (*) | 30,689 | 0.735 | 0.976 ± 0.029 | 1.20 | 0.00 | 0.00 | 2 |
| **late** (*) | 73,670 | 0.799 | 0.970 ± 0.024 | 0.37 | 0.22 | 0.09 | 2 |
| **early** (*) | 48,449 | 0.931 | 0.932 ± 0.016 | 0.50 | 0.29 | 0.17 | 2 |
| **lmf** (*) | 78,085 | 0.843 | 0.945 ± 0.058 | 0.44 | 0.29 | 0.10 | 2 |
| **tfn** (*) | 354,225 | 0.836 | 0.963 ± 0.037 | 0.47 | 0.22 | 0.06 | 2 |
| **mult** (*) | 459,979 | 0.908 | 0.943 ± 0.017 | 0.51 | 0.31 | 0.12 | 2 |

**Brittleness index** (clean vs AUDC across architectures): Pearson -0.93, Spearman -1.00 over n=6 architectures. **This says nothing about H1.** On synthetic features the trend is circular: the generator plants a text x audio interaction that the sophisticated architectures exploit for their clean-data advantage, and that same interaction is what corruption removes first. The coefficient confirms the measurement chain works; it is not evidence.

(*) = on the robustness Pareto frontier. AUDC is the area under the chance-corrected retention curve — **higher is more robust**. Feature provenance: `synthetic`.
<!-- RESULTS_TABLE_END -->

## Reproduction gate

Clean-data numbers must land within 1–2 points of published results before making any
degradation claim. See the comparison table in [Reproduction](docs/REPRODUCTION.md).

## Quick start

Install dependencies and run the tests:

```bash
uv sync
uv run pytest
```

Build the feature cache. If the real corpus is unavailable, the command works offline by
falling back to a deterministic synthetic dataset with the same shape contract:

```bash
uv run wfb-data --dataset mosi
```

Train one model and evaluate the full corruption grid:

```bash
uv run wfb-train model=mult data=mosi seed=0
uv run wfb-eval model=mult data=mosi
```

On Windows without `make`, use the bundled task runner:

```powershell
pwsh ./tasks.ps1 test
```

## Repository structure

| Path | Contents |
|---|---|
| `src/wfb/data` | SDK, local, and synthetic loaders; frozen splits; Lightning data module |
| `src/wfb/corruption` | Corruption interface, registry, and audio/text/visual/temporal operators |
| `src/wfb/models` | Unimodal, early fusion, late fusion, TFN, LMF, and MulT models |
| `src/wfb/training` | Training entry point and modality-dropout regularization |
| `src/wfb/evaluation` | Metrics, retention, AUDC, MRS, and significance tests |
| `src/wfb/serving` | FastAPI application, warm model registry, and Pydantic schemas |
| `configs` | Hydra configuration for all hyperparameters |
| `experiments` | Sweep orchestration and committed JSON results |
| `web` | React and TypeScript demo |
| `paper` | LaTeX source and generated figures |

## Documentation

- [Research specification](PROJECT_SPEC.md)
- [Build plan and design decisions](PLAN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data](docs/DATA.md)
- [Corruption protocol](docs/CORRUPTION.md)
- [Reproduction](docs/REPRODUCTION.md)
- [Progress log](docs/PROGRESS.md)

## License

The code is licensed under the MIT License. Datasets retain their original licenses; see
[Data](docs/DATA.md) for details.
