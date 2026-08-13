## Headline results

> **WARNING - these numbers come from SYNTHETIC data, not a real corpus.** They demonstrate that the pipeline runs end to end; they are not results. See [docs/DATA.md](docs/DATA.md) for how to obtain CMU-MOSI/MOSEI, and [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the gate that must pass before any number here is reportable.

| Architecture | Params | Clean acc2_non0 | Mean AUDC | MRS(T) | MRS(A) | MRS(V) | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| **late** | 56,982 | 0.824 | 0.985 | 0.39 | 0.24 | 0.23 | 1 |
| **mult** (*) | 223,069 | 0.908 | 1.003 | 0.50 | 0.31 | 0.12 | 1 |

**Brittleness index** (clean vs AUDC across architectures): Pearson —, Spearman — over n=2 architectures. Too few architectures to read a trend.

(*) = on the robustness Pareto frontier. AUDC is the area under the chance-corrected retention curve — **higher is more robust**. Feature provenance: `synthetic`.


## Modality reliance — the 7-subset removal grid

| Architecture | −T | −A | −V | −TA | −TV | −AV | −TAV |
|---|---|---|---|---|---|---|---|
| **late** | 0.61 | 0.76 | 0.77 | — | — | — | — |
| **mult** | 0.50 | 0.69 | 0.88 | — | — | — | — |

Values are retention of clean skill above chance. `−T` means text was removed. If `−AV` > `−T`, the model is text-dominated (Q2).


## AUDC by corruption axis

| Corruption axis | late | mult |
|---|---|---|
| `audio.gaussian_noise` | 1.00 | 1.01 |
| `text.asr_error` | 0.97 | 1.00 |


## Mitigation: modality dropout

_Mitigation arm not run yet._
