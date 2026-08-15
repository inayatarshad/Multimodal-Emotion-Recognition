## Headline results

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


## Modality reliance — the 7-subset removal grid

| Architecture | −T | −A | −V | −TA | −TV | −AV | −TAV |
|---|---|---|---|---|---|---|---|
| **text_only** | -0.20 | 1.00 | 1.00 | -0.20 | -0.20 | 1.00 | -0.20 |
| **late** | 0.63 | 0.78 | 0.91 | 0.24 | 0.56 | 0.76 | -0.16 |
| **early** | 0.50 | 0.71 | 0.83 | 0.21 | 0.39 | 0.59 | -0.11 |
| **lmf** | 0.56 | 0.71 | 0.90 | 0.23 | 0.47 | 0.67 | -0.14 |
| **tfn** | 0.53 | 0.78 | 0.94 | 0.28 | 0.44 | 0.72 | -0.01 |
| **mult** | 0.49 | 0.69 | 0.88 | 0.26 | 0.39 | 0.66 | 0.12 |

Values are retention of clean skill above chance. `−T` means text was removed. If `−AV` > `−T`, the model is text-dominated (Q2).


## AUDC by corruption axis

| Corruption axis | text_only | late | early | lmf | tfn | mult |
|---|---|---|---|---|---|---|
| `all.misalign` | 1.00 | 0.99 ± 0.01 | 0.98 ± 0.01 | 0.99 ± 0.02 | 1.02 ± 0.02 | 1.00 ± 0.01 |
| `audio.burst_dropout` | 1.00 | 0.93 ± 0.02 | 0.87 ± 0.01 | 0.94 ± 0.01 | 0.95 ± 0.03 | 0.87 ± 0.02 |
| `audio.clipping` | 1.00 | 0.92 ± 0.01 | 0.89 ± 0.00 | 0.92 ± 0.02 | 0.93 ± 0.04 | 0.89 ± 0.00 |
| `audio.frame_dropout` | 1.00 | 0.92 ± 0.01 | 0.87 ± 0.00 | 0.89 ± 0.01 | 0.92 ± 0.04 | 0.87 ± 0.01 |
| `audio.gaussian_noise` | 1.00 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.01 | 1.00 ± 0.00 | 0.99 ± 0.01 |
| `text.asr_error` | 0.95 ± 0.02 | 1.01 ± 0.01 | 0.99 ± 0.01 | 0.91 ± 0.06 | 0.96 ± 0.01 | 1.00 ± 0.01 |
| `text.token_dropout` | 0.77 ± 0.03 | 0.93 ± 0.02 | 0.84 ± 0.02 | 0.83 ± 0.04 | 0.86 ± 0.01 | 0.84 ± 0.01 |
| `text.word_shuffle` | 1.02 ± 0.01 | 1.02 ± 0.01 | 1.00 ± 0.00 | 0.99 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.01 |
| `visual.blur` | 1.00 | 0.99 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.01 | 1.00 ± 0.01 |
| `visual.frame_dropout` | 1.00 | 0.98 ± 0.02 | 0.92 ± 0.01 | 0.96 ± 0.01 | 0.98 ± 0.02 | 0.96 ± 0.00 |
| `visual.occlusion` | 1.00 | 0.99 ± 0.01 | 0.92 ± 0.01 | 0.98 ± 0.00 | 0.99 ± 0.01 | 0.96 ± 0.01 |


## Mitigation: modality dropout

_Mitigation arm not run yet._
