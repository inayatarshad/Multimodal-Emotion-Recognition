# The degradation protocol

This is the project's contribution. Task metrics are standard; the systematic corruption
protocol and the summaries of how performance falls apart are not.

## Severity

Every operator takes `severity ∈ [0, 1]` and maps it onto its own physical parameter. One
sweep configuration therefore drives every corruption family, and AUDC is comparable
across families.

**Severity 0 is an exact identity — bitwise, not approximately.** Each operator is
responsible for this and each is unit-tested for it across every registered operator. The
base class deliberately does *not* short-circuit at 0, because then the test would be
testing the base class and the bug it exists to catch — an operator that quietly perturbs
its own clean baseline — would sail through. If that bug ever shipped, every retention
curve would be measured against a corrupted baseline and every AUDC would be wrong by an
unknown amount, while the plots would still look entirely plausible.

## The operators

| Name | Modality | Severity maps to | Models the failure |
|---|---|---|---|
| `gaussian_noise` | any | SNR, `-20·log₁₀(s)` dB | ambient noise |
| `frame_dropout` | any | % frames zeroed | packet loss |
| `burst_dropout` | audio, visual | % contiguous frames lost | a stream cutting out |
| `clipping` | audio | clip threshold, 4× → 0.1× RMS | microphone clipping |
| `asr_error` | text | WER 0 → 40% | transcription errors |
| `token_dropout` | text | % tokens zeroed | unrecognised words |
| `word_shuffle` | text | permutation window 0 → 12 | syntax destroyed, bag preserved |
| `occlusion` | visual | % contiguous frames mean-filled | face leaves frame |
| `blur` | visual, audio | Gaussian σ 0 → 6 frames | low frame rate / motion blur |
| `feature_noise` | any | noise as a multiple of feature std | tracker estimation error |
| `shift` | any | 0 → 10 frames | stream misalignment |
| `zero` / `mean` / `mask` | any | fraction removed | modality absent |

### Why the noise map is linear in amplitude

`SNR(dB) = -20·log₁₀(severity)` gives 20 dB at s=0.1, 8 dB at s=0.4, 0 dB at s=1.0 — the
20 → 0 dB range the protocol asks for. An SNR-linear map would need a special case at
s=0 (SNR = ∞), and special cases are where the identity-at-0 invariant would break.
Noise is scaled per feature dimension by the training-set RMS, so a naturally
large-range feature is not preferentially destroyed.

### Why ASR substitutions use real vectors

A substitution swaps in another word vector drawn from the *same utterance* rather than
random noise. This keeps the corrupted input on the data manifold. A model that merely
detects out-of-distribution garbage would otherwise look spuriously robust, because
"reject the weird input" is not the same skill as "still understand the utterance".

Deletions shift the sequence left and insertions duplicate-and-shift right, so the
*alignment* between text and the other streams degrades too — which is exactly the
coupling cross-modal attention depends on.

### Why the three removal variants are all reported

`zero` puts the model at a point that may be far outside its input distribution; `mean`
puts it at the distribution centre; `mask` lets a model that was trained with mask tokens
*know* the modality is gone. Papers pick one and rarely say which. Comparing them is
itself a small result, so all three are in the grid.

### Why misalignment moves only audio and visual

Shifting every stream equally is a no-op for any time-invariant architecture. The
corruption of interest is *relative* misalignment, which is what actually happens when
streams are buffered separately, so text is held as the reference.

## The grid

`standard_grid()` produces:

* **10 graded axes** — the (modality, operator) pairs above, at 6 severities each;
* **1 misalignment axis**;
* **21 removal axes** — all 7 non-empty subsets of {T, A, V} × 3 removal variants.

Plans are deduplicated across axes before evaluation (all 32 axes share the same
severity-0 anchor), which removes ~30 redundant passes over the test set.

## The metrics

**Retention(c)** — fraction of skill *above chance* that survives:

```
Retention(c) = (metric(c) − chance) / (metric(clean) − chance)
```

The chance correction is essential, not cosmetic. Binary accuracy floors at 0.5, so an
uncorrected retention bottoms out near `0.5 / clean` ≈ 0.6 for a typical MOSI model. A
text-only model with its text removed would then report a Modality Reliance Score of ~0.3
despite being completely incapacitated by construction — and every reliance and AUDC
number would be squeezed into the top third of its range, distorting comparisons between
architectures with different clean scores. This was caught by exactly that test case
during development.

**AUDC** — trapezoidal area under the retention curve, normalised by the severity range.
One number per (model, axis). **Higher is more robust.** Retention is clipped to
`[0, 1.5]` for the integral: the floor treats "at chance" and "below chance" as the same
amount of useful skill, and the ceiling prevents one flattering ladder point from
dominating the area.

**Critical threshold** — severity at which retention first crosses below 0.9, linearly
interpolated. `None` means it never does.

**MRS(m)** = `1 − Retention(remove m)`. 1.0 means the model is worthless without that
modality. Q2's text-dominance claim is `MRS(text) ≫ MRS(audio) + MRS(visual)`, read
directly off the 7-subset lattice.

**Brittleness index** — the correlation *across architectures* between clean performance
and mean AUDC. H1 predicts it is negative. With 6–8 architectures the Spearman version is
the honest one to quote, and neither deserves a p-value.

## Statistical protocol

* 5 seeds minimum per configuration; mean ± std everywhere; never a single run.
* Comparisons use a **paired bootstrap** over per-sample errors, not a t-test: absolute
  errors are strongly right-skewed and 0/1 misclassification indicators are not remotely
  normal, so a t-test's assumptions fail exactly where the differences are most
  interesting. A Wilcoxon signed-rank test is reported alongside as a cross-check —
  when a rank test and a resampling test disagree, the difference is not robust and is
  not claimed.
* Pairing is valid because `plan_generator` guarantees every architecture is evaluated on
  bit-identical corrupted inputs.
* Multiple comparisons are corrected with Holm–Bonferroni: uniformly more powerful than
  plain Bonferroni and, unlike Benjamini–Hochberg, requiring no independence assumption.
