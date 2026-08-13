# Reproduction gate

**No degradation claim is made until clean-data numbers land within 1–2 points of
published results for at least MulT and LMF.** If they do not, the degradation curves
measure our bugs rather than the architectures.

This gate is not optional and it is not skippable. It is also the single practice most
portfolio projects omit and most reviewers check for.

## Status

| Model | Dataset | Metric | Published | Ours | Δ | Within gate |
|---|---|---|---|---|---|---|
| MulT | MOSI | Acc-2 | 83.0 | — | — | ⬜ pending real data |
| MulT | MOSI | MAE | 0.871 | — | — | ⬜ |
| MulT | MOSI | Corr | 0.698 | — | — | ⬜ |
| LMF | MOSI | Acc-2 | 82.5 | — | — | ⬜ |
| LMF | MOSI | MAE | 0.917 | — | — | ⬜ |
| TFN | MOSI | Acc-2 | 80.8 | — | — | ⬜ |
| MulT | MOSEI | Acc-2 | 82.5 | — | — | ⬜ |
| LMF | MOSEI | Acc-2 | 82.0 | — | — | ⬜ |

Published figures are the aligned-setting numbers from Tsai et al. (ACL 2019) Table 1 and
Liu et al. (ACL 2018); fill in the exact citations and the convention each used when the
table is populated. **Verify each number against the paper before entering it here** —
these are placeholders for the shape of the table, not vetted values.

## The convention trap

Acc-2 is reported two ways in this literature and papers frequently do not say which:

* `acc2_has0` — every test sample, `pred ≥ 0` counted as positive;
* `acc2_non0` — neutral samples (`label == 0`) excluded first.

`acc2_non0` runs 1–2 points higher. That is the *entire width of the reproduction gate*,
so comparing our `non0` against a published `has0` would either fake a pass or fake a
failure. `wfb.evaluation.metrics` computes both, always. When filling in the table above,
record which convention the published number used.

The same care applies to the aligned vs unaligned setting: MulT reports both, and the
unaligned numbers are lower. Our pipeline uses word-aligned features.

## How to run it

```bash
# once the real archive is in data/raw/
uv run wfb-data --dataset mosi --no-synthetic
uv run python experiments/run_all.py --preset main
```

Then read the clean columns of `experiments/results/REPORT.md` and fill in the table.

## If a number misses

Check, in this order — roughly by how often each is the culprit:

1. **Metric convention** (`has0` vs `non0`, aligned vs unaligned). Cheapest to check and
   most often the answer.
2. **NaNs in COVAREP.** MOSEI genuinely ships them. `sanitize()` handles it; confirm it
   actually ran by checking `bundle.stats` for finite values.
3. **Normalisation.** Without z-scoring, COVAREP's scale spread dominates every gradient.
4. **Sequence length and padding direction.** We keep the tail; some implementations keep
   the head.
5. **Split.** Must be the standard video-level folds, not a random sample-level split —
   the same speaker appearing in train and test inflates every number.
6. **Capacity and schedule.** Our defaults are compute-honest compromises (MulT uses 2
   crossmodal layers, not the paper's 4–5). If everything else checks out, this is where
   the remaining gap lives, and it should be stated in the paper rather than tuned away.

Record whatever you find here. A documented near-miss with an identified cause is worth
considerably more to a reviewer than an undocumented exact match.
