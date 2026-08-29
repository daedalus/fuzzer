# Item 4 real-corpus sensitivity sweep (round 9)

Sequencing step 6 of `docs/handover/handover_skittercreek_tailslayer_port.md`: run
`LiveBitMaskEstimator`'s convergence-threshold sensitivity sweep (previously
only run synthetically, in `tests/test_live_bit_mask.py`) against real
coverage-diff samples from an actual campaign, before trusting the
`_LIVENESS_DEAD_WEIGHT` down-weight / padding-verdict path at full strength.

## Method

- Target: `targets/zlib_read.so` (`fuzz_zlib`), built locally with the SHM
  edge-coverage shim, in-process mode.
- Corpus: real gzip/zlib streams — compressed copies of this repo's own
  `.py` source files (1.5–22 KB compressed), not synthetic fixtures.
- Ran ~140K total execs across several bounded, resumed campaign rounds
  (`fuzzer-tool fuzz ... --inprocess --inprocess-func fuzz_zlib -m 32768
  --resume`), saturating at the target's full 11 reachable edges within the
  first few hundred execs.
- Temporarily instrumented `OperatorEngine.record_coverage_diff` (env-gated,
  zero-cost when unset, and fully reverted after this sweep — not present in
  the tree) to log every real `(region_idx, diff_bits)` pair it computed
  during the campaign, i.e. the exact input the production
  `LiveBitMaskEstimator.observe()` call consumes.
- Raw output: `docs/sweeps/item4_zlib_real_corpus_samples.tsv`, 3,192
  samples. Format: `region_idx<TAB>comma-separated set-bit positions` (a
  sparse encoding of `diff_bits`, not the raw decimal integer — a 65536-bit
  mask's decimal form runs to ~20,000 digits per row and bloats the file
  for no benefit; an empty second column means `diff_bits == 0`).
- Replayed that real sequence, in order, through a fresh
  `LiveBitMaskEstimator(n_bits=65536, switch_after=N)` for
  `N ∈ {50, 100, 200, 400, 800}` — the same sweep range as the synthetic
  test — and recorded convergence point, final mask, and whether the
  estimator ever produced a confirmed-dead transition.

## Result

| switch_after | samples to first convergence | final mask popcount | ever converged-dead |
|---|---|---|---|
| 50  | 53  | 9 | no |
| 100 | 103 | 9 | no |
| 200 (current default) | 312 | 9 | no |
| 400 | 512 | 9 | no |
| 800 | 912 | 9 | no |

91.1% of the 3,192 real samples had a nonzero diff; the longest run of
consecutive zero-diff samples was 6, far short of even the smallest
`switch_after` tested. Every threshold converged to the same 9-bit live
mask and none ever reached a converged-dead verdict — the down-weight path
never fired at any threshold on this target.

## What this does and doesn't validate

**Validates:** the estimator's mask is threshold-stable on real data — the
current production default (`switch_after=200`) doesn't produce a
qualitatively different live-set than nearby values, so there's no sign of
threshold sensitivity in the range already shipped. This is a real,
non-synthetic exercise of the exact code path in production (same
`record_coverage_diff` call, same default `map_size`), not just the
`live_bit_mask.py` unit in isolation.

**Doesn't validate:** the false-negative case the sweep was meant to
stress-test — a *genuinely dead* region being wrongly declared dead too
early. This target never gave the estimator a real dead region to get
wrong: compressed/incompressible payloads correctly have no padding, every
byte matters to something, so the mask never approached empty. Two
compounding scope limits on this run specifically:

- `profile_buffer`'s 4096-byte window collapsed every seed in this corpus
  into a single region (`_REGION_MIN_LEN=512` and default `window=4096`;
  our seeds mostly ran 1.5–22 KB, below or barely above one window), so
  this sweep only ever exercised region 0 — no multi-region comparison.
- The target only has 11 reachable edges total, so the live-bit space
  being estimated was extremely sparse (9 live bits out of 65,536) relative
  to what a richer target (e.g. `png_read`'s chunk structure, which has
  real reserved/padding fields) would exercise.

## Conclusion

Not a full close of Sequencing step 6 — a genuine dead-region false-negative
test needs a target with real structural padding, which this run didn't
have. But it's real evidence, not synthetic-only, that the shipped
`switch_after=200` default doesn't misbehave differently from nearby
thresholds on an actual campaign, which is exactly the risk the
conservative `_LIVENESS_DEAD_WEIGHT=0.1` (down-weight, not exclusion) was
chosen to bound. Recommend keeping the current conservative weighting and
re-running this same methodology against `png_read` or another
chunk-structured target with known padding fields as the next step, rather
than treating this as the final word.
