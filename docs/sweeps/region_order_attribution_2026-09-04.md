# Region-order attribution: measured, 2026-09-04

Answers a question the tree could not previously ask: when an ordering
mutation moves coverage, does the region liveness estimator hear about it in
the right region?

## Why it could not be asked before

`OperatorEngine.record_coverage_diff` folds a coverage-edge diff into
`LiveBitMaskEstimator` for whichever region one byte offset falls in, and its
docstring names that offset as "the byte position the mutation touched"
(`f._last_mutation_offset`). The mutation loop assigned that from
`select_position()` before dispatch, for every operator — including the four
that rewrite the whole buffer and never read it. 139 of 156 handlers declare
`_byte_idx`; the four here are the ones with no true offset to report at all.

No target in the tree has known per-region order-sensitivity, so there was
nothing to check the attribution against. `tools/gen_synthetic_target.py`
produces a value-live/value-dead split, not an order-live/order-dead one.

## The bench

`targets/order_sensitivity.c`: 16384 bytes = 32 records of 512, so the region
profiler's 4096-byte cut gives 8 records per region. Manual guards (gcc has no
`-fsanitize-coverage=trace-pc-guard`), `-O1`, ASLR disabled per child.

Ground truth, 200 samples per cell, deterministic over 5 reruns:

| region | bytes | 1 adjacent record swap | free permutation | truth |
|---|---|---|---|---|
| r0 | [0, 4096) | 177/200 | 190/200 | LIVE |
| r1 | [4096, 8192) | 0/200 | 0/200 | DEAD |
| r2 | [8192, 12288) | 0/200 | 0/200 | DEAD |
| r3 | [12288, 16384) | **0/200** | **24/200** | **LIVE** |

r3 is a cold-but-live region: live, but quiet enough to look dead under a weak
mutator. `docs/sweeps/synthetic_liveness_calibration_2026-08-29.md` names that
exact case as the one its synthetic target could not exhibit and therefore
could not validate. This one exhibits it by construction.

## Result

Five arms, 1200 observations per arm per seed (300 per region;
`_LIVENESS_SWITCH_AFTER` is 200), five seeds. "Sound" means the reported
offset's region is the only region the mutation modified. Per-region columns
show one character per seed, `.` for a verdict matching ground truth.

| arm | sound | r0 | r1 | r2 | r3 | wrong |
|---|---|---|---|---|---|---|
| `chunk_shuffle` whole buffer + fabricated offset (status quo) | 0.4% | `.....` | `XXXXX` | `XXXXX` | `.....` | 2/4 ×5 |
| random adjacent record swap + true offset | 100% | `.....` | `.....` | `.....` | `XXXXX` | 1/4 ×5 |
| SJT adjacent swap from the identity + true offset | 100% | `.....` | `.....` | `.....` | `XXXXX` | 1/4 ×5 |
| SJT adjacent swap, randomised start + true offset | 100% | `.....` | `.....` | `.....` | `..X.X` | 0/4 ×3, 1/4 ×2 |
| free permutation confined to one region + true offset | 100% | `.....` | `.....` | `.....` | `.....` | **0/4 ×5** |

The whole win comes from confining the reorder to one region and reporting its
true offset. Both landed: `_DELOCALISED_OPS` stops the fabrication and
`region_shuffle` supplies the confined mutation.

## What this rules out

The measurement was set up to test whether a Steinhaus-Johnson-Trotter walk
over records — an adjacent-transposition Gray code — earns a place here, on
the argument that it is the only generator that is both exhaustive over all
n! orderings and moves by a single localized edit each step
(`docs/handover/handover_sjt_adjacent_transpositions_2026-09-04.md`). It does
not. Deciding whether a region is order-live wants **independent samples** of
S_n, not a **connected walk** through it; the Gray-code property trades
independence for locality and locality is not the scarce thing.

Mechanically: SJT's path is highly ordered. At n=8 the content of slot 0
first changes at step 7, but the specific value that fires r3 (`loc[0] == 3`)
does not reach slot 0 until **step 5376 of 40320**, and zero times in the
first 300. Its exhaustiveness is asymptotic; inside any bounded observation
budget it is a local sampler. The free-permutation arm hits the same condition
at ~12% per draw.

## Not addressed here

The 135 single-site operators that also declare `_byte_idx` and pick their own
position still feed the estimator the loop's unread draw. They *have* a true
offset; letting handlers report one is a wider change and is not in this
series.
