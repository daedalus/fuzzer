# Synthetic target: measured answers to three open questions

Date: 2026-08-19. Generator: `tools/gen_synthetic_target.py`. Assertions
kept green by `tests/test_synthetic_target.py`.

Three questions had been open for several rounds not because they were hard
but because no target in the matrix could answer them. Each needed a
property no real format provides, so each got argued from simulation or
inference instead. The synthetic target supplies all three by construction:
a controllable guard count, a byte region that is read but provably never
reaches a branch, no checksum anywhere, and optional ASLR-gated blocks.

Built here with `gcc -DSYNTH_MANUAL_GUARDS` (gcc cannot do
`-fsanitize-coverage=trace-pc-guard`, so the generated blocks call the guard
callback themselves — the same shim entry point clang's instrumentation
targets). Under clang the switch gives the guard count and the fanout loop
contributes ~1 guard rather than N; per-execution edge count is therefore
lower under clang than the numbers below.

## 1. A genuinely coverage-dead region exists (handover item B)

The blocker on Sequencing step 6 was never the estimator — it was that four
real campaigns produced zero true-dead regions, so the false-negative rate
had nothing to be measured against. Two of those failures are structural
rather than accidental: compressed data has no padding, and any CRC-covered
format rules out coverage-dead bytes outright, since mutating any byte moves
the CRC-check edge regardless of semantic relevance.

Measured on the deterministic variant (`--unstable 0`, 20000 blocks,
fanout 64, 64 edges per execution), single-bit flips against a fixed 256-byte
seed:

| region | bytes | mutations changing coverage |
|---|---|---|
| live prefix | [0, 32) | **60 / 60** |
| **dead region** | **[32, 96)** | **0 / 60** |

Identical-input reruns: 0/20 differed, so the target is deterministic and
"the dead region changed coverage" is falsifiable rather than swamped by
noise.

This is the case that closes step 6. `_LIVENESS_DEAD_WEIGHT = 0.1` and
`_LIVENESS_SWITCH_AFTER = 200` can now be calibrated against a known-dead
region instead of left as conservative guesses.

**That calibration has since been run** (2026-08-29):
`docs/sweeps/synthetic_liveness_calibration_2026-08-29.md`. Both constants
are retained, and the reason they are retained rather than tuned is recorded
there — the dead side is measured at 0 false negatives and 0 false
positives across the whole `switch_after` grid, but the case that justifies a
high `switch_after` floor is a real *cold-but-live* region, which this target
cannot exhibit by construction.

## 2. The probe-window trade, measured rather than simulated

Previously argued from a Python simulation. Measured against the real shim,
201,279 distinct random edge ids into a 262,144-entry map — `ffmpeg_read`'s
shape, load 0.77:

| window | edges lost | loss | drop counter | ns/edge |
|---:|---:|---:|---:|---:|
| 8 | 8,839 | 4.39% | 65,535 (saturated) | 27.1 |
| 16 | 3,129 | 1.55% | 34,320 | 28.6 |
| 32 | 761 | 0.38% | 8,272 | 29.1 |
| 64 | 86 | 0.04% | 847 | 31.8 |
| 128 | 12 | 0.01% | 33 | 30.4 |
| 1024 | 9 | 0.00% | 0 | 29.7 |

The loss column matches the simulation closely (16 → 1.55% measured vs
1.60% simulated), so the simulation was sound and `__AFL_PROBE_MAX = 64` is
the right pick on the loss axis.

**But the cost column does not support the change.** ns/edge is flat from
window 8 to 1024 — the spread is within noise, and window 8 is not reliably
faster than 1024 despite probing 128× less far. At this load the hot path is
dominated by cache misses over a 2 MB table, not by probe count, and lookups
of already-present edges terminate at the matching entry regardless of the
bound.

So the honest reading is that the bounded window buys no measurable
throughput here while costing 0.04% of edges. It is retained because it
converts an unbounded worst case into a constant, which still matters for a
table that is genuinely full — but the §2 premise that probe cost was worth
attacking is not supported by measurement. Anyone tempted to lower the
window to 16 for speed should note there is no speed to gain.

Note this measurement is only meaningful post-`7119364`. Before that fix the
table saturated within a few hundred executions, so any load-factor
reasoning was describing a symptom.

## 3. `--calibrate-stability 3` is not enough (handover item D)

The unstable variant (`--unstable 4`) has ASLR-gated blocks, so ground truth
is known. Over 60 runs of one fixed input: **3 unstable edges** (26653,
26655, 26899).

Detection rate by `n_runs`, resampling that 60-run pool, 400 trials each:

| n_runs | detects any | mean found | recovers all 3 |
|---:|---:|---:|---:|
| 2 | 70.2% | 1.19 | 11.5% |
| 3 | 94.0% | 1.97 | 35.0% |
| 5 | 99.2% | 2.49 | 61.0% |
| 8 | 100.0% | 2.86 | 86.8% |
| 12 | 100.0% | 2.95 | 95.2% |
| 16 | 100.0% | 3.00 | 99.5% |
| 24 | 100.0% | 3.00 | 100.0% |

Found the hard way: a live 3-run calibration returned **zero** unstable
edges on an input that varies across 12 runs, which looked like a bug in the
implementation and was not — 3 runs miss the instability entirely 6% of the
time and recover the full set only 35% of the time.

This matters because the failure is silent and asymmetric. A missed unstable
edge is not a one-off: it stays unmasked for the rest of the campaign and
keeps absorbing energy every iteration, which is the exact cost the feature
exists to remove. Detecting "at least one" is not enough either, since the
remaining unmasked edges go on producing endless false novelty.

Recommendation: the `--calibrate-stability N` help text should point at 8,
not 3, and the handover's original "run each new seed 3×" suggestion should
be treated as folklore rather than a measured default. Note this is one
input on one target with 4 designated unstable blocks; a target whose
instability is rarer per-run would need more runs still. What is measured
here is the shape of the curve, not a universal constant.

## Postscript: the flakiness bit back immediately

`tests/test_synthetic_target.py::test_identical_input_gives_varying_coverage`
was written with a fixed 12 runs and went red on a full-suite pass shortly
after being added — the ASLR bits happened to agree across all 12. That is
the same failure mode as section 3, arriving unprompted within an hour of
being measured, and it is decent evidence the effect is real rather than an
artifact of the resampling.

The test now collects runs until instability appears (up to 40) and skips,
rather than fails, when an environment shows no address randomisation at
all. Worth generalising: any test that asserts nondeterminism *exists* needs
either a generous run budget or an early exit, and any code that decides
"this edge is stable" from a small fixed sample is making the same mistake
the test did.
