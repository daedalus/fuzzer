# Liveness thresholds: calibrated against the synthetic known-dead region

Date: 2026-08-29. Closes handover item B (`_LIVENESS_SWITCH_AFTER` /
`_LIVENESS_DEAD_WEIGHT` calibration), which had been "unblocked, one run
left" since the synthetic target landed.

Reproduce:

    python tools/sweep_liveness_thresholds.py --synthetic-target --unstable 0

## What was actually blocking it

The handover said the run was a formality — build the synthetic target, run
`tools/sweep_liveness_thresholds.py` against it, read off the numbers. It
wasn't, and the reason is worth recording: **the sweep tool could not run
against a target at all.** Its `--target` argument was accepted, stored, and
never read; both of its paths drove `FormatLearner` over fabricated
transitions and answered a different question (are the padding *hypotheses*
stable across the grid — settled long ago over four real campaigns). There
was no `--unstable` flag, so the handover's own "use `--unstable 0`"
instruction had nothing to attach to. The production consumer
(`operators.record_coverage_diff`) even carries an env-gated
`FUZZER_LIVENESS_LOG` hook "for the item-4 real-corpus sensitivity sweep" —
the producer side of a log the sweep tool never learned to read.

So item B needed a tool, not just a run. `--synthetic-target` is that tool:
it builds `gen_synthetic_target.py`'s target with the same gcc +
`-DSYNTH_MANUAL_GUARDS` recipe the ground-truth test uses, then drives the
*real* `LiveBitMaskEstimator` against the dead and live regions with real
coverage diffs, replaying each region's diff sequence through the estimator
at every threshold in the grid.

## Measured (`--unstable 0`, 400 blocks, fanout 32, 900 samples/region)

Deterministic (identical-input reruns 8/8 identical), dead region `[32, 96)`,
live region `[0, 32)`:

- dead-region mutations moving coverage: **0 / 900**
- live-region mutations moving coverage: **900 / 900**, first edge on sample 1

| switch_after | dead verdict | dead conv@ | live verdict | live mask bits | live leading-zero run |
|---:|---|---:|---|---:|---:|
| 50  | DEAD | 50  | LIVE | 210 | 0 |
| 100 | DEAD | 100 | LIVE | 210 | 0 |
| 200 | DEAD | 200 | LIVE | 210 | 0 |
| 400 | DEAD | 400 | LIVE | 210 | 0 |
| 800 | DEAD | 800 | LIVE | 210 | 0 |

**False-negative rate 0, false-positive rate 0, at every threshold.** The
dead region converges to a DEAD verdict in exactly `switch_after` samples
(every observation is no-growth, so `is_converged` trips precisely at the
threshold, with an empty mask). The live region reveals an edge on sample 1,
so its mask is never empty and the dead verdict (`is_converged and mask == 0`)
cannot fire on it no matter how low `switch_after` goes.

## The reading, and why the default is *not* lowered

On the synthetic target `switch_after` is unconstrained from below by
correctness — 50 classifies both regions correctly. It only sets how many
wasted mutations elapse before the down-weight engages. Lowering it would
make the down-weight act sooner on a truly-dead region and cost nothing
*here*.

It is kept at 200 anyway, because the case that justifies a high floor is one
this target cannot produce by construction: a real **cold-but-live** region —
live, but so rarely-triggering that it emits a long run of no-growth samples
before its first edge. The synthetic live region's leading-zero run is 0, so
this target sets no lower bound on how long a live region can *look* dead. The
dead side is now measured; the cold-live floor remains a stated assumption
about real targets, and 200 is the conservative choice for it.

`_LIVENESS_DEAD_WEIGHT = 0.1` is likewise not tuned by this run. The sweep
confirms the down-weight fires on the dead region and never on the live one,
but 0.1-vs-0.0 is a recoverability choice — the synthetic target cannot price
the cost of a wrong verdict against the benefit of a right one. It stays a
soft down-weight so a misclassified real cold-live region is recoverable.

## ASLR: the harness was measuring a target the fuzzer never runs

The handover warned that `--unstable 0` is mandatory because ASLR-gated
edges destroy the liveness signal. That reproduced — but only under a
condition production does not have.

`services/fuzzer.py` calls `disable_aslr()` at startup, and
`personality(ADDR_NO_RANDOMIZE)` is inherited across both fork and execve,
so **every target the fuzzer runs executes with ASLR off.** The first
version of this calibration used a plain `subprocess.run`, i.e. ASLR on, so
it was characterising a target the fuzzer never actually runs. The sweep now
disables it per-child (`preexec_fn`) rather than by calling
`disable_aslr()`; `--keep-aslr` reproduces the old condition.

The per-child detail is not incidental. `personality()` is process-global
and irreversible, so calling `disable_aslr()` inside the sweep leaves every
later test in the same pytest process running without ASLR — which makes
`tests/test_synthetic_target.py`'s unstable-variant ground-truth checks
**skip** instead of run, depending on file order. That was observed, not
theorised: 7 passed alone, 5 passed + 2 skipped when the sweep tests ran
first. A suite that quietly loses coverage is worse than a red one, so the
tool sets the personality in the forked child, where it is inherited across
execve and dies with the process.

Same `--unstable 4` variant, same seed, both conditions:

| | identical-input unstable edges (30 runs) | dead-region mutations moving coverage | dead verdict |
|---|---:|---:|---|
| ASLR on (`--keep-aslr`) | 1–3 | 56–66 / 120 | LIVE (wrong) |
| **ASLR off (production)** | **0** | **0 / 120** | **DEAD (correct)** |

So with ASLR off the *unstable* variant is a valid calibration target. The
blocks have not been removed or silenced: distinct edges over 40 varied
inputs is 167 either way, and the high guard ids are still present. The heap
address the gates test is simply constant now, so each gate is a fixed
predicate per input rather than a coin flip.

Two consequences worth keeping:

- The `--unstable 0` requirement is really a *determinism* requirement, and
  ASLR is only one way to lose it. The tool now keys its warning and its
  INCONCLUSIVE verdict on observed rerun stability rather than on the
  `--unstable` flag, because the flag was never the thing that mattered.
- The headline calibration is unaffected: `--unstable 0` is deterministic
  under both conditions, and re-running it with ASLR off reproduces the
  table above exactly (0/900, 900/900, DEAD at every threshold). The item-B
  result stands.

This also puts a caveat on §3 of
`synthetic_target_ground_truth_2026-08-19.md`: the
`--calibrate-stability` curve was measured against ASLR-gated instability,
which is not instability the fuzzer would ever encounter, since it disables
ASLR before running anything. The shape of the detection curve is still
informative, but the specific "3 unstable edges" ground truth it resamples
does not exist under production conditions. Whatever `--calibrate-stability`
is worth catching in a real campaign, it is not this.
