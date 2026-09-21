# OpKuramotoScheduler: fixing permanent lock-in in `select_op`

## Background

`handover_op_kuramoto_scheduler_2026-09-20.md` built `OpKuramotoScheduler`
and left "run `tests/support/bandit_env.py`'s convergence harness" as its
first suggested next step. This is that run, and what it found.

## What the harness found

`OpKuramotoScheduler` vs. Thompson sampling (`MonteCarloScheduler`),
`StationaryBernoulli`, 30 seeds x 20,000 rounds:

| | tail-share mean | tail-share median | paired win-rate vs. Thompson |
|---|---|---|---|
| OpKuramotoScheduler (pre-fix) | 0.267 | 0.000 | 8/30 |
| Thompson | 0.996 | 0.996 | 22/30 |

The pre-fix tail-share distribution was **strictly bimodal** — exactly
`0.0` or `1.0` across all 30 seeds, never a partial value. Inspecting a
losing seed directly (seed 1) showed the scheduler locking onto a p=0.05
base-rate arm for 19,945/20,000 picks while the true p=0.30 best arm was
starved after early exploration.

## Root cause

`select_op` copied `OpKatzScheduler.select_op`'s shift-and-sample draw
verbatim: `shifted = vals - vals.min() + 1e-9`, normalized to
probabilities. Every arm starts at score 0 (rate=0 with no track record).
The instant *any* arm registers its first success, its score jumps to
something on the order of its raw rate while every still-unpulled arm
stays pinned at the bare `1e-9` shift constant — a ratio of several orders
of magnitude, which the weighted draw reads as "pick this arm essentially
forever," independent of whether that arm is actually the best one. The
Kuramoto phase/coupling machinery is not implicated: `r` in the confirmed
failing run was a moderate 0.25, and the amplification term
`(1 + r*cos(theta-psi))` only ever scales an already-dominant rate term, it
doesn't create the dominance.

**This is not unique to this module.** The identical shift-and-sample draw
in `OpKatzScheduler.select_op` (which this one was copied from) produces
the same qualitative failure when run through the same harness — confirmed
separately, tail-shares landing at 0.0/~0.5/1.0 rather than converging
cleanly. Not fixed here; out of scope for this patch, flagged for whoever
picks up `op_katz` next.

It is also not a novel failure mode for this codebase: `op_cmaes.py`'s own
`_softmax` docstring already documents the identical symptom for CMA-ES
(bimodal 0.947/0.004 outcomes that got *worse* with more rounds) and
already carries the fix this patch reuses.

## Fix

Mixed a uniform floor into `select_op`'s probability vector, factored into
a new `_select_probs(ops)` helper for direct testability:

```python
floor = self.explore_floor / len(ops)
floored = np.maximum(probs, floor)
return floored / floored.sum()
```

`explore_floor` (new constructor argument, default `0.06`) is a *fraction*
of uniform, not an absolute per-arm constant — the same convention
`op_cmaes.py`'s `_softmax(floor_frac=0.06)` uses, for the same reason: an
absolute floor is negligible at a handful of arms and dominates the whole
distribution once the registry grows to the full operator count (currently
~200). The default `0.06` is borrowed from `op_cmaes.py` unchanged, not
independently tuned for this scorer — flagged in both docstrings as
something a fresh sweep should confirm rather than assume transfers as-is.
`explore_floor=0.0` restores the exact old unfloored draw, kept for anyone
who wants to reproduce the pre-fix numbers above.

## Post-fix measurement

Same harness, same seeds:

| | tail-share mean | tail-share median | regret slope |
|---|---|---|---|
| OpKuramotoScheduler (post-fix), Stationary | 0.360 | 0.354 | 1.016 |
| Thompson, Stationary | 0.996 | 0.996 | 0.200 |
| OpKuramotoScheduler (post-fix), DecayingBest | 0.254 | 0.254 | 0.876 |
| Thompson, DecayingBest | 0.180 | 0.092 | 3.577 |

The bimodal 0.0/1.0 signature is gone (tail-shares now cluster near their
mean, no seed at either extreme). Two honest observations, neither
flattering nor damning:

- On `StationaryBernoulli` this scheduler still trails Thompson by a wide
  margin (0.36 vs. 0.996) — the permanent `explore_floor` exploration tax
  caps how sharply it can converge, the same tradeoff every floored
  scheduler in this pool accepts. Regret slope ~1.0 (not sublinear) is
  consistent with that: a constant floor means constant per-round regret
  contribution from the floor itself, not a bug in the fix.
- On `DecayingBest` it now *beats* Thompson (0.254 vs. 0.180) — the same
  floor that costs it convergence sharpness on a stationary environment
  buys it recovery capacity once the early-best arm's yield collapses,
  because the floor keeps every arm (including the eventual new best)
  getting pulled instead of a converged posterior refusing to let go.

Neither number says anything new about whether operators actually behave
like phase oscillators — that empirical question, and the real
`tools/bench_paired.py` A/B against a live campaign, are unchanged from
`handover_op_kuramoto_scheduler_2026-09-20.md`'s open items. This patch
only fixes a selection-mechanics bug that was drowning out any signal the
phase model might have.

## Status

`tests/test_op_kuramoto.py`: 25 tests (was 21), all passing. New coverage:
`_select_probs` floor bounds, `explore_floor=0.0` restoring the old draw,
constructor validation for `explore_floor`, and an end-to-end regression
reproducing the exact seed-1 lock-in scenario from the harness finding
above and asserting the true best arm is no longer starved to zero in the
tail. `test_cold_start_gives_every_op_nonzero_probability` updated to
assert through the real `_select_probs` pipeline instead of reproducing
the old formula inline.

Full relevant suite (`test_op_kuramoto.py`, `test_kuramoto.py`,
`test_regression_scheduler_fallback_precedence.py`,
`test_regression_scheduler_operator_reach.py`): 112/112 pass. No change to
default behavior of any other scheduler; `OpKuramotoScheduler` remains
off by default, Elo-only, absent from `_FALLBACK_PRECEDENCE`.

## Suggested next steps

1. Apply the identical floor fix to `OpKatzScheduler.select_op` (confirmed
   affected, not touched here).
2. A fresh `explore_floor` sweep for this scorer specifically, rather than
   inheriting `op_cmaes.py`'s `0.06` unchanged.
3. The still-outstanding items from `handover_op_kuramoto_scheduler_2026-09-20.md`:
   a real `tools/bench_paired.py` A/B against the live Elo arbiter, and the
   fuzzgoat+clang calibration run neither this session nor that one had
   the tooling to perform.
