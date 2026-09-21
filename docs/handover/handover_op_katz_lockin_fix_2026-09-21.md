# OpKatzScheduler: applying the same lock-in fix as OpKuramotoScheduler

## Background

`handover_op_kuramoto_lockin_fix_2026-09-21.md` found and fixed a
permanent first-success lock-in in `OpKuramotoScheduler.select_op`, and
flagged that `OpKatzScheduler.select_op` -- the scheduler
`OpKuramotoScheduler` copied this exact draw from -- has the identical
bug. This is that follow-up.

## Confirmation

Same harness (`tests/support/bandit_env.py`, `StationaryBernoulli`, 30
seeds x 20,000 rounds) against unpatched `OpKatzScheduler`:

```
tail_share (sorted): [0.0]*19 + [0.505, 0.512, 0.515, 0.527, 0.529,
                       0.531, 0.533, 0.539] + [1.0]*3
mean: 0.240
```

Not as cleanly bimodal as `OpKuramotoScheduler`'s exact 0.0/1.0 split
(Katz centrality's neighbor-amplification term smears a few seeds toward
~0.5 by letting a second arm ride the first one's transition edges), but
the same qualitative failure: 19/30 seeds locked onto a wrong arm with
0% tail share for the true best, driven by the identical root cause --
every unpulled arm's beta_i is 0 with no incoming Katz injection either,
so the shift-and-sample draw (`vals - min + 1e-9`) leaves it at the bare
`1e-9` floor the instant any other arm has a nonzero score.

## Fix

Identical to the `OpKuramotoScheduler` fix: a uniform floor mixed into
`select_op`'s probability vector, `explore_floor / n` (fraction of
uniform, not absolute), default `0.06` matching both `op_cmaes.py` and
the already-fixed `OpKuramotoScheduler`. Factored into `_select_probs()`
for direct testability, same as the sibling fix. `explore_floor=0.0`
restores the exact old draw.

## Post-fix measurement

Same harness, same seeds:

```
tail_share (sorted): 0.266 .. 0.384, mean 0.322
```

The bimodal/trimodal spread (0.0 / ~0.5 / 1.0) collapses to a tight band
around the mean -- same qualitative outcome as the `OpKuramotoScheduler`
fix. No seed at either extreme. This scheduler is landed off by default
regardless (per its own module docstring's beta-sign-fix note, it needs a
real campaign before trusting it for anything) -- this patch only removes
a selection-mechanics bug from whatever future measurement decides that
question, it doesn't answer it.

## Status

`tests/test_op_katz.py`: 25 tests (was 20), all passing. New coverage
mirrors `test_op_kuramoto.py`: `_select_probs` floor bounds,
`explore_floor=0.0` restoring the old draw, constructor validation, and
an end-to-end regression reproducing a permanent-lock-in scenario and
asserting the true best arm is no longer starved to zero in the tail.

Full relevant suite (`test_op_katz.py`, `test_op_kuramoto.py`,
`test_kuramoto.py`, `test_regression_scheduler_fallback_precedence.py`,
`test_regression_scheduler_operator_reach.py`): 132/132 pass. No change
to default behavior -- `OpKatzScheduler` remains off by default, Elo-only.
The one existing call site (`services/fuzzer.py`'s
`OpKatzScheduler(rng=..., alpha_fraction=...)`) is unaffected since
`explore_floor` is a new keyword-only-in-practice argument with a default.

## Suggested next steps

Both shift-and-sample schedulers in this pool (`op_katz`, `op_kuramoto`)
now share the same floor convention with `op_cmaes.py`. Worth checking
whether any *other* scheduler in `core/schedulers/` uses the same
unfloored shift-and-normalize idiom without inheriting this fix --
`grep -rn "vals.min() + 1e-9"` across the package turns up only these two
plus `op_cmaes.py`'s own (already-floored) `_softmax`, so nothing else
appears to need it, but that grep hasn't been re-run after this patch.
