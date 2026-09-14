# Handover — GradientBanditScheduler: two real bugs fixed, moved back to Elo-only

**Date:** 2026-09-14
**Base:** `74ae2f5` (`Add Exp4Scheduler (expert-advice bandit over operator categories)`)
**Status: FIXED AND TESTED.**

Follow-up to `handover_non_ucb_schedulers_2026-09-13.md`'s T2-1 proposal
(gradient bandit with baseline, "cheapest correct implementation"). Between
that handover and this one, a different session implemented it directly as
`core/schedulers/gradient.py` and wired it as a first-class fallback-precedence
scheduler, without running it against this project's own convergence harness
(`tests/support/bandit_env.py`) first. Running it turned up two real defects,
both now fixed; this doc records what they were and why the fix looks the way
it does, so it isn't rediscovered from scratch next time this file gets
touched.

---

## 1. What was broken

Measured on seed 92, before any fix in this doc:

| Environment | Metric | Before fix |
|---|---|---|
| `StationaryBernoulli` (12 arms) | tail_share(best) | **1.0** |
| `DecayingBest` (12 arms, decay@10000) | tail_share(best_late) | **0.0** |

Neither number is good, in opposite directions. `1.0` on the stationary case
means *zero* residual exploration — worse hedging than every scheduler in
`RELIABLE`, none of which reach 1.0. `0.0` on the decay case means the
scheduler never once reselected the revived best arm in the entire
10000-round recovery window — not slow recovery, no recovery.

**Root cause 1 — the baseline never forgets.** `record()` used a plain
sample average: `avg += (reward - avg) / n`. A 1/n average weights every
sample identically regardless of age, so after a long stable run the
baseline is dominated by history and takes proportionally longer to reflect
a regime change than it took to build up. This is the "global clock"
problem `handover_non_ucb_schedulers_2026-09-13.md`'s §2 already identified
in D-UCB/SW-UCB/Thompson-decay/CUSUM-UCB — reintroduced here through the
baseline instead of the confidence width.

**Root cause 2 — `min_temperature` does not floor any arm's selection
probability, despite the name.** It floors *temperature*, but the exponent
clamp at `z > 50` means a preference gap of a handful of units at
`min_temperature=0.05` already saturates the softmax to ~1e-20-level
probabilities for the loser. That is not a residual floor in any practical
sense. This is why root cause 1's fix alone (tried first, in isolation) was
*not* sufficient: with a correctly-tracking baseline the scheduler still
recovered 0/1000 times in a 2-arm toy replay of the regime switch, because
once the softmax is close enough to one-hot, the update magnitude for
*every* arm (chosen and unchosen alike) scales with a probability that has
already collapsed to ~0, so there is no gradient left to escape with.

## 2. The fix

1. Baseline switched to an exponentially-weighted update at the same
   step size as the preference update (`avg += alpha * (reward - avg)`),
   consistent with a scheduler whose whole design point is a constant step
   size instead of a window.
2. Added an explicit uniform-probability floor, mixed in *after* the
   temperature-scaled softmax: `pi = (1-floor)*softmax + floor/K`. This is
   the same fix `Exp3Scheduler` already applies in this tree for the
   identical reason. New `floor` parameter, default `0.05`, wired through
   to `--gradient-floor`.
3. While re-sweeping 40 seeds on `StationaryBernoulli` to set a defensible
   floor for `RELIABLE`, found a third issue: at the original default
   `alpha=0.1`, one seed (23, out of 40) collapsed to `tail_share=0.004` on
   a 0.30-vs-0.18 gap — not a close call. Both neighboring values tried
   (0.05, 0.15) converged normally on that same seed, so this reads as a
   narrow resonance between that specific step size and that draw sequence
   (bigger step sizes compound early REINFORCE preference swings) rather
   than a structural bug. Lowered the default to `alpha=0.05`, which passed
   all 40 seeds with `min share 0.938`.

Measured after all three changes, seed 92:

| Environment | Metric | After fix |
|---|---|---|
| `StationaryBernoulli`, 40-seed sweep | min tail_share / max slope | **0.938 / 0.267** |
| `DecayingBest` | tail_share(best_late), floor swept 0.02–0.5 | **0.002–0.05** |

The stationary number is now competitive with `ConsolidatedScheduler`
(0.970) and the scheduler is added to `RELIABLE` in
`test_scheduler_convergence.py` on that basis.

**The decay number is still bad, and no amount of floor tuning fixes it** —
tried up to `floor=0.5` (half the softmax mass replaced by uniform noise
over 12 arms), which also wrecks the stationary case, and recovery was
still under 5%. The floor bounds *instantaneous* selection probability; it
is not a forgetting mechanism, and nothing in this scheduler discounts
confidence built up before a regime switch. Recovery time scales with both
arm count and pre-switch confidence, and a 12-arm campaign with a 10000-round
recovery budget is nowhere near enough to undo confidence built over
15000 pre-switch toy rounds, let alone a real campaign's much longer
stable-then-fatigued lifecycle. This is a structural property of
preference-gradient methods without an explicit recency term, not a tuning
bug — the same root cause `EpsilonGreedy`/`MonteCarlo` are already in
`STUCK` for. `Gradient` now joins them there.

**Net assessment, unchanged from the prior handover's framing:** this
remains a legitimate, now-correctly-behaving *stationary* control arm
(T2-1's original pitch), and it does not answer, and structurally cannot
answer, the non-stationary rotting-bandit question (T1) that motivated
looking at this family in the first place.

## 3. Discipline correction: moved out of `_FALLBACK_PRECEDENCE`

`operators.py` documents, in its own comment, that unproven exploratory
arms (`op_katz`, `op_tang`) should only ever be reached via Elo explicitly
choosing them, precisely so a defect like the two above cannot become a
live campaign's silent default selector just because someone passed
`--gradient` without also passing `--elo`. The commit that added `gradient`
put it in `_FALLBACK_PRECEDENCE` anyway, immediately above the comment that
argues against doing exactly that. Moved back out, following the same
pattern as `op_katz`/`op_tang`: still reachable via the Elo ballot, no
longer the live selector by default. `exp4` and `successive_elim` were
*not* touched — they are proven/reachable schedulers by design in that same
commit family and this doc takes no position on whether they belong in
`_FALLBACK_PRECEDENCE`; only `gradient`'s placement was in scope here.

## 4. Unrelated fixture gaps fixed in passing

The commits that added `Exp4Scheduler`/`SuccessiveEliminationScheduler`
(alongside `gradient`) never updated the shared test fixtures that
enumerate every scheduler by name:

- `tests/support/operator_env.py`'s `BALLOT_SCHEDULERS` was missing
  `"exp4"` (`"gradient"` and `"successive_elim"` were already present).
- `test_regression_scheduler_fallback_precedence.py`'s `_SCHEDULER_ATTRS`
  and its hand-pinned `_FALLBACK_PRECEDENCE` list were missing all three
  of `exp4`, `gradient`, `successive_elim`.
- `test_regression_scheduler_operator_reach.py`'s `_all_operator_schedulers()`
  was missing all three schedulers entirely, meaning none of them were
  covered by this project's operator-reachability regression suite at all.

Fixed all three (mechanical, same one-line-per-file pattern already used
for `gradient` above); confirmed via before/after diff of the exact failing
node ID set that these were 100% pre-existing and unrelated to this
session's `gradient.py` changes before being fixed.

## 5. Also found and fixed: `parallel.py` gaps for the whole `gradient`/`exp4`/`successive_elim` family

`services/parallel.py`'s two parallel-execution entry points forwarded no
`gradient_*`, `exp4*`, or `successive_elim*` keyword arguments at all before
this session — `--gradient`, `--exp4`, or `--successive-elim` alone would
silently do nothing under parallel execution (confirmed via
`test_regression_parallel_kwargs.py`, the regression test this project
already has for exactly this class of gap). Fixed for all three schedulers,
mirroring the existing `fpl`/`op_katz` forwarding pattern already in that
file.

`test_regression_mypy_ratchet.py::test_every_exempt_module_exists` also
fails on this base commit (`pyproject.toml`'s mypy override list still
references a deleted `fuzzer_tool.core.allan_variance` module) — confirmed
unrelated to any scheduler in this doc, left alone as out of scope.
