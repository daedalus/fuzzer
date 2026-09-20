# `--shaped-reward` A/B: no gain, off by default, out of `--hail-mary`

**Question.** `handover_matrix_schedulers_2026-09-19.md` calls it "the cheapest
A/B available": scale the shared `op_rewards` weight every scheduler reads by the
fraction of a round's new edges that are independent canonical classes, so a
45-edge duplicate chain pays 1/45 instead of 45 (F10). The wiring landed the same
day as this run (`--shaped-reward`, `--shaped-reward-floor`). The decision this
run had to make was narrow: default on or off, and in or out of `--hail-mary`.

**Answer: off, and out.** Neither form beat the baseline. The faithful form's
point estimate is a *loss*.

## Setup

Rule 52: `targets/fuzzgoat_read_noasan.so`, clang 18 `--clang-scov` build
(compiler-inserted edge coverage), in-process, `-m 65536`. Paired cells over
seeds 0-11, 2,000 execs per campaign, budget counted in execs not wall-clock, one
replicate per cell, run sequentially on a single CPU. Metric: `Edges discovered`
at budget, the same line `tools/bench_paired.py` parses. Arms:

| arm | flags |
|---|---|
| `elo` (baseline) | `--elo --mc-bandit` |
| `shaped` | `+ --shaped-reward` (floor 0.0, the faithful form) |
| `shaped-floor25` | `+ --shaped-reward --shaped-reward-floor 0.25` |

Run outside `bench_paired.py` because the locked matrix has no fuzzgoat cell; the
arms are registered there (`elo-shaped-reward`, `elo-shaped-reward-floor25`) so a
matrix run can be added without re-deriving the flags.

Baseline: median 145 edges (130-163 across the twelve seeds).

## Result

| arm | cells | W/L | median Δ | IQR | mean Δ | 95% CI on mean Δ | McNemar exact |
|---|---|---|---|---|---|---|---|
| `shaped` | 12 | 5/7 | **-7.0** | [-15.5, +3.5] | -5.9 | [-13.5, +1.6] | p = 0.774 |
| `shaped-floor25` | 12 | 5/7 | -2.5 | [-8.8, +10.2] | +0.8 | [-8.9, +10.4] | p = 0.774 |

Read it as the Boltzmann result reads (`2026-08-30-boltzmann-ab-result.md`): not
"it does nothing", but "it does not move edges by more than about ±10 (7% of a
145-edge baseline) on this target at this budget, and the faithful form's
interval sits mostly below zero". Twelve cells at sd ≈ 12 cannot resolve a
two-point effect, and the harness reports the cell count so that is visible
rather than assumed.

The shaping was not inert: the banner carried `shaped-reward` on every shaped
cell, and fuzzgoat's own summary shows what it was biting on — *72 profiles for
139 edges, 67 duplicates, largest class 13*. On a target with no duplicate
classes the arm would be a no-op by construction; that is not this target.

## Decision

* `--shaped-reward` stays **off by default**.
* It stays **out of `_HAIL_MARY_FLAGS`**, now for a measured reason and not only
  a structural one. The structural argument still holds independently (it is not
  a strategy, it rescales what every strategy is paid, and folding it in would
  destroy the single-variable run it exists for).
* `--shaped-reward-floor` default stays **0.0**. The floor moved the median from
  -7.0 to -2.5, which is the direction the clamp was built for, but it buys that
  by turning the shaping off on exactly the rounds the shaping is about. Nothing
  here justifies shipping a non-zero default; the knob exists so the question is
  re-askable once `derived` is populated (P1-2), which is when the floor starts
  protecting against a factor of exactly 0 rather than 1/n.

## Two process notes, both earned here

**Three cells had to be thrown away and re-run.** Seeds 9-11 of the
`shaped-floor25` arm returned in under a second with `corpus=1` and ~29 edges —
the calibration edges and nothing else — because the source tree was edited while
those campaigns were launching. They looked like a catastrophic arm effect
(-121, -118, -100) and would have inverted the reported conclusion for that arm.
`rc` was **0** for all three, so a return-code check would not have caught them;
what caught them was that a 145-edge baseline does not produce a 29-edge cell.
Screen cells on corpus size and wall-clock, not just exit status, and never edit
the tree a benchmark is running from.

**The first read of the faithful arm was 5W/7L and the second was identical.**
Noise that lines up with a mechanism is the failure mode the Boltzmann write-up
warned about (7W/2L at ten grep seeds, exactly 5W/5L for the next ten). Here the
direction was consistent, which is weak evidence of a small real cost — plausible
mechanically, since dividing a discovery round's reward by the chain length makes
every arm slower to learn that the operator found anything at all.
