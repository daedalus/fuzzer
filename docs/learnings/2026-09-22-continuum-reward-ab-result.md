# `--continuum-reward` A/B: no gain, off by default, out of `--hail-mary`

**Question.** Promoting `ContinuumField` (the Navier-Stokes analyzer,
`analyzer_navier_stokes.py`) to its own operator-mutation scheduler was rejected
on rank-equivalence grounds: any score built from an operator's own
successes/failures plus a global scalar ranks operators the same as raw success
rate (768/20,000 sampled disagreements, all cap ties). The alternative that
survived that argument was a *reward* factor instead of a *selector*: scale a
discovery's reward by `frontier_weight`, the mean pressure of the edges co-hit
alongside it — close to 1 in unvisited territory, close to 0 filling a gap
between well-owned edges. That reaches every scheduler through the same
multiplicative seam `--shaped-reward` uses, so it is measurable the same way.
This run asks the same narrow question `--shaped-reward`'s did: default on or
off, and in or out of `--hail-mary`.

**Answer: off, and out.** The point estimate is a loss, and it is not resolved
at this cell count.

## Setup

Same protocol as the `--shaped-reward` run (`2026-09-20-shaped-reward-ab-
result.md`), so the two are comparable: `targets/fuzzgoat_read_noasan.so`,
clang 18 `--clang-scov` build (compiler-inserted edge coverage), in-process,
`-m 65536`. Paired cells over seeds 0-11, 2,000 execs per campaign, one
replicate per cell, run sequentially on a single CPU. Metric: `Edges
discovered` at budget. Arms:

| arm | flags |
|---|---|
| `elo` (baseline) | `--elo --mc-bandit` |
| `continuum-reward` | `+ --continuum-reward` (floor 0.0, the faithful form) |

Run outside `bench_paired.py` (no fuzzgoat cell in the locked matrix, same as
the shaped-reward run); the arm is registered there (`elo-continuum-reward`)
so a matrix run can be added without re-deriving the flags.

Baseline: median 146 edges (115-159 across the twelve seeds) — noticeably
noisier floor-to-ceiling than the shaped-reward run's baseline (130-163) despite
being nominally the same target and arm; the two runs are ~48 hours apart on
whatever else is live in `torch`/mutation registries by then, which this
harness cannot control for.

## Result

| arm | cells | W/L | median Δ | IQR | mean Δ | 95% CI on mean Δ | McNemar exact |
|---|---|---|---|---|---|---|---|
| `continuum-reward` | 12 | 4/8 | **-3.5** | [-8.2, +6.0] | -1.6 | [-10.8, +7.6] | p = 0.388 |

Same reading as the shaped-reward and Boltzmann results: not "it does nothing",
but "it does not move edges by more than about ±11 (7% of a 146-edge baseline)
on this target at this budget, and the point estimate sits on the loss side".
Twelve cells at this variance cannot resolve an effect this size either way.

The shaping was not inert: the banner carried `continuum-reward` on every
shaped cell (`Scheduling: continuum-reward, bandit, elo`), confirmed directly
rather than inferred from the result. Whether it was pricing anything
*meaningfully differently* from `surprisal_weight` on this particular target is
not separately measured here — `continuum_reward_stats()` exists for that
follow-up (mean factor, shaped-round count, neutral-round count) but nothing
calls it yet, same gap `shaped_reward_stats()` has.

## Decision

* `--continuum-reward` stays **off by default**.
* It stays **out of `_HAIL_MARY_FLAGS`**, for the same structural reason
  `--shaped-reward` is out (rescales what every strategy is paid; folding it in
  destroys the single-variable run it exists for) — this run adds no further
  reason to reconsider that, since the measured direction is a loss, not a gain
  being left on the table.
* `--continuum-reward-floor` default stays **0.0**. No cell here reached the
  fully-saturated-neighbourhood case the floor is a hedge against; there is
  nothing in this data that argues for a non-zero default.

## One thing worth flagging, not yet resolved

`frontier_weight` and `shaped_reward`'s `derived`-chain factor are answering
different questions (*where* a discovery landed vs *how duplicated* it was) and
compose multiplicatively at the same call site (`fuzz_one`), but this run only
tested `continuum-reward` alone against `elo`. Whether the two combined move
edges in either direction — including whether they cancel, compound, or fight
each other on rounds where both fire — is untested. That is a third arm
(`elo-continuum-reward-shaped-reward` or similar), not yet registered in
`tools/bench_paired.py`, and would need its own single-variable framing against
one of the two alone rather than against `elo`.
