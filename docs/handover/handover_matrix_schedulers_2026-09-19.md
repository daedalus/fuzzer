# Matrix schedulers: `seed_residual` and `op_credit`

Built from `handover_edge_id_axis_2026-09-18.md` ("A scheduler built on all of
this": P2-3, P3-3, P3-4), on top of `core/scheduler_substrate.py`. Both arms
are **off by default and unproven**. What is here is the arms and the
instruments; whether either is any good is a `bench_paired.py` question that
nothing in this document answers.

```
--seed-residual     Elo seed arm "residual"       core/schedulers/seed_residual.py
--op-credit         Elo operator arm "op_credit"  core/schedulers/op_credit.py
                    shared: core/edge_matrix.py (MatrixSubstrate, MatrixFold)
tools/bench_paired.py  arms: elo, elo-seed-residual, elo-op-credit
```

## What each arm does

**`seed_residual`** scores a seed by its incidence mass on the canonical edge
space (`sum 1/owners` over the classes it touches), then regresses out
`rank(total hits)` and keeps the residual. Energy is `0.05 + percentile`, so no
seed is unreachable. It carries its own falsification: at every refit it logs
`rho_total`/`rho_degree` (how much of the score is still volume) and, given
per-seed productivity (`coverage_edges`), the partial correlation of score with
productivity controlling for volume and for degree. A partial near zero means
the arm is a row sum again. It is logged, not thresholded, because the handover
gives no threshold.

**`op_credit`** is Thompson over `Beta(1 + credit, 1 + stale)` where `credit`
is the number of distinct canonical classes the operator has found. The
selector is deliberately stock; the reward is the change, so an Elo A/B against
the existing operator arms isolates it. `2^H` collapsing with a flat edge count
widens exploration and decays stale failure evidence (never credit, never a
reset).

## How each finding lands

| Finding | Where |
|---|---|
| F1 per-process ids make every edge a singleton | `MatrixSubstrate.set_stability` feeds `coverage_trust`; both arms abstain (`available()`), and `op_credit` leaves the ballot |
| F11 uninstrumented target reads as saturated | same gate, same single decision point (`coverage_trust`) |
| F10 duplicate-profile edges | `EdgeCanonicalizer` classes; mass and credit are per class, so a 45-edge chain pays once |
| F6 owners are incidence, volume is not | `1/owners`, never hit volume |
| F5 `2^H` | `MatrixSubstrate.saturation_signal` (shape from the handover, scale uncalibrated) |
| P3-3 paper question: a class splits mid-campaign | credit is a function of the current partition, never an accumulator (test: `test_a_split_raises_credit_because_nothing_is_stored`) |
| P1-2 independent-coordinate mask | `MatrixSubstrate.derived`, empty; nothing fills it until the relations are confirmed against `core/icfg.py` |

Absent on purpose, as excluded by measurement: any low-rank seed score (SVD
leverage, PC1-3; PC2/PC3 wait on P1-3), l2-magnitude sampling, GF(2)/LLL as a
minimiser, every id-axis statistic, and a neural scorer. An earlier draft of
this build carried opt-in SVD leverage, a GF(2) rank flag, a greedy-cover boost
and a context-family bonus. All four were removed after reading the handover's
exclusion list: the first two are on it, and the last two are new hypotheses
the design does not contain. They can come back as separate single-variable
arms if wanted.

## Preflight, and one carve-out worth knowing

The gate is `coverage_trust(target, ..., id_stability=jaccard)`. That function
returns "trusted" early when there is no target path (an in-process callable),
**before** it looks at the stability figure, so a target-less run is never
gated on F1. That is the substrate's own decision and this module does not
second-guess it (`test_no_target_has_nothing_to_distrust` pins it).

The stability probe runs at campaign start. Until it has run the gate reads
`unverified` and the arms are live; a run that skips the probe is never gated
on F1.

## Verified

- 63 tests over the helpers, the fold, both arms and the wiring, with
  hand-derived oracles and a rank-correlation control against itself (Hard
  Rule 46).
- End to end on a real clang `trace-pc-guard` build of `targets/test_target`
  (`tests/test_matrix_wiring.py`, skipped unless the target is built with
  `--clang-scov`): the fold forms from the tracker (3 seeds, 9 edges, 6
  classes), `op_credit` is elected by the real Elo loop, and its credit is in
  class units. The same run against a gcc build (no compiler coverage) shows
  the gate closed with the F11 reason and `op_credit` off the ballot.
- Not verified: any effect on coverage. There is no bench result.

## Known limits

- The fold is over the tracker's `seed_hit_counts`, i.e. one row per recorded
  input, bounded by tracker pruning. It is the matrix the handover measured
  only to the extent the tracker keeps the same rows.
- The fold is skipped above `CELL_BUDGET` (2M seeds x edges, the same bound as
  the summary's class scan), then the arms abstain rather than score stale
  data. Large campaigns get no arm until the fold is made incremental.
- `SATURATION_SCALE` (0.25) and `op_credit.DECAY` (0.5) are uncalibrated. The
  handover gives the direction, not the size.
- Operator credit state (`_found`, `_pulls`) is in memory and not resumed.
- ~~The reward-shaping form the handover calls the cheapest A/B, scaling the
  shared `op_rewards` weight for **every** scheduler by
  `OpCreditScheduler.shaped_weight`, exists and is tested but is not wired.~~
  **Wired 2026-09-20** as `--shaped-reward` (`Fuzzer._credit_reward_shape`, one
  multiply in the single shared `op_rewards` loop), with `--shaped-reward-floor`
  as the clamp on the two ways the factor collapses. Measured on fuzzgoat and
  **not adopted**: see §"A/B result" below. The arithmetic moved to a module-level
  `shaped_weight(substrate, edges, floor)`; the method is a thin bind, because the
  shaping is for every arm and must not require electing this selector.
- Nothing populates `derived` (P1-2) and PC2/PC3 are unused (P1-3).

## A/B result (2026-09-20): shaped reward measured, not adopted

`--shaped-reward` was wired and run against plain `--elo` on a clang
`--clang-scov` fuzzgoat build (Rule 52), paired over seeds 0-11 at 2,000 execs.

| arm | W/L | median delta | 95% CI on mean | McNemar |
|---|---|---|---|---|
| `--shaped-reward` (floor 0) | 5/7 | -7.0 edges | [-13.5, +1.6] | p = 0.774 |
| `+ --shaped-reward-floor 0.25` | 5/7 | -2.5 edges | [-8.9, +10.4] | p = 0.774 |

Baseline median 145 edges. No gain in either form, and the faithful form's
interval sits mostly below zero. Off by default, floor default 0.0, and out of
`--hail-mary` on the strength of this rather than on caution. Full write-up,
including the three cells that had to be discarded and re-run, in
`docs/learnings/2026-09-20-shaped-reward-ab-result.md`.

This closes the cheapest of the arms' A/Bs. It says nothing about `op_credit`'s
own selector or about `seed_residual`: those move the selector, not the reward,
and are still unmeasured.

## Protocol before it can be on by default

Same shape as the other paired arms: `tools/bench_paired.py run --arms
elo,elo-seed-residual` and `elo,elo-op-credit` over the locked matrix, analysed
by McNemar on discordant cells plus median per-cell edge delta. The maintainer
freezes the threshold **before** the first run; observational correlation has
been wrong twice on this exact question. Preconditions per campaign: ASLR off
and the stability probe at Jaccard 1.0, on a compiler-instrumented build (both
are what the gate checks, so a run that trips the gate is a run whose arm was
silent and must not be counted as an arm result). Also record the
falsification line for `seed_residual`: if the partial correlations sit at
zero the arm is a row sum whatever the win count says.
