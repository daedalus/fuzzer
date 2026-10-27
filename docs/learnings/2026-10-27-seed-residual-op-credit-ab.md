# `--seed-residual` and `--op-credit` vs plain `--elo` on fuzzgoat: exploratory, not adopted

**Question.** `handover_matrix_schedulers_2026-09-19.md` leaves both selector arms
unmeasured ("those move the selector, not the reward"). This is a first paired run on
a real clang build. It is **exploratory**: no threshold was frozen beforehand, so by that
handover's own protocol it cannot adopt either arm.

## Setup

`fuzzgoat_read_noasan.so` built with Ubuntu clang 18.1.3, `-fsanitize-coverage=trace-pc-guard`,
`-fPIC`, linked with `adapters/afl_shim.c` (recipe of handover_coverage_gram_matrix §11,
shared-object variant). `fuzz -c --max-execs 2000 --inprocess-direct -m 65536 -s SEED`
plus the arm flags from `tools/lib/bench_paired.py` `ARMS`. Seeds 0-23, one replicate per cell,
run sequentially. Metric: `Edges discovered` at budget. Run outside `bench_paired.py`
(the locked matrix has no fuzzgoat cell), same as the shaped-reward run. Not run under
`bench_lock`; the ASLR / Jaccard stability preconditions of the handover were not checked.

| arm | flags |
|---|---|
| `elo` (baseline) | `--elo --mc-bandit` |
| `elo-seed-residual` | `+ --seed-residual` |
| `elo-op-credit` | `+ --op-credit` |

Both arms were active: the end-of-run selector table lists `residual` under seed selection and an
`op_credit` row. Baseline: median 340 edges, range 207-427 over the 24 seeds. (The shaped-reward
run's baseline was 145; that build and seed corpus differ, so the numbers do not compare.)

## Result

| arm | cells | W/L | median delta | mean delta | 95% CI on mean (bootstrap) | McNemar exact |
|---|---|---|---|---|---|---|
| `seed-residual` | 24 | 14/10 | +9.0 | +2.2 | [-17.5, +22.7] | p = 0.541 |
| `op-credit`, seed 8 excluded | 23 | 14/9 | +40.0 | +84.7 | [+30.0, +143.1] | p = 0.405 |
| `op-credit`, seed 8 = 594 (rerun) | 24 | 15/9 | +42.0 | +92.4 | [+37.8, +149.2] | p = 0.307 |

**`seed_residual`: nothing.** Interval straddles zero both ways. Its log line does not
carry the `rho_total` / partial-correlation falsification output the handover asks to record,
so the "is it a row sum" question is unanswered here.

**`op_credit`: a tail, not a shift.** Baseline never exceeds 427 edges; `op_credit` reaches
490-643 on 8 of 24 seeds (2, 3, 6, 7, 9, 22, 23, plus seed 8 on rerun) and is at or below baseline on
most of the rest. The mean interval excludes zero because of that tail; the win count does not
(14/9). Read it as "sometimes reaches a region the baseline does not reach in 2,000 execs", not
"better on a typical seed". What is in that region, and whether it is one fixed region, was not
checked.

## Caveats that matter

- **One cell did not finish.** `op-credit` seed 8 died with SIGSEGV; the top Python frame was
  `op_credit.py` `credit()`, a dict comprehension. Two reruns of the same cell finished (594 and
  559 edges) and no other cell in 72 crashed. fuzzgoat is in-process with planted memory bugs and no
  ASAN, so heap corruption surfacing in unrelated code is possible; the arm's own logic is not
  implicated by this evidence either way. The cell is excluded above, with the rerun values as
  sensitivity. Not diagnosed.
- **Seeds are not deterministic.** The same `(arm, seed)` gave 594 and 559 edges. Pairing by seed
  does not remove run-to-run noise at this size.
- One target, 24 seeds, one replicate, 2,000 execs.

## What it does and does not license

Neither arm is adopted, defaults unchanged. `op_credit` earned a pre-registered run: freeze the
threshold first, then `bench_paired.py run --arms elo,elo-op-credit` over the locked matrix with
`--reps` above 1, ASLR off and the stability probe at Jaccard 1.0, and count a gate-tripped cell as
a failed cell. Open: the seed 8 segfault; what the 490-643-edge region is.
