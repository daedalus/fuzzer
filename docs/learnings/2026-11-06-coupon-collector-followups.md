# Coupon-collector follow-ups (2026-11-06)

Target: `fuzzgoat_read_noasan.so`, clang, no ASAN (this sandbox has no ASAN
runtime), `-D dictionaries/json.dict`.

## (a) ETA calibration

`GoodTuringSeedStrategy.execs_to_edges(k)` logged every 250 execs; observed =
execs until the global table gains k distinct edges. 10 seeds x 20k execs per
estimator. `r = log2(pred / observed)`:

| est | k | n | median r | IQR | within 2x | "> window" held |
|---|---|---|---|---|---|---|
| gt | 1 | 779 | +0.92 | [-0.11, +2.34] | 44% | 0/0 |
| gt | 10 | 723 | +1.52 | [+0.96, +2.18] | 27% | 2/11 |
| gtoul | 1 | 782 | +1.10 | [+0.08, +2.34] | 42% | 0/2 |
| gtoul | 10 | 568 | +1.63 | [+0.98, +2.48] | 26% | 6/173 |

Bias is stable by phase (5k-exec bins: +0.7..+1.4 at k=1, +1.3..+1.8 at k=10).
Reading: the forecast is pessimistic ~2x (k=1) to ~3x (k=10). Good-Turing /
Good-Toulmin assume a stationary sampler; coverage-guided admission keeps
changing the sampler and finds edges faster than that model allows. The
`> window` claim is mostly wrong for gtoul k=10 (6/173 held). No correction
applied: one target, no basis for a factor.

## (b) Forced first pass: `--round-robin` vs baseline

`bench_paired.py --set fuzzgoat`, 20 seeds x 10k, rep 1.

| arm | W/L | McNemar | med delta |
|---|---|---|---|
| round-robin | 9/11 | 0.824 | -14.5 |
| round-robin-ctl (A/A) | 8/12 | 0.503 | -4.5 |

NULL. Expected: the uniform picker's coupon tail costs ~n*H_n picks once per
run, a small share of 10k execs.

## (c) Reachability budgets

`coverage_budget(n)` (uniform-picker bar, miss <= 1e-9) as the reach budget,
317 registered ops (bar 8,395): every scheduler passes except CMA-ES (covers at
10,479 draws) and IDS (9,537), ~5x the uniform coupon time n*H_n (~2,000) under
flat 5% rewards. The reach test keeps its 20k cap; Tsallis (covers 252 arms by
draw 478) now uses the derived bar.
