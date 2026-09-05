# Order-statistics analysis: applying uniform-order-statistic theory to the fuzzer

**Date:** 2026-07-29
**Context:** fuzzer-new repo, from an `order_statistics.py` analysis script.
That script was never committed and does not exist in any commit of this
repo — five code comments cited it by path and by "Part 3"/"Part 4" until
2026-09-05. This file is the artefact; cite it, not the script.

## Problem

The fuzzer already uses distributions heavily (Beta for Thompson sampling, Dirichlet for CEM, Exponential for schedules) but these had grown organically without a unified understanding of the underlying uniform-order-statistics framework. Could the theory from `order_statistics.py` — closed forms, Beta equivalence, Dirichlet spacings, extreme-value asymptotics — inform improvements to existing algorithms or suggest new operators?

## Rejected

- **Replace GA tournament selection with spacings trick** — the GA already uses the optimal `rank = N * (1 - U^(1/k))` identity from order-statistics theory. No improvement possible.
- **Replace crossover/splice cut-point generation with spacings** — these operators generate cut points without sorting (`cut2 = uniform(cut1+1, len-1)`). Adding the spacings trick would add complexity for zero gain.
- **Replace Pareto-front weighted selection with spacings** — the seed picker already uses `random.choices()` with precomputed weights. No sorting overhead, no spacings win.

## Approach

Three actions were taken:

1. **CEM Dirichlet docstring** — documented the connection between the CEM's per-byte Dirichlet concentration `α₀` and the order-statistics result that spacings of uniform draws are jointly Dirichlet(1,...,1) ≡ normalized Exponentials. Both use the same Dirichlet family over different categorical structures (byte values vs. gap positions).

2. **Extreme-value stale seed comment** — noted that `n·min(U₁..Uₙ) → Exp(1)` from extreme-value theory gives a simpler stale-seed heuristic: `P(stale) ≈ 1 - exp(-fuzz_count · 0.01)`, matching the existing Beta CDF asymptotically while avoiding the integral.

3. **`block_shuffle_variable` operator** — a new mutation operator using the spacings trick (Part 3 of order_statistics.py): generate k+1 i.i.d. Exponential(1) draws, normalize by their sum, and cumsum to produce k sorted cut points without a sort. The resulting variable-width blocks are then fully shuffled. Complemented the existing `chunk_shuffle` (fixed-width chunks from honggfuzz) with a variable-width counterpart.

## Key insight

The spacings trick (Dirichlet(1,...,1) = normalized Exponentials) produces
sorted uniform cut points without sorting. The other results (Beta
equivalence, extreme-value asymptotics) primarily validate existing
implementations rather than enabling new ones.

**Corrected 2026-09-05.** Both halves of that paragraph were wrong.

The spacings trick was called "the most actionable result" on the grounds
that it avoids an O(N log N) sort. It does, and the sort is not what costs.
Measured against `sorted(r.random() for _ in range(k))` in
`block_shuffle_variable`, the spacings path loses by 2.3-2.6x at every size
from k=1 to k=32, because the Python cumsum it needs is dearer than the sort
it removes. Batching the draws through `expovariate_list` does not save it:
`sorted-unif` wins k=1..8 and `np.sort` over `random_list` wins k=16..1024;
the scalar spacings path never wins at any measured size. The operator keeps
it — it is that operator's defining construction and its draw sequence is
part of what `--seed` reproduces — but it should not be ported anywhere on
speed grounds.

"Beta equivalence primarily validates existing implementations" was the more
costly error: it treated the identity as a *check* and never as a cheaper
*sampler*. `Beta(a,1) == U**(1/a)` and `Beta(1,b) == 1-(1-U)**(1/b)` are one
uniform draw and a pow against two gammavariate draws. Whether that helps
depends entirely on how often a posterior still has a parameter at the prior,
which nobody measured at the time:

- operator bandits — 100% degenerate at 1k pulls, 42% at 100k, 0% once the
  discovery rate is healthy. Closed form goes 4.09x -> 1.05x -> 0.97x across
  that range, i.e. it becomes a loss. One vectorized `Generator.beta` call
  over all 155 arms is 9-11x in *every* regime and ignores parameter shape.
  Layering the identity on top of the vectorized call is slower than the
  vectorized call alone.
- seed bandit — 83% degenerate at 500 seeds, 95% at 2000, 99% at 8000, and it
  does not decay, because per-seed discovery is rare. Shipped in
  `core/seed_quality.py::_beta_sample`, 2.15x/2.80x/3.12x on the draw loop.
  Vectorizing measured faster and was rejected: `--seed` does not reach a
  `default_rng` Generator.

The general lesson is that an identity is only a speedup where the branch it
special-cases is common, and "common" is a property of the workload, not the
algebra. Measure the parameter distribution before believing either answer.

## Method

`stats.chisquare(observed, expected)` cannot compare two *samples* — it
treats one as fixed frequencies, halves the variance, and roughly doubles
chi2. It reported `chi2=526` on 264 arms (df 264, p=0.0000) for two
selection policies that were identical. `stats.chi2_contingency` on the 2xk
table gives p=0.74 for the same data. This is now Hard Rule 46: always run
the control against itself.

## Verification

- Operator smoke test (`test_all_ops_fire`) passes — the new operator dispatches and executes without crashing.
- Full test suite: 2530 passed, 14 skipped, 8 failures all pre-existing (confirmed by git stash comparison). Zero regressions.
- `block_shuffle_variable` produces valid output on 32-byte input (established by the dispatch smoke test's sample buffer).

## Generalizes to

The spacings/Dirichlet trick is a general method for generating sorted random points without sorting. Any algorithm in any domain that needs N sorted uniform random values in an interval can use it: sample N+1 Exponential(1) draws, normalize, cumsum. Avoids an O(N log N) sort and is trivially parallelizable.
