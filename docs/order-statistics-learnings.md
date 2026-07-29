# Order-statistics analysis: applying uniform-order-statistic theory to the fuzzer

**Date:** 2026-07-29
**Context:** fuzzer-new repo, from order_statistics.py analysis

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

The spacings trick (Dirichlet(1,...,1) = normalized Exponentials) is the most actionable result from order-statistics theory for fuzzing. It produces sorted uniform cut points without sorting — useful wherever a mutation operator needs to segment input at random boundaries. The other results (Beta equivalence, extreme-value asymptotics) primarily validate existing implementations rather than enabling new ones.

## Verification

- Operator smoke test (`test_all_ops_fire`) passes — the new operator dispatches and executes without crashing.
- Full test suite: 2530 passed, 14 skipped, 8 failures all pre-existing (confirmed by git stash comparison). Zero regressions.
- `block_shuffle_variable` produces valid output on 32-byte input (established by the dispatch smoke test's sample buffer).

## Generalizes to

The spacings/Dirichlet trick is a general method for generating sorted random points without sorting. Any algorithm in any domain that needs N sorted uniform random values in an interval can use it: sample N+1 Exponential(1) draws, normalize, cumsum. Avoids an O(N log N) sort and is trivially parallelizable.
