# structured-roundtrip-determinism: seeded pools, aligned writes, and whole-buffer statistics

**Date:** 2026-08-10
**Context:** `fuzzer-new`, `src/fuzzer_tool/core/mutations/structured.py` + `tests/test_structured_mutations.py`, commit `8a55ef1`

## Problem

Three tests in the dieharder-inverse operator suite failed, apparently as flaky
statistical assertions: `test_spacings_collapse` (pval 0.052 vs 0.01),
`test_produces_near_maximal_euclid_step_counts` (50 Euclid steps vs 80), and
`test_de_bruijn_fill_leaves_no_missing_kmers` (0.20 occupancy vs 0.01). The
"seeded" fixtures were supposed to pin these to a fixed sample, so a flake
looked impossible — which meant the seed was a lie.

## Rejected

- **Rerunning to reproduce the segfault at `test_structural_constraints` (same session)** — looked plausible because one full-suite run died at 91%; dropped because it never reproduced again (isolation + 3 full-suite runs), and rerunning tells you nothing about a state-dependent bug. Left as an unconfirmed transient.
- **"Seeding the numpy global generator before constructing the pool"** — the existing fixture (`np.random.seed(SEED)` then `RandPool()`); dropped because `RandPool()` calls `np.random.default_rng(None)`, an *independent* generator fed by OS entropy. The global seed is never consulted.
- **Fixing the de Bruijn test with a hits-count convention (like its sibling `kmer_starve`)** — looked consistent with the module's aggregate-threshold style; dropped because it launders an operator that cannot move its own statistic into a "mostly fires" claim. The round-trip discipline in the file docstring says a construction that fails to move its statistic is worthless; the operator was broken relative to its own name ("k-mer saturation", "leaves no missing k-mers").
- **Adding separate local snaps to each operator** — the `rank_deficient`/`degenerate_geometry` pattern; dropped in favor of one shared `align` parameter on `_region` (AGENTS.md rule: fix the convention in one place), migrating the two existing local snaps into it.

## Approach

1. **Pin the RNG object that actually draws:** fixture becomes `RandPool(SEED)` — the pool's own `default_rng(seed)`, not the global API.
2. **Align word writes in the shared region picker:** `_region(..., align=1)` snaps `offset -= offset % align`, clamps `length` to `data_len - offset`, returns `(0,0)` when the snapped window is shorter than `min_len`. Every word-width operator passes `align=width`; `rank_deficient`/`degenerate_geometry` pass their block/point size.
3. **Make the local construction move the whole-buffer statistic it targets:** `de_bruijn_fill` now tiles the sequence across the entire buffer (unchanged below 16 bytes), instead of splicing a random region the detector mostly ignores.

## Key insight

Only one of the three failures was random. Once the fixture actually pinned the
sample, the other two became hard, deterministic bugs — an alignment mismatch
(`fibonacci_pairs` wrote at offset 562; `_worst_pair` read 8-byte-aligned u64s
and saw 47 steps instead of 91) and a scope mismatch (a region fill against a
whole-buffer detector). The flake was *masking* two deterministic defects;
fixing the determinism first turned them from intermittent noise into
reproducible, debuggable failures.

## Verification

- With the fixture pinned and `_region` aligned, `test_produces_near_maximal_euclid_step_counts` reads the Fibonacci pair at offset 560 and measures 91 steps; `test_spacings_collapse` median pval 0.0.
- Full-buffer de Bruijn tiling measures occupancy 0.0 for every shape at 4096 bytes; the all-20-draws assertion passes unmodified.
- Multi-seed sweep (1, 42, 20240609, 777, 123456): fibonacci 92 everywhere, de_bruijn 0.0 everywhere, birthday median < 0.01 for the pinned seed (seed 42 gives 2.7e-02 — the test's designed median-of-15 tolerance, unchanged).
- Full suite: 4114 passed before staging, 4113 after (one pre-existing urandom calibration flake, passes 5/5 in isolation, unrelated to the diff).

## Generalizes to

- A "seeded" test fixture must seed the RNG object that consumes the draws — check whether it inherits global state or holds an independent `Generator`; seeding a parent API does nothing to the latter.
- When a writer and a reader disagree on alignment, the misaligned output is a *silent no-op*: no error, just a statistic that doesn't move. Round-trip tests are the only thing that notices, and only if they read at the same alignment as production.
- If a detector measures a whole-buffer property, a construction must dominate the whole buffer (or the assertion must use the module's aggregate convention). Small-region effects only survive when the effect is global — starving k-mers works locally, saturating them does not.
- Rerunning a flaky test is worthless as a diagnostic: pin the entropy first, and whatever still fails afterward is a real bug worth chasing.
