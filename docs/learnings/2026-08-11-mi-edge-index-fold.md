# mi-edge-index-fold: raw 32-bit shim hashes used as dense array indices

**Date:** 2026-08-11
**Context:** fuzzer-new repo; `fuzzer-tool fuzz targets/png_read.so ... --elo all` (in-process direct_lite, sparse-entry SHM) crashed with `MemoryError` in `core/mi.py:141` (`array("Q", [0]) * (edge + 1 - size)`).

## Problem
A normal fuzz run died instantly with `MemoryError` in `MutualInformationTracker.record()` while extending `edge_marginal`. The run had never even reached steady state — a pure data-path blowup (second MI-tracker OOM in this repo; the first, 2026-08-03, was the joint-cell volume).

## Rejected
- **Blame a recent perf commit** — looked plausible (array.array work on `edge_marginal` landed recently); dropped by reading the alloc site: the exponential/repeated-growth perf change was a red herring, the failure is the same `extend()` regardless of container.
- **Grow a sparse dict instead of a dense array** — considered; rejected as over-scope for this fix. The dense `array("Q")` + zero-copy `np.frombuffer` sum is a deliberate perf choice (49-76x over Python sum), and the id space can be bounded without changing the container.
- **Skip edges >= map_size** — looked plausible (just drop out-of-range ids); dropped because it silently zeroes MI signal for the majority of edges on sparse-shim targets (raw hashes are spread across 32 bits, so almost everything is "out of range"); folding keeps every edge's signal.

## Approach
In `record()`, fold each raw edge hash into `[0, map_size)` before counting: AND-mask when `map_size` is a power of two (AFL convention), modulo otherwise. The `map_size` parameter — whose docstring already said "Maximum edge index to consider" — was being passed by the caller (`fuzzer.py:3015`) but never used. Folding is a pure transform done once per record on the set, so the joint keys, the dense marginal, and the JSON round-trip stay coherent; collision-folding mirrors exactly what AFL does with `cur ^ prev & (MAP_SIZE-1)`.

## Key insight
The system had two edge-id namespaces and they were being conflated: the C shim deliberately stores **full 32-bit hashes** (`caller_ctx ^ prev_loc ^ cur_loc`, unmasked for collision-free SHM identity via linear probing), while `edge_marginal` assumes **dense indices** 0..map_size. Any code that (a) receives IDs from a hash-based producer and (b) does `array[len(index)]`-style dense indexing must fold at the boundary — the parameter carrying the bound existed and was ignored.

## Verification
- Mechanism reproduced pre-fix: `record(b'abc', {1<<24}, map_size=16384)` grew `edge_marginal` to 16,777,217 entries (128 MB) — the log shows the user's run died via the same line; real id space is 2^31+, i.e. multi-GB.
- Post-fix: same call yields `len <= 16384`, counts match an independent masking oracle (`{e & (map_size-1)}`); modulo path verified for non-power-of-two maps; small dense ids remain identity under power-of-two maps (existing tests unchanged).
- Regression tests added (`test_regression_mi_tracker_bounded.py`): power-of-two mask fold, modulo fold, oracle-driven counts. Full suite: **4164 passed, 19 skipped** (pre-change baseline not recorded separately; MI-related files re-run green after the change).

## Generalizes to
When a consumer treats opaque hash keys as dense array indices, fold to the container's domain at the ingestion boundary — use the bound the caller already passes rather than re-deriving one. If a value can't be dropped without losing most of the signal (sparse 32-bit ids vs 16K map), fold instead of filter: collisions are the same lossy-but-useful tradeoff every coverage bitmap already makes. When a docstring states a contract ("Maximum edge index to consider"), a code path that ignores the parameter is where the bug is.
