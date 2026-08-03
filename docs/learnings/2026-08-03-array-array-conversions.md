# array-array-conversions: when array.array wins, and when it doesn't

**Date:** 2026-08-03
**Context:** fuzzer-new repo (src/fuzzer_tool); task "analyze where we can replace list for array.array" — planned, implemented in two phases, third phase dropped on benchmark evidence.

## Problem
AGENTS.md's style rule says "prefer array.array over Python lists for homogeneous numeric data," but the rule gives no guidance on WHERE that's actually a win. A full audit was needed before converting anything, and the naive reading ("all numeric lists → array.array") is wrong on the hot paths.

## Rejected
- **Dense array('Q') rewrite of EdgeTracker's edge-count maps** (the "big memory win") — looked plausible because 5 sparse dicts at ~100B/entry cost more than 8B/slot dense arrays above ~50% map saturation; dropped after the user's array-vs-list-vs-numpy benchmark showed `array.array` is *slower* than dict on scalar read-modify-write (its getitem boxes a fresh PyLong; `dict.get` returns the stored one) and slower on Python-level iteration, while the heavy readers (shannon/simpson entropy) were already numpy-vectorized via `np.fromiter(hits.values())`. A modulo-remap (`eid % map_size`) would also have introduced collision-merge semantics — a metric change. The plan's own "record_edges gets faster" claim was wrong and was retracted.
- **Converting hot-path lists** (execution_time._sorted, walked every fuzz_one in the CRPS computation) — looked plausible on homogeneity alone; dropped because array access is slower than list and the real cost is the `bisect.insort`/middle-pop O(n) memmove, an algorithm issue no container swap fixes.
- **`array.array` for deques** (running_stats, execution_time windows) — needs popleft/maxlen eviction; array has neither.
- **Two-positional-arg signature** for stats_reporter's discovery functions during Phase B — rejected by the impactguard pre-commit hook as a breaking positional change; resolved by passing a paired `(execs, edges)` tuple as the single existing arg (TYPE_WIDENED = non-breaking).

## Approach
Converted only cold, bounded, append-heavy histories — the structures where memory compactness is real and runtime is not on any measured path:
- corpus-size history → `array("I")`; the four tuple histories (discovery, crash-rate, entropy, coverage timeline) → parallel `array("Q")`/`array("d")` pairs with lockstep append/trim and `zip(..., strict=True)` windows; elo prediction-error lists → `array("d")`; redqueen pair-length index → `array("I")` (bisect consumer unchanged).
- JSON state formats kept byte-identical: save converts back to plain lists (`list(arr[-500:])`, `list(zip(execs, edges))`), load rebuilds arrays.
- Dropped the report.py `isinstance(history, list)` guard (a hard break on array) in favor of hasattr on the renamed array attributes.
- Every conversion shipped with a regression test whose expectations are derived independently (reference arithmetic, not the code under test).

## Key insight
"Homogeneous numeric" is necessary but nowhere near sufficient. array.array is a memory-only optimization in CPython: it never wins on speed (unboxing a C value into a fresh PyObject per read is strictly more work than list's pointer dereference). A conversion pays only for large persistent stores, append-heavy structures, or memory pressure — and per-element hot loops should never touch it. The structure that wins BOTH memory and speed for the read paths is numpy (vectorized), which this codebase already used over the sparse dicts.

## Verification
- Container-level `sys.getsizeof` benchmarks: 3.8x-8.6x memory per container (~283KB → ~37KB for the histories at cap).
- Interleaved 3-state runtime benchmark (base vs Phase A vs Phase B; fresh corpus copy per run, 3 timed + 1 profiled 10k run each): wall-clock medians 8.9/8.7/8.3s (within ±2s noise); every touched function flat in cProfile (worst: elo `_effective_k` +0.1ms per whole run); peak RSS flat ~60.6-60.9MB.
- Full pytest suite green at each phase (3041 → 3045 → 3050 passed).

## Generalizes to
Before applying a "prefer X over Y" container/style rule, audit where the rule's *premise* holds: measure the operation profile (hot scalar RMW vs cold bulk read vs incremental build) and pick the structure per operation — list/dict for Python-level scalar building, numpy for vectorized compute, array.array only for memory-constrained storage without a vectorized consumer. Benchmark the payoff (both axes) before and after, and let the evidence override both the style rule and the plan. Also: independent-derivation regression tests catch container-swap semantic drift, and pre-commit API guards (impactguard) treat arity changes on module-level functions as breaking even when every caller is updated in-repo.

## Follow-up (2026-08-03): vectorize the Python-level numeric walk, keep the rewrite dead-code-aware
Applied the same pattern to the next-two-hot sites. Reusable specifics:
1. **A per-exec sequential recurrence can collapse to a one-shot numpy expression — name the recurrence, not the loop.** `_compute_crps`'s per-fuzz_one walk was a prefix sum `crps = Σ (i/n − 𝟙[vᵢ≥obs])² · (vᵢ₊₁−vᵢ)` + a tail term; recognizing that (vs. "a loop over a list") let it become `np.arange(n)/n − (arr>=obs)` then `sum(d²·diff)` — 3.9x at n=200. The old `gap>0` guard was provably dead (sorted ⇒ gap≥0); don't carry dead branches into the rewrite. Equivalence-tested against a verbatim legacy copy.
2. **Only vectorize above a dispatch threshold — the crossover is measurable, and it's tiny.** `count_nonzero(frombuffer(a,uint8) != frombuffer(b,uint8))` for hamming has ~1.3µs fixed cost vs ~30ns/byte for the genexpr; measured crossover ~16 bytes. Gate at 64 bytes (reusing the file's existing `levenshtein_align` `na<64` rule rather than inventing one); numpy loses 1.5x at 16B but wins 5.5x/17x/97x at 256B/1KB/8KB. `np.frombuffer(bytes→uint8)` is itemsize-1, so no length-divisibility guard.
3. **Cached zero-copy views must be released before a resizing `array.array.extend`** (BufferError otherwise) and reset whenever the array is reassigned (a stale view dangles into freed memory) — the real cost of the "view over a growable buffer" pattern.
