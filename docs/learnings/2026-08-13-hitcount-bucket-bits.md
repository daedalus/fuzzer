# hitcount-bucket-bits: representative bucket values are not virgin-map bits

**Date:** 2026-08-13
**Context:** fuzzer repo at `ff7061a`; closing "no hit-count bucketing on the SHM path" from the edge-coverage analysis. `ShmCoverage._check_new_coverage` decided novelty by `ids - self._seen_edge_ids` — set membership only — while `core/count_class.py` had held a complete, vectorized port of AFL's count classification since it was written. The two halves had simply never been connected.

## Problem
An input driving a loop 2 times and one driving it 128 times produced the same edge set, so the second was judged old and discarded. Bucketed hitcounts are the mechanism by which AFL crosses loop-count-guarded branches (`if (n > 16)`, buffer-growth paths, parser backtrack limits), and none of that signal reached the scheduler.

The obvious fix — wrap the existing `classify_single` in a virgin map — is wrong, and quietly so.

## Rejected
- **`virgin[eid] |= classify_single(count)`** — the shape the analysis document sketched. `_classify_byte` returns a *representative value* per bucket (0, 1, 2, 3, 4, 8, 16, 32, 64, 128), and those are not disjoint: class 3 is `0b11`, which is class 1 OR'd with class 2. A virgin map accumulates by OR, so an edge seen once and then twice leaves virgin at `0b11`, and a later count of exactly 3 reports no new bucket and is dropped. Confirmed directly before writing any of the fix.
- **Replacing the set-membership test rather than OR-ing with it** — a new edge whose `count` is 0 occupies no bucket. The C shim never writes such an entry (a claimed slot starts at 1), but hand-built test tables and torn reads do, and `cumulative_edges` has to stay an edge count regardless.
- **Extending the ladder upward for the uint32 `count` field** — tempting, since counts here are real values rather than AFL's wrapped uint8, and `if (n > 1000)` guards exist. Rejected for this patch: it changes which inputs are judged interesting, and shipping an unmeasured deviation from AFL inside an already-unmeasured change makes the A/B uninterpretable. One-line edit to `bucket_bit` when someone wants to measure it.
- **A dict keyed by edge_id** — correct and simple, but it is an interpreter loop over every active entry on every coverage-changing execution. Measured at 108ms per exec on 200k active edges. edges/second is the metric; the novelty test cannot cost more than the executions it is judging.
- **Sorted parallel arrays + `searchsorted`** — vectorized, but `searchsorted` is a cache-missing binary search per query and dominated everything else at 28.7ms on 200k edges (~100% on top of the set-diff already on that path). Sorting the queries first bought 10%.

## Approach
`bucket_bit`/`bucket_bits` in `count_class.py` as a separate ladder, bit-identical to AFL's `count_class_lookup8`, leaving `classify_*` semantics untouched. Virgin map keyed by `edge_id` directly into a dense `uint8` array: 1.8ms on 200k active edges, 6–12% on top of the existing `set(...) - _seen_edge_ids` diff.

Direct indexing is affordable because `trace_pc_guard_init` hands out small sequential guard values, so `edge_id = prev_loc ^ cur_loc` stays in a range of roughly `2 * guard_count`, and XOR with an `__AFL_CTX_BITS`-wide context term cannot widen a value past its wider operand — the default 8-bit context does not change the picture. `__AFL_CTX_BITS` in the 24..32 range is the one configuration that scatters ids across the whole u32 space; those go to a dict (`VIRGIN_DENSE_MAX`).

`record_edge` folds its counts in too. It stands in for both the writer and the reader, so leaving the reader half stale would make a recorded edge report as new coverage on the next scan.

## Key insight
Two representations of "bucket" were being conflated. Comparing two classified traces byte-for-byte only needs each bucket to map to a *distinct* value; accumulating into a virgin map needs each bucket to map to a *disjoint bit*. AFL's table satisfies both, which is why the distinction is invisible when reading AFL — this port satisfies only the first, and the failure mode is silent coverage loss on one specific count with no error anywhere.

The fast path survives untouched for an unrelated reason worth writing down: the shim advances `path_hash` on *every* edge fire, count bump and full-table miss included. An unchanged hash therefore means no count can have moved, so an O(1) header comparison is still a sound guard over a multiplicity-sensitive test. The `path_hash == 0` fallback is blind to multiplicity and is test-only.

## Verification
- Aliasing reproduced pre-fix: `classify_single(3) & ~(classify_single(1) | classify_single(2)) == 0`.
- `bucket_bit` checked against a written-out `count_class_lookup8` for all 256 inputs; all eight non-zero values confirmed single-bit.
- A/B against unpatched `HEAD`, same edge set with only the loop body's count moving — before: `True` once then `False` for 3, 5, 40, 128. After: `True` at each bucket crossing, `False` on a repeat of 2 and on 5000 (same 128+ bucket as 128).
- Cost measured against the set-diff already on that path, not in isolation: 48.7% added at 1k active edges, 11.5% at 10k, 6.1% at 200k.
- 30 new tests in `test_hitcount_buckets.py`. Full suite **4209 passed, 155 skipped** (baseline 4179); the one failure (`TestSmtRequiresCmplog`, z3 absent) reproduces identically on unpatched HEAD. mypy: 55 errors before and after, ratchet unchanged.
- **Not measured:** no coverage delta against a real target. No clang in the container, so no sancov build was fuzzed. Whether bucketing pays for itself in edges/second, and how much corpus growth it causes, is unknown — `tools/bench_paired.py` before this becomes a default.

## Generalizes to
When porting a lookup table, port the property the *consumer* depends on, not the values. A table that round-trips correctly under equality can still be wrong under bitwise accumulation, and nothing will raise. Before wrapping an existing helper in a new algebra, check that its outputs satisfy that algebra's assumptions — here, one line of arithmetic on three constants.

Second: when a fix lands on a per-execution path, benchmark it against what that path already costs rather than in isolation. The absolute number (1.8ms) looks alarming; the number that matters (6% on top of an existing 28.9ms) is the one that decides whether the change is shippable.
