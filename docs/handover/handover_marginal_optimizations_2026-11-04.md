# Handover: operator marginal coverage / Shapley bookkeeping optimizations (2026-11-04)

Scope: `core/marginal_coverage.py` (`marginal_table`), `core/shapley.py`
(`ShapleyAttribution.record`, `_prune_edges`, `marginal_coverage`), report section
`services/report.py::_marginal_coverage`. Everything below is measured on one box with a
synthetic workload (264 operators x ~2000 edges each, ids in a 2^16 space, half the
operators sampling a shared 6,000-edge pool); run-to-run noise is about +/-10 ms, so
differences under ~15% are not evidence.

## Delivered (all upstream)

| Commit (upstream) | Change | Measured |
|---|---|---|
| `047e278b` | Greedy cover in `_greedy` is incremental numpy: picking an operator marks its newly covered edges and decrements each holder's gain once (CSR edge->holders, 16-bit radix key when <= 65536 columns). Exact: same order as the old lazy heap, ties by name. | `marginal_table` 112 -> 62 ms |
| `3a2e9add` | `ShapleyAttribution.marginal_coverage()` reads the live per-operator sets instead of `operator_edge_sets()` copies. | -28 ms (the copy cost as much as the flatten) |
| `f6ecfcc4` | `_prune_edges` removes via C-level `dict.pop` (`deque(map(counts.pop, hit), maxlen=0)`); `Counter.__delitem__` is a Python-level override (2.2M calls in the stress run). | 5.4 -> 4.7 ms/record under constant churn |
| `c3921fac` | Per-operator edges stored once: `_operator_edges` is a read-only `Mapping` (`_KeyViews`) of live key views of `_op_edge_count` (the two always held identical sets). Reads never create keys. | `record` 1353 -> 720 us steady; 55 -> 28 MB at 264 ops x 400 records |
| `297998a0` | Gap fixes (below). | see below |

## Gaps found by re-reading, and fixed (`297998a0`)

1. Ids outside int64 (>= 2^63) raised `OverflowError` and would have killed the whole report. They are now factorized (only identity matters). Real ids come from SHM entries and should never be that large; this is hardening.
2. The direct-count path (`bincount`, ids < `DIRECT_MAX_ID`) allocated O(max id) even for a handful of entries: 3 entries with ids near 4M took 25 ms vs 0.1 ms. Direct path now requires `max id < min(DIRECT_MAX_ID, 64*n + 4096)`; otherwise sort.
3. The radix-key branch (`<= 65536` dense columns, uint16) vs int64 fallback at 65536/65537 columns had no test; boundary tests against a brute-force greedy added (results were already correct).
4. Report could mislead: `SHAPLEY_EDGES_MAX` (10,000) evicts the rarest edges first, which are exactly the edges that give an operator `sole`/`gain`. After pruning, zero-gain operators are a weak prune signal. `ShapleyAttribution.prune_count` added; the report prints a note when it is > 0.

Checked, no change needed: removing the set copy is only a race if another thread mutates the sets during the report; the only other thread is the stack-heartbeat daemon and reports run on the main thread. `_KeyViews` survives pickle/deepcopy and keeps pointing at the same `_op_edge_count`. The greedy tie-break matches the old heap version on random cases. The pick loop is O(n^2) only matters at thousands of operators.

## Where the time goes now (264 x 2000, ~62 ms)

flatten sets to numpy ~24 ms; greedy ~16-20 ms (of which the radix argsort ~8 ms, operator->rank gather ~4 ms); dense index 3 ms; holder counts + share + sole ~7 ms. The report runs once at the end of a campaign and the edge cap bounds the worst case at ~2.6M entries (~0.5 s); a 264 x 20,000 input (1.2 s) cannot occur under the cap.

## Tried and rejected (no code change; do not re-litigate without new evidence)

| Idea | Result |
|---|---|
| Flatten with per-set `np.fromiter` instead of one chained iterator | 15% faster as a microbenchmark (21.7 vs 25.6 ms), no gain end to end (66 vs 62 ms; within noise). Plain `list(chain)` alone is 14 ms, so the floor is close. `array.array` variants were slower. |
| Dense bool grid greedy (no sort), row-major or edge-major | 114-146 ms vs 62: cost scales with operators x distinct edges (here ~65k edges), the CSR path with entries (528k). Only wins when edges are few. |
| Early exit once every edge is covered (rest are zero-gain, by name) | Saves ~1 ms of ~11 ms: the cost is building the edge->holders index, not the pick loop. |
| Lazy-scan greedy without CSR (gather over all entries per positive pick) | 8.9 vs 11.2 ms with 4 positive picks, 2x slower at 14 picks, 4x at 51. Real runs have a handful to ~50 positive picks; switching on the pick count would save a few ms in one corner. |
| Holder counts from the cached edge-major view (`_edge_op_count`) | 17.6 vs 12.0 ms; the view costs ~86 ms cold and the marginal report never builds it. |
| Per-operator bitmaps maintained in `record()` | `record` +23% (729 -> 894 us), +17 MB at a 2^16 id space (~1 MB/operator at 2^20), and flatten from bitmaps is slower (30.7 vs 22.2 ms: scanning 264 x 65,536 bytes beats nothing). Compact slot ids need a dict lookup per edge, costlier still. |

## Open items

- If `ShapleyAttribution` ever stores edges as numpy arrays instead of sets (touches `record`, `_prune_edges`, `operator_kernel`, Jaccard helpers and tests), flatten drops to a concatenate; not worth it for a once-per-run report.
- Pruning under constant churn remains the dominant `record` cost (~1.3M set lookups per prune on the adversarial workload); real runs mostly repeat edges, so prunes are rare.
- A benchmark script is not committed; the numbers above come from ad hoc harnesses (synthetic overlap, 264 ops x 2000 edges). If one is wanted, `tools/` is the place.
