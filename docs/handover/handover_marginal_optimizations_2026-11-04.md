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
| `c60e1ea` | `_all_edges` is a property (key view of `_edge_total`), not a second set updated per exec; prune reads `_edge_total` keys/values directly. | `record` -10 to -20% (3.5 -> 2.8 us at 10 edges, 54 -> 49 at 200, 818 -> 761 at 2000) |
| `fd4b478` | Edge cap is batch Misra-Gries (`core/misra_gries.py::reduce_counts`): on overflow subtract the (k+1)-th largest `_edge_total` count, evict non-positive. Edges above n/(k+1) observations survive; totals undercount by <= n/(k+1); ties evict together. Replaced drop-rarest-half (lost 2 of 8 heavy edges in the 8x3 + 3 newcomers, cap 10 case). | see §Misra-Gries numbers |

## Gaps found by re-reading, and fixed (`297998a0`)

1. Ids outside int64 (>= 2^63) raised `OverflowError` and would have killed the whole report. They are now factorized (only identity matters). Real ids come from SHM entries and should never be that large; this is hardening.
2. The direct-count path (`bincount`, ids < `DIRECT_MAX_ID`) allocated O(max id) even for a handful of entries: 3 entries with ids near 4M took 25 ms vs 0.1 ms. Direct path now requires `max id < min(DIRECT_MAX_ID, 64*n + 4096)`; otherwise sort.
3. The radix-key branch (`<= 65536` dense columns, uint16) vs int64 fallback at 65536/65537 columns had no test; boundary tests against a brute-force greedy added (results were already correct).
4. Report could mislead: `SHAPLEY_EDGES_MAX` (10,000) evicts the rarest edges first, which are exactly the edges that give an operator `sole`/`gain`. After pruning, zero-gain operators are a weak prune signal. `ShapleyAttribution.prune_count` added; the report prints a note when it is > 0.

Checked, no change needed: removing the set copy is only a race if another thread mutates the sets during the report; the only other thread is the stack-heartbeat daemon and reports run on the main thread. `_KeyViews` survives pickle/deepcopy and keeps pointing at the same `_op_edge_count`. The greedy tie-break matches the old heap version on random cases. The pick loop is O(n^2) only matters at thousands of operators.

## Where the time goes now (264 x 2000, ~62 ms)

flatten sets to numpy ~24 ms; greedy ~16-20 ms (of which the radix argsort ~8 ms, operator->rank gather ~4 ms); dense index 3 ms; holder counts + share + sole ~7 ms. The report runs once at the end of a campaign and the edge cap bounds the worst case at ~2.6M entries (~0.5 s); a 264 x 20,000 input (1.2 s) cannot occur under the cap.

## Misra-Gries numbers (`fd4b478` vs `c60e1ea`)

Harness: 300 records, min of 7, 264 ops; `steady` = 200 paths from an 8k-edge pool (never prunes), `big` = 30k pool (prunes often), `churn` = every exec all-new edges. us per `record`; round 2 cut short after two rows.

| Workload | before | after | `shapley_values` before -> after |
|---|---|---|---|
| steady 25 x <=4 | 8.5 / 8.7 | 8.6 / 8.6 | 2.26 -> 2.22 ms |
| steady 2000 x <=8 | 734 / 753 | 739 | 101 -> 101 ms |
| big 500 x <=4 | 475 | 368 | 10.1 -> 5.1 ms |
| big 2000 x <=8 | 4043 | 2429 | 37.1 -> 4.8 ms |
| churn 50 x <=4 | 25.9 | 26.4 | 3.9 -> 2.8 ms |
| churn 2000 x <=8 | 3158 | 2006 | 8.6 -> 4.2 ms |

No prune: unchanged. Prune-heavy: faster partly because ties at the cut evict together, so fewer edges stay tracked (fewer for the report too). Skewed workload (half of each exec from a 3k hot core, 3000 execs): both 94 prunes, ~7.5k tracked, 2937/2937 heavy edges kept (old: 2999/3000 hot core, new: 3000/3000).

## Keep every edge (supersedes the Misra-Gries cap)

Misra-Gries batch reduction evicts every edge tied at the cut, so one exec wider than the cap (19,000 edges, all count 1) left 0 edges; repeated wide execs also drained hot counts. Cap removed. Store: `_CounterCounts` (master's Counters, exact) until, after 1000 records, mean edges x (ops + 1) >= 250 or pairs >= 1M; then one migration to `_EdgeCounts` (uint32 lanes x slots + int64 totals, pending Counters for small execs, folded with numpy before reads).

Warm harness (1300 records warm, 2000 timed + one read, min of 5), us/exec record+read, Misra-Gries master vs keep-all:

| Workload | master | keep-all | mem MB master / keep-all |
|---|---|---|---|
| steady 10 x 1 | 3.5 | 3.9 | 1.2 / 1.2 |
| steady 25 x <=4 | 16.8 | 16.2 | 7.3 / 7.3 |
| steady 60 x <=8 | 56.6 | 37.5 | 21 / 40 |
| steady 200 x <=4 | 98.9 | 61.4 | 37 / 41 |
| steady 500 x <=2 | 165.0 | 99.9 | 56 / 49 |
| steady 2000 x <=8 (always-dense run) | 952 | 330 | 75 / 65 |
| 30k pool 500 x <=4 (always-dense run) | 453, keeps 8,077 edges | 146, keeps 28,950 | 10 / 119 |
| 30k pool 2000 x <=8 (always-dense run) | 2653, keeps 3,656 | 1665, keeps 30,000 | 25 / 226 |

End-of-run report slower on the dense store (sets materialised for `operator_families`): 2000-wide 485 -> 900 ms, 30k pool 51 -> 3177 ms (8x more edges kept). Not measured in-run.

## Tried and rejected (no code change; do not re-litigate without new evidence)

| Idea | Result |
|---|---|
| Flatten with per-set `np.fromiter` instead of one chained iterator | 15% faster as a microbenchmark (21.7 vs 25.6 ms), no gain end to end (66 vs 62 ms; within noise). Plain `list(chain)` alone is 14 ms, so the floor is close. `array.array` variants were slower. |
| Dense bool grid greedy (no sort), row-major or edge-major | 114-146 ms vs 62: cost scales with operators x distinct edges (here ~65k edges), the CSR path with entries (528k). Only wins when edges are few. |
| Early exit once every edge is covered (rest are zero-gain, by name) | Saves ~1 ms of ~11 ms: the cost is building the edge->holders index, not the pick loop. |
| Lazy-scan greedy without CSR (gather over all entries per positive pick) | 8.9 vs 11.2 ms with 4 positive picks, 2x slower at 14 picks, 4x at 51. Real runs have a handful to ~50 positive picks; switching on the pick count would save a few ms in one corner. |
| Holder counts from the cached edge-major view (`_edge_op_count`) | 17.6 vs 12.0 ms; the view costs ~86 ms cold and the marginal report never builds it. |
| Dense (operator x edge slot) int64 count matrix replacing the per-operator Counters in `record()` (fused numpy scatter per exec; small execs buffered in Counters and flushed on read) | Wins only on wide execs: `record` 810 -> 326 us at 2000 edges x <=8 ops, 4360 -> 2090 us on a 30k-edge pool. Loses where fuzzgoat lives (<75 edges/exec, 56% single-op, measured): 3.3 -> 10 us at 10 edges, 10 -> 20 us at 25 x 4; `shapley_values` 2-3x slower at small sizes, report 332 -> 595 ms at 2000 (materializing sets). Edge-major layout was worse (each edge row = one 4 KB page). Nothing beats C-level `Counter.update` below ~250 increments/exec. Code kept out of tree; numbers from ad hoc harnesses (300 records, min of 7). |
| Per-operator bitmaps maintained in `record()` | `record` +23% (729 -> 894 us), +17 MB at a 2^16 id space (~1 MB/operator at 2^20), and flatten from bitmaps is slower (30.7 vs 22.2 ms: scanning 264 x 65,536 bytes beats nothing). Compact slot ids need a dict lookup per edge, costlier still. |
| Split `record`: Counters below `FUSED_MIN_WORK` increments/exec, dense operator-major matrix above, folded into the Counters before any read or prune (exact; 20 equivalence tests). Patch not in tree. | Measured as record + one read per 2000 execs (stats-line cadence), us/exec, Counter-only vs threshold 250 / 500 / 1000: 10x1 3.8 vs 3.9/3.9/3.6; 25x<=4 12.6 vs 12.3/12.9/12.6; 60x<=8 53.3 vs 69.6/65.4/54.1; 100x<=2 36.4 vs 50.2/33.7/34.3; 200x<=4 97.7 vs 139.5/137.3/125.5; 500x<=2 151.6 vs 196.3/197.7/207.3; 2000x<=8 922.8 vs 544.2/541.4/527.7. Fold costs ~100 ns per distinct (op, edge) pair, so it pays only when pairs repeat between reads: wins only near 2000 edges/exec. 30k-pool 2000-edge rows void (baseline killed; split rows 8368/8254/2622 disagree 3x, run concurrent with another bench). Pitfall found: `set.difference(Counter)` skips the exact-dict fast path and scans the whole Counter (~90 us at 8k keys). Ship only gated near 2000 edges/exec, after an in-run check on FFmpeg/sqlite with `--shapley`. |

## Open items

- If `ShapleyAttribution` ever stores edges as numpy arrays instead of sets (touches `record`, `_prune_edges`, `operator_kernel`, Jaccard helpers and tests), flatten drops to a concatenate; not worth it for a once-per-run report.
- Pruning under constant churn remains the dominant `record` cost (~1.3M set lookups per prune on the adversarial workload); real runs mostly repeat edges, so prunes are rare.
- A benchmark script is not committed; the numbers above come from ad hoc harnesses (synthetic overlap, 264 ops x 2000 edges). If one is wanted, `tools/` is the place.
