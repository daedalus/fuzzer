# Handover: TheAlgorithms/C survey (2026-11-19)

Source: `github.com/TheAlgorithms/C` (shallow clone, ~400 `.c`). Survey only — no code copied or ported. Each pick names the fuzzer gap it fills; overlap checked by grep over `src/fuzzer_tool/` and `docs/` (incl. the TheAlgorithms/Python survey, `docs/TODO.md` 2026-11-18).

TODO: `docs/TODO.md` §TheAlgorithms/C survey.

## Verdict

Most of the repo is already covered or irrelevant (sorting zoo, search variants, leetcode, project_euler, games). Six items are new; four are worth building.

## Tier 1 — build

### 1. Thompson construction (`misc/mcnaughton_yamada_thompson.c`)

Regex → ε-NFA. No regex→NFA code exists here (`format_fsm.py` "ce-regex" is a constraint label, not an automaton).

Why:
- **Regex-typed dictionary entries.** A token like `[0-9]{1,10}\.[0-9]+` expands to many concrete strings; today dict tokens are literals only.
- **Near-miss generation.** Walk the NFA, then apply one edit at an accept boundary → inputs that almost match. Parsers branch on exactly this.
- **grep target.** Its pattern argument is a regex; NFA sampling yields valid patterns of controlled shape.

Sketch: `core/regex_nfa.py` — parse (subset: literal, class, `|`, `*`, `+`, `?`, `{m,n}`) → NFA arrays (`array('i')` transitions) → `sample(nfa, rng)` with bounded star unrolling (Hard Rule 54). Op `dict_regex` gated on regex tokens in `_AVAILABLE`.

Wiring: `operator_registry.py` (category `dictionary`) → `_op_dict_regex` in `services/operators.py` → dictionary loader tags lines `regex:` → DEEP_DIVE §Dictionaries.

### 2. Dijkstra on likelihood-weighted CFG (`greedy_approach/dijkstra.c`)

`analyzer_distance._harmonic_bfs` uses unweighted BFS hop counts. A 1-in-10⁶ branch costs one hop, same as a fall-through.

Why: weight edge `u→v` by `−log p(v|u)` from observed hit counts (Laplace-smoothed). Shortest path = most probable path to target. Directed fuzzing then prefers seeds near *reachable* targets, not merely close ones.

Sketch: `heapq` Dijkstra, reverse graph from targets, same harmonic aggregation. Bind weighted vs unweighted once at setup (Hard Rule 57). Recompute on a cadence, not per exec.

Wiring: `analyzer_distance.py` → flag `--distance-weight {hops,likelihood}` (enum) → `waypoints.py` consumers unchanged. Needs A/B on a directed target (Hard Rule 59).

### 3. Hierholzer Euler path (`data_structures/graphs/euler.c`)

Not present (`debruijn_cache.py` builds sequences, not paths over learned graphs).

Why: one synthesized seed that traverses **every observed transition** of `format_fsm` (or the byte-bigram graph of the corpus). Non-Eulerian graph → add minimal duplicate edges (Chinese postman via degree imbalance matching). Cheap seed that exercises all known state pairs at once.

Wiring: `core/euler_walk.py` → seed synthesis next to `format_seed_generator.py` → CLI flag `--euler-seed`.

### 4. Union-find consolidation (`misc/union_find.c` as prompt only)

Seven ad-hoc copies: `shapley.py`, `crash_graph.py`, `crash_metadata.py`, `lineage.py`, `overlap_density.py`, `token_superstring.py`, plus others matched by `parent[`. Differing path compression, some recursive (Code Style: prefer iterative).

Why: Hard Rule 1 — fix once. One `core/dsu.py` (`array('i')` parent/size, path halving, union by size). Not a port; the repo's own duplication motivates it.

Wiring: replace each local `find`/`union`; behavior-preserving, Hard Rule 58 bench on the hottest caller.

## Tier 2 — consider, measure first

| # | Source | Use here | Gap / risk |
|---|--------|----------|-----------|
| 5 | `audio/alaw.c` (G.711 companding) | Mutator: decode A-law/µ-law → mutate linear PCM → re-encode. Semantically small sample changes for FFmpeg | `riff.py` emits random PCM only. Niche; FFmpeg only. |
| 6 | `machine_learning/kohonen_som_*.c` | Report: 2-D topology-preserving map of seed coverage vectors | Overlaps `byte_pca.py`, `edge_components.py`, `seed_dpp`. Visual only. |

## Skipped — already covered or no fit

| TheAlgorithms/C | Reason |
|-----------------|--------|
| Floyd cycle detection | `core/cycle_detect.py` |
| CRC32, Adler32, XOR8, djb2, sdbm, BLAKE2b | `crc_engine.py`, `checksum_sites.py`, `byte_hash_mitm.py`, `hashing.py` |
| Boyer-Moore, Rabin-Karp, naive search | `aho_corasick.py` + `bytes.find` (C, two-way) faster from Python |
| Binary/interpolation/jump/exponential/Fibonacci search | `bisect` (C) beats Python reimplementations |
| All sorting algorithms | `sorted`/numpy; no hot-path sort bottleneck |
| LCS DP | Myers diff in `similarity.py` |
| Welford / realtime stats | `running_stats.py` |
| k-means | `kmeans1d.py` (exact DP) |
| PID | `pi_controller.py` |
| Newton, secant, bisection | `fox_newton.py`, `gradient_cmp.py` |
| Durand-Kerner, QR eigen, LU, Gauss | Complex roots don't map to bytes; `dc_eigh.py`, `poly_mod.py` cover the numeric needs |
| Catalan | `tree_mutator.py`, `lineage.py` |
| Kruskal / Prim | `seed_kruskal_count`, `op_kruskal_count` |
| Tarjan SCC, topological sort | `dominators.py`, `articulation.py`; natural loops via dominators suffice |
| Floyd-Warshall, transitive closure | O(V³) on 10⁵-block CFGs — infeasible; BFS per target suffices |
| Bellman-Ford | No negative weights (−log p ≥ 0) |
| Hamiltonian path | NP-hard, no use |
| Shunting yard, infix→postfix | Grammar engine + `sql_text.py` cover expression inputs |
| Affine, ROT13 | `affine_mod.py`, `text_codec.py` |
| Base conversions, atoi | `ascii_num*`, `_rq_hex`, `radamsa_num` ops |
| `malloc_dbg` | ASAN/LSan |
| `client_server` TCP/UDP | `services/tcp_ingest.py` |
| Lexicographic permutations | `permutation.py` (SJT walk) |
| Trie | `aho_corasick.py` |
| Multikey quicksort / TST corpus substrings | Subsumed by Python survey's *suffix automaton tokens* (O(n)) |
| Quickselect (`rselect.c`), segment tree | Python survey skip: `np.partition`, Fenwick + numpy |
| leetcode, project_euler, games, exercism, geometry, graphics | No fuzzing relevance |

## Next session

1. Build Tier 1 in order 4 → 1 → 3 → 2 (cheapest/safest first). TDD per Hard Rule 38; one falsification + one adversarial test each (Hard Rule 23).
2. #2 and #3 need A/B rows in `docs/AB_LEDGER.md` before claiming value.
