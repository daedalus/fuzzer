# faiss: what can the fuzzer port? (analysis only)

**Status (2026-11-09, master 0bc9d23f):** analysis, no layout bump. #1 was
implemented afterwards (section at the end); #2 to #5 are not.

## Trigger

"Git pull daedalus fuzzer and analyze what can we port from
https://github.com/facebookresearch/faiss" → "Create a handover MD with all
the findings."

## Method

Shallow clone of faiss (`--depth 1`, HEAD f34c0b8, 2026-11-09) and a read of
`faiss/{impl,utils}` plus the top-level index classes. For each technique,
grep the fuzzer for an existing equivalent before proposing a port
(keyword counts were noisy: `lsh`/`minhash` substring-match unrelated words,
so the word-boundary greps below are the ones to trust).

## Already in the fuzzer (nothing to port)

| faiss | Fuzzer equivalent |
|---|---|
| MinHash / LSH style approximate similarity | `MinHashLSH` in `core/edge_tracker.py` (`find_similar`, `corpus_minhash`) |
| `VisitedTable` (generation counter, O(1) clear) | `__afl_generation` tag-based reset in `afl_shim.c` |
| Partition / quickselect top-k (`utils/partitioning.cpp`) | numpy 2.5 `np.argpartition` already does this; the fuzzer just does not call it |
| Popcount Hamming (`utils/hamming.cpp`) | `np.bitwise_count` (numpy >= 2.0) and `int.bit_count`; `similarity._hamming_dist` |
| SIMD in-register LUT (`impl/fast_scan`) | AVX2 probe of the SHM table (`__AFL_PROBE_ISA`) |
| Hadamard / random rotation (RaBitQ) | `core/hadamard.py` |

## Candidates, ranked

### 1. NN-descent kNN graph over seeds (`impl/NNDescent.cpp`, 550 lines)

Builds an approximate kNN graph in sub-quadratic time by repeatedly
exploring neighbours-of-neighbours. Seeds are points in edge space; the
distance is Jaccard (or Hamming on the touched bitmap).

- Fits: `core/seed_cut.py` (max-k-cut, currently a greedy pass over an
  edge x group table), `core/edge_components.py` (B = A·Aᵀ), the Fiedler
  code in `core/shapley.py`, and `core/gram_matrixfree.py`. All want a
  sparse overlap graph; none should need the SxS matrix at 100k+ seeds.
- Open question: kNN truncation drops long-range overlap, so connected
  components and cuts will differ from the exact ones. Needs a paired
  comparison on fuzzgoat (393 edges, 250 seeds) against the exact
  `--edge-components` result before adopting.
- Effort: medium (Python + numpy, no C). Risk: recall of the graph.

### 2. Exact 1-D k-means (`impl/kmeans1d.cpp`, 301 lines)

Optimal 1-D clustering by dynamic programming, no Lloyd iteration.

- Fits: data-driven count buckets instead of the fixed AFL ladder in
  `core/count_class.py` (the 10-class ladder is a hand choice); bucketing
  of exec times, energies or scheduler scores.
- Caveat: `--count-aware` (`core/count_items.py`) ids are `id<<8|tag`, so
  changing the ladder changes ids and needs a scheme bump. Start as a
  report-only comparison (fixed ladder vs DP buckets on real hit-count
  histograms from `tools/edge_hit_spectrum.py`).
- Effort: small-medium. Risk: low if report-only.

### 3. Multi-index Hamming search (`IndexBinaryHash.cpp`, 285 lines)

Exact radius search over binary codes by substring hashing (pigeonhole).

- Measured here (numpy 2.5, synthetic): a flat Hamming scan of 100,000
  seeds x 8,192-bit codes (102 MB) costs about 85 ms per query. Fine for
  batch jobs (minimize, report), too slow per exec.
- Only worth it if a "nearest seed in coverage space" lookup is wanted in
  the hot path. No such consumer exists today. Not recommended yet.

### 4. faiss as a fuzz target (`faiss/impl/index_read.cpp`)

`read_index` parses an untrusted binary format. The repo already has
`*_read` targets in `targets/` (C and one C++, `tailslayer_read.cpp`).

- `tests/test_read_index_deserialize.py` has byte-level builders for
  crafted payloads (header, PQ with M=0, byte limits, lattice nsq=0, RQ
  nbits mismatch, ...). They are ready-made seeds and a dictionary of the
  struct layout (`i`, `q`, `Q`, `?`, `f`, `d`, `b`, `B`).
- The C API exposes `faiss_read_index_custom` (IOReader), `faiss_read_index`
  (FILE*) and `faiss_read_index_fname`; a memory-reader target needs either
  the custom IOReader or the C++ API (`faiss::read_index(IOReader*)`).
- faiss has validation tests for deserialization but no harness in-tree
  (grep for "fuzz" finds only partitioning tests). Whether upstream runs one
  elsewhere was not checked.
- Effort: medium (build faiss with sancov, new `targets/faiss_read.cpp`,
  `build_targets.sh` entry, seed generator). Risk: build time and libomp.

### 5. `sorted(...)[:k]` -> `np.argpartition` (small)

About 23 `sorted(...)[:k]` sites and 3 `heapq.nlargest/nsmallest` sites,
zero `argpartition`. None looked hot in a read of `seed_picker.py`,
`fuzz_round.py`, `fuzzer.py`, `core/shapley.py`, `core/set_cover.py`; the
only candidate in a hot module is `services/corpus_manager.py:1653`
(`sorted(picked, key=...)[:budget]`). Profile before touching: past rounds
of this kind (`np.lexsort` in `_prune_edges`) were -8% and discarded.

## Not worth porting

- PQ / OPQ / RaBitQ / IVF / SQ: the seed-edge incidence matrix is already
  sparse and binary; quantizing it loses exactness for no memory gain.
- HNSW / NSG / IndexNNDescent as an index: `MinHashLSH` already covers
  approximate lookup; the graph build of #1 is the part that is useful.
- Polysemous training, lattice Zn, additive/residual quantizers, neural
  codecs, GPU and `svs` backends: no counterpart in the fuzzer's data.
- Panorama / PDX / AdSampling (early-abandon distance scans): the scans
  here are over bitmaps, where a full popcount pass is already cheap.
- Runtime SIMD dispatch (`impl/simd_dispatch.h`): the shim chooses the ISA
  at build time (`__AFL_PROBE_ISA`); runtime dispatch would only matter for
  prebuilt shims shipped across machines.

## Recommendation

#1 first (feeds three existing features), then #4 (new bug-finding surface),
#2 as a report-only experiment. Skip #3 and #5 unless a profile asks.

## Not measured

- No faiss code was compiled or run. The only measurement is the flat
  Hamming scan in #3 (synthetic random codes, one query, numpy 2.5).
- No A/B of any candidate. #1 and #2 claims about fit come from reading
  the consumers, not from data.
- The existing-capability table was built by grep and a read of the named
  modules, not by running them.

## Open items

- [x] #1: `--seed-knn` shipped; fuzzgoat comparison below.
- [x] #2: `tools/count_buckets.py` shipped; fuzzgoat result below. png_read/ffmpeg still open.
- [ ] #4: decide on a build route for faiss under sancov.

## #1 implemented: `minimize -c --seed-knn K` (2026-11-09)

`core/seed_knn.py`, tests `tests/test_seed_knn.py` (20). Report only.

- **Premise corrected.** The analysis said a kNN graph is a cheaper input
  for `seed_cut`, `edge_components` and the Fiedler code. Reading them
  again: `edge_components` propagates labels over the seed-edge incidence and
  `seed_cut` keeps an edge x group table; neither builds S x S. The graph is
  a separate view (Jaccard-weighted, K nearest), not a speedup for them.
- **Method.** MinHash (64 perms, hash of `dense id + 1`) after the
  `STRIP_FRAC` strip; NN-descent rounds over neighbours, 5 reverse
  neighbours and their neighbours; exact scan at <= 2048 seeds.
- **Bug found by the tests.** Hashing the raw dense id gave 0 for id 0 in
  every perm, so every seed holding that edge matched every other (sim = all
  perms). Fixed with `id + 1`; `test_estimate_tracks_exact_jaccard` caught it.
- **Fuzzgoat** (clang 18, 250 seeds -> 110 after twin collapse, 328 edges,
  exhaustive mode, `STRIP_FRAC` 10%): exact components 14.

  | K | graph components | exact comps split | seeds off main piece |
  |---|---|---|---|
  | 1 | 39 | 1 | 82 (75%) |
  | 2 | 18 | 1 | 25 (23%) |
  | 3 | 17 | 1 | 25 (23%) |
  | 5 | 15 | 1 | 10 (9%) |
  | 10 | 14 | 0 | 0 |

  K >= 10 reproduces the exact components here; small K fragments them.
  Edge ids are not reproducible between replays, so the K=5 percentage moved
  between 9.1% and 9.2% across two runs.
- **Synthetic, descent mode** (numpy 2.5, K=10, one core): 20,000 seeds x
  ~290 edges: 6.5 s, 11 rounds, recall 96%, peak RSS 417 MB. 100,000 seeds x
  ~190 edges: 38.8 s, hit the 12-round cap (0.33% changed), recall 86.3%,
  peak RSS 1.25 GB (mostly the synthetic book). In the 20k case the exact
  graph is one component (random edges bridge clusters) while the kNN graph
  recovers the 50 planted clusters; the report's "off main piece" figure
  reads 98% there and is not a loss.
- **Not measured:** descent mode on a real corpus; png_read and ffmpeg; any
  consumer of the graph; estimate error beyond the 512-perm test (mean
  abs error < 0.03).
- Rule 41: opt-in report, no per-exec path touched. Rule 44:
  `architecture.png` not updated.

## #2 implemented: `tools/count_buckets.py` (2026-11-09)

`core/kmeans1d.py` (DP, `optimal_buckets`, `partition_cost`) + the tool. Tests
`tests/test_kmeans1d.py` (42, vs brute force over every partition) and
`tests/test_count_buckets.py` (19). Report only.

- **Data.** Per-exec count of every (edge, exec) pair, saturated at 255; not
  the summed spectrum of `edge_hit_spectrum.py`, which mixes execs.
  `replay` got an optional `tap`.
- **Fuzzgoat** (clang 18 sancov, `fuzzgoat_read_noasan`, 250 inputs, 24,407
  pairs, 82 distinct counts, max 233, 92.6% of pairs at count <= 3). Control:
  two replays give identical histograms.

  | scale | ladder unexplained | DP (k=9) unexplained | k_match |
  |---|---|---|---|
  | log | 0.695% | 0.368% | 8 |
  | linear | 3.717% | 1.660% | 7 |

  DP boundaries (log): `1 2 3 4-5 6-8 9-14 15-29 30-68 71-233`. On log, DP
  needs 7 buckets for 0.77% and 5 for 2.4%.
- **Reading.** The ladder spends 9 buckets and the DP matches its error with
  8 (log): one bucket saved, not a different shape. The DP puts its cuts
  lower (4-5, 6-8 vs 4-7, 8-15) because this corpus has little mass above 30.
  That is a property of one small target.
- **Bugs found.** (1) `-o` on an existing file kept the old data
  (`EdgePickleWriter` skips keys already on disk): a first, failed run left an
  all-zero histogram that two later runs reloaded. `save` now replaces the
  file. (2) `k_match` used an absolute tolerance; singleton buckets cost
  ~1e-13 from rounding, so it is now relative to total variance.
  (3) `.json` files are skipped by `discover_seed_files`
  (`_SEED_SKIP_SUFFIXES`); `tools/corpus_fuzzgoat.py` writes `.json`, so the
  replay found 0 inputs until the corpus was staged as `seeds/id_*`.
- **Not measured:** png_read, ffmpeg or any corpus with high counts; any
  effect on discovery (no fuzz run, so no `AB_LEDGER` row); `--count-aware`
  with other boundaries. Rule 44: `architecture.png` not updated (report tool).
