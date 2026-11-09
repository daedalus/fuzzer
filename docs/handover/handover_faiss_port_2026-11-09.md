# faiss: what can the fuzzer port? (analysis only)

**Status (2026-11-09, master 0bc9d23f):** analysis, no code change, no
layout bump. Nothing is implemented; the ranking below is a proposal.

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

- [ ] #1: paired comparison of exact `--edge-components` vs kNN-graph
  components on fuzzgoat.
- [ ] #2: report-only DP buckets vs fixed ladder on real hit-count data.
- [ ] #4: decide on a build route for faiss under sancov.
