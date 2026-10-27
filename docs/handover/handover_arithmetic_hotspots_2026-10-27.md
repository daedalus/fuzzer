# Handover: Arithmetic-heavy code in daedalus/fuzzer

**Date:** 2026-10-27
**Repo state:** `master` @ `0fb976c1` (`perf(shim): sticky flags instead of per-edge counters in the ctx collapse detector`). `git pull` was a no-op (already up to date).
**Scope:** Static survey only. No code was changed, nothing was profiled.

---

## 1. Question

Where in the codebase is the code arithmetic heavy?

## 2. Method

An AST pass over every `.py` file (556 files), excluding `tests/`, `vendor/` and `.git/`.

Counted per file:
- `ast.BinOp` nodes and `ast.AugAssign` nodes whose operator is one of
  `+ - * / // % ** << >> ^ & |`
- Separately: number of `%` (Mod) and `**` (Pow) operators
- Lines of code, to derive ops/line (density)

Two rankings were produced: absolute operator count, and density (min 80 LOC).
For numpy usage, files importing numpy were ranked by count of calls to
`dot, matmul, linalg, fft, einsum, cumsum, log, exp, sqrt`.
For C, `afl_shim.c` was grepped for hash/mix/context functions.

### Caveats
- Operator count is a **proxy**. It does not measure runtime cost, loop
  nesting, or whether a file is on the per-exec hot path.
- `+`, `-` and `*` on indices, counters and strings are counted too, so
  large "orchestration" files (e.g. `services/fuzzer.py`: 312 ops in 10,608
  LOC, 0.03/line) show up in absolute counts but are not numerically heavy.
- Operator overloads on numpy arrays count as one `BinOp` regardless of size.
- Only Python was AST-analysed. The C files were only grepped.

---

## 3. Findings

### 3.1 Highest absolute operator counts

| Ops | LOC | Ops/line | `%` | `**` | File |
|----:|----:|---------:|----:|-----:|------|
| 726 | 3965 | 0.18 | 6 | 6 | `src/fuzzer_tool/core/mutations/generic.py` |
| 681 | 7234 | 0.09 | 38 | 0 | `src/fuzzer_tool/services/operators.py` |
| 582 | 3184 | 0.18 | 19 | 1 | `src/fuzzer_tool/core/mutations/structured.py` |
| 439 | 4027 | 0.11 | 3 | 14 | `src/fuzzer_tool/core/edge_tracker.py` |
| 402 | 1178 | 0.34 | 0 | 20 | `src/fuzzer_tool/core/randomness.py` |
| 312 | 10608 | 0.03 | 9 | 0 | `src/fuzzer_tool/services/fuzzer.py` |
| 299 | 592 | 0.51 | 9 | 3 | `tools/fxt_gates_ref.py` |
| 266 | 1947 | 0.14 | 1 | 4 | `src/fuzzer_tool/core/schedulers/op_monte_carlo.py` |
| 262 | 2757 | 0.10 | 0 | 2 | `src/fuzzer_tool/services/seed_picker.py` |
| 259 | 2927 | 0.09 | 0 | 4 | `src/fuzzer_tool/services/report.py` |
| 254 | 2352 | 0.11 | 1 | 2 | `src/fuzzer_tool/core/elf.py` |
| 245 | 622 | 0.39 | 1 | 0 | `src/fuzzer_tool/core/mutations/lz4.py` |
| 243 | 1521 | 0.16 | 0 | 21 | `src/fuzzer_tool/core/analyzers/analyzer_elo.py` |
| 235 | 3231 | 0.07 | 8 | 9 | `tools/edge_diagnostic.py` |
| 231 | 557 | 0.41 | 51 | 9 | `src/fuzzer_tool/core/numtheory.py` |
| 179 | 487 | 0.37 | 2 | 1 | `src/fuzzer_tool/core/mutations/int_encodings.py` |
| 178 | 824 | 0.22 | 0 | 0 | `src/fuzzer_tool/core/similarity.py` |
| 166 | 1840 | 0.09 | 3 | 0 | `src/fuzzer_tool/adapters/shm.py` |
| 162 | 2257 | 0.07 | 1 | 0 | `src/fuzzer_tool/services/stats.py` |
| 159 | 344 | 0.46 | 0 | 1 | `src/fuzzer_tool/core/running_stats.py` |
| 158 | 320 | 0.49 | 6 | 1 | `src/fuzzer_tool/core/mutations/hash_len_ext.py` |
| 140 | 2261 | 0.06 | 0 | 0 | `src/fuzzer_tool/services/corpus_manager.py` |
| 139 | 419 | 0.33 | 1 | 0 | `tools/measure_operators.py` |
| 139 | 387 | 0.36 | 0 | 3 | `src/fuzzer_tool/core/integer_relation.py` |
| 137 | 1580 | 0.09 | 0 | 5 | `src/fuzzer_tool/core/grammar.py` |
| 136 | 529 | 0.26 | 1 | 0 | `src/fuzzer_tool/core/mutations/fxt.py` |
| 133 | 770 | 0.17 | 0 | 3 | `src/fuzzer_tool/core/schedules.py` |
| 131 | 698 | 0.19 | 0 | 0 | `src/fuzzer_tool/core/mutations/avif.py` |
| 130 | 1140 | 0.11 | 0 | 0 | `src/fuzzer_tool/core/analyzers/analyzer_distance.py` |
| 130 | 690 | 0.19 | 2 | 0 | `src/fuzzer_tool/core/intel_pt.py` |

### 3.2 Densest files (≥ 80 LOC)

| Ops/line | Ops | LOC | File |
|---------:|----:|----:|------|
| 0.69 | 59 | 86 | `core/ks_pvalue.py` |
| 0.63 | 89 | 141 | `tools/gen_ffmpeg_803_seeds.py` |
| 0.60 | 95 | 158 | `core/hashing.py` |
| 0.58 | 84 | 146 | `core/gaussian.py` |
| 0.51 | 56 | 110 | `core/mutations/murmur3.py` |
| 0.51 | 299 | 592 | `tools/fxt_gates_ref.py` |
| 0.49 | 158 | 320 | `core/mutations/hash_len_ext.py` |
| 0.46 | 159 | 344 | `core/running_stats.py` |
| 0.42 | 53 | 127 | `core/nn_saliency.py` |
| 0.41 | 231 | 557 | `core/numtheory.py` |
| 0.39 | 245 | 622 | `core/mutations/lz4.py` |
| 0.37 | 179 | 487 | `core/mutations/int_encodings.py` |
| 0.37 | 105 | 286 | `tools/gen_malformed_png.py` |
| 0.36 | 139 | 387 | `core/integer_relation.py` |
| 0.36 | 85 | 239 | `core/lattice.py` |

(All `core/` paths are under `src/fuzzer_tool/`.)

### 3.3 Grouped by kind of arithmetic

**Number theory / modular arithmetic**
- `core/numtheory.py`: 51 `%` operators, the most of any file. Likely
  home for the `(a*x+b) % m == t` roadblock work.
- `core/integer_relation.py`, `core/lattice.py`: integer-relation and
  lattice code.

**Hashing / bit mixing**
- `core/hashing.py`, `core/mutations/murmur3.py`,
  `core/mutations/hash_len_ext.py`.

**GF(2) / finite-field**
- `core/gf2_common.py` (603 LOC): GF(2) linear algebra and `GF2n`
  (Zech log/antilog tables). Operators are mostly XOR/AND, so they are
  under-represented by the simple op-count ranking.
- `tools/fxt_gates_ref.py`, `core/mutations/fxt.py`: FXT gates reference
  and mutator.

**Statistics / probability**
- `core/ks_pvalue.py`, `core/gaussian.py`, `core/running_stats.py`,
  `core/randomness.py` (20 `**`), `core/analyzers/analyzer_elo.py`
  (21 `**`).

**Byte / integer mutation arithmetic**
- `core/mutations/generic.py`, `core/mutations/structured.py`,
  `services/operators.py` (38 `%`), `core/mutations/int_encodings.py`,
  `core/mutations/lz4.py`, `core/mutations/avif.py`.

**Coverage bookkeeping with numpy math**
- `core/edge_tracker.py`: 439 ops, 14 `**`, and the most numpy math
  calls of any file (23).

### 3.4 numpy math-call counts (top files importing numpy)

| Calls | File |
|------:|------|
| 23 | `core/edge_tracker.py` |
| 17 | `core/schedulers/seed_entropy_kl.py` |
| 12 | `core/schedulers/op_monte_carlo.py` |
| 9 | `core/schedulers/seed_entropy_zscore.py` |
| 8 | `core/randomness.py` |
| 8 | `core/periodicity.py` |
| 7 | `core/zipf.py` |
| 7 | `core/schedulers/seed_tang.py` |
| 6 | `core/nn_saliency.py` |
| 5 | `core/spectral_kl.py` |
| 5 | `core/schedulers/seed_entropy_shapley.py` |
| 5 | `core/schedulers/seed_dpp.py` |

### 3.5 C side

- `src/fuzzer_tool/adapters/afl_shim.c`: 6,821 lines. Arithmetic is the
  per-edge hot path:
  - `__afl_get_caller_ctx()` (line ~1671): call-stack context.
  - `__afl_edge_hash_as()` (line ~1722): edge hash.
  - `__afl_guard_mix()` (declared line ~1717): guard mixing.
  - Multiplicative map-slot hash (documented at line ~1741):
    `slot = ((uint32)(edge_id * 0x9E3779B1) * size) >> 32`.
- `adapters/perf_shim.c`: 100 lines, not analysed further.
- Other C sources: `adapters/antifuzz_evade.c`, `adapters/fuzz_loader.c`,
  and targets under `targets/` (`gzip_read.c`, `fuzzgoat_read.c`,
  `gf2_gates.c`, `cmplog_exercise.c`, `unrar_read.c`,
  `tracecmp_target.c`). The target files were listed but not analysed.

---

## 4. Interpretation

Two different notions of "heavy" apply:

1. **Per-exec hot path (runtime cost):** `afl_shim.c` hashing and
   context code, and `core/edge_tracker.py`. These run for every
   execution or every edge, so small changes matter most for throughput.
2. **Mathematical content (logic density):** `numtheory.py`,
   `gf2_common.py`, `hashing.py`, `integer_relation.py`, `lattice.py`,
   `ks_pvalue.py`. These hold the most arithmetic per line but are not
   necessarily called often.

The large mutation files (`generic.py`, `structured.py`, `operators.py`)
lead on raw operator count because they contain many small
integer/byte manipulations, not because they are mathematically deep.

## 5. Not done / open items

- No profiling, so the hot-path ranking above is inferred from structure,
  not measured. A `cProfile`/`py-spy` run on a real fuzzing session would
  confirm which of these modules dominate wall time.
- GF(2) bit operations are likely under-counted relative to their
  importance. A weighted count (XOR/AND/shift weighted higher) or a
  loop-depth-aware metric would refine the ranking.
- No vectorization review was done for the numpy-heavy schedulers listed
  in 3.4.
- C targets and `perf_shim.c` were not examined for arithmetic content.

## 6. Reproduction

The AST scan script is a short standalone Python file (walks `.`,
skips `.git`, `vendor`, `tests`, counts `BinOp`/`AugAssign` with the
operators listed in section 2, prints top-30 by count and top-15 by
density with LOC ≥ 80). Run from the repo root:

```bash
git clone https://github.com/daedalus/fuzzer && cd fuzzer
python3 arith.py
```

Suggested location for this document: `docs/handover/handover_arithmetic_hotspots_2026-10-27.md`.
