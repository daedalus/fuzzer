# Handover: Arithmetic-heavy code in daedalus/fuzzer

**Date:** 2026-10-27
**Repo state:** `master` @ `0fb976c1` (`perf(shim): sticky flags instead of per-edge counters in the ctx collapse detector`). `git pull` was a no-op (already up to date).
**Scope:** Sections 1-6 are a static survey (no code changed, nothing profiled).
Section 7 (2026-10-28) records the optimization pass that followed.

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

- The Section 3 ranking is still structural. Section 7 measured individual
  functions and one real run, but not a representative long campaign; see
  7.5 for what that run can and cannot show.
- GF(2) bit operations are likely under-counted relative to their
  importance. A weighted count (XOR/AND/shift weighted higher) or a
  loop-depth-aware metric would refine the ranking.
- Vectorization of the numpy-heavy schedulers in 3.4 was not reviewed
  (only `_compute_weights` was looked at, see 7.3).
- C targets and `perf_shim.c` were not examined for arithmetic content.
- A `py-spy record -f raw --nonblocking` run on the target and mode the
  user actually fuzzes (the docs use ffmpeg `nosan` `.so`) is still the
  right base for the next optimization.

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

---

## 7. Update 2026-10-28: optimization pass

Repo at `ca49cb39` after `git pull --rebase`; later pulls added the SARIF
export (`749f8590`). Python 3.12, numpy available, gcc only (no clang).

### 7.1 Shipped: MinHash one-shot block

`MinHashLSH.compute_signature` (`core/edge_tracker.py`) now builds one
`(num_perm, n)` block (outer product, in-place add, row min) for
`n <= _MINHASH_ONE_SHOT_MAX` (1024). Larger sets keep the banded path so the
temporary stays bounded. Same uint64 wrapped values. Landed upstream as
`84d999ac`, with `tests/test_minhash_one_shot.py` (matches the pure-Python
reference on both sides of the cutoff, near 2**64, empty set, and
one-shot vs banded on the same set).

| Edges in set | Before | After |
|---:|---:|---:|
| 1 | 43 us | 5 us |
| 100 | 67 us | 29 us |
| 400 | 105 us | 56 us |
| 1000 | 186 us | 126 us |
| 4000 | 454 us | 448 us |

Why: per-admission updates fold in only newly owned edges, so n is small
and the 8-row banding costs more in numpy call overhead than arithmetic.

### 7.2 Micro-benchmarks (single-call, this host)

| Function | Time |
|---|---:|
| `xxh64` 4 KiB / 200 B | 210 us / 16 us |
| `MinHashLSH.compute_signature` n=4000 / n=100 (before 7.1) | 469 us / 68 us |
| `ks_two_sample` 200x200 | 123 us |
| `RunningMoments.update` (x1000) | 571 us (1436 us with window 256) |
| `numtheory.is_prime(2**89-1)` | 351 us |
| `numtheory.factorint` of a 60-bit semiprime | 7.4 ms |
| `numtheory.sqrt_mod` / `modinv` | 22 us / 1.6 us |
| `gaussian.norm_cdf` / `norm_ppf` | 0.2 us / 0.6 us |
| `ks_pvalue.kolmogorov_pvalue_two_sample` | 3.8 us |

None of these except MinHash had slack worth a patch. `seed_key` callers in
`CorpusManager` are memoized, so `xxh64` runs on admissions, not per exec.

### 7.3 Looked at and not shipped

- **`_EdgeVectors.stats` (`services/seed_picker.py`)**: variants with
  `np.add.reduce`, `take(..., mode="clip")` and a fused two-column gather
  were within 4-12% of the current code (interleaved min of 25 rounds,
  us/seed, current vs best variant: 4.31/3.88 at 30 edges, 5.10/4.73 at 300,
  12.25/10.84 at 3000). A fused gather was 2-7x slower. Not worth a patch.
  Single-shot timings were unusable: the same variant varied about 2x between
  runs on this host, so only interleaved min-of-N numbers were trusted.
- **`_compute_weights`**: synthetic corpus, bounded tracker kept 980 of 3000
  seeds, about 13 us/seed warm. Under cProfile the edge penalty and `stats`
  path was the largest slice (about 30%). `docs/DEEP_DIVE.md` already records
  that batching `stats` with `reduceat` was slower, so it was left alone.
- **`ks_two_sample` (`core/edge_tracker.py`)**: only caller is
  `services/differential.py`, once per 10 inputs in differential mode, over
  windows of up to 500 samples. Negligible amortized cost. A bit-identical
  vectorization would also have to reproduce the walk's tie handling exactly.

### 7.4 Pre-existing test failures

`tests/test_regressions.py::TestEloDecayInitIndependentOfChi2` (two tests)
failed on clean `master` at `0fb976c1`, with and without the MinHash change.
Not re-checked on later commits.

### 7.5 One real run (gzip target, py-spy)

Built `gzip_read.so` with `gcc -O2 -g -shared -fPIC -include
src/fuzzer_tool/adapters/afl_shim.c ... -lz -Wl,--export-dynamic`, 2-4 seed
corpus.

- Plain `--inprocess` runs the target through `_run_c_subprocess`, one
  subprocess per exec: about 25 execs/s, `tgt` about 34 ms, fuzzer-side
  `diff` about 4.5 ms.
- `--inprocess-direct`, with or without `--inprocess`, showed the same
  `tgt` of about 34 ms, so it was not confirmed that a direct mode engaged on
  this host.
- `py-spy record -f raw --nonblocking` (`--native` cannot be combined with
  `--nonblocking`), 45 s, 706 non-import samples of 1294. Inclusive shares of
  the non-import samples: cmplog token collection `collect_tokens` 16.3%
  (`_collect_tokens_fifo` 14.7%, `_parse_lines` 7.6%); target path
  `_run_target` 14.2% (`_spawn_capture` 7.9%); `mutate` 7.5%;
  `_feed_learners` 6.2%.
- Caveat: tiny corpus, subprocess-per-exec, few samples. It shows the
  cmplog parse path is worth a look; it does not show steady-state cost on a
  large corpus. Do not read it as a ranking.

### 7.6 Tooling notes

- `fuzzer-tool fuzz <target.so> -d CORPUS -o CRASHES` (`-c` toggles
  coverage; it does not take a path).
- Use `date` arithmetic for wall time inside `sh -c`; `time` is not a binary
  there.
- `pip install -e .` and `pip install pytest ruff py-spy
  --break-system-packages` were needed in a fresh container.

### 7.7 Update 2026-10-29: cmplog parse path (7.5 lead)

Drains are capped at `CMPLOG_MAX_LINES_PER_READ` (10k lines) and run every
1/5/20 iterations by pool size (`fuzz_round._collect_due`). Synthetic drains,
interleaved min-of-N, this host. Warm `_parse_lines` share at 10k lines /
9k distinct: cond parse 41%, `_ingest_conds` 14%, `_evict_pairs` 13%,
operand parse 10%.

Shipped: `cond_tuples_from_cmplog_text` splits `"CMP "` lines once, no
`strip`; stripped-line set dropped (the key set dedupes). Regression test
`tests/test_regression_cmp_parse_fastpath.py` (pre-change oracle, Rule 46
control, adversarial whitespace; a naive `parts[0] == "CMP"` variant fails 3/4).

| Distinct lines | Parse before / after (ms) | Warm drain before / after (ms) |
|---:|---:|---:|
| 300 | 0.173 / 0.140 | 0.356 / 0.322 |
| 2470 | 1.559 / 1.206 | 3.553 / 3.188 |
| 3911 | 2.701 / 2.039 | 6.589 / 5.751 |

Not shipped:
- `_ingest_conds` with `itemgetter` + `dict.fromkeys` bulk updates: 5-13%
  slower (the per-pair loop still runs for new-pair detection).
- Cross-drain line→tuple cache: would remove most of the parse on repeat
  lines, but needs a memory bound and eviction for ≤1 ms per drain at the
  saturated 1/20 cadence. Not built.

### 7.8 Update 2026-10-29: 7.5% bar

Bar set by the user: a gain > 7.5% on the function is worth shipping.
Re-measured at 10k-line drains (interleaved min-of-N):

| Change | Gain | Shipped |
|---|---:|---|
| `detect_hash_candidates`: zero bytes of `int(a) ^ int(b)` | 62% | yes |
| pending-redqueen prune: `filter(set.__contains__)` | 47% | yes |
| `_eviction_victims`: `islice` over zero-credit generator | 14-20% | yes |
| `_evict_pairs`: `difference_update` + bound `pop` per map | 16% | yes (and `_evict_tokens`) |
| occurrence count: bound locals | 12% | yes |
| `pairs_from_operand_records` split once | 5-9% | no: under the bar at 3000+ distinct |

Warm `_parse_lines`, master vs branch, best of 3 process pairs:
0.449 → 0.416 ms (300 distinct), 3.494 → 2.984 (3000), 6.401 → 5.159 (9000).

Tests: `tests/test_regression_cmplog_drain_loops.py` (pre-change hash oracle,
Rule 46 control, leading-zero / 2-vs-3-match cut, eviction state under 3x
cap churn). A variant that skips re-padding in `to_bytes` fails 2/4.
`_eviction_victims` stays covered by `tests/test_cmplog_eviction_victims.py`
(full-sort oracle, 3000 random pools).

### 7.9 Update 2026-10-29: fuzzgoat profile pass

§3.4 numpy schedulers surveyed: 7 of 10 are flag-gated or never called at
runtime (`seed_entropy_kl`/`zscore` `--entropy-kl`/`--entropy-zscore`,
`seed_tang` `--tang`, `seed_dpp`, `op_monte_carlo` `--mc-*`/`--mopt`;
`seed_entropy_shapley` and `spectral_kl` tests/tools only). Default runs
touch only `periodicity.estimate_record_size` (per admission) and `zipf`
(gated). Not optimized: no default-run cost.

Real profile instead: fuzzgoat `fuzzgoat_read_noasan.so` built with clang
(Hard Rule 52; no ASAN runtime on this host), direct_lite, py-spy 200 Hz,
60k execs. Shares of the round loop: `mutate` 40% (`build_ops` 11.5%,
handlers 12%), coverage check 13%, `_feed_models` 10%, target call ~5%.

| Change | Gain | Shipped |
|---|---:|---|
| `_SNIFF_CACHE_MAX` 256 → 4096 (miss 18% → 2%) | `available` 71.3 → 59.7 us in-run (16%) | yes |
| `CrashMITracker.record`: flat gather, cached row offsets, count bound picks the step table | 40-44% | yes |
| `RandPool.randint_list` `a == 0`: no int64 cast/add | 35-45% | yes |
| static-flag gates decided once | ~6% of `available` | no |
| `_live_entries` contiguous u64 gen compare | +19% at 65k entries, -12..36% at 262k | no |
| `katz.observe`, `_regularity`, `_is_repeat` | fixed numpy/hash overhead or operator work | not attempted |

`available` costs 24.7 us warm vs ~60 us in-run: cache-cold effects, not
one line. End-to-end 20k-exec wall time (18.6-23.1 s per run) varies more
than the expected few-percent gain; not claimed.

Open: `_trickle` geometric skip (needs ~2 draws instead of ~48 per call)
changes the seeded stream that `tests/test_regression_registry_plan.py`
pins; needs a decision.
