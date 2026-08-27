# fgrep Fuzzing Findings

## Target

[fgrep](https://github.com/daedalus/fgrep) — a SIMD-accelerated grep implementation with AVX2 fixed-string search, POSIX regex support, and BMH (Boyer-Moore-Horspool) pattern matching.

## Fuzz Targets

Three ASAN-instrumented fuzz targets were created in `targets/` to cover fgrep's key attack surfaces:

| Target | Attack Surface | Input Format |
|--------|---------------|--------------|
| `fuzz_regex_compile` | `regcomp()` with adversarial patterns | Raw pattern bytes |
| `fuzz_pattern_match` | `regexec()` / SIMD search with fuzzed data | Byte 0 selects pattern, rest is data |
| `fuzz_search_pipeline` | Full `search_data()` end-to-end | Bytes 0-3 config flags, rest is file content |

**Compilation flags**: `-O2 -g -fsanitize=address -mavx2 -lpthread`

## Fuzzing Configuration

- **Iterations**: 10,000 per target
- **Coverage**: AFL SHM bitmap (`-c`)
- **Engine**: fuzzer-tool with Markov byte generation, Thompson sampling bandit, and grammar-aware mutations

## Results

| Metric | Value |
|--------|-------|
| Total crashes | 8 |
| Unique signatures | 1 |
| Target | `fuzz_search_pipeline` |
| Time to first crash | ~3 seconds (432 execs) |
| Exploitability | MEDIUM |

## Bugs

- **[unsigned_underflow_avx2_fixed_string_search](unsigned_underflow_avx2_fixed_string_search.md)** — Unsigned integer underflow in AVX2 fixed-string search when pattern length exceeds data length. 32-byte OOB read via `size_t` underflow in `search_data()`. Severity: Medium.
- **[heap_buffer_overflow_avx2_dual_load](heap_buffer_overflow_avx2_dual_load.md)** — Heap-buffer-overflow in AVX2 fixed-string dual-load search. Loop bound doesn't account for second `_mm256_loadu_si256` offset, causing 32-byte read past allocation. Severity: Medium.
- **[heap_buffer_overflow_fixed_string_insensitive_match](heap_buffer_overflow_fixed_string_insensitive_match.md)** — Heap-buffer-overflow in fixed-string insensitive match. Missing bounds check after `memchr` advances `i`, causing read past heap allocation. Severity: Medium.
