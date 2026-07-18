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

Compilation flags: `-O2 -g -fsanitize=address -mavx2 -lpthread`

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

## Bug: Unsigned Integer Underflow in AVX2 Fixed-String Search

**File**: `src/search.c:111`
**Severity**: Heap/stack buffer over-read (32 bytes)
**Root cause**: Unsigned integer underflow in `size_t` arithmetic

### Description

The fixed-string multi-byte search path in `search_data()` computes the search bound as:

```c
size_t end = len - nlen + 1;
```

When the input data length (`len`) is smaller than the pattern length (`nlen`), this expression underflows because `size_t` is unsigned. The result is a value near `SIZE_MAX`, causing the AVX2 SIMD loop to read far past the buffer:

```c
while (pos + 32 <= end) {
    __m256i cf = _mm256_loadu_si256((const __m256i *)(data + pos));  // OOB read
    __m256i cl = _mm256_loadu_si256((const __m256i *)(data + pos + nlen - 1));  // OOB read
```

### Reproducer

```bash
printf '\x00\x2c\x40\x59' | ./fuzz_search_pipeline
```

Input breakdown:
- `buf[0] = 0x00` — selects pattern index 0 (`"test"`, 4 bytes)
- `buf[1] = 0x2c` — sets `fixed_string=true`, `count_only=true`, `line_number=true`
- `data_len = 0` — empty search data

Pattern `"test"` (4 bytes) is longer than the data (0 bytes), triggering the underflow.

### ASAN Output

```
==53102==ERROR: AddressSanitizer: unknown-crash on address 0x7ffd2b152134
READ of size 32 at 0x7ffd2b152134 thread T0
    #0 _mm256_loadu_si256
    #1 search_data
    #2 main
Address 0x7ffd2b152134 is located in stack of thread T0 at offset 65924
  [416, 65952) 'buf' <== Memory access at offset 65924 partially overflows
```

### Fix

Added an early return guard when the pattern is longer than the input data:

```c
// Before:
if (nlen == 0) { if (match_count_out) *match_count_out = 0; return FGREP_OK; }

// After:
if (nlen == 0 || nlen > len) { if (match_count_out) *match_count_out = 0; return FGREP_OK; }
```

**Commit**: `f5ac4fd` in `daedalus/fgrep`

### Impact

This bug affects any caller of `search_data()` where the pattern is longer than the input buffer. In fgrep's CLI usage, the pattern comes from argv and data from file/stdin, so exploitation requires a specially crafted file shorter than the search pattern. The AVX2 read accesses 32 bytes past the buffer end, which could leak stack/heap data or crash.

### Mitigation

The fix is a single-line guard that returns early when the pattern cannot possibly match. No functional behavior changes for valid inputs.
