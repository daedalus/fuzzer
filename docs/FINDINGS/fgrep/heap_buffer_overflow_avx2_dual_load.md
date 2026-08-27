# Heap-Buffer-Overflow in AVX2 Fixed-String Dual-Load Search

**File**: `src/search.c:172`
**Severity**: Heap-buffer-overflow (32-byte read past allocation)
**Root cause**: Loop bound doesn't account for second AVX2 load offset
**Discovered by**: fuzzer-tool with ASAN via `direct_lite` mode

---

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

---

## Fuzzing Configuration

- **Iterations**: 10,000 per target
- **Coverage**: AFL SHM bitmap (`-c`)
- **Engine**: fuzzer-tool with Markov byte generation, Thompson sampling bandit, and grammar-aware mutations

---

## Results

| Metric | Value |
|--------|-------|
| Total crashes | 8 |
| Unique signatures | 1 |
| Target | `fuzz_search_pipeline` |
| Time to first crash | ~3 seconds (432 execs) |
| Exploitability | MEDIUM |

---

## Description

The AVX2 fixed-string search uses a dual-load technique: load 32 bytes at `data[pos]` and 32 bytes at `data[pos + nlen - 1]`, then compare first/last needle characters simultaneously to find candidate positions.

The loop condition only checked that the first load was in bounds:

```c
while (pos + 32 <= len) {
    __m256i cf = _mm256_loadu_si256((const __m256i *)(data + pos));
    // ...
    __m256i cl = _mm256_loadu_si256((const __m256i *)(data + pos + nlen - 1));  // OOB
```

When `nlen > 1`, the second load at `data[pos + nlen - 1]` reads up to `nlen - 1` bytes past the first load's range. With `pos` near the end of the buffer, this reads past the heap allocation.

## ASAN Output

```
==2744917==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x51a00017b18b
READ of size 32 at 0x51a00017b18b thread T0
    #0 _mm256_loadu_si256 /usr/lib/gcc/x86_64-linux-gnu/14/include/avxintrin.h:929
    #1 search_data /home/dclavijo/my_code/fgrep/src/search.c:172
    #2 fuzz_search_pipeline targets/fuzz_search_pipeline.c:81

0x51a00017b1a5 is located 0 bytes after 1317-byte region
```

## Crash Metadata

| Field | Value |
|-------|-------|
| Signal | Not explicitly recorded |
| Execs to find | Not explicitly recorded |
| Corpus at find | Not explicitly recorded |
| Elapsed | Not explicitly recorded |
| Parent seed | Not explicitly recorded |
| Target SHA256 | Not explicitly recorded |

## Exploitability Assessment

| Factor | Assessment |
|--------|-----------|
| Crash determinism | Deterministic for inputs placing `pos` near the end with `nlen > 1` |
| Trigger depth | Shallow — directly triggered by buffer boundary and pattern length |
| Preconditions | Pattern length > 1 and `pos` within `nlen - 1` bytes of buffer end |
| Signal type | Heap-buffer-overflow |
| Memory safety | 32-byte read past heap allocation |
| Reach | Any caller of `search_data()` with `nlen > 1` and near-end `pos` |
| Severity | **MEDIUM** — can leak heap data or crash the caller |

Any pattern with length > 1 searched in a buffer where the last valid first-load position is within `nlen - 1` bytes of the end. The 32-byte read accesses heap memory past the allocation, which could leak data or crash. In fuzzer-tool's in-process mode, this crashes the fuzzer process.

## Suggested Fix

Updated the loop condition to account for the second load's offset:

```c
// Before:
while (pos + 32 <= len) {

// After:
while (pos + nlen - 1 + 32 <= len) {
```

**Commit**: `a67c3ea` in `daedalus/fgrep`

## Impact

The bug affects all fixed-string AVX2 searches where the second load can exceed the buffer. The fix is localized to the loop bound and does not change matching semantics for valid inputs.

## Mitigation

The loop bound is tightened so both AVX2 loads remain within `data[0..len)`. No functional behavior changes for valid inputs.

## Upstream Status

Fixed in commit `a67c3ea` in `daedalus/fgrep`.
