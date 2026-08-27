# Heap-Buffer-Overflow in Fixed-String Insensitive Match

**File**: `src/regex_engine.c:50`
**Severity**: Heap-buffer-overflow (single-byte read past allocation)
**Root cause**: Missing bounds check after `memchr` advances position
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

`fixed_string_insensitive_match()` uses `memchr` to find the first character of the needle, then advances `i` to that position and checks the remaining characters in a tight loop:

```c
for (size_t i = 0; i + pat->fixed_len <= len; i++) {
    void *found = memchr_fn(data + i, (int)needle_char, len - i);
    if (!found) return false;
    i = (size_t)((const char *)found - data);  // i jumps forward

    bool match = true;
    for (size_t j = 1; j < pat->fixed_len; j++) {
        if ((unsigned char)data[i + j] != (unsigned char)pat->fixed_str[j]) {  // OOB
```

After `memchr` finds the first character, `i` is updated to that position. The outer `for` loop's condition (`i + pat->fixed_len <= len`) is only checked at the **top** of the next iteration, not after the `i` assignment. The inner loop then accesses `data[i + j]` without verifying that `i + pat->fixed_len <= len`.

## ASAN Output

```
==2749355==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x52100119b4ee
SUMMARY: AddressSanitizer: heap-buffer-overflow
  /home/dclavijo/my_code/fgrep/src/regex_engine.c:50 in fixed_string_insensitive_match
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
| Crash determinism | Deterministic when first character is near buffer end |
| Trigger depth | Shallow — directly triggered by buffer boundary and pattern length |
| Preconditions | Case-insensitive fixed-string search where first character appears near the buffer boundary |
| Signal type | Heap-buffer-overflow |
| Memory safety | Single-byte read past heap allocation |
| Reach | Any caller of `fixed_string_insensitive_match()` with near-boundary matches |
| Severity | **MEDIUM** — can leak heap data or crash the caller |

When `memchr` finds the first needle character near the end of the buffer, the inner loop reads past the allocation. This affects case-insensitive fixed-string searches where the first character appears near the buffer boundary.

## Suggested Fix

Added a bounds check after `memchr` advances `i`:

```c
// Before:
i = (size_t)((const char *)found - data);

// After:
i = (size_t)((const char *)found - data);
if (i + pat->fixed_len > len) return false;
```

**Commit**: `c260d80` in `daedalus/fgrep`

## Impact

The bug affects case-insensitive fixed-string searches where the first character is found near the end of the buffer. The fix adds a single bounds check and does not change matching semantics for valid inputs.

## Mitigation

The bounds check ensures the inner loop never accesses `data[i + j]` past the allocation. No functional behavior changes for valid inputs.

## Upstream Status

Fixed in commit `c260d80` in `daedalus/fgrep`.
