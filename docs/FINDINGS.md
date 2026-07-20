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

---

## Bug: Heap-Buffer-Overflow in AVX2 Fixed-String Dual-Load Search

**File**: `src/search.c:172`
**Severity**: Heap-buffer-overflow (32-byte read past allocation)
**Root cause**: Loop bound doesn't account for second AVX2 load offset
**Discovered by**: fuzzer-tool with ASAN via `direct_lite` mode

### Description

The AVX2 fixed-string search uses a dual-load technique: load 32 bytes at `data[pos]` and 32 bytes at `data[pos + nlen - 1]`, then compare first/last needle characters simultaneously to find candidate positions.

The loop condition only checked that the first load was in bounds:

```c
while (pos + 32 <= len) {
    __m256i cf = _mm256_loadu_si256((const __m256i *)(data + pos));
    // ...
    __m256i cl = _mm256_loadu_si256((const __m256i *)(data + pos + nlen - 1));  // OOB
```

When `nlen > 1`, the second load at `data[pos + nlen - 1]` reads up to `nlen - 1` bytes past the first load's range. With `pos` near the end of the buffer, this reads past the heap allocation.

### ASAN Output

```
==2744917==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x51a00017b18b
READ of size 32 at 0x51a00017b18b thread T0
    #0 _mm256_loadu_si256 /usr/lib/gcc/x86_64-linux-gnu/14/include/avxintrin.h:929
    #1 search_data /home/dclavijo/my_code/fgrep/src/search.c:172
    #2 fuzz_search_pipeline targets/fuzz_search_pipeline.c:81

0x51a00017b1a5 is located 0 bytes after 1317-byte region
```

### Fix

Updated the loop condition to account for the second load's offset:

```c
// Before:
while (pos + 32 <= len) {

// After:
while (pos + nlen - 1 + 32 <= len) {
```

**Commit**: `a67c3ea` in `daedalus/fgrep`

### Impact

Any pattern with length > 1 searched in a buffer where the last valid first-load position is within `nlen - 1` bytes of the end. The 32-byte read accesses heap memory past the allocation, which could leak data or crash. In fuzzer-tool's in-process mode, this crashes the fuzzer process.

---

## Bug: Heap-Buffer-Overflow in Fixed-String Insensitive Match

**File**: `src/regex_engine.c:50`
**Severity**: Heap-buffer-overflow (single-byte read past allocation)
**Root cause**: Missing bounds check after `memchr` advances position
**Discovered by**: fuzzer-tool with ASAN via `direct_lite` mode

### Description

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

### ASAN Output

```
==2749355==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x52100119b4ee
SUMMARY: AddressSanitizer: heap-buffer-overflow
  /home/dclavijo/my_code/fgrep/src/regex_engine.c:50 in fixed_string_insensitive_match
```

### Fix

Added a bounds check after `memchr` advances `i`:

```c
// Before:
i = (size_t)((const char *)found - data);

// After:
i = (size_t)((const char *)found - data);
if (i + pat->fixed_len > len) return false;
```

**Commit**: `c260d80` in `daedalus/fgrep`

### Impact

When `memchr` finds the first needle character near the end of the buffer, the inner loop reads past the allocation. This affects case-insensitive fixed-string searches where the first character appears near the buffer boundary.

---

## Bug Discovery Method

Both bugs were found using fuzzer-tool's ASAN-instrumented in-process mode:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 fuzzer-tool fuzz \
    targets/fuzz_search_pipeline.so -d /tmp/fgrep_test -c -n 10000
```

Key technical details:
- **ASAN .so targets** require `LD_PRELOAD` set before process start for `direct_lite` mode
- `direct_lite` uses ctypes `CDLL` in-process — ASAN must be loaded first
- Without external `LD_PRELOAD`, fuzzer-tool falls back to persistent loader (fork-per-call)
- Both bugs were in cold paths (AVX2 SIMD, fixed-string insensitive match) not hit by simple mutations
- Coverage-guided fuzzing with Markov byte generation reached these paths after ~432-1000 execs
