# ASAN Direct-Lite Limitation

**Date**: 2026-07-29
**Status**: Unresolved — root cause not isolated

## Executive Summary

ASAN-instrumented shared libraries (`.so` files built with `-fsanitize=address`) can be loaded and executed in direct_lite (`--inprocess-direct`) mode via `ctypes.CDLL(mode=RTLD_GLOBAL)` with a `verify_asan_link_order=0` shim. However, **ASAN memory bug detection does not trigger** — all three crash types (heap-buffer-overflow, use-after-free, stack-buffer-overflow) return `0` silently without ASAN aborting.

The same `.so` loaded via `dlopen` from a C program with `LD_PRELOAD=libasan.so.8` at process start correctly detects all three crash types. The failure is specific to mid-process ctypes/RTLD_GLOBAL loading.

## What Works

- **ASAN `.so` loads and runs** in direct_lite mode without crashing from the "ASan runtime does not come first in initial library list" error (resolved by `verify_asan_link_order=0` shim)
- **`malloc`/`free` PLT entries resolve to ASAN's versions** — confirmed by address comparison via `ctypes.cast()` between the target `.so`'s free and `libasan.so.8`'s free
- **Shadow memory is properly mapped and accessible** — `__asan_shadow_memory_dynamic_address` = `0x7fff8000` at runtime, matching the compile-time hardcoded offset in the `.so`
- **Shadow IS properly poisoned after free** — all 8 freed bytes read `0xfd` (ASAN's freed-memory marker)
- **`libc.free()` on ASAN-allocated pointer crashes with `munmap_chunk(): invalid pointer`** — proves ASAN's allocator IS active and the `.so`'s `malloc`/`free` go through ASAN
- **`__asan_memset` resolves to libasan's implementation** — the compile-time ASAN transform of `memset()` → `__asan_memset()` works correctly
- **Direct `__asan_report_load1(0x42)` call from Python produces full ASAN diagnostic dump and aborts** — the error-reporting infrastructure is functional
- **AFL SHM edge coverage works** — the target's AFL instrumentation runs and reports coverage normally

## What Doesn't Work

- **Heap-buffer-overflow** (`BUG!H` in `asan_target.c`): writes past `malloc(8)` via `memset(p + 8, 'X', 1)` — returns 0 silently
  - **Update**: clang `-O2` dead-code-eliminates this path entirely (the write has no observable side effects before the `free`)
- **Use-after-free** (`BUG!U`): frees `malloc(8)` then reads `p[0]` — returns 0 silently. Shadow byte at `p/8` = `0xfd` (correctly poisoned), but the shadow check at the load instruction does not trigger
- **Stack-buffer-overflow** (`BUG!S`): writes 8 bytes into `char small[4]` — returns 0 silently
  - **Update**: clang `-O2` dead-code-eliminates this path entirely

For the use-after-free case (the only one that survives compiler optimization), the compiled assembly at the load site reads shadow byte `0xfd` correctly but the conditional branch (`jne __asan_report_load1`) is NOT taken despite the non-zero shadow value. A trace shim that interposes `__asan_report_load1` via `dlsym(RTLD_NEXT)` confirmed the function is **never called** during execution of `fuzz(b"BUG!U", 5)`.

## Root Cause Analysis

### What Was Ruled Out

| Hypothesis | Evidence Against |
|---|---|
| `-Bsymbolic` prevents ASAN interposition | ❌ `malloc`/`free` DO resolve to ASAN's versions (address comparison confirmed). `-Bsymbolic` only binds symbols **defined within** the `.so`, not undefined externals. |
| Shadow memory not initialized | ❌ Shadow base confirmed at `0x7fff8000` (matches compile-time). Shadow bytes are accessible and correct (`0xfd` after free). |
| ASAN's `__asan_report_load1` is broken/unreachable | ❌ Direct call `__asan_report_load1(0x42)` from Python produces full ASAN dump and aborts. |
| ASAN's allocator not active | ❌ `libc.free()` on ASAN pointer → `munmap_chunk()` crash. Shadow is poisoned (ASAN's `free` does this). |
| Lazy binding vs eager binding | ❌ Both `dlopen(RTLD_NOW)` and `RTLD_LAZY` produce the same result (no detection). |
| Compiler optimization removes the crash path | ❌ Only H and S paths are dead-code-eliminated. The U path's shadow check IS present in the compiled assembly. |
| Wrong ASAN library version | ❌ Same `.so`, same `libasan.so.8` — works from C with LD_PRELOAD, fails from Python ctypes. |

### Remaining Hypotheses (ordered by likelihood)

1. **PLT resolution reordering**: When the `.so` is loaded mid-process via `ctypes.CDLL(RTLD_GLOBAL)`, the dynamic linker resolves PLT entries differently than at process start with `LD_PRELOAD`. The target's call to `free()` may resolve to ASAN's `free` (as confirmed by ctypes address comparison), but the shadow check's boundary conditions or a related helper function (e.g., `__asan_stack_malloc` or a thread-local state flag) may resolve to a different version or fail to initialize. The shadow-check code at `0x3f1a` reads `0x7fff8000(%rax)` where `%rax` is computed from the pointer — if a PLT-resolved helper used during that computation resolves to a non-ASAN version, the address calculation could be wrong even though the final shadow read appears correct.

2. **ASAN initialization order**: When loaded mid-process, ASAN's constructors run after Python and all its dependencies are already initialized. ASAN's thread-local state, fake stack, or quarantine may be in a degraded mode that allows memory operations but skips crash reporting. The `verify_asan_link_order=0` shim bypasses the startup check but doesn't fully reinitialize ASAN as if it were loaded at process start.

3. **afl_shim.c signal handler override**: The shim's `__afl_auto_init()` constructor installs `SIGSEGV`/`SIGABRT` handlers via `sigaction()` that call `_exit(128+sig)`. ASAN's own SIGABRT handler is NOT installed by default (requires `handle_abort=1`), so this likely doesn't affect ASAN detection. However, the shim's SIGSEGV handler could intercept ASAN's crash signal before ASAN's error-reporting machinery captures it. Unlikely since the shim chains the old handler.

4. **GCC vs clang ASAN ABI mismatch**: The target `.so` is compiled with GCC but linked with clang's ASAN runtime (libasan). If there is a subtle ABI mismatch in the shadow-check instrumentation between GCC and clang, the generated code might not correctly interface with clang's libasan. This was not fully tested with a clang-only pipeline.

## Investigation Timeline

### Round 1: Basic connectivity
- Compiled `asan_target_asan.so` with GCC + `-fsanitize=address`
- Loaded via ctypes RTLD_GLOBAL with `verify_asan_link_order=0` shim
- All 3 crash types returned 0
- Initial hypothesis: `-Bsymbolic` prevents ASAN interposition in `.so` builds

### Round 2: Check ASAN interposition
- Built `asan_target_asan.so` **without** `-Bsymbolic` — same result (return 0)
- Checked PLT resolution: `malloc`/`free` DO resolve to ASAN (address comparison confirmed)
- Checked shadow memory: `__asan_shadow_memory_dynamic_address` = `0x7fff8000`, shadow bytes correct
- Checked signal handlers: afl_shim.c overrides SIGABRT/SIGSEGV
- Conclusion: initial `-Bsymbolic` hypothesis was wrong — more investigation needed

### Round 3: Deep analysis with GDB
- GDB stepping at `fuzz+0x5a` (offset `0x3f1a`): `movzbl 0x7fff8000(%rax),%eax` reads correct shadow byte
- `jne` at `0x3f23` is the conditional branch to `__asan_report_load1`
- Observed: `movzbl` gets `0xfd` but `jne` does NOT trigger — contradicts basic logic
- Observed: GDB stepping behavior may differ from full-speed execution (ptrace vs native)

### Round 4: Trace shim insertion
- Compiled `/tmp/trace_asan.so` that interposes `__asan_report_load1` via `dlsym(RTLD_NEXT)` with `fprintf` trace
- Loaded trace shim before target `.so`
- `fuzz(b"BUG!U", 5)` produced NO trace output — confirmed `__asan_report_load1` is NEVER called
- But `fuzz(b"BUG!X", 5)` (no-match, returns early) also produced no trace — correct
- Conclusion: the shadow check to `__asan_report_load1` chain is broken at the conditional branch

### Round 5: Standalone C reproducer
- Wrote `/tmp/c_test_fuzz.c`: C program that `dlopen`s `asan_target_asan.so` with `LD_PRELOAD=libasan.so.8`
- Compiled without ASAN: `gcc -o /tmp/c_test_fuzz /tmp/c_test_fuzz.c -ldl`
- **All 3 crash types correctly detected** with full ASAN diagnostic output
- Proved: the `.so`'s ASAN instrumentation is correct; the failure is specific to mid-process Python ctypes loading

### Round 6: Disambiguate lazy vs eager binding
- `ctypes.CDLL(path, mode=RTLD_GLOBAL)` uses `dlopen` with `RTLD_LAZY` by default
- `ctypes.CDLL(path, mode=RTLD_GLOBAL | RTLD_NOW)` forces eager resolution
- **Both modes produce the same result** — no ASAN detection
- Conclusion: not a lazy-vs-eager binding issue

### Round 7: Direct `__asan_report_load1` test
- Called `__asan_report_load1(0x42)` directly via ctypes from Python
- **Full ASAN diagnostic dump produced**, process aborted
- Proved: ASAN's error-reporting infrastructure IS functional when called directly
- The failure is specifically in the shadow-check-to-`abort()` chain, not in ASAN's reporting machinery

### Round 8: `libc.free()` on ASAN memory
- Loaded target `.so`, called `lib.free(lib.malloc(8))` via ctypes
- `ctypes.CDLL("libc.so.6", RTLD_GLOBAL)` to get libc's `free`
- Freed ASAN pointer with `libc.free(lib.malloc(8))` — **crashed with `munmap_chunk(): invalid pointer`**
- Proved: ASAN's allocator IS active and `malloc`/`free` in the `.so` resolve to ASAN's versions
- `libc.free()` cannot operate on ASAN-managed memory — this is expected and confirms correct interposition

## Code Paths

### Working path (process-start LD_PRELOAD):

```
exec(fuzzer, LD_PRELOAD=libasan.so.8)
  → ld.so loads libasan.so.8 at process start (first library)
  → ASAN constructors in .preinit_array initialize shadow memory
  → dlopen(asan_target_asan.so, RTLD_NOW)
  → PLT entries resolve (malloc→ASAN, free→ASAN, memset→__asan_memset)
  → Shadow checks work correctly
  → ASAN bug → __asan_report_load1 → ASAN diagnostic dump → abort()
```

### Broken path (mid-process ctypes loading):

```
Python process already running (no LD_PRELOAD)
  → ctypes.CDLL(verify_asan_link_order=0.so, RTLD_GLOBAL)
  → ctypes.CDLL(libasan.so.8, RTLD_GLOBAL)
  → ASAN constructors run (shadow memory initialized, base = 0x7fff8000)
  → ctypes.CDLL(asan_target_asan.so, RTLD_GLOBAL)
  → PLT entries resolve (malloc→ASAN? confirmed ✓)
  → Shadow checks read correct values (confirmed ✓)
  → BUT: conditional branch to __asan_report_load1 is NOT taken
  → Either the branch condition is wrong, or the instrumentation's
    internal state disagrees with what we observe in shadow memory
```

## Impact

### Direct-lite mode with ASAN `.so`:

| Metric | Value |
|---|---|
| Throughput (png_read_asan.so) | ~124k eps |
| Subprocess fallback | Eliminated (this fix) |
| ASAN bug detection | **Not functional** |
| AFL edge coverage | Working |
| Crash isolation | None (afl_shim.c `_exit(128+sig)` provides exit-code signal) |

### Workarounds for ASAN crash detection:

1. **Use a standalone executable target** instead of `.so` — fork+exec with `LD_PRELOAD=libasan.so.8` works correctly
2. **Use persistent subprocess mode** (`--inprocess` without `--inprocess-direct`) — fork + `LD_PRELOAD` at process start; ASAN detects bugs in the child
3. **Run with an external LD_PRELOAD wrapper**: `LD_PRELOAD=libasan.so.8 fuzzer-tool fuzz targets/target_asan.so ...` — ASAN initializes at process start; crash kills the fuzzer but ASAN diagnostics are emitted

## Reproducer Commands

```bash
# Build the ASAN test target
cd targets && gcc -g -fsanitize=address -shared -fPIC \
  -o asan_target_asan.so asan_target.c

# Test from Python (direct_lite mode — broken)
python3 -c "
import ctypes, subprocess
# verify_asan_link_order=0 shim
r = subprocess.run(['cc', '-shared', '-fPIC', '-o', '/tmp/v0.so', '-x', 'c', '-'],
    input=b'const char *__asan_default_options(void) { return \"verify_asan_link_order=0\"; }',
    capture_output=True, timeout=30)
ctypes.CDLL('/tmp/v0.so', mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libasan.so.8', mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL('targets/asan_target_asan.so', mode=ctypes.RTLD_GLOBAL)
lib.fuzz.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.fuzz.restype = ctypes.c_int
data, buf = b'BUG!U', ctypes.create_string_buffer(b'BUG!U')
print(lib.fuzz(ctypes.cast(buf, ctypes.c_void_p), 5))  # prints 0 — broken
"

# Test from C (process-start LD_PRELOAD — works)
cat > /tmp/c_test.c << 'EOF'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*fuzz_t)(const unsigned char *, size_t);
int main(void) {
    void *h = dlopen("./targets/asan_target_asan.so", RTLD_NOW);
    fuzz_t f = dlsym(h, "fuzz");
    unsigned char buf[] = "BUG!U";
    int r = f(buf, 5);
    printf("result=%d\n", r);
    return 0;
}
EOF
gcc -o /tmp/c_test /tmp/c_test.c -ldl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 /tmp/c_test
# → ASAN: heap-use-after-free on address ... — works!
```

## References

- `src/fuzzer_tool/services/fuzzer.py` — ASAN ctypes preloading logic (lines 1010-1064)
- `src/fuzzer_tool/cli/commands.py` — Removed ASAN `--inprocess-direct` fallback (lines 212-218)
- `targets/asan_target.c` — ASAN crash test target (3 crash modes)
- `/tmp/c_test_fuzz.c` — standalone C reproducer (deleted after testing)
- `/tmp/trace_asan.so` — `__asan_report_load1` trace shim (deleted after testing)
- `docs/inprocess-limitations.md` — General in-process mode limitations
