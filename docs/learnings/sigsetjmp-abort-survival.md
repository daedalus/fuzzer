# Learnings

## 2026-07-29: SIGABRT in direct_lite mode cannot be caught by Python signal handlers

**Context:** `src/fuzzer_tool/adapters/afl_shim.c`, inprocess.py
**Problem:** `fuzzer-tool fuzz targets/ffmpeg_read_asan.so` would die with "Aborted" in direct_lite mode — the in-process ASAN target calls `abort()` and the entire fuzzer process dies.

### Rejected

- **Python `signal.signal(SIGABRT, handler)`** — looks like it should work because Python can install C-level signal handlers via `signal.signal()`. But glibc's `abort()` implementation first does `raise(SIGABRT)`, and if the handler returns, it calls `signal(SIGABRT, SIG_DFL)` and `raise(SIGABRT)` again — the second raise uses SIG_DFL which kills the process. The handler's return value or flag-setting doesn't help because the finally block never executes.

- **`ctypes.CDLL` + signal handlers in `_run_c_direct`** — same underlying issue: abort()'s handler-reset-and-re-raise happens before the ctypes call returns, so Python's signal-handling path (which depends on the C function returning to the interpreter) never gets a chance to process the crash flag. Verified by observing that `_run_c_direct`'s `finally` block with `crashed` flag check never fires for SIGABRT (but does work for SIGSEGV which doesn't have the re-raise behavior).

- **Separate `sigguard.so` compiled as standalone .so** — would work but adds a runtime build dependency and a separate compilation step. The user correctly asked to integrate into the existing `afl_shim.c` which is already compiled into every target via `-include`.

### Approach

Modified `afl_shim.c` to use `sigsetjmp`/`siglongjmp` instead of signal handler return-and-re-raise:

1. Added `#include <setjmp.h>` and a global `sigjmp_buf` to the shim
2. Changed `__afl_crash_handler` to call `siglongjmp(__afl_jmp_buf, sig)` instead of the old restore-and-`raise(sig)` pattern — this completely escapes the signal handler without returning, so abort() never gets to reset the handler
3. Added `__afl_guarded_call(entry, data, size)` which performs `sigsetjmp` before calling the entry function and returns `-sig` (negated signal number, e.g. `-6` for SIGABRT) when `siglongjmp` fires
4. In `InProcessRunner._run_c_direct_lite`, if the target .so exports `__afl_guarded_call`, route calls through it instead of direct ctypes invocation

Also removed the `#ifndef __SANITIZE_ADDRESS__` guard on the `abort()` preprocessor override — the `__asan_default_options` shim (loaded before libasan) sets `halt_on_error=0:abort_on_error=0`, so ASAN's own bugs produce reports via stderr and return normally without calling `abort()`. The override is now safe in all builds and catches library-internal assertions (FFmpeg `av_assert0`).

Also expanded the guarded signals from 2 (SIGSEGV, SIGABRT) to 7 (added SIGFPE, SIGBUS, SIGILL, SIGPIPE, SIGSYS) — using an array and loop instead of individual static variables, since all signals route to the same `siglongjmp`.

### Key insight

`abort()` in glibc performs a two-step kill: `raise(SIGABRT)`, then if the handler returns, `signal(SIGABRT, SIG_DFL)` + `raise(SIGABRT)`. Any signal-handler approach that returns from the handler cannot survive abort() — glibc ensures the process termines. `siglongjmp` from within the handler is the only reliable way to survive abort() in-process because the handler never returns to abort()'s re-raise logic.

### Verification

`fuzzer-tool fuzz targets/ffmpeg_read_asan.so -n 10` completed without "Aborted" — 5 crashes detected (SIGABRT, correctly mapped to `signal:6`), full run summary printed. Without the fix, the same command dies silently with "Aborted" after the first crash.

### Generalizes to

- In-process crash survival (ctypes calls, JNI, embedded interpreters) must use non-local jumps (`sigsetjmp`/`siglongjmp` or platform equivalent) to escape signal handlers that are subject to abort()'s two-step re-raise. Python's `signal.signal()` cannot provide this.
- Preprocessor `#define abort()` overrides only affect the target's own source code — pre-compiled libraries (libasan, libc, third-party .so files) call the REAL abort() which requires signal-level handling.
- `__asan_default_options` with `halt_on_error=0:abort_on_error=0` makes ASAN safe for in-process fuzzing — ASAN bugs produce stderr reports without terminating, so intercepting separate abort() calls from library assertions won't interfere with ASAN's own diagnostic output.
