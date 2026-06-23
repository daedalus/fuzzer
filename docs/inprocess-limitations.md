# In-Process Fuzzing — Limitations & Design Notes

## Overview

The fuzzer supports three in-process execution modes that bypass the default
subprocess-per-iteration model. Each trades off speed, crash isolation, and
coverage support differently.

| Mode | Flag | execs/sec | Crash isolation | Coverage |
|------|------|-----------|-----------------|----------|
| Direct ctypes | `--inprocess-direct` | ~2k–34k | None (crash kills fuzzer) | No |
| Persistent subprocess | `--inprocess` | ~2.9k | Full (separate process) | Partial* |
| Per-call subprocess | `--inprocess` | ~21 | Full | Partial* |

\* Coverage bitmap is returned by the loader but the sanitizer's inline
instrumentation does not populate it (see Coverage section below).

---

## Direct ctypes (`--inprocess-direct`)

**How it works:** Loads the target `.so` via `ctypes.CDLL` and calls
`LLVMFuzzerTestOneInput` directly in the fuzzer process. A minimal C shim
provides the undefined `__sanitizer_cov_8bit_counters_init` symbol so the
`.so` can load.

**Speed:** 2,000–34,000 execs/sec depending on target complexity. The
overhead is only the ctypes FFI call + target execution time.

**Limitations:**

- **No crash isolation.** A SIGSEGV, SIGABRT, or any other signal in the
  target kills the fuzzer process immediately. The target *must* handle
  errors internally via `setjmp`/`longjmp` (like our libpng wrapper does)
  or be compiled with sanitizers that trap via `__asan_on_error` instead
  of signals.

- **No coverage.** The sanitizer's inline instrumentation requires the
  sanitizer runtime to be linked, which doesn't happen when loading a `.so`
  via ctypes. A no-op shim provides the `__sanitizer_cov_8bit_counters_init`
  symbol but the inline code never initializes. The coverage bitmap
  remains all zeros.

- **Thread safety.** The target must be thread-safe if used with parallel
  workers. Each worker gets its own `ctypes.CDLL` handle.

- **`longjmp` across FFI.** If the target uses `longjmp` to recover from
  errors (as our libpng shim does), the jump must not cross the ctypes
  FFI boundary in a way that corrupts the Python stack. Our shim handles
  this correctly by keeping `setjmp`/`longjmp` within the same C function.

---

## Persistent Subprocess (`--inprocess` with `-c`)

**How it works:** Spawns one Python subprocess that stays alive across
all iterations. The fuzzer sends input data via stdin and receives the
return code + coverage bitmap via stdout. This eliminates Python startup
and `ctypes.CDLL` load overhead on every iteration.

**Speed:** ~2,900 execs/sec — 130× faster than per-call subprocess.

**Limitations:**

- **Coverage bitmap is empty.** The subprocess loads the target `.so` via
  ctypes, which means the sanitizer runtime is not linked. The no-op
  `__sanitizer_cov_8bit_counters_init` shim prevents the sanitizer from
  initializing inline instrumentation. The bitmap is returned (correct
  size) but all bytes are zero.

- **Single-threaded.** The persistent subprocess is a single process.
  Parallel fuzzing spawns multiple subprocess instances.

- **Pipe overhead.** Each iteration still involves stdin/stdout pipe I/O
  (~50μs per round-trip). This is the bottleneck vs direct ctypes.

---

## Coverage Limitations

**Root cause:** Sanitizer coverage (`-fsanitize-coverage=inline-8bit-counters`)
requires the sanitizer runtime to be linked into the binary. When a `.so` is
loaded via `ctypes.CDLL`, the runtime is not linked — only the `.so`'s own
undefined symbols need resolution.

The shim provides `__sanitizer_cov_8bit_counters_init` so the `.so` can load,
but this function is a no-op. The sanitizer's inline instrumentation (compiled
into the `.so` by clang) calls this function during library initialization.
With a no-op, the inline code never sets up the bitmap pointer, so all
counter writes go nowhere.

**What would fix this:**

1. **Link the sanitizer runtime statically** into the target `.so`. This
   requires `-static-libsan` or equivalent, which clang doesn't support
   for coverage-only builds.

2. **Use a standalone executable** instead of a `.so`. When compiled as
   an executable with `-fsanitize-coverage=inline-8bit-counters`, the
   linker automatically links the sanitizer runtime. The fuzzer can then
   run it as a subprocess and read the `.profraw` file.

3. **Use a custom coverage mechanism** that doesn't depend on the sanitizer
   runtime — e.g., Intel Pin, DynamoRIO, or manual edge instrumentation.

**Current workaround:** For coverage-guided fuzzing, compile the target as
a standalone executable (not a `.so`) and use the fuzzer's default
subprocess mode with `-c`.

---

## Platform Requirements

- **Linux x86_64** required. The ELF parser assumes 64-bit little-endian.
  The `BitmapReader` parses LOAD segments and resolves virtual addresses.
- **clang** required for sanitizer-instrumented targets. GCC does not
  support `-fsanitize-coverage=inline-8bit-counters`.
- **Python 3.10+** required for `match` syntax and type hints used in
  the shim factory.

---

## Known Issues

1. **Per-call subprocess is slow (~21 execs/sec).** The persistent loader
   optimizes this to ~2,900 execs/sec but only activates with `-c`. Without
   coverage, each iteration spawns a new Python process.

2. **`--inprocess-direct` + `-c` doesn't populate the bitmap.** The shim
   factory builds a minimal shim, but the sanitizer's inline instrumentation
   doesn't initialize. The fuzzer runs but coverage-guided decisions are
   based on an empty bitmap.

3. **The `shm-edges` display shows 0.** This is cosmetic — the bitmap
   from the persistent loader is 0 bytes of actual coverage data because
   the sanitizer isn't initialized. The fuzzer still runs and finds crashes
   via signal detection.

4. **ELF parsing fragility.** The `BitmapReader` and persistent loader
   parse ELF symbol tables to find `__start___sancov_cntrs`. This works
   for standard clang-built `.so` files but may fail for:
   - Stripped binaries (no `.symtab`)
   - LTO-built binaries (different symbol layout)
   - Binaries with custom linker scripts

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  Fuzzer Process                                     │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  InProcessRunner                              │   │
│  │                                               │   │
│  │  direct=True ──► ctypes.CDLL ──► fn_ptr()    │   │
│  │                  (shim preload via RTLD_GLOBAL)│   │
│  │                                               │   │
│  │  direct=False ─► PersistentLoader             │   │
│  │                   │                           │   │
│  │                   ▼                           │   │
│  │            ┌─────────────┐                    │   │
│  │            │ Subprocess  │ stdin/stdout pipes  │   │
│  │            │ Python +    │                     │   │
│  │            │ ctypes.CDLL │                     │   │
│  │            └─────────────┘                    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  ShimFactory                                  │   │
│  │  • Inspects target ELF for sancov symbols     │   │
│  │  • Builds minimal C shim (one-time)           │   │
│  │  • BitmapReader reads counters from memory    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Coverage (SHM bitmap)                        │   │
│  │  • reset before each call                     │   │
│  │  • read bitmap after each call                │   │
│  │  • copy into SHM for fuzz_one() decisions     │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```
