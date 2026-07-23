# Trace-Cmp Comparison Tracing — Vendor Build How-To

## What this is

Compiler-IR comparison tracing (`-fsanitize-coverage=trace-cmp`) intercepts
**every comparison in the binary** — including ones that the compiler inlines
and folds into integer instructions. This catches comparisons that the
symbol-based `cmplog_shim` (which intercepts `memcmp`/`strcmp`) misses.

When you compile a library (like libpng or zlib) with `trace-cmp`, every
`if`, `switch`, `==`, `!=` inside that library fires a
`__sanitizer_cov_trace_cmp*` callback. At runtime, the `cmplog_shim`
provides the logging implementations that write operand pairs to the
`_CMPLOG_OUT` file. The fuzzer's `CmplogCollector` reads them — producing
dictionary tokens from comparisons in library code, not just your wrapper.

## Prerequisites

- **Clang** (GCC ignores `-fsanitize-coverage=trace-cmp`). Debian: `apt install clang`
- **Vendor library sources** in `vendor/zlib` and `vendor/libpng`
  (run `apt-get source zlib libpng-dev` to fetch Debian source packages)

## Quick start

```bash
# One-shot: rebuild vendor libs and all trace-cmp targets
tools/build_targets.sh --vendor-tracecmp

# With ASAN:
tools/build_targets.sh --vendor-tracecmp --asan
```

Output goes to `targets/*_tracecmp.so` and `targets/*_tracecmp` (executables),
leaving the regular (non-trace-cmp) builds untouched.

## How `--cmplog` builds work

When you pass `--cmplog` to `build_targets.sh`, the build script:

1. **Compiles `cmplog_shim.c` with Clang** (not GCC) into a separate `.o`
2. **Compiles the target with Clang + `-fsanitize-coverage=trace-cmp`**
   so the compiler generates `__sanitizer_cov_trace_cmp*` calls
3. **Links the shim `.o` into the target `.so`** — the shim provides the
   callback implementations that the target's code calls
4. **Adds `-Wl,-Bsymbolic`** to prevent ASAN from overriding the callbacks

The shim must NOT be compiled with `-fsanitize-coverage=trace-cmp` — it
**provides** the callbacks, it must not **call** them.

## Key compiler flags

| Flag | Purpose |
|------|---------|
| `-fsanitize-coverage=trace-cmp` | Clang-only: insert callbacks before every comparison |
| `-fsanitize-coverage=trace-pc-guard` | Required companion: without it, `trace-cmp` generates zero callbacks |
| `-Wl,-Bsymbolic` | Force intra-.so symbol resolution, prevent ASAN LD_PRELOAD override |
| `-shared -fPIC` | Required for .so targets (in-process mode) |

**Important**: `trace-cmp` alone does nothing. You **must** combine it with
`trace-pc-guard`. GCC's `-fsanitize-coverage=trace-cmp` does not generate the
`__sanitizer_cov_trace_cmp*` callbacks that the shim implements.
**Always use Clang.**

## ASAN + tracecmp: the `-Bsymbolic` fix

### The bug

When the target `.so` is compiled with both `-fsanitize=address` (ASAN) and
`-fsanitize-coverage=trace-cmp,trace-pc-guard`, the ASAN runtime
(`libasan.so`) provides its own `__sanitizer_cov_trace_cmp*` **no-op stubs**
that override the cmplog shim's logging implementations.

This happens because ASAN is loaded via `LD_PRELOAD` before the target `.so`.
The dynamic linker resolves the trace-cmp PLT/GOT entries to ASAN's stubs
instead of the shim's implementations. Result: cmplog shows `0t 0p` — the
callbacks fire but write nothing.

**How to reproduce**: compile a target with ASAN + trace-cmp, run it under
`LD_PRELOAD=libasan.so.8`, and check the `_CMPLOG_OUT` file. It will be
empty even though comparisons execute.

### The fix: `-Wl,-Bsymbolic`

Link the target `.so` with `-Wl,-Bsymbolic`. This tells the linker to resolve
references to global symbols **within the .so itself**, bypassing the PLT/GOT
and preventing ASAN's LD_PRELOAD from intercepting them.

```bash
clang -O2 -g -fsanitize=address -fsanitize-coverage=trace-cmp,trace-pc-guard \
    -shared -fPIC -Wl,-Bsymbolic \
    -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target.so targets/my_target.c \
    /tmp/cmplog_shim.o vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm -ldl
```

`build_targets.sh --asan --cmplog` applies this automatically.

### Why this works

Without `-Bsymbolic`:
```
vendored libpng code → PLT → GOT → (resolved at runtime)
                                        ↓
                              ASAN's no-op stub wins (loaded first via LD_PRELOAD)
```

With `-Bsymbolic`:
```
vendored libpng code → direct call to shim's implementation (binding resolved at link time)
                                        ↓
                              cmplog shim's buffer_cmp() fires, writes to _CMPLOG_OUT
```

### Verification

```bash
# Build with -Bsymbolic
clang -O2 -g -fsanitize=address -fsanitize-coverage=trace-cmp,trace-pc-guard \
    -shared -fPIC -Wl,-Bsymbolic -include src/fuzzer_tool/adapters/afl_shim.c \
    -o /tmp/test.so targets/png_read.c /tmp/cmplog_shim.o \
    vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm -ldl

# Run with ASAN — should produce cmplog output
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 _CMPLOG_OUT=/tmp/cmp.log \
    python3 -c "
import ctypes; lib = ctypes.CDLL('/tmp/test.so')
fn = lib.fuzz_shm_run; fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 64
buf = (ctypes.c_uint8 * len(data))(*data)
fn(buf, len(data)); lib.__tracecmp_flush()
"
wc -l /tmp/cmp.log  # Should show >0 CMP lines
```

### Alternative: `-fvisibility=hidden`

If `-Bsymbolic` causes issues with other symbol resolution, an alternative is
to compile the cmplog shim with hidden visibility so its trace-cmp symbols are
local to the `.so`:

```bash
clang -O2 -g -fsanitize=address -fvisibility=hidden -fPIC -c \
    src/fuzzer_tool/adapters/cmplog_shim.c -o /tmp/cmplog_shim.o
```

This prevents the symbols from appearing in the dynamic symbol table, so
ASAN's LD_PRELOAD can't see or override them. The lifecycle symbols
(`__cmplog_reset`, `__tracecmp_flush`) need explicit `visibility("default")`
attributes (already present in the shim).

## Why `.so` not executable for in-process mode

When clang builds an **executable** with `-fsanitize-coverage=trace-cmp`, the
compiler-rt runtime links in **weak** definitions of
`__sanitizer_cov_trace_cmp*`. These are no-ops — they don't log anything —
and they cannot be overridden by LD_PRELOAD.

When clang builds a **shared library** (`.so`) with the same flags, the
`__sanitizer_cov_trace_cmp*` symbols are left **undefined (`U`)**. They
are resolved at load time by the linked `cmplog_shim.o`, which provides the
strong definitions that actually log.

**Always use the `.so` variant** when fuzzing in-process.

## Adding a new target

### If your target uses libpng and/or zlib

1. Add your target source to `targets/your_target.c` with a `fuzz_shm_run()`
   entry point and AFL edge coverage (see `targets/png_read.c` for a template).
2. Add a build rule in `build_vendored_tracecmp_targets()` in
   `tools/build_targets.sh`. Follow the existing pattern:

```bash
# In build_vendored_tracecmp_targets(), after the "Build .so targets" section:
if [ -f "$TARGETS/your_target.c" ]; then
    $CC -O2 -g $ALL_FLAGS -shared -fPIC -Wl,-Bsymbolic -include "$SHIM" \
        -o "$TARGETS/your_target${OUT_SUFFIX}.so" \
        "$TARGETS/your_target.c" $VENDOR_LIBS $VENDOR_INC 2>/dev/null && \
        ok "your_target${OUT_SUFFIX}.so" || warn "failed: your_target${OUT_SUFFIX}.so"
fi
```

### If your target needs a different vendor library

1. Add the source to `vendor/` (e.g. `vendor/libfoo/`).
2. Compile it with clang + `-fsanitize-coverage=trace-cmp,trace-pc-guard`:
   ```bash
   cd vendor/libfoo
   CC=clang CFLAGS="-O2 -g -fPIC -fsanitize-coverage=trace-cmp,trace-pc-guard" \
       ./configure --enable-shared=no && make -j$(nproc)
   ```
3. Link against the static `.a` or `.o` files in your target build command.

### If your target uses NO vendor libraries (standalone)

Build directly:

```bash
clang -O2 -g -fsanitize-coverage=trace-cmp,trace-pc-guard -shared -fPIC \
    -Wl,-Bsymbolic \
    -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target_tracecmp.so targets/my_target.c \
    /tmp/cmplog_shim.o
```

Comparisons in your target code will be captured, but comparisons in any
dynamically linked system libraries will not.

## How the fuzzer uses it at runtime

The fuzzer's `CmplogCollector` (in `src/fuzzer_tool/core/cmplog.py`) handles
the cmplog shim automatically:

1. On `start()`, it compiles `cmplog_shim.c` into a `.o` and links it
2. `_CMPLOG_OUT` is set in `os.environ` before the target loads (both
   `direct_lite` and `persistent` modes inherit it)
3. The `.so` constructor opens `_CMPLOG_OUT` for appending at load time
4. After each target execution, the fuzzer calls `__tracecmp_flush()` to
   push the buffered CMP lines to disk
5. `collect_tokens()` reads all `CMP` lines and extracts operand tokens for
   the dictionary and input-to-state matching

## Verifying your build

### Static check: are trace-cmp callbacks in the library?

```bash
# Count trace-cmp callbacks in the vendor library
nm vendor/zlib/libz.a | grep -c 'U.*trace_cmp'
# Expected: 20 (zlib 1.3.1)

# In the target .so — should show exported (T) symbols
nm -D targets/png_read.so | grep 'trace_cmp'

# Check for cmplog lifecycle symbols
nm -D targets/png_read.so | grep -E 'cmplog_reset|tracecmp_flush'
```

### Runtime check: are comparisons being logged?

```bash
# Quick smoke test (no ASAN)
clang -O2 -g -fsanitize-coverage=trace-cmp,trace-pc-guard -shared -fPIC \
    -Wl,-Bsymbolic -include src/fuzzer_tool/adapters/afl_shim.c \
    -o /tmp/test_cmp.so targets/png_read.c /tmp/cmplog_shim.o \
    vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm -ldl

_CMPLOG_OUT=/tmp/cmp.log python3 -c "
import ctypes, os
os.environ['_CMPLOG_OUT'] = '/tmp/cmp.log'
lib = ctypes.CDLL('/tmp/test_cmp.so')
fn = lib.fuzz_shm_run; fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 64
buf = (ctypes.c_uint8 * len(data))(*data)
fn(buf, len(data)); lib.__tracecmp_flush()
"
wc -l /tmp/cmp.log  # Should show >0 CMP lines
```

## Known limitations

- **libjpeg-turbo** is not vendored yet — `jpeg_read_tracecmp.so` only
  captures comparisons in the wrapper itself (1 callback). To instrument
  libjpeg, add `vendor/libjpeg-turbo` and compile with trace-cmp.
- **Double counting**: if the same comparison operand appears from both the
  symbol-based shim (`cmplog_shim`) and the IR-level tracing, the
  `CmplogCollector` deduplicates by token value, so there's no harm.
- **Performance**: `trace-cmp` adds a callback per comparison instruction,
  which is more overhead than just `trace-pc-guard`. Expect slower execution
  per fuzz iteration. Use it alongside the regular build, not as a replacement.
- **ASAN + tracecmp requires `-Bsymbolic`**: ASAN's LD_PRELOAD overrides
  trace-cmp callbacks with no-ops. The `-Wl,-Bsymbolic` linker flag forces
  intra-.so resolution, preventing the override.
- **Vendored libs must be rebuilt with trace-cmp**: system libraries (libpng,
  zlib) are not compiled with trace-cmp. You must rebuild from source using
  `--vendor-tracecmp` or manually with `CC=clang CFLAGS="...-fsanitize-coverage=trace-cmp,trace-pc-guard"`.
