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
# Daily workflow: rebuild vendored libs with ASAN + trace-cmp, then all .so targets
tools/build_targets.sh --cmplog

# Standalone trace-cmp targets (separate *._tracecmp.so builds, no ASAN):
tools/build_targets.sh --vendor-tracecmp

# With ASAN on the standalone trace-cmp builds:
tools/build_targets.sh --vendor-tracecmp --asan
```

**`--cmplog`** is the primary workflow. It links every .so target against vendored
libpng+zlib compiled with **both ASAN and trace-cmp**, so comparisons inside
library code fire `__sanitizer_cov_trace_cmp*` callbacks that the cmplog shim
logs. Vendor libs must be compiled with these flags first (see "Rebuilding
vendored libs with ASAN + trace-cmp" below). When the `.a` files exist in
`vendor/`, the build script auto-detects them and prints:
```
Using vendored trace-cmp libraries
```

**`--vendor-tracecmp`** produces `targets/*_tracecmp.so` and `targets/*_tracecmp`
(executables), leaving the regular builds untouched. Useful for A/B testing
trace-cmp vs non-trace-cmp performance.

## How `--cmplog` builds work

When you pass `--cmplog` to `build_targets.sh`, the build script:

1. **Compiles `cmplog_shim.c` with Clang** into a separate `.o`
2. **Detects vendored `.a` files** (`vendor/zlib/libz.a`,
   `vendor/libpng/.libs/libpng16.a`) compiled with trace-cmp — these contain
   `U` references to `__sanitizer_cov_trace_cmp*` from every comparison in
   library code
3. **Links the shim `.o` + vendored `.a` files into the target `.so`** —
   the linker resolves the vendored libs' `U` trace-cmp references to the
   shim's `T` implementations within the same `.so`
4. **Adds `-Wl,-Bsymbolic`** so those resolutions stay internal and ASAN's
   LD_PRELOAD can't override them with no-op stubs

The target wrapper (e.g. `png_read.c`) is **not** compiled with trace-cmp —
only the vendored libraries are. This keeps the wrapper lightweight while
capturing all comparisons inside the library code that does the real work.

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

## Rebuilding vendored libs with ASAN + trace-cmp

Vendored libraries must be compiled with `clang`, ASAN, and trace-cmp so their
comparisons produce callbacks that the cmplog shim can log. The vendor source
lives in `vendor/zlib` and `vendor/libpng/` (copy from `fuzzer_old/vendor/`
if missing).

```bash
# zlib
cd vendor/zlib
make clean 2>/dev/null
CC=clang CFLAGS="-O2 -g -fPIC -fsanitize=address \
    -fsanitize-coverage=trace-cmp,trace-pc-guard" \
    ./configure --static && make -j$(nproc)
# Expected: ~20 U trace_cmp symbols across all .o files

# libpng (depends on zlib — ./configure finds ../zlib headers)
cd vendor/libpng
make clean 2>/dev/null
CC=clang CFLAGS="-O2 -g -fPIC -fsanitize=address \
    -fsanitize-coverage=trace-cmp,trace-pc-guard -I../zlib" \
    LDFLAGS="-L../zlib" \
    ./configure --enable-shared=no --quiet && make -j$(nproc)
# Expected: ~36 U trace_cmp symbols across all .o files
```

After rebuilding, run `build_targets.sh --cmplog`. The build script auto-detects
the `.a` files and prints "Using vendored trace-cmp libraries".

## Verifying your build

### Build-time diagnostics (automated)

The build script runs these checks after every `--cmplog` build:

```
Verifying vendored trace-cmp resolution...
  OK: vendor/zlib: 20 trace-cmp callers (U)
  OK: vendor/libpng: 36 trace-cmp callers (U)
  Vendor callers: 56 | .so implems: 84 | .so unresolved: 0
  OK: 21/21 .so targets: trace-cmp fully resolved
  OK: 21/21 .so targets: -Bsymbolic present
```

- **Vendor callers**: `U` (undefined) references to `__sanitizer_cov_trace_cmp*`
  in the vendor `.a` files — each one is a comparison site in libpng/zlib.
- **.so implems**: `T` (text/defined) callback implementations from
  `cmplog_shim.o` linked into each `.so`.
- **.so unresolved**: any remaining `U` after linking — must be 0.

### Manual static checks

```bash
# Count trace-cmp callers in vendor libraries
nm vendor/zlib/libz.a | grep -c 'U.*trace_cmp'
# Expected: 20 (zlib 1.3.1)
nm vendor/libpng/.libs/libpng16.a | grep -c 'U.*trace_cmp'
# Expected: 36 (libpng 1.6.x)

# Check the final .so has no unresolved trace-cmp (only T definitions)
nm targets/png_read.so | grep 'trace_cmp'
# Should show: 4 T symbols, 0 U symbols

# Check -Bsymbolic is present
readelf -d targets/png_read.so | grep SYMBOLIC
# Should show: 0x0000000000000010 (SYMBOLIC) 0x0

# Check for cmplog lifecycle symbols
nm -D targets/png_read.so | grep -E 'cmplog_reset|tracecmp_flush'
```

### Runtime check: are comparisons being logged?

Targets built with `--cmplog` are ASAN-instrumented, so loading them via
`ctypes.CDLL` requires preloading libasan first. Use the
`verify_asan_link_order=0` shim to bypass ASAN's load-order check:

```bash
_CMPLOG_OUT=/tmp/cmp.log python3 << 'PYEOF'
import ctypes, os, subprocess, tempfile

# 1. Compile ASAN link-order bypass shim
shim_src = b'const char *__asan_default_options() { return "verify_asan_link_order=0"; }'
fd, shim_path = tempfile.mkstemp(suffix=".so", prefix="asan_opts_")
os.close(fd)
subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", shim_path, "-xc", "-"],
    input=shim_src, capture_output=True, check=True)

# 2. Preload ASAN shim + libasan with RTLD_GLOBAL
ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libasan.so.8", mode=ctypes.RTLD_GLOBAL)
os.unlink(shim_path)

# 3. Load the cmplog target and run a minimal PNG
os.environ["_CMPLOG_OUT"] = "/tmp/cmp.log"
lib = ctypes.CDLL("targets/png_read.so", mode=ctypes.RTLD_GLOBAL)
fn = lib.fuzz_shm_run
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 256
buf = (ctypes.c_uint8 * len(data))(*data)
fn(buf, len(data))
lib.__tracecmp_flush()
PYEOF
wc -l /tmp/cmp.log  # Should show >0 CMP lines (~85 for minimal PNG)
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
- **Vendored libs must be rebuilt with ASAN + trace-cmp**: system libraries (libpng,
  zlib) are not compiled with either. The `--cmplog` build links against vendored
  `.a` files but does NOT rebuild them. Rebuild manually (see "Rebuilding vendored
  libs with ASAN + trace-cmp" above) or the `.so` targets will have zero trace-cmp
  callers and produce `0t 0p` at runtime.
- **`--vendor-tracecmp` does NOT add ASAN** to the vendored libs — use
  `--vendor-tracecmp --asan` if you need ASAN in standalone trace-cmp builds.
- **nop_target.c** is required for `build_targets.sh` to reach the verification
  steps (the missing source causes `set -e` to abort). Copy from `fuzzer_old/`
  if missing.
