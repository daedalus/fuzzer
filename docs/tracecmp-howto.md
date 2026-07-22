# Trace-Cmp Comparison Tracing — Vendor Build How-To

## What this is

Compiler-IR comparison tracing (`-fsanitize-coverage=trace-cmp`) intercepts
**every comparison in the binary** — including ones that the compiler inlines
and folds into integer instructions. This catches comparisons that the
symbol-based `cmplog_shim` (which LD_PRELOADs `memcmp`/`strcmp`) misses.

When you compile a library (like libpng or zlib) with `trace-cmp`, every
`if`, `switch`, `==`, `!=` inside that library fires a
`__sanitizer_cov_trace_cmp*` callback. At runtime, the `tracecmp_shim.so`
LD_PRELOAD intercepts these callbacks and writes the operands to the
`_CMPLOG_OUT` file. The fuzzer's `CmplogCollector` reads them — producing
dictionary tokens from comparisons in library code, not just your wrapper.

## Prerequisites

- **Clang** (GCC ignores `-fsanitize-coverage`). Debian: `apt install clang`
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

## Output files

| File | Description | trace-cmp callbacks |
|---|---|---|
| `targets/png_read_tracecmp.so` | libpng fuzz target + vendored zlib+png | 4 (wrapper) + 36 (png) + 20 (zlib) |
| `targets/zlib_read_tracecmp.so` | zlib fuzz target + vendored zlib | 2 (wrapper) + 20 (zlib) |
| `targets/gzip_read_tracecmp.so` | gzip fuzz target + vendored zlib | 2 (wrapper) + 20 (zlib) |
| `targets/jpeg_read_tracecmp.so` | libjpeg fuzz target (system libjpeg) | 1 (wrapper only) |
| `targets/png_read_tracecmp` | Executable variant (subprocess mode) | same |
| `targets/zlib_read_tracecmp` | Executable variant | same |
| `targets/gzip_read_tracecmp` | Executable variant | same |

## Why `.so` not executable for in-process mode

When clang builds an **executable** with `-fsanitize-coverage=trace-cmp`, the
compiler-rt runtime links in **weak** definitions of
`__sanitizer_cov_trace_cmp*`. These are no-ops — they don't log anything —
and they cannot be overridden by LD_PRELOAD.

When clang builds a **shared library** (`.so`) with the same flags, the
`__sanitizer_cov_trace_cmp*` symbols are left **undefined (`U`)**. They
are resolved at load time by LD_PRELOAD'ing `tracecmp_shim.so`, which
provides the strong definitions that actually log.

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
    $CC -O2 -g $ALL_FLAGS -shared -fPIC -include "$SHIM" \
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
    -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target_tracecmp.so targets/my_target.c
```

Comparisons in your target code will be captured (via `tracecmp_shim.so` at
runtime), but comparisons in any dynamically linked system libraries will not.

## How the fuzzer uses it at runtime

The fuzzer's `CmplogCollector` (in `src/fuzzer_tool/core/cmplog.py`) handles
both shims automatically:

1. On `start()`, it compiles `tracecmp_shim.so` via `shim_factory.build_tracecmp_shim()`
2. On `setup_env()`, it prepends both `cmplog_shim.so` and `tracecmp_shim.so`
   to `LD_PRELOAD`, and sets `_CMPLOG_OUT` to a temp file
3. After each target execution, `collect_tokens()` reads all `CMP` lines from
   the log file and extracts operand tokens for the dictionary and input-to-state
   matching

## ASAN + tracecmp (two-step build)

When the target is compiled with both `-fsanitize=address` and
`-fsanitize-coverage=trace-cmp`, the ASAN library (`libasan.so`) provides
its OWN `__sanitizer_cov_trace_cmp*` no-op stubs that override the tracecmp
shim's logging implementations. This happens because ASAN is LD_PRELOAD'd
first and its symbols take priority over the shim's.

**Fix**: compile `tracecmp_shim.c` into the target `.so` with
`-fvisibility=hidden`, so the tracecmp callbacks have local binding and
bypass the PLT/GOT:

```bash
# Step 1: compile tracecmp_shim.c with hidden visibility + ASAN
clang -O2 -g -fsanitize=address -fvisibility=hidden -fPIC -c \
    src/fuzzer_tool/adapters/tracecmp_shim.c \
    -o /tmp/tracecmp_shim.o

# Step 2: compile target with trace-cmp + ASAN
clang -O2 -g \
    -fsanitize=address \
    -fsanitize-coverage=trace-cmp,trace-pc-guard \
    -shared -fPIC \
    -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target_asan.so \
    targets/my_target.c /tmp/tracecmp_shim.o \
    vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm

# Step 3: run with ASAN LD_PRELOAD'd externally
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 \
python -m fuzzer_tool fuzz \
    targets/my_target_asan.so \
    -c --cmplog -d corpus -n 10000
```

The fuzzer auto-detects tracecmp and ASAN, uses `direct_lite` mode, and
collects comparison tokens from the target's own tracecmp callbacks
without ASAN interference.

The `build_targets.sh --vendor-tracecmp --asan` script does this
automatically for vendored libpng+zlib targets.

## Verifying your build

### Static check: are trace-cmp callbacks in the library?

```bash
# Count trace-cmp callbacks in the vendor library
nm vendor/zlib/libz.a | grep -c 'U.*trace_cmp'
# Expected: 20 (zlib 1.3.1)

# In a standalone shim .so — should show exported (T) symbols
nm -D targets/png_read_tracecmp.so | grep 'trace_cmp'

# In an ASAN-compiled .so — hidden (t) symbols, not in dynamic table
nm targets/png_read_tracecmp_asan.so | grep 'trace_cmp'
```

### Runtime check: are comparisons being logged?

```bash
# Build tracecmp_shim.so
gcc -shared -fPIC -O2 -g -o /tmp/tracecmp_shim.so \
    src/fuzzer_tool/adapters/tracecmp_shim.c

# Run the target with a valid input
_CMPLOG_OUT=/tmp/cmp.log LD_PRELOAD=/tmp/tracecmp_shim.so \
    ./targets/png_read_tracecmp /path/to/input.png

# Check the log
wc -l /tmp/cmp.log
head -20 /tmp/cmp.log
```

If you see `CMP <hex> <hex> <result> <len>` lines, it's working.

## Known limitations

- **libjpeg-turbo** is not vendored yet — `jpeg_read_tracecmp.so` only
  captures comparisons in the wrapper itself (1 callback). To instrument
  libjpeg, add `vendor/libjpeg-turbo` and compile with trace-cmp.
- **Double counting**: if the same comparison operand appears from both the
  symbol-based shim (`cmplog_shim`) and the IR-level shim (`tracecmp_shim`),
  the `CmplogCollector` deduplicates by token value, so there's no harm.
- **Performance**: `trace-cmp` adds a callback per comparison instruction,
  which is more overhead than just `trace-pc-guard`. Expect slower execution
  per fuzz iteration. Use it alongside the regular build, not as a replacement.
- **ASAN + tracecmp requires two-step build**: the tracecmp callbacks must be
  compiled with `-fvisibility=hidden` and linked into the target `.so` to
  prevent ASAN's LD_PRELOAD from overriding them.
