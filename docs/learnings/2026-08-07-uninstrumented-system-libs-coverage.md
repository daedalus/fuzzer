# Uninstrumented system libs: fuzzing the wrapper, not the library

**Date:** 2026-08-07
**Context:** fuzzer-new `tools/build_targets.sh`, reported "why does libpng give only 120 edges?" on `targets/png_read.so`

## Problem
A libpng campaign plateaued at a very low edge count. libpng is a large library with deep chunk-parsing, filter-reconstruction and inflate paths — a few hundred edges is implausibly low for it, and the number barely moved no matter how long the campaign ran or which schedulers were enabled.

## Rejected
- **"The mutators aren't reaching deep parser states"** — the natural first guess, and the one that sends you tuning schedulers and dictionaries for a day. Dropped because the ceiling was structural: the edges being counted did not exist in the binary at all, so no mutation strategy could ever have found them.
- **"`--clang-scov` is required to instrument vendored libs"** — believed this mid-investigation and it produced a real, working `png_read_scov.so`. It was a red herring: `--cmplog` is ON by default, and `build_simple_so_targets()` already auto-detects vendored archives and links them into the *plain* target. `_scov` and plain came out byte-for-byte equivalent, which is what exposed the misunderstanding.

## Approach
Symbol inspection settles it immediately, and is much faster than reasoning about coverage numbers:

```
readelf -d targets/png_read.so | grep NEEDED
nm targets/png_read.so | grep -c ' [Tt] png_'    # defined here
nm -D targets/png_read.so | grep -c 'U png_'     # resolved at runtime
```

|                       | system-linked | vendored + instrumented |
|-----------------------|---------------|-------------------------|
| `NEEDED`              | `libpng16.so.16`, `libz.so.1` | none |
| png symbols defined   | 2             | 418                     |
| png symbols undefined | 17            | 0                       |
| edge bitmap           | 8,192 B       | 16,384 B                |
| **edges (45 s)**      | **304**       | **1,444**               |

The AFL shim's edge callbacks only exist in code the build compiled. `targets/png_read.c` is a 175-line wrapper — open file, call `png_read_png`, check the error path. Everything interesting lived behind a `.so` boundary the instrumentation could not see. The campaign was measuring the wrapper.

The vendored build needed two `build_targets.sh` fixes:

1. **Wrong relative path.** `compile_vendored_libs()` runs `cd vendor/libpng` and passed `-I../../zlib` / `-L../../zlib` — two levels up resolves *outside the repo*. Correct is `../zlib` (`VENDOR="vendor"` is repo-relative). Also needs `CPPFLAGS`: libpng's `pnglibconf.h` generation step does not inherit `CFLAGS`, so it silently probed the *system* zlib and failed the `ZLIB_VERNUM != PNG_ZLIB_VERNUM` guard.
2. **Object glob swept up test binaries.** `ls vendor/libpng/*.o` picks up `pngtest.o`, and zlib's `example.o` / `minigzip.o` / `example64.o` / `minigzip64.o` — build byproducts of `make all` that each define `main()`, colliding with the target wrapper's `main()`. Link the built `.a` archives instead; the archive is exactly the library's object subset, and also includes the `mips/intel/powerpc/` objects a flat top-level glob misses entirely.

## Key insight
Coverage instrumentation is a property of *how each object was compiled*, not of the target binary as a whole. A target can be fully instrumented, report "AFL instrumentation: detected", auto-size a bitmap, and produce a healthy-looking rising edge curve — while 95% of the code under test is a dynamically-linked stranger contributing zero edges. The fuzzer cannot tell the difference: it faithfully reports every edge it can see, and there is no signal distinguishing "explored this thoroughly" from "never had visibility into it".

The trap is sharpened by `vendor/` being gitignored: a fresh clone silently falls back to system libs and *works*, producing plausible numbers. Nothing fails; the ceiling is just quietly 5x lower.

## Verification
Measured both ways on the same corpus and time budget by moving `vendor/libpng` and `vendor/zlib` aside and rebuilding: 304 edges (system libs, 8 KB bitmap) vs 1,444 edges (vendored, 16 KB bitmap) in 45 s — 4.75x. Confirmed the mechanism, not just the correlation, via the `NEEDED` / defined / undefined symbol counts above.

## Generalizes to
Any target linking a system-provided library — libpng, libz, libjpeg, libxml2, openssl. Before trusting an edge count as a measure of *exploration*, confirm the code you think you are fuzzing is actually compiled into the target: `readelf -d` for `NEEDED`, and `nm` for whether the library's symbols are defined locally or undefined. A suspiciously low plateau on a large, branchy library is this until proven otherwise — check the link before tuning the fuzzer. Related: the README already notes MSAN skips targets linking uninstrumented system libs for the same underlying reason (false positives), so the codebase knew about this class of problem in one place but not the other.
