# rebuild-targets-surfaced-regressions: the build gate that found a shim SEGV

**Date:** 2026-08-11
**Context:** `fuzzer-new`, `tools/build_targets.sh` + `src/fuzzer_tool/adapters/afl_shim.c`, commits `392b37b` / `451131c`

## Problem

A routine `tools/build_targets.sh` rebuild ("rebuild targets") silently produced 4 broken targets — `png_read_nosan`, `zlib_read_nosan`, `gzip_read_nosan`, `ffmpeg_read_nosan` — and, once rebuilt, the full test suite crashed: `test_shm_distance_channel` SEGV'd inside the coverage shim during SHM-map setup. Two distinct defects, both hidden by `2>/dev/null` and by a pre-commit hook stack that runs **no pytest** (AGENTS.md's claim that hooks run the full suite is false — verified in this session).

## Rejected

- **"The nosan link failure is a flake"** — dropped when the same four lib-linked executables failed on every run while no-lib executables (asan_target, grep_read) succeeded; the failure set was consistent, so it was structural.
- **"The SEGV is a shim-info problem introduced by my build change"** — dropped after reproducing it with a *test-self-compiled* binary that my build never touches: the distance tests compile their own target at `-O1`. The blame belonged to the shim, not the build.
- **"Guard the frame walk with a NULL check on the frame pointer"** — looked sufficient; dropped because the crash is `__builtin_return_address(1)` dereferencing the *caller's* return-address slot, which is garbage when the caller omitted its frame — the guard reads a non-NULL register and the load still faults.
- **"Cover just the startup window with a mapping flag"** — implemented and still crashed: the walk fires on *every* edge, and at `-O1/-O2` (clang x86-64 defaults to frame-pointer omission) any caller without a frame produces garbage, not just the constructor path. Only reproducible on a machine where a rebuild mixes old and new binaries — hence the hours of "which binary is gdb looking at" confusion.

## Approach

1. **Believe the build matrix, not the summary line.** The rebuild's `tail` said "44 new targets OK"; the full log had four `failed:` lines. Subsequent grepping of the whole log (not the tail) is what surfaced them — and rerunning the half (loader subsets) confirmed they were gone before deducing the leftover warnings were pre-existing skips (vendor trees absent is by design).
2. **Reproduce the field symptom in one command with the same artifact.** The distance tests spawn a target; the failing path was reproduced standalone by compiling the exact test C file with real SysV SHM env — making the SEGV deterministic instead of test-ordering-dependent.
3. **Root cause #1 (build):** `build_target` (executables) never linked the cmplog shim, but vendored `libpng.a`/`libz.a` are compiled with comparison tracing → undefined `__sanitizer_cov_trace_const_cmp*` at executable link time (a `.so` link tolerates undefined symbols, which is why the `.so` variants built fine). Fix mirrors `build_so_target`: compile+link the shim object, excluded for MSAN/TSAN (per their "every linked unit must be instrumented" rule). `ffmpeg_read*` additionally pointed at system headers that aren't installed → routed to the vendored coverage tree.
4. **Root cause #2 (shim):** concurrent commit `0df346a` enabled call-stack-sensitive edge hashing (`caller_ctx = __builtin_return_address(1)`), whose docstring claimed coverage builds "already default to -fno-omit-frame-pointer". False: clang/gcc omit them at `-O1/-O2`. The walk then dereferences the caller's return-address slot, which doesn't exist → SEGV on default builds. Fix: default `__AFL_CTX_SENSITIVE` to 0 (opt-in with `-D__AFL_CTX_SENSITIVE=1 -fno-omit-frame-pointer`), guard the walk during the constructor window and on NULL frame pointer, and make the feature's own test opt in explicitly — so the feature is still verified under the flags it requires.

## Key insight

The "rebuild targets" request was a cheap, high-signal gate: it recompiled everything with the current sources, and the differences from the previous binary state exposed a **concurrent regression that would have crashed the fuzzer's own targets at runtime** (the ctx-hash feature, authored by another session, segfaulted the standard `-O1/-O2` build configurations). A rebuild is a differential test: "same sources, fresh binaries — is the world still coherent?" The hidden gems were the asymmetry (executables fail while `.so` pass — the undefined-symbol tolerance in shared links) and the wrong docstring contract (a feature whose safety cage was a false claim about compiler defaults).

## Verification

- After the build fix: all four nosan executables + ASAN executables link; rebuild log has zero `failed:` except the not-vendored secp256k1 skip; feature matrix fully ON.
- After the shim fix: standalone distance repro rc 139→0; `test_shm_distance_channel` 7/7; `test_shm` 52/52 (with the ctx test opted in); full suite **4162 passed, 0 failed**.
- The earlier 15-minute suite "hang" was a self-inflicted artifact (concurrent-stray pytest processes); the clean re-run is 107s.

## Generalizes to

- **A rebuild is a regression gate.** "Rebuild targets" after source changes is cheap differential testing — compare the full log (skips vs. failures), and treat a fresh-build suite run as authoritative, not the last incremental run's memory.
- **Compiler flag defaults are facts to verify, not folklore to repeat.** "Coverage builds keep frame pointers" was false; every rule derived from it (NULL-returning return_address) was also false. When a docstring's safety contract is wrong, the feature is broken by default — make it opt-in, and encode the requirement in its tests, not in prose.
- **Undefined symbols: an executable link fails, a `.so` link might not.** Any link-time failure that appears only for executables but not for the shared variants is usually this.
- **Errors hidden by `2>/dev/null` and summary tails are the same bug twice.** The build script's silent-flag and the `tail -40` both truncated the truth; grep the full log for the failure lines the summary doesn't print.
