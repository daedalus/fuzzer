# ctx-collapse-selfcheck: call-context that fails silently when frame pointers are omitted

**Date:** 2026-10-27
**Context:** `src/fuzzer_tool/adapters/afl_shim.c` (`__afl_caller_ctx_as`), `tools/build_targets.sh`, `src/fuzzer_tool/core/shim_health.py`; compared gcc 13.3, clang 18.1 and zig cc 0.17 on x86-64

## Problem

CTX hashes the return address one frame above the instrumented function by walking saved frame pointers. The 2026-08-11 learning established that omitted frame pointers can crash the walk and that `build_targets.sh` must pass `-fno-omit-frame-pointer`. What it did not pin down is the quiet case: a walk that succeeds and returns the wrong thing. Nothing reports it, and edge ids stay valid-looking.

## Measurements

A call chain (`leaf` reached from `a_call`, `b_call`, and tail-call wrappers `a_tail`, `b_tail`), instrumented code built per flag set, per-edge context recovered by XORing the edge ids of a ctx-on and a ctx-off build, ASLR off. 8-bit context.

- **Frame pointers kept**: `a_call` and `b_call` get distinct contexts on all three compilers.
- **Frame pointers omitted**: every edge gets the same context (0x93 here) on all three compilers. The walk reads the frame of whatever last linked rbp (here `main`), passes the bounds checks, and returns a constant.
- **Defaults at -O2 and -O3**: gcc and clang omit frame pointers, so the above is what an unflagged build gets. zig keeps them by default. At -O0 all three keep them.
- **Tail calls**: `return f(x)` becomes a `jmp` at -O2/-O3 on all three, so `a_tail` and `b_tail` hash alike. At -O0 they differ.
- **Junk rbp**: when the omitted-frame-pointer code uses rbp as a general register, the walk fails a bounds check and every edge gets context 0 (covered by a test with inline asm, x86-64 only).
- **Edge counts are not comparable across builds**: the same 150-line gate target gave 24 to 95 raw edges depending on compiler and -O level, with identical semantics. clang -O3 unrolls the 16-byte sum loop (95 edges). zig turns UBSan on at -O0 (134 coverage call sites against 31 with `-fno-sanitize=undefined`), and its -O3 object was byte-identical to -O2.

## Rejected

- **A synthetic two-level chain run at shim init.** It checks the shim's own build, not the target's. The target is what is built without the flag.
- **A hot-path counter of distinct contexts.** Even one inlined increment on the miss and bail paths cost +3% on `shim-micro-benchmark full` under clang (medians 5.54 vs 5.38 ns). Moved into `noinline, cold` helpers, as `__afl_slot_claim` already is: 5.16 vs 5.18 ns (clang), 5.51 vs 5.57 ns (gcc), indistinguishable from the old shim on this machine.

## Approach

`CTX_RA_CHANGES` counts main-stack walks that missed the return-address cache, `CTX_BAILS` counts walks that failed a bounds check. The detector fires when at most one distinct return address was seen, the walk ran at least once, and at least 128 distinct edge slots were claimed. It reads as a health field (`CTX_COLLAPSED`, direct mode) and as one stderr line per process from an atexit hook (tag `__afl_shim:`, so `ShimWatch` dedups it).

Companions: FFmpeg's configure now gets `$FRAME_POINTER` (it was the one vendored library without it) and a post-configure check that the last frame-pointer flag in `config.mak` is `-fno-omit-frame-pointer`; `--no-sibling-calls` (or `NO_SIBLING_CALLS=1`) adds `-fno-optimize-sibling-calls`; `cov_flag_for_cc` knows `zig cc` and adds `-fno-sanitize=undefined`; the unused-function and gcc visibility warnings are gone.

## Limits

- Main-thread frames only. A target that runs entirely on a worker thread with healthy frames is never judged; one with broken frames is (bails).
- A harness whose whole coverage sits in one function behind one call site trips it legitimately, so the message says "looks collapsed".
- Forkserver children that leave through `_exit()` skip the atexit line; direct mode reads the counter.
- Not run against a real FFmpeg build (no vendored tree in the sandbox). Whether FFmpeg's own `-fomit-frame-pointer` lands after ours is exactly what the post-configure check reports; it has not been observed.
- zig support is the flag quirk only. No end-to-end zig build of the targets was run.

## Verification

`tests/test_ctx_collapse_selfcheck.py` (31 tests, clang guard and gcc trace-pc): detector silent with frame pointers, with CTX off, and under 128 edges; fires on constant and junk-rbp shapes; one stderr line; C and Python field indices agree. 184 related tests pass (`test_shim_health`, `test_shim_ctx_*`, `test_regression_ctx_*`, `test_gcc_trace_pc`, `test_regression_build_flags`).

## Generalizes to

- A check that can only fail loudly is not a check. Walks that read stale-but-valid memory need a statistical signature, because there is no error to catch.
- Compiler defaults differ per compiler and per -O level; measure them with the target's own flags, not the script's.
- Raw edge counts compare only within one compiler and -O level.
