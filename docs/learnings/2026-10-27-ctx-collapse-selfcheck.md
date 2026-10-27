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
- **Per-edge counters** (the first version): the first A/B used `shim-micro-benchmark`, where almost every edge hits the return-address cache, and showed nothing. It never ran the paths the counters sit on. A 150-call chain built three ways (frame pointers kept: every edge a cache miss; omitted with a constant context: all hits; omitted with junk rbp: every edge bails) showed it: out-of-line counters cost +8 to +19%, inline counters +2 to +11%. A counter is a read-modify-write on one address, a loop-carried dependency through memory, on exactly the targets it exists to catch.

## Approach

Three sticky byte flags set by plain stores, on the miss and bail paths only: a walk happened, a second distinct caller was seen (a miss while `last_ra` is already set), a walk failed a bounds check. The hit path is what it was. The health fields `CTX_RA_CHANGES` (0, 1, or 2 meaning two or more) and `CTX_BAILS` (0 or 1) are derived from them at read time. The detector fires when no second caller was seen, a walk ran or bailed, and at least 128 distinct edge slots were claimed. It reads as `CTX_COLLAPSED` (direct mode) and as one stderr line per process from an atexit hook (tag `__afl_shim:`, so `ShimWatch` dedups it). `-D__AFL_CTX_CHECK=0` compiles all of it out.

Companions: FFmpeg's configure now gets `$FRAME_POINTER` (it was the one vendored library without it) and a post-configure check that the last frame-pointer flag in `config.mak` is `-fno-omit-frame-pointer`; `--no-sibling-calls` (or `NO_SIBLING_CALLS=1`) adds `-fno-optimize-sibling-calls`; `cov_flag_for_cc` knows `zig cc` and adds `-fno-sanitize=undefined`; the unused-function and gcc visibility warnings are gone.

## Speed

Median-of-trials is useless on this shared single core (the same binary moved 5 to 10% between runs); the table is the minimum of 25 interleaved trials, ns per `run()` of the 150-call chain (~300 edge callbacks), against the shim before the detector. Worst cases, not typical targets: a target that mostly hits the cache sees none of it.

| target build | inline counters | sticky flags | `-D__AFL_CTX_CHECK=0` |
|---|---|---|---|
| clang, frame pointers kept (all misses) | +7.9% | +3.6% | -1.7% |
| clang, constant context (all hits) | +10.5% | +2.4% | -4.9% |
| clang, junk rbp (all bails) | +12.4% | +4.3% | -0.7% |
| gcc, frame pointers kept | +3.4% | +4.1% | -1.7% |
| gcc, constant context | +3.7% | +1.7% | -0.2% |
| gcc, junk rbp | +5.4% | +2.4% | +0.8% |

Run-to-run noise is about 3 to 5%, so the gcc rows do not separate the first two columns. Removing the bookkeeping entirely (counters as no-ops) brought clang back to the old shim, so the cost is the bookkeeping, not code layout; on gcc the junk-rbp row stayed +6.5% with no bookkeeping, so part of that cost is something else that was not isolated. What is left on the default build is a few percent in the worst cases; use `-D__AFL_CTX_CHECK=0` for raw-speed runs.

## Limits

- Main-thread frames only. A target that runs entirely on a worker thread with healthy frames is never judged; one with broken frames is (bails).
- A harness whose whole coverage sits in one function behind one call site trips it legitimately, so the message says "looks collapsed".
- Forkserver children that leave through `_exit()` skip the atexit line; direct mode reads the counter.
- Not run against a real FFmpeg build (no vendored tree in the sandbox). Whether FFmpeg's own `-fomit-frame-pointer` lands after ours is exactly what the post-configure check reports; it has not been observed.
- zig support is the flag quirk only. No end-to-end zig build of the targets was run.

## Verification

`tests/test_ctx_collapse_selfcheck.py` (clang guard and gcc trace-pc): detector silent with frame pointers, with CTX off, with `__AFL_CTX_CHECK=0`, and under 128 edges; fires on constant and junk-rbp shapes; one stderr line; C and Python field indices agree. 304 related tests pass (`test_shim_health`, `test_shim_ctx_*`, `test_regression_ctx_*`, `test_regression_shim_*`, `test_sancov_modes`, `test_gcc_trace_pc`, `test_regression_build_flags`).

## Generalizes to

- A check that can only fail loudly is not a check. Walks that read stale-but-valid memory need a statistical signature, because there is no error to catch.
- Compiler defaults differ per compiler and per -O level; measure them with the target's own flags, not the script's.
- Raw edge counts compare only within one compiler and -O level.
