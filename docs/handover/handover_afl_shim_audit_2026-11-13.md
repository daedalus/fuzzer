# afl_shim.c audit (2026-11-13)

Read-only audit of `src/fuzzer_tool/adapters/afl_shim.c` (8,289 lines, at `d9d67d14`). No fix shipped.
Items in `handover_afl_shim_audit_2026-11-12.md` and `docs/TODO.md` excluded.
R = reproduced (clang 18, no sanitizer runtime on the host); ✔ = repro rerun or code path re-read by the merging session.

## High

| # | Lines | Defect | Fix |
|---|-------|--------|-----|
| 1 R✔ | 6030-6058, 6219 | Crash handler runs `backtrace()` before `siglongjmp`; a smashed saved `rbp` faults the unwinder inside the handler, signal blocked: the kernel kills the process, in direct/direct_lite the fuzzer. `CrashSymbolSink` is always on (`services/fuzzer.py:2307`). ASAN builds mostly escape. | Frame-pointer walk bounded to the thread stack, or nested `sigsetjmp` around the capture. |

## Medium

| # | Lines | Defect | Fix |
|---|-------|--------|-----|
| 2 R✔ | 6319-6345 | `sigaction`/`signal` interposers guard only the crash signals; a target `signal(s, SIG_DFL)` loop resets `__AFL_TIMEOUT_SIG`/`__AFL_EXIT_SIG`. Next tick kills the host (`Real-time signal 5`); `SIG_IGN` disables timeouts for the process. | Swallow installs on both signals while their handlers are live. |
| 3 R✔ | 7365-7373 | `__afl_exit_handler` jumps the owner from any pc, without the tick's deferrals. Worker `exit()` while the owner is in glibc malloc: arena lock held, host deadlocks (4/10 and 15/20 runs; timeout-path control 0/20). Also breaks `__afl_ph_big_compact` mid-move. | Same deferral as the tick; keep `__afl_exit_pending` set. |
| 4 R | 7928-7938 | `__afl_persist_sync` skips `__afl_sancov_fold()`: `__AFL_LOOP` on inline-8bit-counter/bool-flag builds reports 0 edges per parked iteration (`0,0,0,7` vs `7` fork-per-exec). | Fold in `__afl_persist_sync`. |
| 5 R✔ | 7564-7576 | `_exit()` (or a crash) in a `vfork()` child of a guarded entry jumps to the parent's guard frame; child returns into the host, SEGV. | Record `gettid()` per guard frame; jump only on a match. |
| 6 R✔ | 959-975, 847-857 | `__afl_wguard_arm` zeroes the shared `__afl_wguard_depth` on re-attach; a thread mid-bracket wraps it to `0xFFFFFFFF`, the next open skips `mprotect(RW)`, store faults. `__AFL_WGUARD=mprotect` + direct-mode resize. | Keep the count across re-arm; leave the new mapping writable while depth > 0. |

## Low

- R✔ 4742-4748: `__afl_resolve_pending` reads `__afl_pending[i]` after the lookup; a nested miss reuses slot `i`, A's pointer lands in B's `real_*` (613cbb2 fix). Synthetic `dlsym` wrapper only; stock glibc did not re-enter. Copy `slot`/`name` to locals first.
- R✔ 8246-8263: preload-only crash handler keys "hardware fault" on the signal number; `raise(SIGSEGV)` is swallowed, exit 0. Use `si_code > 0` as `__afl_is_hw_fault` does.
- R✔ 1793-1800: main-stack ctx cache pairs `ra` with a signal handler's ctx when the handler lands between the `last_ctx` and `last_ra` stores (1,077 wrong of 24.4M; control 0). Re-check `last_ctx` after the `last_ra` store.
- R 1540-1559: dladdr cache "seqlock" (f85d139) has no writer exclusion; two writers on one slot leave tag A with base B (35 wrong of 2.4e10, `FUZZER_KEEP_ASLR=1`). CAS-claim the slot.
- R 1187-1188, 1266-1268: `reuse` path (f85d139) `memcpy`s the bitmap magic into the write-locked mapping. Latent: no caller re-attaches the same id. Skip the write on `reuse`.
- R 7861-7876, 7917-7922: forkserver never checks or reaps the parked persistent child: a killed child yields rc -9 for the next input unrun; with SIGHUP ignored one stopped target leaks per loader restart. Reproduced before `1073762a` (`--persistent` changes): recheck. `waitpid(WNOHANG)` before resume, kill on server exit, `PR_SET_PDEATHSIG`.
- 7928-7937, 4535: `__afl_persist_sync` never calls `__afl_cmplog_new_scope`; inputs 2..N drop records an earlier input emitted. Conflicts with TODO "Cmplog pool frozen" (b): decide the contract first.
- 3649-3664: `__ltl_open` waiters spin on `LTL_FD_OPENING` unbounded, no atfork reset: a child forked mid-open hangs on its first LTL event.
- 6232-6236: `alarm(2)`/`alarm(0)` in the no-guard crash path cancels the process's pending alarm; the path also runs before chaining to a recovering target handler. Restore the previous value.
- 7054-7060, 7096-7097: a jump between `mmap` and the table insert (`__afl_ph_big_new`) or between unlock and `munmap` (`__afl_ph_big_del`) leaks a >1 MiB mapping. Do both under the heap lock.
- 8044-8083: `__afl_fuzz_read_alt` reads into the range the previous call ASAN-poisoned (fallback delivery + persistent + longer input): false use-after-poison. Unpoison the whole buffer before the read. Unrun (no ASAN runtime).
- 3411-3413: node bitmap plain `|=`; block bitmap (2473) is atomic. Lost node visits under threads.
- 1052-1067 vs 2540-2561: refused re-attach NULLs `__afl_gen_word`/`__afl_path_hash` under a thread already past the `__afl_fast` test: NULL deref blamed on the input. Keep header pointers on the retired segment.
- 2847-2860, 2904-2912: sancov dump fd cached for the process, no stolen-fd check. Route through the log-fd helpers.

## Checked fine (agent-reported)

SHM header/entry/tail/bitmap layout vs `shm.py`; attach size math; refusal memo, grave ring; publish order; PKRU math. Frame walk; probe bounds (scalar, SSE2, AVX2, MSI); generation/reclaim; synthetic id mask; guard numbering; sancov fold and record layout; aux segments vs Python. Cmplog wire formats vs `cmp_shm.py`/`track_parser.py`/`core/cmplog.py`; record and line buffers; flush progress; lock word; dedup table; operand read clamps. Signal routing, altstack, tick epoch and deferrals, `__afl_tick_slow` on all guard exits; private heap tables, realloc keep path, fork reset. Forkserver wire protocol vs `fuzz_loader.c`; mode env; testcase SHM layout; ctypes widths. 14 gate combos compile and run.

Not exercised: `-fsanitize-coverage` and ASAN builds (no compiler-rt), pkey guard (no PKU).

## Repros

Not in the repo (scratch, session-local): `a.c`/`b.c` (wguard), `pend.c` (pending), `bt.c` (backtrace), `rt.c` (RT signals), `t.c`/`host.c` (worker exit), `v.c`/`vhost.c` (vfork), `p.c` (preload raise), `sig.c`, `seq.c` (ctx caches). Each fix needs its regression test written first (Hard Rule 37).
