# afl_shim.c audit (2026-11-12)

Read-only audit of `src/fuzzer_tool/adapters/afl_shim.c` (7,982 lines). Nothing run.
✔ = checked against the code by the merging session; others are agent-reported, line-cited.

## Medium

| # | Lines | Defect | Fix |
|---|-------|--------|-----|
| 1 ✔ | 1241-1264 | Resize re-attach `shmdt`s the old segment right after publishing; other target threads mid-edge write unmapped memory (SEGV attributed to target) or pair old table with new `map_size`. | Snapshot `{area,size}` once per edge; defer `shmdt` to next reset. |
| 2 ✔ | 7199-7217, 6386 | Crash/exit landing skips `__afl_tick_slow()`; `__afl_timer_budget` no-ops on same budget, so the 50 µs tick persists → false TIMEOUT on the next call. | Call `__afl_tick_slow()` on the landing path. |
| 3 | 6801-6846, 6613 | Double free / realloc of a freed big (>1 MiB) private-heap block falls through `foreign_err` (above brk) to glibc `free` on unmapped/host memory. | Refuse page-aligned ptrs above brk; remember freed big blocks. |
| 4 ✔ | 2102, 2275, 2528 | Synthetic state/LTL ids (bit 31) pushed into `prev_loc` (bit 30 after `>>1`); next real edge id > 2^24, off Python's dense path, keyed on event hash. | Push a masked loc from synthetic callers. |
| 5 ✔ | 1281-1286, 3136-3296 | Block bitmap attach gated by `__AFL_DISTANCE_MODE`; `=0` builds report 0 blocks silently. | Move attach out of the gate. |
| 6 | 4060-4072, 4793, 3551 | FIFO `ENXIO` at init is permanent (writers return on `fd<0`, flush on `pos==0`); comments promise a retry. Hidden by `core/cmplog.py` opening the reader first. | Reopen when `fd<0`, or fix comments. |
| 7 | 5998-6005 | Faults the target's own handler recovers (GC barriers, JITs) write the FATAL marker and uninstall shim handlers. | Mark only when the refault kills. |

## Low

- 1483-1484: non-TLS dladdr cache stores `.ra` before `.base`; torn entry under threads (`FUZZER_KEEP_ASLR=1`). Use the 1719-1724 fence sequence.
- 6865: `realloc` same-class keep path skips LIVE/alignment check (freed/interior ptr returned).
- 6922: `__afl_ph.calls` uint32 stamp wraps (~12 h at 100k eps) → false reclaim.
- 7238: atfork child does not clear `__afl_ph.lock` → child spins forever.
- 4096-4104/5459: signal inside own cmplog flush rewrites already-written chunks.
- 2552-2564: LTL fd lacks `O_CLOEXEC`, stolen-fd check, racy `-2` init.
- 1283-1284, 3201, 3233: dist/node segments re-`shmat`ed per resize, never detached.
- 4675-4681: dlsym recursion miss caches naive fallback permanently.
- 4973-4987: `n*sizeof(wchar_t)` wrap (record lost, no overread).
- 1007/1038: 32-byte refused-id buffer; long ids re-counted every retry.
- 1137-1141: silent overflow refusal (32-bit only). 996: `strtol` accepts space/`+`.
- 3270: `*__afl_dist_count` reloaded per probe; cache at attach.

## Stale comments

9-11, 801/810 (PKRU: pre-existing threads are access-disabled, not unlocked), 640-659/1104-1106/1282 (shim is pragma-exempt from instrumentation), 1104 (0 keeps previous size), 1291, 1359 (hardcoded 128), 1434-1438, 2520-2522, 6544, 7362-7393 (abort macro does not reach library `.o` or `__assert_fail`).

## Checked fine

SHM layout vs `shm.py`/`cmp_shm.py`; size/overflow math; probe bounds (scalar, SIMD, MSI); generation/saturation; frame walk; record/buffer bounds; cmplog lock; calloc overflow, alignment family; exit/abort visibility; forkserver I/O; constructor order; gate combos.

## Status (2026-11-12)

Fixed with regression tests: `tests/test_regression_shim_audit_{attach,cmplog,guard}.py`.
Medium #7 kept by design (comment documents the GC/JIT false positive); low "getenv in handler" kept (env is set after dlopen) — both documented in code. Residuals: `docs/TODO.md` "afl_shim audit follow-ups".
