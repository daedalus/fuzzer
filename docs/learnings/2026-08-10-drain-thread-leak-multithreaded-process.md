# drain-thread-leak: the "z3 hang" traced to an unstopped loader thread keeping the process multi-threaded

**Date:** 2026-08-10
**Context:** `fuzzer-new`, `src/fuzzer_tool/adapters/{persistent_loader,forkserver,inprocess}.py`, commit `673162b`

## Problem

`tests/test_structural_constraints.py::TestNonOverlapSoundness::test_true_arithmetic_not_modular` — a z3 solve of a trivially-satisfiable 3-section constraint that takes ~11 ms in isolation — intermittently **hung or segfaulted** in full-suite runs, at ~91%, in a process where the same test passed every time on its own. Root cause unknown; both symptoms pointed at native-state corruption, but everything relevant was order-dependent and non-reproducible on demand.

## Rejected

- **Reproducing by rerunning** — looked plausible because the failure was intermittent and any repro would need the rare condition; dropped because it produced nothing: isolation always passed, and unarmed full-suite runs were green more often than not. A one-shot capture can't catch a 1-in-2 event.
- **"z3 is somehow stateful between tests" (np.random / global z3 params)** — looked plausible because `ConcolicTrace.solve()` calls the global `z3.set_param("timeout", ...)`; dropped after reading `solve_coupled_sections`: it builds a fresh `Solver` and sets the timeout per-solver, and even the worst global-param leak would cause `unknown`→`None` assertion failures, not a hang. No mechanism connects it to a destructor spin.
- **ASAN loaded into the pytest process** — looked plausible as a heap-corruption source; dropped: the only test that would load an ASAN `.so` in-process is `@pytest.mark.skip`, and `_probe_so_function` was already migrated off `ctypes.CDLL` for exactly this reason.
- **Closing the pipe wrapper (`stream.close()`) to unblock the reader** — looked like the obvious teardown; dropped after it *caused* a worse crash: the drain thread blocked inside `readline()` holds the `BufferedReader`'s own lock, so `close()` from another thread (or from GC at interpreter shutdown) spins forever on that lock — a fatal `_enter_buffered_busy` that aborted the fuzzer child (rc=-6). The lock lives on the *wrapper*, so the fd must be closed underneath it.

## Approach

1. **Get the evidence with a watcher, not a rerun.** A script looped full-suite runs, detected non-progress (log size frozen 120 s while pytest alive), and captured both a gdb native backtrace and a faulthandler dump of the hung process — turning the next occurrence into data. The gdb bt pinned the stuck frame inside `libz3.so`: `and_then_tactical::operator()` → `sat_tactic::operator()` → `goal2sat::imp::~imp()` → `ref_vector_core<app,…>::~ref_vector_core()`, stack bottom `0x0` — a destructor spinning on a tiny AST, i.e. corrupted node state in a poisoned process, not a hard solve.
2. **Probe the process state, not the test.** A pytest hook logging `threading.enumerate()` every test found the environment defect: `_drain_stderr` daemon threads (persistent_loader.py:344, forkserver.py:106) born in `test_inprocess_crash.py` and alive for the rest of the suite. The threads exit only at pipe EOF, and a live loader child (or a fork grandchild holding the pipe write end) means EOF never comes; tests that never `stop()` their runners leak both child and thread, so every later `os.fork()` ran in a multi-threaded process (Python 3.13's DeprecationWarning) — the classic deadlock/heap-corruption hazard, and the same family as the prior `malloc_consolidate` SIGSEGV.
3. **Fix the leak** (not the z3 test): `_close_streams()` in both adapters closes the raw pipe fd first (`os.close(fileno())`) — which unblocks the blocked reader without touching the wrapper's lock — then closes the wrapper only outside interpreter finalization; `stop()` and the loader-timeout restart path call it; `InProcessRunner` gained `__del__ → stop()`; the leaking tests now stop their runners. Regression test asserts no `_drain_stderr` thread survives `stop()` or `del`.

## Key insight

The "z3 hang" was never about z3. A destructor spinning for minutes on an 11 ms problem is the signature of a *poisoned process*, and the poison had a concrete, normal-lifecycle source: loader subprocesses whose stderr-drain daemon threads never terminated, keeping the whole suite permanently multi-threaded. Two layers of indirection hid it: the crash manifests at an *arbitrary later allocation site* (any native-heavy test — this run z3, an earlier session `test_fuzzer.py`/`malloc_consolidate`, another run the ptrace fault-addr tests), and the leak itself only surfaces via a thread census, never via a failure at the leak's own site.

## Verification

- Watcher caught the hang in-session: gdb bt of the stuck pid inside z3 tactic teardown (above), hang at exactly the z3 test position.
- Thread probe: `Thread-12/13 (_drain_stderr)` alive across all later tests → after the fix, **0 live threads** across the loader + z3 slice.
- fuzzer-child aborts from the bad `stream.close()` version caught by the test suite during development (rc=-6, `_enter_buffered_busy`) — the fd-first design then passed with unraisable-as-error.
- Full suite: 4116 passed, 0 failed; the two previously order-dependent `test_regression_persistent_loader_fault_addr` tests now pass under full order. (Caveat: the hang was intermittent ~1-in-2, so closure is strongly indicated but not statistically sealed by one green suite.)

## Generalizes to

- **Intermittent native hang/segfault = instrument the environment, don't rerun the test.** A watcher that captures gdb/faulthandler on the first non-progress makes the next occurrence evidence. A probe that snapshots process state (threads, fds, memory) at every test is the fastest way to find order-dependent pollution.
- **"Exit only at EOF" daemon threads are leaks by construction when anyone can forget teardown.** Any background reader on a pipe must also be terminable by closing the fd — and never by closing the buffered wrapper, because a thread blocked inside `readline()` holds the wrapper's own lock (per-object recursive lock), so `close()` from elsewhere deadlocks or fatals at shutdown. Close the raw fd first; that unblocks the reader without acquiring the lock.
- **A crash's backtrace tells you where, not why.** The z3 destructor frame was accurate and useless; the cause lived two files and forty tests earlier, in thread bookkeeping. Diagnose the environment the backtrace executes in, not the frame itself.
- **`__del__ → stop()` on resource-owning wrappers** is cheap insurance against the "forgot to close" class, but the tests that actually leak must stop explicitly — GC timing is not a teardown contract.
