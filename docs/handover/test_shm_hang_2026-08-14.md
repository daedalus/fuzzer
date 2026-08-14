# Handover — `tests/test_shm.py::TestShimEdgeCountEndToEnd::test_shim_updates_edge_count_after_target_call` hang investigation

## Symptom
The test `test_shim_updates_edge_count_after_target_call` is reported to hang.
It exercises a tiny GCC-compiled `.so` built with `-include afl_shim.c`, loaded via `ctypes.CDLL`, whose exported `fuzz_shm_run()` calls `__afl_map_edge()` three times and then returns.

## Local reproduction status
**I could not reproduce the hang on this machine.** The test passes consistently:
- Single invocation: ~0.4s, PASSED
- 5 consecutive invocations: all PASSED in ~0.4s each
- Full `tests/test_shm.py` (55 tests): all PASSED in 0.68s
- `pytest -n auto` (xdist parallelism, 4 workers): 55 passed in 2.55s
- Manual Python reproduction (compile → `ShmCoverage` → `ctypes.CDLL` → call → read edge_count): completes in ~0.12s with `edge_count == 3`

This means either:
- the hang is environmental/CI-specific,
- it requires a specific prior process state,
- or it has already been fixed by a recent commit.

## Recent history of the touched code
```
d588254 feat(shim): add a real AFL-style forkserver to afl_shim.c
16898d9 feat(fuzzer): add SkipDet deterministic stage and fix SHM fast-path edge set
1759165 fix(shim): merge cmplog into afl_shim.c
...
```
The most likely suspect is `d588254`, which introduced `__afl_start_forkserver()` inside the shim constructor.

## What I investigated

### 1. Test mechanics
The test:
1. Compiles `test_edge_count.c` → `test_edge_count.so` using **gcc** (not clang) with `-O2 -g -shared -fPIC -include afl_shim.c`
2. Creates a `ShmCoverage()` instance (sysV SHM, ~65560 bytes)
3. Sets `__AFL_SHM_ID` and `AFL_MAP_SIZE` env vars
4. Loads the `.so` via `ctypes.CDLL`
5. Calls `fuzz_shm_run(buf, len)`
6. Asserts `read_edge_count() > 0` and `is_new_coverage_with_edges()` reports 3 edges
7. Cleans up SHM

### 2. Time distribution
All operations are fast in isolation:
- gcc compile: ~0.12s
- `ctypes.CDLL(str(so))`: ~0ms
- `func(...)`: ~0ms
- `read_edge_count()`: ~0ms

So the hang is not in the test logic itself; if it hangs, it is inside `ctypes.CDLL()` or inside the called function.

### 3. AFL shim constructor / forkserver hypothesis
`afl_shim.c` now has:
```c
__attribute__((constructor))
static void __afl_auto_init(void) {
    __afl_mapping = 1;
    __afl_map_shm();
    __afl_install_crash_handlers();
    __afl_mapping = 0;
    __afl_start_forkserver();   // <-- new in d588254
}
```

`__afl_start_forkserver()`:
- Writes a 4-byte hello to `AFL_FORKSRV_FD+1` (fd 199).
- If that write succeeds, enters a `while(1)` loop:
  - `read(AFL_FORKSRV_FD, cmd, 4)` — blocks waiting for loader commands
  - `fork()`, `waitpid()`, write child pid/status back.

**This constructor runs during `ctypes.CDLL()` load.** If fd 198/199 happen to be a valid pipe in the loading process, the hello write succeeds and the constructor enters the forkserver loop — **the loader never returns from `ctypes.CDLL()`**.

In normal pytest runs fds 0–3 are stdin/stdout/stderr + one more, so 198/199 are closed and `write()` fails → the constructor returns → no hang. The hang would require 198 and 199 to be open pipe ends inherited from a parent that created them.

### 4. What I tried
- Running the test in isolation: passes
- Running the full test_shm.py: passes
- Running with `pytest -n auto`: passes
- Manual reproduction via standalone Python script: passes
- Artificially occupying fd 199 with a regular file: **reproduced a hang at `ctypes.CDLL()`**
- Artificially occupying fds 198/199 with a pipe pair: reproduced repeated `ctypes.CDLL()` → fork-spawn cycles that loop until timeout

This confirms the root-cause mechanism: **if fd 199 is writable at load time, the shim constructor enters the forkserver loop and `ctypes.CDLL()` never returns**.

### 5. Why it may reproduce in CI/other environments
- A parent process (test runner wrapper, coverage tool, debugger, profiling harness) may have fds 198/199 open as a pipe/socketpair.
- Those fds are inherited by the Python test process across `fork()`/`exec()` or even across `pytest` workers in some setups.
- Normal interactive runs don't have this, which matches the local non-reproduction.

## Root cause hypothesis
The new forkserver constructor in `afl_shim.c` (`__afl_start_forkserver`) is **unsafe to run in `ctypes.CDLL`-loaded `.so` test contexts** because:
1. It uses fixed fd numbers (198/199) with no reservation.
2. A successful `write(199, ...)` is treated as “we are being driven by the loader” even when fd 199 is just a coincidentally open file/pipe.
3. Once inside the `while(1)` loop, the process blocks on `read(198, ...)` forever — the Python test hangs at `ctypes.CDLL()` load.

## Potential fixes (not implemented — awaiting decision)

### Option A — Test-only guard (minimal)
In the test's C source, before any shim call, call `__afl_map_reset()` or add an explicit env-var guard that skips the forkserver when `__AFL_SHM_ID` is set (i.e. in-process mode). The constructor currently checks `write(AFL_FORKSRV_FD+1,...)` but doesn't check whether the surrounding process is actually the forkserver loader.

### Option B — Shim-side guard
In `__afl_start_forkserver()`, require an explicit opt-in env var (e.g. `__AFL_FORKSRV=1`) before entering the forkserver loop. Normal in-process `.so` loads via `ctypes.CDLL` would skip it. The real forkserver loader sets the var.

### Option C — fd reservation / safer probe
Before entering the forkserver loop, verify the fd pair is actually a pipe/socket (e.g. `fcntl(fd, F_GETFL)` or `getsockopt`/`pipe2` flags). A regular file or unrelated fd would be rejected.

### Option D — disable forkserver in test builds
Do not compile `__afl_start_forkserver()` into the test `.so` in `test_shm.py`. Add a `-D__AFL_NO_FORKSRV` define for these test compilations.

## Suggested next step
Reproduce the hang deterministically by adding a small Python wrapper that opens a pipe at fds 198/199 and then loads the test `.so — that converts this from “intermittent/environmental” to “always reproducible” and makes the fix easy to validate.

## Files of interest
- `tests/test_shm.py` — `TestShimEdgeCountEndToEnd::test_shim_updates_edge_count_after_target_call`
- `src/fuzzer_tool/adapters/afl_shim.c` — `__afl_auto_init`, `__afl_start_forkserver`, `__afl_map_shm`
- `src/fuzzer_tool/adapters/shm.py` — `ShmCoverage` (Python-side SHM reader)

## Resolution (follow-up, same day)

Option B was taken in `a267ff8`, but only half of it landed: the shim gate went
in and no loader ever set the variable. The sentence above — "The real
forkserver loader sets the var" — described intent, not code. `__AFL_FORKSRV`
appeared in exactly two places in the tree, the gate itself and a docstring.

Consequences, in order of how long they took to notice:

1. `start_forkserver()` in `fuzz_loader.c` could never complete its handshake,
   so `use_forksrv` was always 0 and every RUN fell back to `run_executable()`.
   The forkserver work of `d588254` → `fd59970` → `1cdd9ee` was effectively
   reverted, at the cost of one wasted target execution per INIT to discover
   the fallback. Nothing failed; it just got slower.
2. `tests/test_regression_forkserver_shm.py` did not catch this, because all
   eight of its assertions — SHM coverage, stderr forwarding, timeouts,
   protocol desync — hold identically under both execution paths. The two
   modes were externally indistinguishable over the protocol.

The fix sets the variable in the forkserver child of `start_forkserver()`,
between `fork()` and `execl()`. It deliberately does *not* go into
`ForkserverRunner.env_overrides`: that would put it in the loader's own
environment, where the `dlopen()` path for `.so` targets would enter the
forkserver loop and hang the loader — the identical bug this document is
about, one level up.

To make the failure observable at all, the loader now reports the resolved
mode on its READY line (`READY forkserver` / `READY exec` / `READY dlopen`)
and `ForkserverRunner.exec_mode` carries it. A regression test asserts the
mode is `forkserver` for a shim-built executable, which is the assertion that
would have caught this.

### Separately: target stdout desynced the protocol

Found while verifying the above, and independent of it. Children were spawned
with stdin and stderr redirected but *not* stdout, which is the loader's half
of the RUN/RC protocol. A target that printed anything had its output parsed
as an `RC` header; the adapter returned -2 for a run that succeeded, and the
stream stayed desynced afterwards. Reproduced on both the forkserver and the
fork+exec paths, and on the handshake probe child as well. Children now get
stdout on `/dev/null` (not the stderr pipe — `ExecutionRunner.is_crash` scans
stderr for sanitizer reports, and ordinary chatter there would read as a
crash).
