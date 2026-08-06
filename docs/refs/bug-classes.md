# Recurring Bug Classes

These rules are extracted from ~85 `fix:` commits in project history. Each names the
recurring bug *class*, not just the single instance that surfaced it — recognize the
pattern before it reappears in a new file.

Open this file when touching: process/signal handling, timeouts, ptrace, concurrency,
resource cleanup, hashing/identity, caching, ELF/low-level parsing, numeric edge cases,
state persistence, dispatch tables, error swallowing, shared-library symbol visibility,
widely-used return-value APIs, or test mocks.

## Signals, processes, and timeouts

- **Check syscall return values under signal-based timeouts — don't assume an interrupted syscall failed cleanly.** A `SIGALRM` handler racing `waitpid`/`os.wait` can leave `status` unset on `EINTR`, which then reads as `WIFEXITED(0)` — a false "success" instead of a timeout. Branch explicitly on the wait call's return value; if interrupted, force-kill and re-reap before deciding the outcome. This recurred independently in both the C loader and the Python persistent runner.
- **Put child processes in their own process group before ever using `killpg` on them.** `preexec_fn=os.setsid` (Python) / `setsid()` (C) must run before any code path can `SIGKILL` via `killpg`, or the signal lands on the caller's own group and can kill the fuzzer itself.
- **A timed-out or killed parent can leave an orphaned grandchild running forever.** If a subprocess itself forks/execs (e.g. a loader dlopen-ing and calling a target function), killing the immediate child on timeout isn't enough — track the grandchild's PID explicitly (e.g. via a PID file) and kill its process group too.
- **Guard `kill()`/`os.kill()` against `ESRCH` / `ProcessLookupError` races, and don't let a broad `except` swallow the real result.** A process can exit between a status check and a cleanup `kill()`; if the broad exception handler around that pattern also wraps the actual crash-detection logic, a benign race turns "a real crash" into "no crash detected." Catch the specific race exception close to the call that raises it, not several frames up.
- **`ChildProcessError` in a waitpid path means "already reaped" — not "success."** Returning `rc=0` on `ChildProcessError` silently masks crashes. Return `rc=-2` (unknown) so the crash detection pipeline treats it as suspicious, not clean.
- **Stale loop flags cause redundant waitpid on already-reaped children.** In ptrace mode, `last_action`/`last_sig` record "we last resumed the child after a breakpoint," but the post-loop code treats them as "the child might still be alive." A target that crashes after hitting even one coverage breakpoint (normal for guided fuzzing) gets its correctly-captured crash discarded by a redundant `waitpid` → `ChildProcessError` → `return 0`. Track explicitly whether the loop already reaped the child (`child_reaped` flag) and skip the redundant wait/kill entirely if so.

## Concurrency & resource cleanup

- **Release threads, fds, and temp resources in `finally` — but don't assume setup succeeded before scheduling cleanup.** A `thread.join()` in `finally` raises `NameError` if `fork()`/thread creation failed before the variable was assigned. Guard the cleanup call, or initialize the variable to `None` first.
- **Kill processes before detaching their shared memory.** Detaching SHM while the target is still writing to it can cause SEGV in the target. Always SIGKILL → waitpid → detach SHM, not the reverse.
- **Never use `tempfile.mktemp()`.** It's a TOCTOU/symlink race by construction. Use `mkstemp()`/`mkdtemp()`.
- **Namespace any filesystem path shared across parallel workers by PID.** Compiled shim/loader binaries and other on-disk artifacts written under `-j N` must embed `os.getpid()` (or equivalent), or concurrent workers race to compile/clean up the same file.
- **Return the actual PID on exception, not 0.** `run_target_stdin`/`run_target_file` callers use the returned PID for crash attribution. Returning `pid=0` on exception matches the swapper/idle process, silently discarding real kernel crashes.

## Hashing & identity

- **Never use Python's builtin `hash()` for anything persisted or shared across processes.** `hash(bytes)` is randomized per-process via `PYTHONHASHSEED`. Using it for a seed/edge-tracker key orphans every entry on fuzzer restart and produces divergent keys between `-j N` parallel workers. Always go through `hash_data()`/`hashlib`. Grep for `str(hash(` before adding any new keying scheme.

## Caching

- **Cache invalidation must key off the actual dependency, not a proxy.** Invalidating on every `exec_count` tick makes the cache useless (recomputes on almost every call — cost 80 eps once). Invalidating on the wrong signal, or never, serves stale data instead. Key strictly off the values the cache depends on (e.g. `corpus_version`, `edge_version`), and check both directions: does it recompute on every real change, and skip recomputing when nothing relevant changed?

## Low-level parsing (ELF, ptrace)

- **Use exact bitmask equality for "all these bits must be set" checks, never bare truthiness.** `flags & (PF_R | PF_X)` is truthy if *either* bit is set; a read-only RELRO segment (`PF_R` only) then satisfies a check meant to require both, silently selecting the wrong ELF segment. Write `(flags & mask) == mask` when the intent is "all of these bits."
- **This tool parses attacker-controlled binaries (the fuzz target's own ELF headers) and user-supplied grammar files as part of its own operation.** Bounds-check every offset/count read from a section header, program header, or symbol table before indexing with it. Clamp any grammar-controlled repeat/recursion count to a fixed MAX — an unbounded count is a resource-exhaustion bug in the fuzzer itself, not just in whatever it's fuzzing.

## Numeric & mutation edge cases

- **Clamp any input-derived range before calling `random.randint`/`randrange`.** If bounds come from input length (e.g. `len(raw) // stride - 1`), a small or degenerate input can make `hi < lo`, raising `ValueError` at fuzz time. Guard with `max(0, ...)` or an early return for the degenerate case.
- **Clamp arithmetic-mutation results to their valid range before packing.** `val * 2` or `val ^ (1 << 31)` can overflow the target field width and crash `struct.pack_into` — clamp to `[lo, hi]` for the field width in use.

## State & double-counting

- **Persisted state must have exactly one source of truth.** If a value can either be reloaded from disk (`state.pkl.gz` via `StateStore.get("markov")`, `StateStore.get("edge_tracker")`) or freshly re-derived by a normal-startup code path, the reload path must skip re-derivation — otherwise transition counts / edge stats double up silently across restarts.
- **Reduction and minimization must re-verify the specific property of interest, not just "did something happen."** Crash minimization (`tmin`) that only checks "does it still crash" can drift onto a different bug on a multi-bug target mid-delta-debugging. Pin the original crash signature up front and require every candidate to match it exactly.

## Testing

- **Assert on behavior, not on stale or accidentally-inverted expectations.** Some past bugs shipped *with* a passing test because the assertion checked the wrong direction (e.g. asserting a timeout counts as a crash) or referenced an error string that had since changed. When fixing a bug, re-read the failing test's assertion and confirm it actually encodes the intended behavior, not just "no exception was raised."

- Always add regression tests. When a bug is found and fixed, write a test that reproduces the exact scenario that caused the bug. Name it `test_regression_<brief_description>` (e.g. `test_regression_empty_list_crash`). This ensures the bug cannot silently return. Regression tests are non-negotiable: every fix ships with a test.

- **Tests that assert equivalence between two computed values must derive at least one side independently of the code under test.** The independent reference can be a hand-computed literal (e.g. `10 * 3` instead of `f(10, 3)`), a result from a different algorithm or library, or a known canonical answer. Never call the same function or implementation on both sides of an assertion — that validates the code against itself and passes even when the implementation is buggy. This rule applies to: `assert f(X) == f(Y)` (same function, different inputs), `assert result.some_field == another_call(...)`, and any test where the "expected" value is itself computed rather than a literal constant.

## Dispatch table & half-shipped features

- **Every entry in `_build_dispatch()` must have a corresponding module and class.** If an operator name is registered in `MUTATIONS` / `FORMAT_MUTATIONS` and wired into the dispatch table, but the module it imports doesn't exist, the fuzzer crashes with `ModuleNotFoundError` the moment the scheduler picks that operator. This is invisible to unit tests because they never exercise the live dispatch path. The integration smoke test (`test_operator_smoke.py::test_all_ops_fire`) catches this by calling every handler once — it must pass before any release.

## Silent error swallowing

- **Never `except Exception: pass` in production code paths.** A broad except that swallows errors hides real failures (disk full, permission denied, EMFILE). At minimum, log at `warning` level. Reserve `log.debug` for genuinely expected/recoverable situations only.
- **`except ChildProcessError` in waitpid does NOT mean success.** It means the child was already reaped — return `-2` (unknown), not `0` (success), to avoid masking crashes.

## Shared library symbol visibility (abort() override)

- **Libc function overrides in .so builds need `__attribute__((visibility("hidden")))`.** Without it, the PLT/GOT resolves calls at runtime to libc's version, not the override. `visibility("hidden")` forces direct `call` instructions within the .so, bypassing PLT resolution entirely. This applies to any shim-level override of `abort()`, `malloc`, `free`, etc. when the override is compiled into a `-shared -fPIC` library that gets `ctypes.CDLL`-loaded. Executable builds (non-`.so`) are not affected because the linker resolves from the translation unit first.

## Return-value API changes with many callers

- **When a widely-used function gets a new return value, grep for ALL callers — not just the production ones.** `load_corpus()` is called in 17+ places across 5 test files. Adding a 3rd return element (`irreplaceable_hashes`) silently breaks every `corpus, seen = load_corpus(...)` unpack pattern. The fix is mechanical (add `, _` or `, irr_hash`), but grep must target both `src/` and `tests/` to find them all.

## Mock alignment with production

- **A new attribute on Fuzzer that's read in a shared method (like `auto_minimize_corpus`) must be mirrored in every test mock.** A MockFuzzer that doesn't set `self.irreplaceable_hashes` causes `AttributeError` when the production logic tries to read it — but the error surfaces as a test failure in an unrelated test (the one that exercises the shared method, not the one testing the new feature). When adding a feature that touches shared code paths, audit all mocks for missing attributes before running tests.
