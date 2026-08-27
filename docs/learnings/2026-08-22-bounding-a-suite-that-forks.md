# Bounding a suite that forks: why the obvious pytest timeout was the wrong one

2026-08-22. Closing E1, E2 and CRITICAL #3 from `docs/bugreport_2026-08-21_merged.md`.

## The finding

E1 says the suite can hang forever and wants a timeout. The obvious fix is the
one the bug report itself implies: `pytest-timeout` with `--timeout-method=thread`,
because the hang was inside Z3's `solver.add()` and only the thread method
survives native code — `signal` needs SIGALRM to be delivered at a bytecode
boundary, and a C call that never returns never reaches one.

That fix is wrong here, and the suite said so within one run.

`test_regression_persistent_execve_failure_exits` started failing. It asserts
that running one forking test in a subprocess reports that test's name exactly
once, as a proxy for "no orphaned duplicate pytest session". Under the thread
method the count was 2. The second occurrence was a warnings summary — CPython's
`DeprecationWarning: This process is multi-threaded, use of fork() may lead to
deadlocks in the child`.

The mechanism is that `--timeout-method=thread` arms a `threading.Timer` for
**every** test, so the pytest process is multi-threaded for the whole session.
This suite forks from that process in at least three places (`persistent_signal.py:72`,
`runner.py:311`, the inprocess loader). E3 in the same bug report is precisely
that hazard, and `docs/handover/test_shm_hang_2026-08-14.md` is the time it
actually cost a session.

So the obvious fix for E1 makes E3 worse, permanently, on every test. Measured
on one otherwise-silent forking test:

| method | multi-threaded-fork warnings |
|---|---:|
| `thread` | 1 |
| `signal` | 0 |

Full-suite warning count went 4 → 15 → 4 across the wrong fix and the right one.

## The resolution

Default to `signal` — no thread, bounds anything that reaches a bytecode
boundary (verified: a `time.sleep(120)` test dies at 3s under `--timeout=3`).
Then let the modules that genuinely block in native code opt into `thread`
individually, via `pytest_collection_modifyitems`. Today that is Z3
(`test_structural_constraints`, `test_field_constraints`), whose own `timeout`
parameter bounds `check()` and not assertion processing.

The general shape: **a global default that trades away a property the codebase
depends on is not a default, it is a regression with good intentions.** Scope
the expensive method to the cases that need it.

Applied here rather than in `addopts` so a dev environment without the plugin
still runs the suite instead of dying on an unrecognised argument, matching the
deliberately best-effort style of the `_fuzz_loader_built` fixture.

## Two measurement traps, both of which produced a false FAIL

CRITICAL #3's fix spawns the target into its own process group so a timeout
kill reaches grandchildren. Verifying that took three attempts, and the first
two failures were in the probe, not the code.

**1. `pgrep -f 'while true'` matched the harness's own command line.** The
shell invocation running the check contained the literal string being searched
for. Reported 2 survivors; there were none.

**2. `ps`-based counting flagged processes in state `Z`.** A SIGKILLed orphan
is reparented to PID 1, and this container's PID 1 does not reap. A correctly
killed grandchild therefore lingers as a zombie indefinitely. It is dead — but
a naive survivor count fails a working fix, and would fail it on exactly the
containerised machines CI runs on.

The probe that finally answered the question reads pgid out of `/proc/<pid>/stat`
and excludes state `Z`. It is in `tests/test_regression_fast_path_timeout.py`
with the reasoning attached, because the next person to check process cleanup
will reach for `ps | grep` first.

Generalisation worth keeping: **when a cleanup assertion fails, suspect the
observation before the code.** Process-liveness checks are unusually easy to
get wrong, and both failure modes above report the same symptom as a real leak.

## Why the fast path did not get a watchdog thread

`run_target_stdin` and `run_target_file` both enforce their timeout with a
watchdog thread, and Hard Rule 1 says match the closest existing example. The
fast path does not, for the same reason as the section above: it exists to
create no threads, and adding one puts it back in the E3 blast radius on the
single hottest path in the fuzzer.

Instead it polls the stderr pipe. EOF on that pipe means every writer closed
it, which for a spawned child means it exited — so one `poll()` serves as both
the stderr drain and the liveness wait. Draining concurrently is not an
optimisation either: the pipe holds 64 KiB, and reading only after the reap
(as the original did) deadlocks the child in `write()` against the parent in
`waitpid()`. Old code, 400 KiB of stderr: SIGKILLed at 15s, never returned.
New code: `rc=3`, 65536 B captured, 0.77s.

Cost: none measurable. 980.3 vs 989.2 eps median over 5 interleaved repeats of
400 execs, ranges fully overlapping.

**Residual case, stated rather than papered over.** A target that closes fd 2
and *then* loops forever reaches EOF without exiting, and the reap blocks.
Polling the reap instead would put a sleep on the hot path for every execution
to cover a target that deliberately closes its own stderr. The common hang — a
target that loops without exiting — never reaches EOF and is caught by the poll
deadline. The docstring says so.

## On snapshot-and-restore

`CmplogCollector.restore_env()` snapshots on the *first* `setup_env_for_run()`
call and never again. This matters because `run_target` calls that method
before every execution, not once per run: re-snapshotting each time would
capture the `LD_PRELOAD` the previous call installed, and restore would become
a no-op that leaves the shim in place forever. That is the original bug wearing
the fix's clothes, so it has its own regression test.

The class of bug is worth naming: **a save/restore pair is only correct if the
save happens exactly once across an idempotent setup.** Anything called per-iteration
needs the guard.
