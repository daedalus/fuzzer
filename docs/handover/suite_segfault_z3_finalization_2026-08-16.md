# Handover — intermittent `Segmentation fault` in the full test suite

**Date:** 2026-08-16
**Base:** `725168d`, plus this session's two commits (`88b865c`, `2bc899c`)
**Status:** root cause **identified and captured**; fix **not yet written**

---

## Symptom

A full `python -m pytest tests/ -q` run died with a bare `Segmentation fault`
from the shell. The visible output stopped mid-file at ~90%
(`tests/test_structural_constraints.py`), with no summary line and no Python
traceback.

Reproduction rate on this machine: **1 crash in 8 full-suite runs**.

## Root cause

The crash is **not** where the output stopped. It happens at **interpreter
finalization, after every test has already passed**, inside z3's own
destructor:

```
========== 4566 passed, 114 skipped, 4 warnings in 235.13s (0:03:55) ===========
Fatal Python error: Segmentation fault

Current thread 0x00007fcf5ef7a080 (most recent call first):
  File ".../z3/z3core.py", line 1684 in Z3_del_context
  File ".../z3/z3.py", line 224 in __del__
```

Two things conspired to make this look like a mid-suite crash:

1. **Block-buffered stdout.** With output going to a pipe, the tail of the
   progress dots *and the entire summary line* sat in an unflushed 4 KiB
   buffer when the process died. The last flushed dots named
   `test_structural_constraints.py`, which had nothing to do with it.
2. **`fuzzer.py` installs a Python-level `SIGSEGV` handler at import time**
   (`services/fuzzer.py:194`, `signal.signal(signal.SIGSEGV, _handle_sigsegv)`).
   A Python handler cannot run from a C-level fault in a native library, so
   it produced no output *and* suppressed `faulthandler`'s dump. Confirmed
   still installed in a normal run.

### Why z3 faults

`z3.Context.__del__` calls `Z3_del_context(self.ctx)` on z3's **singleton
main context** (`z3._main_ctx`), during finalization, when z3's own module
globals are being cleared in arbitrary order.

Object census at `atexit` on a full suite run:

```
[z3scan] live z3 objects at atexit:
   Elementaries: 848
   CheckSatResult: 3
   Context: 1
   ContextObj: 1
[z3scan] z3._main_ctx set: True
[z3scan] referrers of live Solver/Context: builtins.list: 1, builtins.dict: 1
```

**No `Solver` objects survive to shutdown**, so this is not the project
leaking solvers into finalization. What dies is z3's own singleton, and the
ordering between its teardown and the 848 `Elementaries` wrappers is not
deterministic — hence the ~1-in-8 rate.

The project creates solvers via bare `z3.Solver()` (main context) in five
places: `core/xor_map_solver.py:224`, `core/path_constraints.py:305`,
`core/smt_solver.py:134`, `core/structural_constraints.py:175`,
`core/field_constraints.py:375`. None constructs its own `z3.Context()`.

### Preconditions

- **Requires the optional `smt` extra.** z3 is not pulled in by
  `pip install -e ".[dev]"`, so a machine without it cannot hit this. It was
  installed in this session to exercise the XOR-map solver.
- **Not caused by this session's commits.** They add no z3 lifecycle code;
  `88b865c`'s fixture constructs a `PathConstraintSolver` only when z3 is
  present, and that solver is collected long before shutdown (census above).
  Timing is coincidental: the crash first appeared on the run immediately
  after z3 was installed.

## How it was found

Prior hypotheses, each **falsified** by measurement — recorded so they are
not re-tried:

| Hypothesis | Test | Result |
|---|---|---|
| Use-after-`cleanup()` on `ShmCoverage` read paths | trip-wire raising on any read after `cleanup()`, full suite | 0 hits |
| Leaked `stack-heartbeat` daemon walking a live foreign frame | stress: 20 Hz–5 kHz sampling vs. deep call/return churn, 6 × 8 s | 0 crashes in ~9 M cycles |
| Daemon threads alive at finalization | 8 heartbeat-shaped daemons + immediate exit, 60 iterations | 0 crashes |
| z3 usage alone | the four z3-touching test files only, 25 iterations | 0 crashes |
| z3 + fork + daemon thread, minimal | 120 iterations | 0 crashes |

What worked: a loop running the full suite under a wrapper that imports
`fuzzer_tool.services.fuzzer` first, then **takes `SIGSEGV` back**
(`signal.signal(signal.SIGSEGV, signal.SIG_DFL)`) and enables
`faulthandler` with `all_threads=True` to a file. That produced the
traceback above on the first crashing run.

## Suggested fix (not implemented)

Two independent changes, in priority order:

1. **Stop masking faults.** `services/fuzzer.py:182-194`'s `_handle_sigsegv`
   is actively harmful: a Python handler cannot run from a native fault, it
   suppressed `faulthandler`'s dump, and it turned a diagnosable crash into a
   silent one. It also installs process-wide at *import*, so it affects every
   consumer of the module including pytest. Replace with
   `faulthandler.enable()`, which handles SIGSEGV correctly at the C level
   and prints every thread's stack. Note `adapters/inprocess.py:469` installs
   its own SIGSEGV handler around in-process runs and restores it at :505 —
   that one is scoped and deliberate; leave it.

2. **Make z3 teardown deterministic.** Drive z3's main context down while the
   interpreter is still fully alive, rather than during finalization. An
   `atexit` hook (they run *before* module globals are cleared) that drops
   project references, `gc.collect()`s, and disposes the main context should
   convert an arbitrary-order teardown into a controlled one. This needs a
   reproduction harness to validate — see below, the crash is ~1-in-8 per
   4-minute run, so a fix needs ~30+ runs to claim anything.

Do **not** attempt to "fix" this by removing z3 from the environment; the
`@requires_z3` skips already handle its absence, and skipping is what hid
this until now.

## Unrelated defect found while investigating (real, unfixed)

`ShmCoverage.cleanup()` (`adapters/shm.py:629`) detaches the segment and sets
`_ptr = None`, but leaves **`_map`, `_entries` and `_tail` bound to the
detached address**:

```
_ptr: None | _map still bound: True | _tail: 139969452310552
```

Reading through them after cleanup is a hard segfault, reproducible in three
lines:

```python
from fuzzer_tool.adapters.shm import ShmCoverage
c = ShmCoverage(); c.cleanup()
c.get_edge_ids()          # -> SIGSEGV (exit 139)
```

Worse than the crash: **the kernel hands back the same address every time.**
Six successive `shmat`/`shmdt` cycles all returned `0x7f98c6b12000`. So a
stale `_map` from a cleaned-up instance *silently aliases the next live
segment* — reads and writes land in another `ShmCoverage`'s coverage table
instead of faulting. It only segfaults in the window where no live segment
occupies the address.

`DistanceTableShm.cleanup()` (:708) has the same shape and is worse: it sets
`_ptr = 0` rather than `None`, so `self._ptr + 4 + pos * entry_bytes` at :699
still computes a valid-looking near-null address instead of raising
`TypeError`.

The full suite does **not** currently exercise either path (the trip-wire
found 0 hits), so this is latent, not active. Fix is small: null out
`_map`/`_entries`/`_tail` in both `cleanup()` methods so post-cleanup use
raises loudly instead of corrupting a live segment. `resize()` (:593-606)
should drop the old views before `shmdt(old_ptr)` for the same reason.

## Reproduction harness

Wrapper that makes the crash diagnosable (was `_crashhunt.py`, deleted):

```python
import faulthandler, signal, sys
import fuzzer_tool.services.fuzzer  # installs its own handlers first
signal.signal(signal.SIGSEGV, signal.SIG_DFL)
faulthandler.enable(file=open("/tmp/fault.txt", "w"), all_threads=True)
import pytest
sys.exit(pytest.main(["tests/", "-q", "-p", "no:cacheprovider"]))
```

Loop it, writing stdout to a **file** (not a pipe) so the tail is not lost,
and treat any `rc > 1` as a crash. Budget ~4 minutes per run.

## State of the tree

Clean. All investigation scaffolding (`_probe.py`, `_crashhunt.py`,
`_z3scan.py`, `_threadscan.py`, `_beat_stress.py`, `_probe_run.py`) removed;
`tests/conftest.py` was temporarily edited during probing and has been
reverted. `git status` is clean at `2bc899c`.
