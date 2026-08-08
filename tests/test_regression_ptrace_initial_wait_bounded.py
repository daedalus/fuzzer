"""Verify the initial-ptrace-stop wait is bounded, and bounded *separately*.

Two regressions live here:

1. The hang. os.fork() clones only the calling thread, so a child forked
   while another thread holds the malloc or loader lock can deadlock before
   reaching execv and never deliver SIGTRAP. A blocking waitpid() on that
   child never returns.

2. The double budget. Bounding the initial wait with the full ``f.timeout``
   closed the hang but charged the timeout twice -- once here, once again in
   the run loop that follows -- so the worst case per exec became 2x the
   configured budget. Reaching the first ptrace stop is fork+execv only, so
   ``_INITIAL_STOP_TIMEOUT`` caps it independently.

Both assertions run against a virtual clock rather than the wall. A
wall-clock threshold measures the machine, not the code: it passes on an
idle box, is coverage-sensitive under ``--cov`` (~9x slower), and can only
ever assert a loose upper bound well above the real one. Driving
``runner``'s view of time directly makes the budget itself observable --
the test asserts that at most ``_INITIAL_STOP_TIMEOUT`` was consumed, which
is the actual invariant, and does so deterministically.
"""

import os
import time
import types

import pytest

from fuzzer_tool.services import runner as R


class _FakeCov:
    def reset_edge_map(self):
        pass

    def install_breakpoints(self, pid):
        pass


class _VirtualClock:
    """Stand-in for the ``time`` module as seen from ``runner``.

    ``sleep`` advances the clock instead of blocking, so the poll loop's
    deadline arithmetic runs exactly as written while the test itself
    finishes in milliseconds.
    """

    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self.start = start

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 1e-4)

    @property
    def elapsed(self) -> float:
        return self.now - self.start


def _make_fuzzer(timeout):
    f = types.SimpleNamespace()
    f.ptrace_cov = _FakeCov()
    f._last_fault_addr = None
    f._last_regs = {}
    f._last_child_pid = None
    f.target = "/bin/true"
    f.timeout = timeout
    f._perf_counters = None
    return f


def _stuck_child_fork():
    """Fork a child that never execs, emulating the loader-lock deadlock."""
    real_fork = os.fork

    def fake_fork():
        pid = real_fork()
        if pid == 0:
            while True:
                time.sleep(3600)
        return pid

    return fake_fork


def test_stuck_child_does_not_hang_parent(monkeypatch):
    """Child never TRAPs -> parent gives up and reports a killed run."""
    monkeypatch.setattr(R.os, "fork", _stuck_child_fork())
    monkeypatch.setattr(R, "time", _VirtualClock())

    f = _make_fuzzer(timeout=2.0)
    rc, msg = R.TargetRunner(f)._run_target_ptrace(b"input")

    assert rc == -2, f"expected killed-run rc=-2, got {rc}"
    assert "timeout" in msg
    # child must be reaped, not orphaned
    with pytest.raises(ChildProcessError):
        os.waitpid(f._last_child_pid, os.WNOHANG)


def test_initial_wait_does_not_spend_the_exec_budget(monkeypatch):
    """The initial stop is capped by _INITIAL_STOP_TIMEOUT, not f.timeout.

    With a timeout well above the cap, a child that never TRAPs must cost
    at most the cap -- otherwise the run loop's own full-timeout deadline
    stacks on top of it.
    """
    monkeypatch.setattr(R.os, "fork", _stuck_child_fork())
    clock = _VirtualClock()
    monkeypatch.setattr(R, "time", clock)

    timeout = 30.0
    assert timeout > R._INITIAL_STOP_TIMEOUT, "test needs a timeout above the cap"
    f = _make_fuzzer(timeout=timeout)
    rc, _ = R.TargetRunner(f)._run_target_ptrace(b"input")

    assert rc == -2
    # One poll interval of slack: the deadline is checked after the sleep.
    assert clock.elapsed <= R._INITIAL_STOP_TIMEOUT + 0.01, (
        f"initial wait consumed {clock.elapsed:.2f}s of a {timeout}s budget; "
        f"cap is {R._INITIAL_STOP_TIMEOUT}s"
    )


def test_short_timeout_still_wins(monkeypatch):
    """A timeout below the cap is honoured -- min(), not a flat constant."""
    monkeypatch.setattr(R.os, "fork", _stuck_child_fork())
    clock = _VirtualClock()
    monkeypatch.setattr(R, "time", clock)

    timeout = 0.05
    assert timeout < R._INITIAL_STOP_TIMEOUT
    f = _make_fuzzer(timeout=timeout)
    rc, _ = R.TargetRunner(f)._run_target_ptrace(b"input")

    assert rc == -2
    assert clock.elapsed <= timeout + 0.01
