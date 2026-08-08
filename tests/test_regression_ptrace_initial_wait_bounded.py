"""Verify the initial-ptrace-stop wait is bounded.

Simulates the deadlock: a child that never reaches execv (so never
delivers SIGTRAP) must NOT hang the parent forever.
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


def test_stuck_child_does_not_hang_parent(monkeypatch):
    """Child sleeps forever instead of exec'ing -> parent must time out."""
    real_fork = os.fork

    def fake_fork():
        pid = real_fork()
        if pid == 0:
            # Emulate a child deadlocked before execv: never TRAPs.
            while True:
                time.sleep(3600)
        return pid

    monkeypatch.setattr(R.os, "fork", fake_fork)

    f = _make_fuzzer(timeout=2.0)
    r = R.TargetRunner(f)

    start = time.monotonic()
    rc, msg = r._run_target_ptrace(b"input")
    elapsed = time.monotonic() - start

    assert elapsed < 15.0, f"parent hung for {elapsed:.1f}s (unbounded wait)"
    assert rc == -2, f"expected killed-run rc=-2, got {rc}"
    assert "timeout" in msg
    # child must be reaped, not orphaned
    with pytest.raises(ChildProcessError):
        os.waitpid(f._last_child_pid, os.WNOHANG)
