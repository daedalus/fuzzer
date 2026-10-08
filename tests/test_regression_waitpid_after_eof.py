"""Regression: ``run_target_fast`` hung on a target that closes fd 2 then loops.

EOF on the stderr pipe ended the deadline-bounded drain, and the reap that
followed was a blocking ``waitpid(pid, 0)``. The reap must honour the rest of
the deadline and report a timeout like the drain does.

Targets are ``/bin/sh`` scripts so no toolchain is needed.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from fuzzer_tool.adapters.process import _child_pids, run_target_fast

TIMEOUT = 1.0
SLACK = 3.0  # generous bound on scheduler noise
SETTLE = 5.0  # wait for SIGKILLed orphans to leave the run state


@pytest.fixture
def script(tmp_path):
    """Write an executable /bin/sh target and hand back its path."""

    def _make(name: str, body: str) -> str:
        p = tmp_path / f"{name}.sh"
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
        return str(p)

    return _make


def _live_pgid_members(pgid: int) -> list[int]:
    """Live (non-zombie) PIDs in *pgid*; orphan zombies may linger under PID 1."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().split(") ")[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            out.append(int(entry))
    return out


def _wait_group_gone(pgid: int) -> list[int]:
    """Poll until *pgid* has no live members or SETTLE expires."""
    end = time.monotonic() + SETTLE
    while time.monotonic() < end and _live_pgid_members(pgid):
        time.sleep(0.05)
    return _live_pgid_members(pgid)


def _run(path: str, timeout: float = TIMEOUT) -> tuple[int, str, int, float]:
    t0 = time.monotonic()
    rc, err, pid = run_target_fast(path, b"x", timeout=timeout)
    return rc, err, pid, time.monotonic() - t0


@pytest.mark.timeout(15)
def test_regression_waitpid_after_eof(script):
    """Closes stderr, then hangs: must time out, be killed and reaped."""
    hang = script("closed_hang", "exec 2>&-\nexec sleep 60\n")
    before = set(_child_pids())

    rc, err, pid, elapsed = _run(hang)

    assert rc == -1
    assert err == "timeout"
    assert elapsed < TIMEOUT + SLACK, f"reap ignored deadline ({elapsed:.1f}s)"
    assert not set(_child_pids()) - before, "pid left tracked"
    assert not os.path.exists(f"/proc/{pid}"), "child not reaped"


@pytest.mark.timeout(15)
def test_closed_stderr_clean_exit(script):
    """Falsification: prompt exit after closing stderr keeps its status."""
    quick = script("closed_exit", "exec 2>&-\nexit 5\n")

    rc, _, pid, elapsed = _run(quick, timeout=10.0)

    assert rc == 5
    assert elapsed < 5.0
    assert not os.path.exists(f"/proc/{pid}")


@pytest.mark.timeout(15)
def test_closed_stderr_signal_exit(script):
    """Adversarial: death by signal after closing stderr stays a crash."""
    segv = script("closed_segv", "exec 2>&-\nkill -SEGV $$\n")

    rc, _, _, _ = _run(segv, timeout=10.0)

    assert rc == -11


@pytest.mark.timeout(15)
def test_closed_stderr_grandchild_exit(script):
    """Adversarial: target forks a grandchild holding no pipe, then exits.

    The grandchild must not stretch the reap; the target's status wins.
    """
    fork = script("closed_fork", "exec 2>&-\nsleep 60 &\nexit 0\n")

    rc, _, pid, elapsed = _run(fork, timeout=10.0)
    os.killpg(pid, signal.SIGKILL)  # Hard Rule 21: the orphan grandchild is ours

    assert rc == 0
    assert elapsed < 5.0
    assert not _wait_group_gone(pid)


@pytest.mark.timeout(15)
def test_closed_stderr_grandchild_hang(script):
    """Adversarial: target and grandchild both hang; group kill reaches both."""
    gp = script("closed_gp", "exec 2>&-\nsleep 60 &\nsleep 60\n")

    rc, _, pid, elapsed = _run(gp)

    assert rc == -1
    assert elapsed < TIMEOUT + SLACK
    assert not _wait_group_gone(pid), "grandchild survived"


def _no_pidfd(pid: int) -> int:
    raise OSError("pidfd_open unsupported")


@pytest.mark.timeout(15)
def test_closed_stderr_hang_without_pidfd(script, monkeypatch):
    """Fallback: kernels without pidfd_open still honour the deadline."""
    monkeypatch.setattr(os, "pidfd_open", _no_pidfd)
    hang = script("closed_hang_nopidfd", "exec 2>&-\nexec sleep 60\n")

    rc, _, pid, elapsed = _run(hang)

    assert rc == -1
    assert elapsed < TIMEOUT + SLACK
    assert not os.path.exists(f"/proc/{pid}")


@pytest.mark.timeout(15)
def test_closed_stderr_exit_without_pidfd(script, monkeypatch):
    """Fallback falsification: prompt exit keeps its status."""
    monkeypatch.setattr(os, "pidfd_open", _no_pidfd)
    quick = script("closed_exit_nopidfd", "exec 2>&-\nsleep 0.2\nexit 4\n")

    rc, _, _, _ = _run(quick, timeout=10.0)

    assert rc == 4
