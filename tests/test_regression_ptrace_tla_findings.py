"""Regressions found by the TLA+ model of the ptrace path (``tools/tla_extract.py ptrace``).

* NoOrphan: after a breakpoint ``cont``, settle polls ``waitpid(WNOHANG)``.
  Python returns ``(0, 0)`` for "no event" and ``WIFEXITED(0)`` is true, so a
  still-running tracee came back as a clean run and was never reaped.
* NoStaleKill: on a fatal stop settle SIGKILLed and reaped the tracee, then
  read the stale stop status and SIGKILLed the same pid again (every crash;
  a reused pid is someone else's process).
* NoUncaught: ``os.fork()`` failing (EAGAIN) escaped ``_ptrace_run`` and
  leaked the stdin pipe.
"""

import errno
import os
import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services import runner as runner_mod
from fuzzer_tool.services.runner import TargetRunner

_DEADLINE_S = 5.0


def _spawn_sleeper() -> int:
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os.execv("/bin/sh", ["/bin/sh", "-c", "sleep 300"])
        os._exit(127)
    return pid


def _spawn_exit(code: int) -> int:
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os._exit(code)
    return pid


def _settle(pid: int) -> tuple[int, str]:
    """Settle right after a breakpoint `cont`: the path that polls with WNOHANG."""
    runner = TargetRunner(SimpleNamespace())
    deadline = time.time() + _DEADLINE_S
    return runner._ptrace_settle(pid, None, deadline, 0, 0, False, "cont", signal.SIGTRAP)


def _spawn_stopped() -> tuple[int, int]:
    """A child stopped by SIGSTOP, and the stop status waitpid reported."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os.kill(os.getpid(), signal.SIGSTOP)
        os._exit(0)
    _, status = os.waitpid(pid, os.WUNTRACED)
    return pid, status


def _sigkills(kill: MagicMock, pid: int) -> int:
    return sum(1 for c in kill.call_args_list if c.args == (pid, signal.SIGKILL))


def _reaped(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def test_regression_ptrace_settle_running_tracee():
    """Falsification: a running tracee is killed, reaped and reported as not run."""
    pid = _spawn_sleeper()
    try:
        rc, _ = _settle(pid)
        assert rc == runner_mod._NOT_RUN
        assert _reaped(pid)
    finally:
        if not _reaped(pid):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


@pytest.mark.parametrize("code", [0, 3])
def test_exited_tracee_keeps_its_code(code):
    """Control: a tracee that already exited reports its own exit code."""
    pid = _spawn_exit(code)
    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)  # exited, not yet reaped
    rc, _ = _settle(pid)
    assert rc == code
    assert _reaped(pid)


def test_regression_ptrace_settle_kills_stopped_once():
    """Falsification: a fatal stop is SIGKILLed once, and its stop signal reported."""
    pid, status = _spawn_stopped()
    runner = TargetRunner(SimpleNamespace())
    libc = MagicMock()
    libc.ptrace.return_value = -1  # crash-state reads fail on a dead pid
    deadline = time.time() + _DEADLINE_S
    try:
        with patch.object(os, "kill", wraps=os.kill) as kill:
            rc, _ = runner._ptrace_settle(
                pid, libc, deadline, status, 0, False, None, signal.SIGSEGV
            )
        assert rc == -signal.SIGSTOP
        assert _sigkills(kill, pid) == 1
        assert _reaped(pid)
    finally:
        if not _reaped(pid):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def test_regression_ptrace_fork_failure_is_not_run():
    """Adversarial: EAGAIN from fork is a failed run; both pipe ends are closed."""
    fds: list[int] = []
    real_pipe = os.pipe

    def pipe() -> tuple[int, int]:
        r, w = real_pipe()
        fds.extend((r, w))
        return r, w

    f = SimpleNamespace(ptrace_cov=MagicMock(), target="/bin/true", timeout=_DEADLINE_S)
    runner = TargetRunner(f)
    with (
        patch.object(os, "pipe", side_effect=pipe),
        patch.object(os, "fork", side_effect=OSError(errno.EAGAIN, "fork")),
    ):
        rc, _ = runner._ptrace_run(b"x", 2)
    assert rc == runner_mod._NOT_RUN
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
