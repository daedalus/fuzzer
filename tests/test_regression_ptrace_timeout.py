"""Regression: ptrace-mode timeouts were reported as crashes, never as timeouts.

``Fuzzer`` decides a run hung with ``is_timeout = returncode == -1``
(``services/fuzzer.py:3298``).  The ptrace wait loop could not produce -1 on
any path.  On deadline expiry it fell out of ``while time.time() < deadline``
and reconstructed a return code from ``status``, which still held the LAST
CONSUMED event:

* with at least one breakpoint handled, the post-loop ``waitpid(WNOHANG)``
  returned ``(0, 0)`` and the stale ``SIGTRAP`` stop yielded ``rc = -5``,
  reported as "crash signal 5";
* with none, the else-branch SIGKILLed and the SIGKILL death status yielded
  ``rc = -9``.

Either way a slow input was filed as a crash, taking a slot in signature
dedup and a file in ``crashes/``.

These tests assert the exact ``(-1, "timeout")`` pair rather than "not a
crash": -2 is the infrastructure sentinel and -9 is a legitimate SIGKILL
crash, so a fix that merely stopped returning -5 would still be wrong.
"""

import os
import signal
import time
from unittest.mock import patch

import pytest

from fuzzer_tool.services.runner import _ptrace_report_timeout


def _spawn_sleeper() -> int:
    """A child that outlives any test deadline, for the reaping tests."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os.execv("/bin/sh", ["/bin/sh", "-c", "sleep 300"])
        os._exit(127)
    return pid


class TestPtraceReportTimeout:
    def test_returns_the_timeout_sentinel(self):
        pid = _spawn_sleeper()
        assert _ptrace_report_timeout(pid) == (-1, "timeout")

    def test_reaps_the_tracee(self):
        """No zombie and no orphan: waitpid on a reaped pid raises ECHILD."""
        pid = _spawn_sleeper()
        _ptrace_report_timeout(pid)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_kills_rather_than_leaking_the_process(self):
        pid = _spawn_sleeper()
        _ptrace_report_timeout(pid)
        # The pid is gone; signal 0 probes existence without delivering.
        with pytest.raises((ProcessLookupError, PermissionError)):
            for _ in range(50):
                os.kill(pid, 0)
                time.sleep(0.01)

    def test_survives_an_already_dead_tracee(self):
        """The deadline can expire in the window where the child just died."""
        pid = _spawn_sleeper()
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        assert _ptrace_report_timeout(pid) == (-1, "timeout")

    def test_survives_an_unkillable_pid(self):
        """A PermissionError from kill must not escape as a run failure."""
        with (
            patch("os.kill", side_effect=PermissionError),
            patch("os.waitpid", side_effect=ChildProcessError),
        ):
            assert _ptrace_report_timeout(1) == (-1, "timeout")


class TestTimeoutContract:
    def test_minus_one_is_what_the_fuzzer_reads_as_a_timeout(self):
        """Pin the contract this fix depends on, so a change to it breaks here.

        ``_ptrace_report_timeout`` is only correct because rc -1 alone sets
        ``is_timeout``.  An earlier version of that check also required
        ``stderr == "timeout"``, which is why ptrace's empty stderr went
        unnoticed for so long.
        """
        from pathlib import Path

        import fuzzer_tool

        src = (Path(fuzzer_tool.__file__).parent / "services" / "fuzzer.py").read_text()
        assert "is_timeout = returncode == -1" in src, (
            "the timeout sentinel moved; _ptrace_report_timeout must follow it"
        )
