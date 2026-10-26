"""Regression: ptrace mode sent the target's stderr to /dev/null.

Sanitizer reports never reached is_crash: an ASAN build that exits 1 after
its report was a clean run, and signal crashes lost their stack.
"""

from __future__ import annotations

import platform

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.services.ptrace_coverage import PtraceCoverage
from fuzzer_tool.services.runner import TargetRunner

pytestmark = pytest.mark.skipif(platform.machine() != "x86_64", reason="ptrace runner is x86-64")

REPORT = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1"


class _FakeFuzzer:
    def __init__(self, target: str):
        self.ptrace_cov = PtraceCoverage(target)
        self.target = target
        self.timeout = 5
        self._last_child_pid = None
        self._last_fault_addr = None
        self._last_regs = {}


def _run(tmp_path, body: str) -> tuple[int, str]:
    script = tmp_path / "target.sh"
    script.write_text(body)
    script.chmod(0o755)
    return TargetRunner(_FakeFuzzer(str(script)))._run_target_ptrace(b"data")


def test_regression_ptrace_keeps_sanitizer_report(tmp_path) -> None:
    rc, stderr = _run(tmp_path, f"#!/bin/sh\necho '{REPORT}' >&2\nexit 1\n")

    assert rc == 1
    assert REPORT in stderr


def test_ptrace_quiet_target_empty_stderr(tmp_path) -> None:
    # Falsification: nothing written, nothing returned.
    rc, stderr = _run(tmp_path, "#!/bin/sh\nexit 0\n")

    assert (rc, stderr) == (0, "")


def test_ptrace_stderr_keeps_tail(tmp_path) -> None:
    # Edge: noise past the cap, report last. Shell builtins only: a forked
    # helper would stop the tracee on SIGCHLD.
    chunk = "n" * 1024
    reps = 2 * process._STDERR_CAP // len(chunk)
    body = (
        f"#!/bin/sh\ni=0\nwhile [ $i -lt {reps} ]; do printf '{chunk}' >&2; i=$((i+1)); done\n"
        f"echo >&2\necho '{REPORT}' >&2\nexit 1\n"
    )
    rc, stderr = _run(tmp_path, body)

    assert rc == 1
    assert len(stderr) <= process._STDERR_CAP
    assert REPORT in stderr


def test_regression_child_exit_signal_not_crash(tmp_path) -> None:
    # A helper process's SIGCHLD stopped the tracee and was reported as rc -17.
    rc, _ = _run(tmp_path, "#!/bin/sh\n/bin/true\nexit 3\n")

    assert rc == 3


def test_fault_signal_still_crash(tmp_path) -> None:
    # Falsification: a fault signal still ends the run as a crash.
    rc, _ = _run(tmp_path, "#!/bin/sh\n/bin/true\nkill -SEGV $$\n")

    assert rc == -11
