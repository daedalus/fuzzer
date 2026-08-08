"""Regression tests: direct_lite crash triage via the ptrace-attached loader.

direct_lite runs the target in-process via ctypes; a crash is reported only as
a negative signal code from __afl_guarded_call, with no fault address. The
triage path re-runs the crashing input once through the subprocess loader
script self-traced with PTRACE_TRACEME, so TargetRunner can read si_addr +
registers at the fatal-signal stop via PTRACE_GETSIGINFO.
"""

import contextlib
import ctypes
import os
import platform
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.services.runner import TargetRunner
from tests.conftest import requires_clang

pytestmark = [
    pytest.mark.skipif(
        platform.machine() != "x86_64", reason="siginfo_t layout is x86-64 specific"
    ),
    requires_clang,
]

TARGETS_DIR = Path(__file__).parent.parent / "targets"

PTRACE_TRACEME = 0


def _ptrace_available() -> bool:
    """True if PTRACE_TRACEME works here (yama ptrace_scope can block it)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    libc.ptrace.restype = ctypes.c_long
    pid = os.fork()
    if pid == 0:
        try:
            if libc.ptrace(PTRACE_TRACEME, 0, None, None) != 0:
                os._exit(42)
            os._exit(0)
        except BaseException:
            os._exit(127)
    try:
        _, status = os.waitpid(pid, os.WUNTRACED)
        return not (os.WIFEXITED(status) and os.WEXITSTATUS(status) in (42, 127))
    except ChildProcessError:
        return False
    finally:
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def _compile_shared(src: Path, out: Path):
    subprocess.run(
        ["clang", "-shared", "-fPIC", "-o", str(out), str(src)],
        check=True,
        capture_output=True,
    )


class _RunnerStub:
    """Minimal stand-in for InProcessRunner — the triage only needs these."""

    def __init__(self, target: str):
        self.target = target
        self.function_name = "fuzz_shm_run"
        self._loader_path = None
        self.direct_lite = True


class _FakeFuzzer:
    def __init__(self, target: str):
        self.target = target
        self.timeout = 5.0
        self._inprocess_runner = _RunnerStub(target)
        self._last_fault_addr = None
        self._last_regs = {}


@pytest.mark.skipif(not _ptrace_available(), reason="ptrace not permitted in this environment")
class TestDirectLiteTriage:
    def test_triage_segv_captures_fault_addr(self, tmp_path):
        so = tmp_path / "tgt.so"
        _compile_shared(TARGETS_DIR / "test_target.c", so)
        f = _FakeFuzzer(str(so))

        rc, _ = TargetRunner(f)._run_triage_ptrace(b"CRASHS")

        assert rc < 0
        assert f._last_fault_addr == 0  # NULL-deref: si_addr == 0
        assert f._last_regs.get("rsp", 0) != 0

    def test_triage_abrt_no_fault_addr(self, tmp_path):
        so = tmp_path / "tgt.so"
        _compile_shared(TARGETS_DIR / "test_target.c", so)
        f = _FakeFuzzer(str(so))

        rc, _ = TargetRunner(f)._run_triage_ptrace(b"CRASHA")

        assert rc < 0
        assert f._last_fault_addr is None  # SIGABRT not in the capture set
        assert f._last_regs  # registers are still captured at the stop

    def test_triage_safe_input_no_capture(self, tmp_path):
        so = tmp_path / "tgt.so"
        _compile_shared(TARGETS_DIR / "test_target.c", so)
        f = _FakeFuzzer(str(so))

        rc, _ = TargetRunner(f)._run_triage_ptrace(b"SAFEXXX00")

        assert rc == 0
        assert f._last_fault_addr is None
        assert f._last_regs == {}

    def test_triage_writes_loader_on_demand(self, tmp_path):
        # direct_lite never creates _loader_path; the triage must write one.
        so = tmp_path / "tgt.so"
        _compile_shared(TARGETS_DIR / "test_target.c", so)
        f = _FakeFuzzer(str(so))
        assert f._inprocess_runner._loader_path is None

        TargetRunner(f)._run_triage_ptrace(b"SAFEXXX00")

        assert f._inprocess_runner._loader_path is not None
        assert Path(f._inprocess_runner._loader_path).exists()
