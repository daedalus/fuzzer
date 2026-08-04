"""Regression tests: ptrace crash fault-address extraction via PTRACE_GETSIGINFO.

Spawn a real traced child that faults (SIGSEGV) and assert that
runner._get_fault_addr recovers the kernel-reported si_addr — the same
mechanism the ptrace runner uses to enrich crash signatures and metadata.
This guards against regressions that would conflate the instruction pointer
(rip) with the faulting memory address.
"""

import contextlib
import ctypes
import os
import platform
import signal
import struct

import pytest

from fuzzer_tool.services.ptrace_coverage import PtraceCoverage
from fuzzer_tool.services.runner import TargetRunner, _get_fault_addr

pytestmark = pytest.mark.skipif(
    platform.machine() != "x86_64", reason="siginfo_t layout is x86-64 specific"
)

PTRACE_TRACEME = 0
PTRACE_CONT = 7


def _libc():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.ptrace.argtypes = [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    libc.ptrace.restype = ctypes.c_long
    return libc


def _fault_stop(fault_fn):
    """Fork a traced child running *fault_fn*; return (stop_sig, fault_addr).

    The child PTRACE_TRACEMEs itself, syncs via SIGSTOP, then runs the fault.
    Returns ``pytest.skip`` if ptrace is unavailable (e.g. tests under a
    tracer), and (None, None) if the child exits without a signal stop.
    """
    libc = _libc()
    pid = os.fork()
    if pid == 0:
        try:
            if libc.ptrace(PTRACE_TRACEME, 0, None, None) != 0:
                os._exit(42)  # ptrace not permitted here
            os.kill(os.getpid(), signal.SIGSTOP)  # sync: tracer knows we're traced
            fault_fn()
            os._exit(0)
        except BaseException:
            os._exit(127)
    try:
        _, status = os.waitpid(pid, os.WUNTRACED)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 42:
            pytest.skip("ptrace not permitted in this environment")
        assert os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP
        libc.ptrace(PTRACE_CONT, pid, None, None)
        _, status = os.waitpid(pid, os.WUNTRACED)
        if not os.WIFSTOPPED(status):
            return None, None
        return os.WSTOPSIG(status), _get_fault_addr(pid, libc)
    finally:
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def _deref(addr: int):
    ctypes.string_at(addr)


class TestGetFaultAddr:
    def test_null_deref(self):
        sig, fault = _fault_stop(lambda: _deref(0))
        assert sig == signal.SIGSEGV
        assert fault == 0

    def test_wild_pointer(self):
        # Canonical unmapped address (below the x86-64 47-bit boundary; a
        # non-canonical address would raise #GP and report si_addr == 0).
        target = 0xDEADBEEF000
        sig, fault = _fault_stop(lambda: _deref(target))
        assert sig == signal.SIGSEGV
        assert fault == target

    def test_user_raised_signal_returns_none(self):
        # kill(SIGSEGV) has si_code == SI_USER (0): si_addr is meaningless.
        sig, fault = _fault_stop(lambda: os.kill(os.getpid(), signal.SIGSEGV))
        assert sig == signal.SIGSEGV
        assert fault is None

    def test_non_fault_signal_returns_none(self):
        sig, fault = _fault_stop(os.abort)
        assert sig == signal.SIGABRT
        assert fault is None

    def test_clean_exit_returns_none(self):
        sig, fault = _fault_stop(lambda: None)
        assert sig is None
        assert fault is None

    def test_register_offsets_sane(self):
        # Regression: x86-64 user_regs_struct rsp is at offset 152, not
        # 176 (gs_base, 0 for the main thread). The old 128+48 offset made
        # the breakpoint handler's `rsp > 0x1000` check always-false, so it
        # never restored RIP past a breakpoint and skipped the first
        # instruction of every instrumented function.
        libc = _libc()
        pid = os.fork()
        if pid == 0:
            try:
                if libc.ptrace(PTRACE_TRACEME, 0, None, None) != 0:
                    os._exit(42)
                os.kill(os.getpid(), signal.SIGSTOP)
                os._exit(0)
            except BaseException:
                os._exit(127)
        try:
            _, status = os.waitpid(pid, os.WUNTRACED)
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 42:
                pytest.skip("ptrace not permitted in this environment")
            assert os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP
            regs_buf = (ctypes.c_char * (27 * 8))()
            assert libc.ptrace(12, pid, None, regs_buf) == 0  # PTRACE_GETREGS
            regs = bytes(regs_buf)
            rsp = struct.unpack_from("<Q", regs, 152)[0]
            gs_base = struct.unpack_from("<Q", regs, 176)[0]
            assert rsp > 0x1000, f"rsp@152 should be a stack address, got {rsp:#x}"
            assert gs_base == 0  # the old buggy offset read gs_base here
        finally:
            with contextlib.suppress(ProcessLookupError, ChildProcessError):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)


class TestPtraceRunnerFlow:
    """End-to-end runs through TargetRunner._run_target_ptrace.

    Uses shebang shell scripts as targets (no C compiler needed). Ptrace
    coverage finds no ELF symbols in a script, so these exercise the wait
    loop and signal handling without breakpoint instrumentation.
    """

    def _run_script(self, tmp_path, body: str) -> tuple[int, str, object]:
        script = tmp_path / "target.sh"
        script.write_text(body)
        script.chmod(0o755)
        f = _FakeFuzzer(str(script))
        rc, stderr = TargetRunner(f)._run_target_ptrace(b"data")
        return rc, stderr, f

    def test_clean_exit_reported_as_rc0(self, tmp_path):
        # Regression: the runner's waitpid loop checked only `status == 0`,
        # conflating "no event" (0, 0) with "clean exit rc=0" (pid, 0). Every
        # clean exit was treated as still-running, the next poll hit ECHILD,
        # and the run was misreported as -2 ("exec failed").
        rc, stderr, f = self._run_script(tmp_path, "#!/bin/sh\nexit 0\n")
        assert rc == 0
        assert stderr == ""
        assert f._last_fault_addr is None

    def test_signal_exit_captured(self, tmp_path):
        rc, stderr, f = self._run_script(tmp_path, "#!/bin/sh\nkill -SEGV $$\n")
        assert rc == -11
        assert f._last_fault_addr is None  # user-raised: si_code == SI_USER


class _FakeFuzzer:
    def __init__(self, target: str):
        self.ptrace_cov = PtraceCoverage(target)
        self.target = target
        self.timeout = 5
        self._last_child_pid = None
        self._last_fault_addr = None
        self._last_regs = {}
