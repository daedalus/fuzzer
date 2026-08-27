"""Regression tests: fault-address capture in the persistent loader.

The ptrace fault-address extraction (PTRACE_GETSIGINFO) originally lived only
in runner.py's dedicated ptrace runner. The persistent loader's grandchild now
self-traces (PTRACE_TRACEME) so its direct parent can read si_addr + registers
at the fatal-signal stop; the loader relays them to the fuzzer through the RC
line. These tests pin that capture end-to-end through InProcessRunner.run_one.
"""

import contextlib
import ctypes
import os
import platform
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.inprocess import InProcessRunner
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


def _make_runner(so: Path) -> InProcessRunner:
    return InProcessRunner(
        target=str(so),
        function_name="fuzz_shm_run",
        timeout=5.0,
        shm_size=4096,
        direct_lite=False,
        coverage_env_id=None,
        cov=False,
        debug=False,
        use_ptrace=True,  # capture is opt-in (--ptrace); these tests pin that path
    )


@pytest.mark.skipif(not _ptrace_available(), reason="ptrace not permitted in this environment")
class TestPersistentLoaderFaultAddr:
    def test_nosan_segv_captures_fault_and_regs(self, tmp_path):
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)
        runner = _make_runner(nosan_so)

        rc, _ = runner.run_one(b"CRASHS")
        assert rc < 0  # crash detection preserved through the waitpid-loop refactor
        assert runner._last_fault_addr == 0  # NULL-deref: si_addr == 0
        assert runner._last_regs.get("rsp", 0) != 0  # real stack address captured
        assert runner._last_regs.get("rbp", 0) != 0
        runner.stop()

    def test_nosan_abrt_has_no_fault_addr(self, tmp_path):
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)
        runner = _make_runner(nosan_so)

        rc, _ = runner.run_one(b"CRASHA")
        assert rc < 0
        # SIGABRT is not in the SEGV/BUS/ILL/FPE capture set — address stays None.
        assert runner._last_fault_addr is None
        assert runner._last_regs  # registers are still captured at the stop
        runner.stop()

    def test_safe_input_no_capture(self, tmp_path):
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)
        runner = _make_runner(nosan_so)

        rc, _ = runner.run_one(b"SAFEXXX00")
        assert rc == 0
        assert runner._last_fault_addr is None
        assert runner._last_regs == {}
        runner.stop()

    def test_fault_state_reset_between_runs(self, tmp_path):
        # A crash followed by a safe input must not leak stale fault info.
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)
        runner = _make_runner(nosan_so)

        runner.run_one(b"CRASHS")
        assert runner._last_fault_addr is not None
        runner.run_one(b"SAFEXXX00")
        assert runner._last_fault_addr is None
        assert runner._last_regs == {}
        runner.stop()


AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"


def _compile_shared_with_cmplog(src: Path, out: Path, tmp_path: Path, cc: str = "gcc"):
    """Build a .so the way build_targets.sh does by default: AFL shim
    -include'd with the cmplog layer enabled."""
    subprocess.run(
        [
            cc,
            "-O2",
            "-g",
            "-shared",
            "-fPIC",
            "-Wl,-Bsymbolic",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(out),
            str(src),
            "-ldl",
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not _ptrace_available(), reason="ptrace not permitted in this environment")
class TestFaultAddrWithCmplog:
    """The cmplog layer flushes its comparison buffer before the process
    dies. It must NOT re-raise hardware faults with raise(): raise()
    generates a *software* signal, and the kernel only fills
    siginfo_t.si_addr for hardware faults — so a ptrace tracer would read
    si_addr=0 instead of the true faulting address, silently collapsing
    NULL-deref and wild-pointer crashes into one dedup bucket.

    In an edge build the flush now happens inside __afl_crash_handler,
    which siglongjmps out rather than re-raising anything, so si_addr is
    never at risk from this path at all. The LD_PRELOAD-only build keeps
    the restore-and-return handler for the same reason it always had it.

    cmplog is ON by default in build_targets.sh, so this is the configuration
    real campaigns actually run; the pre-existing tests above build without it
    and therefore could not catch the regression.
    """

    @pytest.mark.parametrize("cc", ["gcc", "clang"])
    def test_segv_fault_addr_survives_cmplog_handler(self, tmp_path, cc):
        so = tmp_path / f"test_cmplog_{cc}.so"
        _compile_shared_with_cmplog(TARGETS_DIR / "test_target.c", so, tmp_path, cc=cc)
        runner = _make_runner(so)

        rc, _ = runner.run_one(b"CRASHS")
        assert rc != 0
        # The real bug: this was None when crash_handler re-raised via raise().
        assert runner._last_fault_addr == 0  # NULL-jump: si_addr == 0
        assert runner._last_regs.get("rsp", 0) != 0
        runner.stop()

    @pytest.mark.parametrize("cc", ["gcc", "clang"])
    def test_safe_input_clean_with_cmplog(self, tmp_path, cc):
        so = tmp_path / f"test_cmplog_safe_{cc}.so"
        _compile_shared_with_cmplog(TARGETS_DIR / "test_target.c", so, tmp_path, cc=cc)
        runner = _make_runner(so)

        rc, _ = runner.run_one(b"SAFEXXX00")
        assert rc == 0
        assert runner._last_fault_addr is None
        runner.stop()
