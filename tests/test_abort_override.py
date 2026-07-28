"""Tests for the abort() override in afl_shim.c.

The shim intercepts libc abort() — instead of killing the process it writes
"[shim] abort() intercepted" to stderr and returns. This prevents false crash
detections from FFmpeg's av_assert0 (~1600 call sites) and similar library
assertions during fuzzing.

Two compilation variants:
  - "shim.bin": compiled WITH -include afl_shim.c (override active)
  - "noshim.bin": compiled WITHOUT the shim (libc abort kills process)

Three test cases:
  1. CRASHA through shim.bin via run_target_stdin → rc=0, "[shim] abort()" in stderr
  2. CRASHA through noshim.bin via run_target_stdin → rc=-6 (SIGABRT), no shim message
  3. CRASHS through shim.bin via run_target_stdin → rc=-11 (SIGSEGV still propagates)
  4. CRASHA through shim.so via InProcessRunner (direct_lite) → rc=0 (stderr not captured in-process)
"""

import subprocess
from pathlib import Path

import pytest

TARGETS_DIR = Path(__file__).parent.parent / "targets"
SHIM_PATH = TARGETS_DIR.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

SHIM_STDERR_MSG = b"[shim] abort() intercepted"


@pytest.fixture(scope="module")
def abort_test_binaries(tmp_path_factory):
    """Compile test_target.c with and without the abort() override shim."""
    tmpdir = tmp_path_factory.mktemp("abort_test")

    def _compile_bin(src, out_name, extra_flags=None):
        out = tmpdir / out_name
        cmd = ["gcc", "-o", str(out), str(src)]
        if extra_flags:
            cmd.extend(extra_flags)
        subprocess.run(cmd, check=True, capture_output=True)
        return out

    def _compile_so(src, out_name, extra_flags=None):
        out = tmpdir / out_name
        cmd = ["gcc", "-shared", "-fPIC", "-o", str(out), str(src)]
        if extra_flags:
            cmd.extend(extra_flags)
        subprocess.run(cmd, check=True, capture_output=True)
        return out

    shim_bin = _compile_bin(
        TARGETS_DIR / "test_target.c",
        "test_target_shim",
        extra_flags=["-include", str(SHIM_PATH)],
    )
    noshim_bin = _compile_bin(
        TARGETS_DIR / "test_target.c",
        "test_target_noshim",
    )
    shim_so = _compile_so(
        TARGETS_DIR / "test_target.c",
        "test_target_shim.so",
        extra_flags=["-include", str(SHIM_PATH)],
    )

    class Binaries:
        SHIM_BIN = shim_bin
        NOSHIM_BIN = noshim_bin
        SHIM_SO = shim_so

    return Binaries()


class TestAbortOverride:
    """Verify the abort() override in afl_shim.c intercepts CRASHA input."""

    def test_abort_caught_by_shim(self, abort_test_binaries):
        """With shim: CRASHA returns rc=0 and writes shim message to stderr."""
        from fuzzer_tool.adapters.process import run_target_stdin

        rc, stderr, pid = run_target_stdin(
            str(abort_test_binaries.SHIM_BIN),
            b"CRASHA",
            timeout=2.0,
        )

        assert rc == 0, f"Expected rc=0 (abort intercepted), got rc={rc}. stderr: {stderr}"
        assert SHIM_STDERR_MSG in stderr.encode(), (
            f"Expected shim abort message in stderr, got: {stderr!r}"
        )

    def test_abort_sigabrt_without_shim(self, abort_test_binaries):
        """Without shim: CRASHA kills process with SIGABRT (rc=-6)."""
        from fuzzer_tool.adapters.process import run_target_stdin

        rc, stderr, pid = run_target_stdin(
            str(abort_test_binaries.NOSHIM_BIN),
            b"CRASHA",
            timeout=2.0,
        )

        # SIGABRT = signal 6, encoded as -6
        assert rc == -6, f"Expected rc=-6 (SIGABRT), got rc={rc}. stderr: {stderr}"
        assert SHIM_STDERR_MSG not in stderr.encode(), (
            "Shim abort message should NOT appear without the shim"
        )

    def test_segv_still_propagates_with_shim(self, abort_test_binaries):
        """With shim: CRASHS still produces SIGSEGV (rc=-11).
        The abort override only intercepts abort(), not segfaults."""
        from fuzzer_tool.adapters.process import run_target_stdin

        rc, stderr, pid = run_target_stdin(
            str(abort_test_binaries.SHIM_BIN),
            b"CRASHS",
            timeout=2.0,
        )

        assert rc == -11, f"Expected rc=-11 (SIGSEGV), got rc={rc}. stderr: {stderr}"
        # CRASHS should NOT trigger the abort message
        assert SHIM_STDERR_MSG not in stderr.encode(), (
            "SIGSEGV should not produce shim abort message"
        )

    def test_shim_stderr_message_exact_text(self, abort_test_binaries):
        """Verify the exact stderr text from the abort override via subprocess."""
        from fuzzer_tool.adapters.process import run_target_stdin

        rc, stderr, pid = run_target_stdin(
            str(abort_test_binaries.SHIM_BIN),
            b"CRASHA",
            timeout=2.0,
        )

        assert "[shim] abort() intercepted" in stderr, (
            f"Expected exact shim message in stderr, got: {stderr!r}"
        )

    def test_shim_so_inprocess_abort_does_not_crash(self, abort_test_binaries):
        """With shim .so via InProcessRunner: CRASHA does not crash (rc=0).
        stderr is not captured in direct_lite mode; only rc is checked."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(abort_test_binaries.SHIM_SO),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=True,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, _ = runner.run_one(b"CRASHA")
        assert rc == 0, f"Expected rc=0 (abort intercepted via .so), got rc={rc}"
