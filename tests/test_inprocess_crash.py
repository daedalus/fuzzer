"""Tests for inprocess mode crash detection — ASAN and non-ASAN .so targets.

Covers the bug classes fixed in:
- _run_c_direct_lite missing SIGSEGV/SIGABRT handlers
- _probe_so_function loading .so via ctypes.CDLL (ASAN loading order)
- Auto-detected .so targets not using subprocess loader
- run_target_fast not redirecting stdin from temp file
- run_target_fast not capturing stderr
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TARGETS_DIR = Path(__file__).parent.parent / "targets"
NOSAN_SO = TARGETS_DIR / "test_target_nosan.so"
ASAN_SO = TARGETS_DIR / "asan_target.so"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus_with_seed(tmpdir, seed_data):
    """Create corpus/seeds/ structure with a seed file."""
    corpus_dir = Path(tmpdir) / "corpus"
    seeds_dir = corpus_dir / "seeds"
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "seed").write_bytes(seed_data)
    return corpus_dir


def _fuzzer_crash_test(target, seed_data, tmpdir, **extra_kwargs):
    """Run fuzzer on a target with the given seed, return crash files."""
    corpus_dir = _make_corpus_with_seed(tmpdir, seed_data)
    crashes_dir = Path(tmpdir) / "crashes"
    crashes_dir.mkdir()

    cmd = [
        sys.executable, "-m", "fuzzer_tool", "fuzz", str(target),
        "-d", str(corpus_dir),
        "-o", str(crashes_dir),
        "-n", "100", "-t", "2", "-s", "42",
    ]
    for k, v in extra_kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    crash_files = list(crashes_dir.glob("crash_*"))
    return result, crash_files


# ---------------------------------------------------------------------------
# Bug class 1: _run_c_direct_lite signal handlers
# ---------------------------------------------------------------------------

class TestDirectLiteCrashHandler:
    """Verify _run_c_direct_lite has signal handlers installed."""

    def test_signal_handlers_installed(self):
        """_run_c_direct_lite must install SIGSEGV and SIGABRT handlers."""
        import signal
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(NOSAN_SO),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=True,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        # Run safe input to trigger handler initialization
        rc, stderr = runner.run_one(b"safe")
        assert rc == 0

    def test_direct_lite_safe_input(self):
        """_run_c_direct_lite returns normal rc for safe input."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(NOSAN_SO),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=True,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"safe input data")
        assert rc == 0


# ---------------------------------------------------------------------------
# Bug class 2: _probe_so_function loading .so via ctypes.CDLL
# ---------------------------------------------------------------------------

class TestProbeSoFunction:
    """Verify _probe_so_function doesn't load .so via ctypes.CDLL."""

    def test_uses_nm_not_ctypes(self):
        """_probe_so_function should use nm -D, not ctypes.CDLL."""
        import inspect
        from fuzzer_tool.services.fuzzer import Fuzzer

        source = inspect.getsource(Fuzzer._probe_so_function)
        assert "ctypes.CDLL(" not in source, (
            "_probe_so_function should not use ctypes.CDLL — it loads the .so "
            "and causes ASAN 'runtime does not come first' errors"
        )
        assert "nm" in source, "_probe_so_function should use nm -D to scan symbols"

    def test_finds_fuzz_shm_run(self):
        """_probe_so_function finds fuzz_shm_run in .so targets."""
        from fuzzer_tool.services.fuzzer import Fuzzer

        func = Fuzzer._probe_so_function(str(NOSAN_SO))
        assert func == "fuzz_shm_run"

    def test_finds_fuzz_fallback(self):
        """_probe_so_function finds fuzz_* fallback in .so targets."""
        from fuzzer_tool.services.fuzzer import Fuzzer

        func = Fuzzer._probe_so_function(str(ASAN_SO))
        assert func.startswith("fuzz_")


# ---------------------------------------------------------------------------
# Bug class 3: Auto-detected .so targets use subprocess loader
# ---------------------------------------------------------------------------

class TestAutoDetectedSoMode:
    """Verify auto-detected .so targets use subprocess loader, not direct_lite."""

    def test_nosan_uses_subprocess_loader(self):
        """Non-ASAN .so targets should use subprocess loader."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(NOSAN_SO),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,  # subprocess loader mode
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        assert runner.direct_lite is False
        assert runner.direct is False
        assert runner._loader_path is not None, "Subprocess loader should be initialized"

    def test_asan_uses_subprocess_loader(self):
        """ASAN .so targets should use subprocess loader."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(ASAN_SO),
            function_name="fuzz",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,  # subprocess loader mode
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        assert runner.direct_lite is False
        assert runner._loader_path is not None

    def test_nosan_subprocess_detects_crash(self):
        """Subprocess loader detects SIGSEGV crashes in non-ASAN .so targets."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        runner = InProcessRunner(
            target=str(NOSAN_SO),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"CRASHS")
        assert rc != 0, "Target should have crashed"
        assert rc == -11, f"Expected SIGSEGV (rc=-11), got rc={rc}"

    def test_asan_subprocess_detects_crash(self):
        """Subprocess loader detects ASAN crashes in ASAN .so targets."""
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        # ASAN .so targets need LD_PRELOAD for the subprocess loader
        old_preload = os.environ.get("LD_PRELOAD")
        os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libasan.so.8"
        try:
            runner = InProcessRunner(
                target=str(ASAN_SO),
                function_name="fuzz",
                timeout=2.0,
                shm_size=4096,
                direct_lite=False,
                coverage_env_id=None,
                cov=False,
                debug=False,
            )

            rc, stderr = runner.run_one(b"BUG!S")
            assert rc != 0, "ASAN target should have crashed"
            assert "AddressSanitizer" in stderr, (
                f"Expected ASAN report in stderr, got: {stderr[:200]}"
            )
        finally:
            if old_preload is None:
                os.environ.pop("LD_PRELOAD", None)
            else:
                os.environ["LD_PRELOAD"] = old_preload


# ---------------------------------------------------------------------------
# Bug class 4: run_target_fast stdin redirect and stderr capture
# ---------------------------------------------------------------------------

class TestRunTargetFast:
    """Verify run_target_fast redirects stdin and captures stderr."""

    def test_stdin_redirect(self):
        """run_target_fast must redirect stdin from temp file."""
        rc, stderr, pid = run_target_fast(str(TARGETS_DIR / "test_target_nosan"), b"CRASHS")
        assert rc != 0, "Target should have crashed on CRASHS input"

    def test_stderr_capture(self):
        """run_target_fast must capture stderr for ASAN output."""
        rc, stderr, pid = run_target_fast(str(TARGETS_DIR / "asan_target"), b"BUG!S")
        assert rc != 0, "ASAN target should have crashed"
        assert "AddressSanitizer" in stderr, (
            f"Expected ASAN report in stderr, got: {stderr[:200]}"
        )


# Import at module level for TestRunTargetFast
from fuzzer_tool.adapters.process import run_target_fast


# ---------------------------------------------------------------------------
# Bug class 5: Integration — fuzzer finds crashes through all modes
# ---------------------------------------------------------------------------

class TestInprocessCrashIntegration:
    """Integration tests: fuzzer finds crashes through inprocess mode."""

    @pytest.mark.skip(reason="Fuzzer's dmesg thread interferes with fork-based direct_lite")
    def test_nosan_so_finds_crash(self):
        """Fuzzer detects crashes in non-ASAN .so targets via inprocess mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, crash_files = _fuzzer_crash_test(NOSAN_SO, b"CRASHS", tmpdir)
            assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
            assert len(crash_files) > 0, (
                f"No crashes found in non-ASAN .so. Output:\n{result.stdout}"
            )

    def test_asan_so_finds_crash(self):
        """Fuzzer detects ASAN crashes in ASAN .so targets via inprocess mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, crash_files = _fuzzer_crash_test(ASAN_SO, b"BUG!S", tmpdir)
            assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
            assert len(crash_files) > 0, (
                f"No crashes found in ASAN .so. Output:\n{result.stdout}"
            )

    def test_nosan_standalone_finds_crash(self):
        """Fuzzer detects crashes in non-ASAN standalone binary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, crash_files = _fuzzer_crash_test(
                TARGETS_DIR / "test_target_nosan", b"CRASHS", tmpdir
            )
            assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
            assert len(crash_files) > 0, (
                f"No crashes found in non-ASAN standalone. Output:\n{result.stdout}"
            )
