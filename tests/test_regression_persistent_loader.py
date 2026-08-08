"""Regression tests: crash detection in persistent loader and subprocess loader.

Covers the bugs fixed on 2026-08-04 where:
- PersistentLoader's grandchild swallowed crashes (caught SIGSEGV as Python
  exception, converted to rc=0, parent never detected the crash)
- The subprocess loader (_LOADER_SCRIPT) had no crash detection at all
  (uncaught SegmentationFault → Python exits with code 1 → parent sees
  rc=1 which is not in SIGNAL_CRASH_CODES)

Both fixes add __afl_guarded_call support so that signal-killed processes
exit with 128+signum, which the parent correctly recognizes as a crash.
"""

import subprocess
import sys
from pathlib import Path

from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES
from tests.conftest import requires_clang

pytestmark = requires_clang

TARGETS_DIR = Path(__file__).parent.parent / "targets"


def _compile_shared(src: Path, out: Path):
    """Compile a C source file to a shared library (.so)."""
    subprocess.run(
        ["clang", "-shared", "-fPIC", "-o", str(out), str(src)],
        check=True,
        capture_output=True,
    )


def _compile_standalone(src: Path, out: Path):
    """Compile a C source file to a standalone executable."""
    subprocess.run(
        ["clang", "-o", str(out), str(src)],
        check=True,
        capture_output=True,
    )


def _make_corpus(tmpdir: Path, seed_data: bytes):
    """Create corpus/seeds/ structure with a seed file."""
    seeds_dir = tmpdir / "seeds"
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "seed").write_bytes(seed_data)
    return tmpdir


def _fuzz_with_loader(target: Path, seed: bytes, tmpdir: Path, *, inprocess: bool = False):
    """Run fuzzer-tool fuzz and return the result."""
    corpus_dir = tmpdir / "corpus"
    crashes_dir = tmpdir / "crashes"
    corpus_dir.mkdir()
    crashes_dir.mkdir()
    _make_corpus(corpus_dir, seed)

    cmd = [
        sys.executable,
        "-m",
        "fuzzer_tool",
        "fuzz",
        str(target),
        "-d",
        str(corpus_dir),
        "-o",
        str(crashes_dir),
        "-n",
        "100",
        "-t",
        "2",
        "-s",
        "42",
    ]
    if inprocess:
        cmd.append("--inprocess")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    crash_files = list(crashes_dir.glob("crash_*"))
    return result, crash_files


class TestPersistentLoaderCrashDetection:
    """Crash detection in the persistent loader (forked grandchild path)."""

    def test_nosan_persistent_detects_segv(self, tmp_path):
        """Persistent loader detects SIGSEGV in non-ASAN .so via crash code."""
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)

        runner = InProcessRunner(
            target=str(nosan_so),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"CRASHS")
        assert rc != 0, f"Expected crash, got rc={rc}"
        assert rc in SIGNAL_CRASH_CODES or rc >= 128, f"Expected crash code, got rc={rc}"

    def test_nosan_persistent_detects_abrt(self, tmp_path):
        """Persistent loader detects SIGABRT in non-ASAN .so via crash code."""
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)

        runner = InProcessRunner(
            target=str(nosan_so),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"CRASHA")
        assert rc != 0, f"Expected crash, got rc={rc}"
        assert rc in SIGNAL_CRASH_CODES or rc >= 128, f"Expected crash code, got rc={rc}"

    def test_nosan_persistent_safe_input_no_crash(self, tmp_path):
        """Persistent loader returns 0 for safe inputs (no crash)."""
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)

        runner = InProcessRunner(
            target=str(nosan_so),
            function_name="fuzz_shm_run",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"SAFEXXX00")
        assert rc == 0, f"Expected success (rc=0), got rc={rc}"


class TestSubprocessLoaderCrashDetection:
    """Crash detection in the subprocess loader (_LOADER_SCRIPT path)."""

    def test_nosan_subprocess_detects_segv(self, tmp_path):
        """Subprocess loader detects SIGSEGV in non-ASAN standalone binary."""
        nosan_bin = tmp_path / "test_nosan"
        _compile_standalone(TARGETS_DIR / "test_target.c", nosan_bin)

        runner = InProcessRunner(
            target=str(nosan_bin),
            function_name="LLVMFuzzerTestOneInput",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"CRASHS")
        assert rc != 0, f"Expected crash, got rc={rc}"
        assert rc in SIGNAL_CRASH_CODES or rc >= 128, f"Expected crash code, got rc={rc}"

    def test_nosan_subprocess_detects_abrt(self, tmp_path):
        """Subprocess loader detects SIGABRT in non-ASAN standalone binary."""
        nosan_bin = tmp_path / "test_nosan"
        _compile_standalone(TARGETS_DIR / "test_target.c", nosan_bin)

        runner = InProcessRunner(
            target=str(nosan_bin),
            function_name="LLVMFuzzerTestOneInput",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"CRASHA")
        assert rc != 0, f"Expected crash, got rc={rc}"
        assert rc in SIGNAL_CRASH_CODES or rc >= 128, f"Expected crash code, got rc={rc}"

    def test_nosan_subprocess_safe_input_no_crash(self, tmp_path):
        """Subprocess loader returns 0 for safe inputs (no crash)."""
        nosan_bin = tmp_path / "test_nosan"
        _compile_standalone(TARGETS_DIR / "test_target.c", nosan_bin)

        runner = InProcessRunner(
            target=str(nosan_bin),
            function_name="LLVMFuzzerTestOneInput",
            timeout=2.0,
            shm_size=4096,
            direct_lite=False,
            coverage_env_id=None,
            cov=False,
            debug=False,
        )

        rc, stderr = runner.run_one(b"SAFEXXX00")
        assert rc == 0, f"Expected success (rc=0), got rc={rc}"


class TestIntegrationCrashDetection:
    """Integration tests: fuzzer finds crashes through persistent and subprocess loaders."""

    def test_nosan_inprocess_finds_crash(self, tmp_path):
        """Fuzzer detects crashes in non-ASAN .so targets via inprocess mode."""
        nosan_so = tmp_path / "test_nosan.so"
        _compile_shared(TARGETS_DIR / "test_target.c", nosan_so)

        result, crash_files = _fuzz_with_loader(nosan_so, b"CRASHS", tmp_path, inprocess=True)
        assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
        assert len(crash_files) > 0, (
            f"No crashes found in non-ASAN .so via inprocess mode. Output:\n"
            f"{result.stdout}\n{result.stderr}"
        )

    def test_nosan_subprocess_finds_crash(self, tmp_path):
        """Fuzzer detects crashes in non-ASAN standalone binary via subprocess loader."""
        nosan_bin = tmp_path / "test_nosan"
        _compile_standalone(TARGETS_DIR / "test_target.c", nosan_bin)

        result, crash_files = _fuzz_with_loader(nosan_bin, b"CRASHS", tmp_path)
        assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
        assert len(crash_files) > 0, (
            f"No crashes found in standalone binary via subprocess loader. Output:\n"
            f"{result.stdout}\n{result.stderr}"
        )
