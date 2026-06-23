"""Integration tests: compile target, fuzz, verify crashes found."""

import subprocess
import tempfile
from pathlib import Path

import pytest

TARGET_SRC = Path(__file__).parent.parent / "targets" / "test_target.c"
TARGET_BIN = Path(__file__).parent.parent / "targets" / "test_target"


@pytest.fixture(scope="module")
def compiled_target():
    """Compile test_target.c if not already built."""
    if not TARGET_BIN.exists() or TARGET_SRC.stat().st_mtime > TARGET_BIN.stat().st_mtime:
        result = subprocess.run(
            ["gcc", "-g", "-o", str(TARGET_BIN), str(TARGET_SRC)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"
    yield str(TARGET_BIN)


class TestIntegration:
    def test_fuzzer_finds_crash(self, compiled_target):
        """Fuzz the test target and verify at least one crash is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            crashes_dir = Path(tmpdir) / "crashes"
            corpus_dir.mkdir()
            crashes_dir.mkdir()

            # Seed with "CRASHS" to trigger SIGSEGV quickly
            (corpus_dir / "seed").write_bytes(b"CRASHS")

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "fuzz",
                    compiled_target,
                    "-d",
                    str(corpus_dir),
                    "-o",
                    str(crashes_dir),
                    "-n",
                    "500",
                    "-t",
                    "2",
                    "-s",
                    "42",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"
            crash_files = list(crashes_dir.glob("crash_*"))
            assert len(crash_files) > 0, (
                f"No crashes found in {crashes_dir}. Output:\n{result.stdout}"
            )

    def test_tmin_minimizes_crash(self, compiled_target):
        """Minimize a crash input and verify it gets smaller."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a crash input
            crash_file = Path(tmpdir) / "crash_input"
            crash_file.write_bytes(b"CRASHS" + b"\x00" * 200)

            output_file = Path(tmpdir) / "minimized"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "tmin",
                    compiled_target,
                    str(crash_file),
                    "-t",
                    "2",
                    "-O",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"tmin failed: {result.stderr}"
            assert output_file.exists(), "Minimized output not created"
            minimized = output_file.read_bytes()
            assert len(minimized) < 206, f"Expected reduction: {len(minimized)} >= 206"

    def test_replay_crash(self, compiled_target):
        """Replay a crash input and verify it reproduces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crash_file = Path(tmpdir) / "crash_input"
            crash_file.write_bytes(b"CRASHS")

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "replay",
                    compiled_target,
                    str(crash_file),
                    "-t",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            assert result.returncode == 0, f"replay failed: {result.stderr}"
            assert "Crash reproduced" in result.stdout

    def test_minimize_corpus(self, compiled_target):
        """Minimize a corpus and verify redundant entries are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            corpus_dir.mkdir()

            # Create duplicate inputs (same content)
            for i in range(5):
                (corpus_dir / f"input_{i}").write_bytes(b"safe input")

            # Create one unique input
            (corpus_dir / "unique").write_bytes(b"CRASHS")

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "minimize",
                    compiled_target,
                    "-d",
                    str(corpus_dir),
                    "-t",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"minimize failed: {result.stderr}"
            remaining = list(corpus_dir.glob("input_*"))
            # At least some duplicates should be removed
            assert len(remaining) < 5, f"Expected fewer inputs, got {len(remaining)}"
