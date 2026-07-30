"""Tests for the 'sweep' subcommand — linearly scan corpus for missed crashes.

The sweep subcommand loads every seed from the corpus, runs it through the
target (no mutations/scheduler/coverage), and reports any that crash.
"""

import subprocess
import sys
from pathlib import Path

import pytest

TARGETS_DIR = Path(__file__).parent.parent / "targets"


@pytest.fixture(scope="module")
def compiled_test_binary(tmp_path_factory):
    """Compile test_target.c as a standalone binary."""
    tmpdir = tmp_path_factory.mktemp("binaries")
    out = tmpdir / "test_target"
    subprocess.run(
        ["gcc", "-o", str(out), str(TARGETS_DIR / "test_target.c")],
        check=True,
        capture_output=True,
    )
    return out


def _make_corpus(tmpdir, seeds_dict):
    """Create a corpus dir with seeds from a {filename: bytes} dict."""
    corpus_dir = Path(tmpdir) / "corpus"
    seeds_dir = corpus_dir / "seeds"
    seeds_dir.mkdir(parents=True)
    for name, data in seeds_dict.items():
        (seeds_dir / name).write_bytes(data)
    return corpus_dir


class TestSweepCommand:
    """Verify the sweep subcommand finds crashes in corpus seeds."""

    def test_finds_crash_in_corpus(self, compiled_test_binary, tmp_path):
        """Sweep detects a CRASHS seed in the corpus and saves the crash."""
        corpus_dir = _make_corpus(
            tmp_path,
            {
                "safe1": b"hello world",
                "crash1": b"CRASHS",
                "safe2": b"abcdef",
            },
        )
        crashes_dir = tmp_path / "crashes"
        crashes_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fuzzer_tool",
                "sweep",
                str(compiled_test_binary),
                "-d",
                str(corpus_dir),
                "-o",
                str(crashes_dir),
                "-t",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        crash_files = list(crashes_dir.iterdir())
        assert result.returncode == 0, (
            f"Sweep failed with rc={result.returncode}. "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert len(crash_files) >= 1, f"Expected at least 1 crash file. stdout:\n{result.stdout}"
        assert "3 seeds" in result.stdout or "3 seeds" in result.stderr or "3" in result.stdout, (
            f"Expected mention of 3 seeds. stdout:\n{result.stdout}"
        )
        # Verify the crashing seed data was saved
        crash_data = b"".join(cf.read_bytes() for cf in crash_files)
        assert b"CRASHS" in crash_data, (
            f"CRASHS data not found in crash files. stdout:\n{result.stdout}"
        )

    def test_no_crash_clean_corpus(self, compiled_test_binary, tmp_path):
        """Sweep on a corpus with no crashing seeds reports 0 crashes."""
        corpus_dir = _make_corpus(
            tmp_path,
            {
                "safe1": b"hello world",
                "safe2": b"abcdef",
                "safe3": b"1234567890",
            },
        )
        crashes_dir = tmp_path / "crashes"
        crashes_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fuzzer_tool",
                "sweep",
                str(compiled_test_binary),
                "-d",
                str(corpus_dir),
                "-o",
                str(crashes_dir),
                "-t",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0
        # No crash files should be created
        crash_files = list(crashes_dir.iterdir())
        assert len(crash_files) == 0, f"Expected no crash files, got {len(crash_files)}"

    def test_empty_corpus(self, compiled_test_binary, tmp_path):
        """Sweep on an empty corpus reports no seeds and exits cleanly."""
        corpus_dir = tmp_path / "empty_corpus"
        corpus_dir.mkdir()
        seeds_dir = corpus_dir / "seeds"
        seeds_dir.mkdir()
        crashes_dir = tmp_path / "crashes"
        crashes_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fuzzer_tool",
                "sweep",
                str(compiled_test_binary),
                "-d",
                str(corpus_dir),
                "-o",
                str(crashes_dir),
                "-t",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0
        assert "No seeds" in result.stdout

    def test_missing_corpus_exits_with_error(self, compiled_test_binary, tmp_path):
        """Sweep on a nonexistent corpus dir reports error and exits non-zero."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fuzzer_tool",
                "sweep",
                str(compiled_test_binary),
                "-d",
                str(tmp_path / "nonexistent"),
                "-t",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode != 0, "Should exit non-zero for missing corpus"
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()
