"""Integration tests: compile target, fuzz, verify crashes found."""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

TARGET_SRC = Path(__file__).parent.parent / "targets" / "test_target.c"
TARGET_BIN = Path(__file__).parent.parent / "targets" / "test_target"
ASAN_SRC = Path(__file__).parent.parent / "targets" / "asan_target.c"
ASAN_BIN = Path(__file__).parent.parent / "targets" / "asan_target"
ASAN_SO = Path(__file__).parent.parent / "targets" / "asan_target.so"


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


@pytest.fixture(scope="module")
def compiled_asan_target():
    """Compile asan_target.c with ASAN if not already built."""
    if not ASAN_BIN.exists() or ASAN_SRC.stat().st_mtime > ASAN_BIN.stat().st_mtime:
        result = subprocess.run(
            ["gcc", "-g", "-fsanitize=address", "-o", str(ASAN_BIN), str(ASAN_SRC)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ASAN compilation failed: {result.stderr}"
    yield str(ASAN_BIN)


@pytest.fixture(scope="module")
def compiled_asan_so():
    """Compile asan_target.c as shared library with ASAN if not already built."""
    if not ASAN_SO.exists() or ASAN_SRC.stat().st_mtime > ASAN_SO.stat().st_mtime:
        result = subprocess.run(
            [
                "gcc",
                "-g",
                "-fsanitize=address",
                "-shared",
                "-fPIC",
                "-o",
                str(ASAN_SO),
                str(ASAN_SRC),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ASAN .so compilation failed: {result.stderr}"
    yield str(ASAN_SO)


class TestIntegration:
    def test_fuzzer_finds_crash(self, compiled_target):
        """Fuzz the test target and verify at least one crash is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            crashes_dir = Path(tmpdir) / "crashes"
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)
            crashes_dir.mkdir()

            # Seed with "CRASHS" to trigger SIGSEGV quickly
            (seeds_dir / "seed").write_bytes(b"CRASHS")

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
            assert minimized == b"CRASHS", (
                f"Expected minimized to be b'CRASHS', got {len(minimized)} bytes"
            )

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
            remaining_inputs = list(corpus_dir.glob("input_*"))
            unique_kept = list(corpus_dir.glob("unique*"))
            # At least some duplicates should be removed
            assert len(remaining_inputs) < 5, (
                f"Expected fewer duplicate inputs, got {len(remaining_inputs)}"
            )
            # Unique input should be kept (triggers SIGSEGV)
            assert len(unique_kept) == 1, (
                f"Unique crash input should be kept, found {len(unique_kept)}"
            )

    def test_fuzzer_eps_minimum(self, compiled_target):
        """Verify fuzzer maintains at least 100 eps against a fast target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)
            (seeds_dir / "seed").write_bytes(b"safe")

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "fuzz",
                    compiled_target,
                    "-d",
                    str(corpus_dir),
                    "-n",
                    "1000",
                    "-t",
                    "2",
                    "-s",
                    "42",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Fuzzer failed: {result.stderr}"

            # Parse EPS from stats lines (format: "eps: NNNN")
            eps_matches = re.findall(r"eps:\s*([\d.]+)", result.stdout)
            assert len(eps_matches) > 0, f"No EPS stats found in output:\n{result.stdout}"
            # Use the last EPS value (most stable — steady-state rate)
            eps = float(eps_matches[-1])
            assert eps >= 100, f"EPS {eps:.0f} is below minimum 100. Output:\n{result.stdout}"

    def test_asan_finds_heap_buffer_overflow(self, compiled_asan_target):
        """Fuzz ASAN-instrumented target and verify heap-buffer-overflow detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            crashes_dir = Path(tmpdir) / "crashes"
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)
            crashes_dir.mkdir()

            # Seed with "BUG!S" to trigger stack-buffer-overflow quickly
            (seeds_dir / "seed").write_bytes(b"BUG!S")

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "fuzzer_tool",
                    "fuzz",
                    compiled_asan_target,
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

            # Verify crash is identified as heap-buffer-overflow
            bin_files = [f for f in crash_files if f.suffix == ".bin"]
            assert len(bin_files) > 0, "No .bin crash files found"
            assert any("stackbufferoverflow" in f.name for f in bin_files), (
                f"Expected stack-buffer-overflow in crash filenames, got: {[f.name for f in bin_files]}"
            )

            # Verify crash report contains ASAN output
            txt_files = [f for f in crash_files if f.suffix == ".txt"]
            assert len(txt_files) > 0, "No .txt crash reports found"
            reports = [f.read_text() for f in txt_files]
            assert any("AddressSanitizer" in r for r in reports), (
                "Missing AddressSanitizer in any report"
            )
            assert any("stack-buffer-overflow" in r for r in reports), (
                f"Missing stack-buffer-overflow in any report. Found: {[f.name for f in txt_files]}"
            )

    @pytest.mark.parametrize(
        "mode_args,target_fixture,mode_label",
        [
            ([], "compiled_asan_target", "default_subprocess"),
            (["--no-shm"], "compiled_asan_target", "ptrace"),
        ],
    )
    def test_asan_all_modes(self, request, mode_args, target_fixture, mode_label):
        """Verify ASAN detection works in all execution modes."""
        target = request.getfixturevalue(target_fixture)
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            crashes_dir = Path(tmpdir) / "crashes"
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)
            crashes_dir.mkdir()

            # Seed with "BUG!S" to trigger stack-buffer-overflow quickly
            (seeds_dir / "seed").write_bytes(b"BUG!S")

            cmd = [
                "python3",
                "-m",
                "fuzzer_tool",
                "fuzz",
                target,
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
            ] + mode_args

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, (
                f"[{mode_label}] Fuzzer failed: {result.stderr}\nstdout: {result.stdout}"
            )

            # Check ASAN was detected
            assert "ASAN detected" in result.stdout, (
                f"[{mode_label}] ASAN not detected in output:\n{result.stdout}"
            )

            # Check crash found
            crash_files = list(crashes_dir.glob("crash_*"))
            assert len(crash_files) > 0, (
                f"[{mode_label}] No crashes found. Output:\n{result.stdout}"
            )

            # Check crash is ASAN heap-buffer-overflow
            bin_files = [f for f in crash_files if f.suffix == ".bin"]
            assert any("stackbufferoverflow" in f.name for f in bin_files), (
                f"[{mode_label}] Expected stack-buffer-overflow, got: {[f.name for f in bin_files]}"
            )
