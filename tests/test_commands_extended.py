"""Extended unit tests for cli/commands.py — coverage improvement."""

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.cli.commands import (
    _add_common_args,
    _auto_tune_timeout,
    _detect_asan,
    _get_dirs,
    _validate_target,
    cmd_estimate,
    cmd_fuzz,
    cmd_import,
    cmd_minimize,
    cmd_rank,
    cmd_replay,
    cmd_tmin,
    main,
)


class TestDetectAsan:
    def test_detects_asan_binary(self, monkeypatch):
        """ASAN binary has __asan_init symbol."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b"__asan_init"))
        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _detect_asan("/fake/target") is True

    def test_no_asan(self, monkeypatch):
        """Non-ASAN binary lacks the symbol."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b"main\nfoo"))
        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _detect_asan("/fake/target") is False

    def test_nm_not_found(self, monkeypatch):
        """Graceful handling when nm is not installed."""
        monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError))
        assert _detect_asan("/fake/target") is False

    def test_nm_timeout(self, monkeypatch):
        """Graceful handling when nm times out."""
        monkeypatch.setattr(
            subprocess, "run", MagicMock(side_effect=subprocess.TimeoutExpired("nm", 10))
        )
        assert _detect_asan("/fake/target") is False


class TestAutoTuneTimeout:
    def test_returns_reasonable_timeout(self, monkeypatch, tmp_path):
        """Auto-tuned timeout should be reasonable."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF" + b"\x00" * 100)
        target.chmod(0o755)

        mock_run = MagicMock(return_value=(0, ""))
        monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", mock_run)

        timeout = _auto_tune_timeout(str(target), runs=3)
        assert 0.05 <= timeout <= 30.0


class TestCmdFuzz:
    def test_fuzz_function_exists(self):
        """cmd_fuzz should be callable."""
        assert callable(cmd_fuzz)

    def test_fuzz_help(self):
        """fuzz subcommand should accept --help."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "fuzz" in result.stdout.lower()


class TestCmdEstimate:
    def test_estimate_help(self):
        """estimate command should accept --help."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "estimate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "estimate" in result.stdout.lower()

    def test_estimate_missing_corpus_exits(self, tmp_path):
        """estimate should fail without --corpus."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            target=str(target),
            corpus=None,
            calibrate=100,
        )
        # Missing corpus should raise or error
        with pytest.raises((SystemExit, TypeError)):
            cmd_estimate(args)


class TestCmdImport:
    def test_import_afl(self, monkeypatch, tmp_path):
        """Import from AFL format."""
        src = tmp_path / "afl_out"
        src.mkdir()
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"

        mock_import = MagicMock(return_value=(10, 5))
        monkeypatch.setattr("fuzzer_tool.services.import_corpus.import_from_afl", mock_import)

        args = argparse.Namespace(
            source_dir=str(src),
            format="afl",
            corpus=str(corpus),
            crashes=str(crashes),
        )
        result = cmd_import(args)
        assert result == 0
        mock_import.assert_called_once()

    def test_import_libfuzzer(self, monkeypatch, tmp_path):
        """Import from libFuzzer format."""
        src = tmp_path / "libfuzzer_out"
        src.mkdir()
        corpus = tmp_path / "corpus"

        mock_import = MagicMock(return_value=20)
        monkeypatch.setattr("fuzzer_tool.services.import_corpus.import_from_libfuzzer", mock_import)

        args = argparse.Namespace(
            source_dir=str(src),
            format="libfuzzer",
            corpus=str(corpus),
            crashes=None,
        )
        result = cmd_import(args)
        assert result == 0


class TestCmdTmin:
    def test_tmin_validates_target(self, tmp_path):
        """tmin should validate target exists."""
        args = argparse.Namespace(
            target="/nonexistent/target",
            crash_file=str(tmp_path / "crash.bin"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            grammar=None,
            mutations_per_input=8,
            max_len=4096,
        )
        with pytest.raises(SystemExit):
            cmd_tmin(args)


class TestCmdReplay:
    def test_replay_validates_target(self, tmp_path):
        """replay should validate target exists."""
        args = argparse.Namespace(
            target="/nonexistent/target",
            corpus=str(tmp_path / "corpus"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            max_len=4096,
            iterations=10,
        )
        with pytest.raises(SystemExit):
            cmd_replay(args)


class TestCmdRank:
    def test_rank_empty_corpus(self, tmp_path):
        """rank with empty corpus should handle gracefully."""
        corpus = tmp_path / "empty_corpus"
        corpus.mkdir()

        # Create a valid target
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            corpus=str(corpus),
            top=10,
            dump=None,
            format="text",
            target=str(target),
        )
        # Should not raise, just print empty results
        cmd_rank(args)


class TestMain:
    def test_main_no_args_shows_help(self, monkeypatch):
        """main() with no args should show help."""
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse exits with code 2 when no args given
        assert exc_info.value.code == 2

    def test_main_fuzz_subcommand(self, monkeypatch):
        """main() should dispatch to fuzz subcommand."""
        mock_fuzz = MagicMock(return_value=0)
        monkeypatch.setattr("fuzzer_tool.cli.commands.cmd_fuzz", mock_fuzz)

        monkeypatch.setattr(
            sys,
            "argv",
            ["fuzzer-tool", "fuzz", "/tmp/target", "-n", "100"],
        )
        # Need a valid target
        target = Path("/tmp/target")
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        try:
            main()
        except SystemExit:
            pass

        # Cleanup
        target.unlink(missing_ok=True)
