"""Tests for cli/commands.py — helper functions and argument parsing."""

from unittest.mock import MagicMock, patch

from fuzzer_tool.cli.commands import _auto_tune_timeout, _get_dirs, _validate_target


class TestGetDirs:
    def test_default_dirs(self):
        args = MagicMock(corpus=None, crashes=None)
        corpus, crashes = _get_dirs(args, "/usr/bin/ls")
        assert "ls" in corpus
        assert "ls" in crashes
        assert "fuzzing" in corpus

    def test_custom_dirs(self):
        args = MagicMock(corpus="/custom/corpus", crashes="/custom/crashes")
        corpus, crashes = _get_dirs(args, "/usr/bin/ls")
        assert corpus == "/custom/corpus"
        assert crashes == "/custom/crashes"

    def test_partial_custom(self):
        args = MagicMock(corpus="/custom/corpus", crashes=None)
        corpus, crashes = _get_dirs(args, "/usr/bin/ls")
        assert corpus == "/custom/corpus"
        assert "fuzzing" in crashes


class TestValidateTarget:
    def test_valid_target(self):
        _validate_target("/bin/true")  # should not raise

    def test_nonexistent_target(self):
        import pytest
        with pytest.raises(SystemExit):
            _validate_target("/nonexistent/binary")

    def test_not_executable(self, tmp_path):
        f = tmp_path / "not_exec"
        f.write_bytes(b"#!/bin/sh\n")
        f.chmod(0o644)
        import pytest
        with pytest.raises(SystemExit):
            _validate_target(str(f))


class TestAutoTuneTimeout:
    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_returns_positive(self, mock_run):
        mock_run.return_value = (0, "", 1)
        result = _auto_tune_timeout("/bin/true", runs=3)
        assert result > 0

    def test_file_mode(self, tmp_path):
        result = _auto_tune_timeout("/bin/true", file_mode=True, runs=2)
        assert result > 0
