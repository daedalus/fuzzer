"""Unit tests for cli/commands.py — CLI utility functions."""

import argparse
import os
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli.commands import _add_common_args, _get_dirs, _validate_target


class TestGetDirs:
    def test_defaults_use_target_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        parser = argparse.ArgumentParser()
        _add_common_args(parser)
        args = parser.parse_args([str(tmp_path / "my_target")])
        corpus, crashes = _get_dirs(args, str(tmp_path / "my_target"))
        assert "my_target" in corpus
        assert "my_target" in crashes
        assert corpus.endswith("corpus")
        assert crashes.endswith("crashes")

    def test_custom_corpus_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        parser = argparse.ArgumentParser()
        _add_common_args(parser)
        custom = str(tmp_path / "custom_corpus")
        args = parser.parse_args(["-d", custom, str(tmp_path / "target")])
        corpus, crashes = _get_dirs(args, str(tmp_path / "target"))
        assert corpus == custom

    def test_custom_crashes_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        parser = argparse.ArgumentParser()
        _add_common_args(parser)
        custom = str(tmp_path / "custom_crashes")
        args = parser.parse_args(["-o", custom, str(tmp_path / "target")])
        corpus, crashes = _get_dirs(args, str(tmp_path / "target"))
        assert crashes == custom


class TestValidateTarget:
    def test_missing_target_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _validate_target("/nonexistent/binary")
        assert exc_info.value.code == 1

    def test_not_executable_exits(self, tmp_path, capsys):
        f = tmp_path / "not_exec"
        f.write_bytes(b"data")
        f.chmod(0o644)
        with pytest.raises(SystemExit) as exc_info:
            _validate_target(str(f))
        assert exc_info.value.code == 1

    def test_valid_target_no_exit(self, tmp_path):
        f = tmp_path / "valid_target"
        f.write_bytes(b"\x7fELF")
        f.chmod(0o755)
        _validate_target(str(f))  # should not raise


class TestAddCommonArgs:
    def test_adds_all_args(self):
        parser = argparse.ArgumentParser()
        _add_common_args(parser)
        args = parser.parse_args(
            [
                "/target",
                "-d",
                "/corpus",
                "-o",
                "/crashes",
                "-t",
                "10",
                "-F",
                "-A",
                "arg1",
                "arg2",
                "-c",
            ]
        )
        assert args.target == "/target"
        assert args.corpus == "/corpus"
        assert args.crashes == "/crashes"
        assert args.timeout == 10.0
        assert args.file_mode is True
        assert args.target_args == ["arg1", "arg2"]
        assert args.coverage is True

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        _add_common_args(parser)
        args = parser.parse_args(["/target"])
        assert args.corpus is None
        assert args.crashes is None
        assert args.timeout == 1
        assert args.file_mode is False
        assert args.target_args is None
        assert args.coverage is False
