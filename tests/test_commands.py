"""Unit tests for cli/commands.py — CLI utility functions."""

import argparse
from pathlib import Path

import pytest

from fuzzer_tool.cli.commands import _get_dirs, _validate_target


def _ns(target, corpus=None, crashes=None):
    """The two fields _get_dirs reads, built directly.

    These used to be produced by parsing argv through `_add_common_args`,
    a helper with no production call sites at all — so the tests exercised
    a parser no user could reach, and its `-c` default was the obvious
    place to edit when flipping coverage on. Editing it would have changed
    nothing and turned this file red, which is the wrong signal twice over.
    Building the namespace directly tests `_get_dirs` and nothing else.
    """
    return argparse.Namespace(target=target, corpus=corpus, crashes=crashes)


class TestGetDirs:
    def test_defaults_use_target_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        args = _ns(str(tmp_path / "my_target"))
        corpus, crashes = _get_dirs(args, str(tmp_path / "my_target"))
        assert "my_target" in corpus
        assert "my_target" in crashes
        assert corpus.endswith("corpus")
        assert crashes.endswith("crashes")

    def test_custom_corpus_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        custom = str(tmp_path / "custom_corpus")
        args = _ns(str(tmp_path / "target"), corpus=custom)
        corpus, _crashes = _get_dirs(args, str(tmp_path / "target"))
        assert corpus == custom

    def test_custom_crashes_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        custom = str(tmp_path / "custom_crashes")
        args = _ns(str(tmp_path / "target"), crashes=custom)
        _corpus, crashes = _get_dirs(args, str(tmp_path / "target"))
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
