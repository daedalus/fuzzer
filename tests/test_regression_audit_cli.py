"""Regressions from the 2026-11-13 audit: CLI exit codes, flag wiring, arg bounds."""

import sys
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.cli.commands import cmd_fuzz, cmd_import, main
from tests import test_commands_extended as ext

_USAGE_ERROR = 2


@pytest.fixture
def fuzz_args(tmp_path):
    return ext.TestCmdFuzzConstruction()._make_default_args(tmp_path)


class TestAbortExitCode:
    """An aborted campaign printed "aborted by an unexpected error" and exited 0."""

    def _run(self, monkeypatch, fuzz_args, aborted):
        fuzzer = MagicMock()
        fuzzer.aborted = aborted
        monkeypatch.setattr(commands, "Fuzzer", lambda **_kw: fuzzer)
        return cmd_fuzz(fuzz_args)

    def test_regression_aborted_run_exits_nonzero(self, monkeypatch, fuzz_args):
        assert self._run(monkeypatch, fuzz_args, True) == 1

    def test_clean_run_exits_zero(self, monkeypatch, fuzz_args):
        """Falsification: a normal stop keeps exit code 0."""
        assert self._run(monkeypatch, fuzz_args, False) == 0

    def test_aborted_run_skips_ledger(self, monkeypatch, fuzz_args):
        """Adversarial: a truncated campaign must not land in the ledger as finished."""
        fuzz_args.ledger_preset = "x"
        recorded = []
        monkeypatch.setattr(commands, "_record_ledger", lambda *a: recorded.append(a))

        assert self._run(monkeypatch, fuzz_args, True) == 1
        assert recorded == []


class TestDifferentialWired:
    """``--differential TARGET_B`` was parsed and never passed to ``Fuzzer``."""

    def _kwargs(self, monkeypatch, fuzz_args):
        seen = {}

        def build(**kwargs):
            seen.update(kwargs)
            fuzzer = MagicMock()
            fuzzer.aborted = False
            return fuzzer

        monkeypatch.setattr(commands, "Fuzzer", build)
        rc = cmd_fuzz(fuzz_args)
        return rc, seen

    def test_regression_differential_reaches_fuzzer(self, monkeypatch, fuzz_args):
        fuzz_args.differential = fuzz_args.target
        rc, seen = self._kwargs(monkeypatch, fuzz_args)

        assert rc == 0
        assert seen["differential_target"] == fuzz_args.target

    def test_no_flag_no_differential(self, monkeypatch, fuzz_args):
        """Falsification: without the flag the fuzzer gets no second target."""
        _rc, seen = self._kwargs(monkeypatch, fuzz_args)

        assert seen.get("differential_target") is None

    def test_missing_second_target_is_an_error(self, monkeypatch, fuzz_args, tmp_path):
        """Adversarial: a typo'd TARGET_B fails up front, not silently."""
        fuzz_args.differential = str(tmp_path / "nope")

        with pytest.raises(SystemExit):
            self._kwargs(monkeypatch, fuzz_args)


class TestArgBounds:
    """``-t -1`` made every exec a timeout; ``-m 0`` / ``-m -1`` were accepted."""

    def _main(self, monkeypatch, tmp_path, *flags):
        target = tmp_path / "t"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)
        called = MagicMock(return_value=0)
        monkeypatch.setattr(commands, "cmd_fuzz", called)
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", str(target), *flags])
        return main, called

    @pytest.mark.parametrize(
        "flags", [("-t", "-1"), ("-t", "0"), ("-m", "0"), ("-m", "-1"), ("-t", "nan")]
    )
    def test_regression_rejected(self, monkeypatch, tmp_path, flags):
        run, called = self._main(monkeypatch, tmp_path, *flags)

        with pytest.raises(SystemExit) as exc:
            run()

        assert exc.value.code == _USAGE_ERROR
        called.assert_not_called()

    def test_valid_values_accepted(self, monkeypatch, tmp_path):
        """Falsification: fractional timeouts and a 1-byte cap still parse."""
        run, called = self._main(monkeypatch, tmp_path, "-t", "0.05", "-m", "1")
        run()

        args = called.call_args[0][0]
        assert (args.timeout, args.max_len) == (0.05, 1)


class TestMissingDirExitCode:
    """``[-] ... not found`` was followed by a success line and exit 0."""

    def test_regression_import_missing_source(self, tmp_path):
        import argparse

        args = argparse.Namespace(
            source_dir=str(tmp_path / "nope"),
            format="afl",
            corpus=str(tmp_path / "c"),
            crashes=str(tmp_path / "x"),
            autotokens=False,
        )

        assert cmd_import(args) == 1

    def test_regression_minimize_missing_corpus(self, tmp_path, monkeypatch):
        target = tmp_path / "t"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)
        monkeypatch.setattr(
            sys, "argv", ["fuzzer-tool", "minimize", str(target), "-d", str(tmp_path / "nope")]
        )

        assert main() == 1
