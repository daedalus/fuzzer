"""`fuzz` budgets by --max-execs only; -n/--iterations is gone."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.cli.commands import _run_fuzzer, main


def _parse(monkeypatch, tmp_path, *flags):
    """Run main() with fuzz flags; return the Namespace cmd_fuzz received."""
    target = tmp_path / "target"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)

    mock_fuzz = MagicMock(return_value=0)
    monkeypatch.setattr(commands, "cmd_fuzz", mock_fuzz)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", str(target), *flags])
    main()
    return mock_fuzz.call_args.args[0]


@pytest.mark.parametrize("flag", ["-n", "--iterations"])
def test_regression_iterations_flag_rejected(monkeypatch, tmp_path, flag):
    """Falsification: the removed flag must fail parsing."""
    with pytest.raises(SystemExit) as exc:
        _parse(monkeypatch, tmp_path, flag, "100")

    assert exc.value.code == 2


def test_regression_max_execs_parsed(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, "--max-execs", "100")

    assert args.max_execs == 100
    assert not hasattr(args, "iterations")


def test_regression_run_gets_only_budget():
    """Adversarial: a stray iterations attr must not reach Fuzzer.run."""
    seen = {}
    fuzzer = SimpleNamespace(run=lambda **kw: seen.update(kw))

    _run_fuzzer(fuzzer, SimpleNamespace(iterations=7, max_execs=42))

    assert seen == {"max_execs": 42}
