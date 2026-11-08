"""`--clock virtual` is the run default: a `--seed` run replays exactly.

`Clock()` / `WALL_CLOCK` stay wall: they are the fallback for partial
fuzzers (`test_clock.py::test_default_is_wall`), not the run default.
"""

from __future__ import annotations

import inspect
import sys

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.clock import ClockMode
from fuzzer_tool.services.fuzzer import Fuzzer


def _parse(monkeypatch, flags: list[str]):
    """Run the real ``main()`` parser and return the Namespace it built."""
    captured: dict[str, object] = {}

    def _spy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    assert commands.main() == 0
    return captured["args"]


def test_cli_default_is_virtual(monkeypatch):
    """No `--clock` flag: the parsed mode is virtual."""
    assert ClockMode(_parse(monkeypatch, []).clock) is ClockMode.VIRTUAL


def test_fuzzer_default_is_virtual():
    """Programmatic callers (benchmarks, tools) get the same default as the CLI."""
    default = inspect.signature(Fuzzer.__init__).parameters["clock"].default
    assert default is ClockMode.VIRTUAL


def test_falsify_wall_opt_out(monkeypatch):
    """`--clock wall` still reaches the host clock: the default is not forced."""
    assert ClockMode(_parse(monkeypatch, ["--clock", "wall"]).clock) is ClockMode.WALL


def test_adversarial_rejects_unknown_mode(monkeypatch):
    """A misspelt mode fails loudly instead of silently picking a default."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--clock", "Virtual"])
