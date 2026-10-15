"""Regressions for the `-M` (mutations per input) default.

`-M 8` stacked eight operators per input. Fuzzgoat, clang ASAN, 20 paired
seeds x 20k execs: `-M 1` found +94.5 median edges (19/20 seeds) at +73%
exec/s; the `-M 8` vs `-M 8` control showed no difference (p=0.50). See
docs/DEEP_DIVE.md "-M default".
"""

from __future__ import annotations

import inspect
import sys

from fuzzer_tool.services.fuzzer import MUTATIONS_DEFAULT, Fuzzer

EXPLICIT_M = 8


def _parsed_mutations(monkeypatch, argv: list[str]) -> int:
    """`args.mutations` from the shipped parser (main() runs, cmd_fuzz is spied)."""
    from fuzzer_tool.cli import commands

    captured: dict[str, int] = {}

    def _spy(args):
        captured["mutations"] = args.mutations
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    assert commands.main() == 0
    return captured["mutations"]


def test_regression_cli_default_is_one(monkeypatch):
    """Falsification: the shipped parser defaults to one operator per input."""
    assert MUTATIONS_DEFAULT == 1
    assert _parsed_mutations(monkeypatch, []) == MUTATIONS_DEFAULT


def test_explicit_m_still_honoured(monkeypatch):
    """Adversarial: an explicit -M overrides the default."""
    assert _parsed_mutations(monkeypatch, ["-M", str(EXPLICIT_M)]) == EXPLICIT_M


def test_fuzzer_ctor_default_matches_cli():
    """Adversarial: library callers get the same default as the CLI."""
    param = inspect.signature(Fuzzer.__init__).parameters["mutations_per_input"]
    assert param.default == MUTATIONS_DEFAULT
