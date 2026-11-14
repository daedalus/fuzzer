"""Bench arms for the GA crossover A/B (`--ga-crossover-rate 0`)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402


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


def test_falsify_arm_changes_only_crossover(monkeypatch):
    base = vars(_parse(monkeypatch, ARMS["ga"]))
    arm = vars(_parse(monkeypatch, ARMS["ga-no-crossover"]))
    diff = {k for k in base if base[k] != arm[k]} - {"func"}  # func: bound per parse
    assert ARM_BASELINES["ga-no-crossover"] == "ga"
    assert base["ga"] is True
    assert diff == {"ga_crossover_rate"}
    assert arm["ga_crossover_rate"] == 0.0


def test_adversarial_control_is_ga_exactly():
    """A/A control (Rule 46): any extra flag would make it a second arm."""
    assert ARMS["ga-ctl"] == ARMS["ga"]
    assert ARM_BASELINES["ga-ctl"] == "ga"
