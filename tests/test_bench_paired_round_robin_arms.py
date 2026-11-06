"""Bench arms for the coupon-collector first-pass A/B (`--round-robin`).

Same contract as ``test_bench_paired_arms.py``: flags reach the real fuzz
parser; the arm is baseline plus one knob; the control is baseline itself.
"""

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


def test_falsify_arm_sets_only_round_robin(monkeypatch):
    """The arm turns on round_robin; the baseline it is scored against does not."""
    assert ARM_BASELINES["round-robin"] == "baseline"
    assert _parse(monkeypatch, ARMS["round-robin"]).round_robin is True
    assert not _parse(monkeypatch, ARMS["baseline"]).round_robin


def test_adversarial_control_is_baseline_exactly():
    """A/A control (Rule 46): any extra flag would make it a second arm."""
    assert ARMS["round-robin-ctl"] == ARMS["baseline"]
    assert ARM_BASELINES["round-robin-ctl"] == "baseline"
