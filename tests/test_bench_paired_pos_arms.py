"""Bench arms for the standalone position schedulers (round-robin, fibonacci).

Same contract as ``test_bench_paired_arms.py``: an arm's flags reach the real
fuzz parser, and it differs from its declared baseline by the one knob.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS, POSITION_ARMS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402

# arm -> the dest that must be True after parsing (the knob under test)
_EXPECTED_DEST = {
    "pos-round-robin": "pos_round_robin",
    "pos-fibonacci": "pos_fibonacci",
}


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


def test_every_position_arm_is_registered_with_a_baseline():
    assert set(_EXPECTED_DEST) == set(POSITION_ARMS)
    for arm in POSITION_ARMS:
        assert arm in ARMS
        assert ARM_BASELINES[arm] in ARMS


@pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
def test_flags_reach_the_real_parser(monkeypatch, arm):
    assert getattr(_parse(monkeypatch, ARMS[arm]), _EXPECTED_DEST[arm]) is True


@pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
def test_only_the_knob_under_test_changes(monkeypatch, arm):
    # Falsification: a baseline already carrying the knob measures nothing.
    base = vars(_parse(monkeypatch, ARMS[ARM_BASELINES[arm]]))
    test = vars(_parse(monkeypatch, ARMS[arm]))
    changed = {k for k in base if k != "func" and base[k] != test[k]}
    assert changed == {_EXPECTED_DEST[arm]}, changed


def test_arms_do_not_enable_the_arena(monkeypatch):
    # Adversarial: with the arena on, Elo picks the proposer and the arm
    # would no longer isolate one position policy.
    for arm in POSITION_ARMS:
        args = _parse(monkeypatch, ARMS[arm])
        assert args.position_arena is False
        assert not args.elo
