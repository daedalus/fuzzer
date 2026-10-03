"""Bench arms for the Good-Turing arms (handover entropy §7.1).

Same contract as ``test_bench_paired_arms.py``: the flags reach the real fuzz
parser, and each arm is its Elo baseline plus the one knob under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS, GOOD_TURING_ARMS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402

# arm -> the dest that must be True after parsing (the knob under test)
_EXPECTED_DEST = {
    "elo-good-turing-seed": "good_turing_seed",
    "elo-op-good-turing": "op_good_turing",
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


def test_arms_registered_with_the_elo_baseline():
    assert set(_EXPECTED_DEST) == set(GOOD_TURING_ARMS)
    for arm in GOOD_TURING_ARMS:
        assert arm in ARMS
        # Against plain `baseline` the Elo arbiter's own effect is credited
        # to the arm.
        assert ARM_BASELINES[arm] == "elo"


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


@pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
def test_arm_is_baseline_plus_additions(arm):
    base = ARMS[ARM_BASELINES[arm]]
    assert ARMS[arm][: len(base)] == base
    assert len(ARMS[arm]) > len(base)


def test_arms_do_not_move_the_prior(monkeypatch):
    # Adversarial: the prior (20) is untuned; an arm that also changed it
    # would conflate the estimator with its shrinkage.
    base = _parse(monkeypatch, ARMS["elo"]).good_turing_prior
    for arm in GOOD_TURING_ARMS:
        assert _parse(monkeypatch, ARMS[arm]).good_turing_prior == base
