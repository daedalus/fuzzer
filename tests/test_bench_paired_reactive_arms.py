"""Bench arms for the reactive gradient A/B (core/schedulers/op_reactive.py).

Same contract as ``test_bench_paired_good_turing_arms.py``. Both arms are
``elo`` plus one scheduler flag; reactive is read against gradient with
``analyse --baseline elo-gradient``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS, REACTIVE_ARMS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402

# arm -> the dest that must be True after parsing (the knob under test)
_EXPECTED_DEST = {
    "elo-gradient": "gradient",
    "elo-reactive": "reactive",
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
    assert set(_EXPECTED_DEST) == set(REACTIVE_ARMS)
    for arm in REACTIVE_ARMS:
        assert arm in ARMS
        assert ARM_BASELINES[arm] == "elo"


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


def test_arms_differ_only_in_the_scheduler(monkeypatch):
    # Adversarial: reactive vs gradient must swap one scheduler for the other,
    # nothing else, or the comparison credits a second knob.
    grad = vars(_parse(monkeypatch, ARMS["elo-gradient"]))
    reac = vars(_parse(monkeypatch, ARMS["elo-reactive"]))
    changed = {k for k in grad if k != "func" and grad[k] != reac[k]}
    assert changed == {"gradient", "reactive"}, changed


def test_control_is_gradient_again():
    # A control that differs from its pair cannot validate the oracle.
    assert ARMS["elo-gradient-ctl"] == ARMS["elo-gradient"]
    assert "elo-gradient-ctl" not in REACTIVE_ARMS
