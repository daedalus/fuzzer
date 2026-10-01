"""Bench arms for the standalone position schedulers (round-robin, fibonacci).

Same contract as ``test_bench_paired_arms.py``: an arm's flags reach the real
fuzz parser, and it differs from its declared baseline by the one knob.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import (  # noqa: E402
    ARENA_TESTABLE,
    ARM_BASELINES,
    ARMS,
    POSITION_ARENA_ARMS,
    POSITION_ARMS,
)

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


# ── arena subset arms (--pos-arena-arms) ───────────────────────────────


def _arena_arm(arm: str) -> str:
    return arm.removeprefix("pos-arena-").replace("-", "_")


def test_arena_arms_registered_with_a_baseline():
    for arm in POSITION_ARENA_ARMS:
        assert arm in ARMS
        assert ARM_BASELINES[arm] in ARMS
    # Every subset arm pairs against the uniform-only control, not `baseline`:
    # against `baseline` it would credit the arena machinery to the arm.
    for arm in POSITION_ARENA_ARMS[1:]:
        assert ARM_BASELINES[arm] == "pos-arena-uniform"
    assert ARM_BASELINES["pos-arena-uniform"] == "elo"


def test_arena_arm_flags_reach_the_real_parser(monkeypatch):
    for arm in POSITION_ARENA_ARMS:
        args = _parse(monkeypatch, ARMS[arm])
        assert args.position_arena is True
        assert args.elo


def test_arena_subset_is_exactly_uniform_plus_the_named_arm(monkeypatch):
    for a in ARENA_TESTABLE:
        arm = "pos-arena-" + a.replace("_", "-")
        assert _parse(monkeypatch, ARMS[arm]).pos_arena_arms == ("uniform", a)
    assert _parse(monkeypatch, ARMS["pos-arena-uniform"]).pos_arena_arms == ("uniform",)
    assert _parse(monkeypatch, ARMS["pos-arena-all"]).pos_arena_arms is None


def test_arena_arms_differ_from_control_by_the_subset_only(monkeypatch):
    # Falsification: any other changed dest means the A/B is confounded.
    ctl = vars(_parse(monkeypatch, ARMS["pos-arena-uniform"]))
    for arm in POSITION_ARENA_ARMS[1:]:
        test = vars(_parse(monkeypatch, ARMS[arm]))
        changed = {k for k in ctl if k != "func" and ctl[k] != test[k]}
        assert changed == {"pos_arena_arms"}, (arm, changed)


def test_arena_arms_do_not_turn_on_other_position_flags(monkeypatch):
    # The arena implies the schedulers inside Fuzzer(); the CLI must not
    # also set them, or the arm stops being the only variable.
    for arm in POSITION_ARENA_ARMS:
        args = _parse(monkeypatch, ARMS[arm])
        for dest in (
            "pos_fractal",
            "pos_levy",
            "pos_kl_ducb",
            "burn_front",
            "pos_context",
            "pos_boundary",
            "pos_token",
            "pos_chunk",
            "pos_changed",
            "pos_rare_mask",
            "pos_consolidated",
        ):
            assert getattr(args, dest) is False, (arm, dest)


def test_arena_testable_arms_need_no_extra_feature_flag():
    from fuzzer_tool.services.position_arena import POSITION_STRATEGY_NAMES

    assert set(ARENA_TESTABLE) <= set(POSITION_STRATEGY_NAMES)
    # gated on their own feature, so they cannot be A/B'd by subset alone
    assert not {"cmplog", "lineage", "effector"} & set(ARENA_TESTABLE)
