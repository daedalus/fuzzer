"""G0: bench arms for the generation group (wfc, mcts, alphabeta, bootstrap).

docs/handover/handover_generators_2026-09-20.md G0: every A/B in that
handover was blocked on `tools/bench_paired.py` having no arm for any of the
four. An arm is only evidence if (a) its flags reach the real fuzz parser --
argparse would otherwise accept a typo'd flag as an abbreviation or reject it
only when the campaign starts, three hours into a matrix -- and (b) it differs
from its declared baseline by the one thing under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bench_paired import ARM_BASELINES, ARMS, GENERATION_ARMS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402

# arm -> the dest that must be True after parsing (the knob under test)
_EXPECTED_DEST = {
    "wfc": "wfc",
    "elo-mcts": "mcts",
    "elo-alphabeta": "alphabeta",
    "bootstrap": "bootstrap",
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


class TestGenerationArmsRegistered:
    def test_every_generation_arm_is_in_arms(self):
        assert set(_EXPECTED_DEST) <= set(GENERATION_ARMS)
        for arm in GENERATION_ARMS:
            assert arm in ARMS, arm

    def test_every_generation_arm_names_a_baseline_that_exists(self):
        for arm in GENERATION_ARMS:
            assert arm in ARM_BASELINES, f"{arm} has no declared baseline"
            assert ARM_BASELINES[arm] in ARMS


class TestGenerationArmsParse:
    @pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
    def test_flags_reach_the_real_parser(self, monkeypatch, arm):
        args = _parse(monkeypatch, ARMS[arm])
        assert getattr(args, _EXPECTED_DEST[arm]) is True

    @pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
    def test_baseline_leaves_the_knob_off(self, monkeypatch, arm):
        """Falsification: the baseline must not already contain the thing
        under test, or the pair measures nothing."""
        args = _parse(monkeypatch, ARMS[ARM_BASELINES[arm]])
        assert getattr(args, _EXPECTED_DEST[arm]) is False

    def test_unknown_flag_is_rejected(self, monkeypatch):
        """Adversarial: the parse check above must be able to fail. argparse
        abbreviation matching would let `--wf` through, so pin that a plain
        typo does not."""
        with pytest.raises(SystemExit):
            _parse(monkeypatch, ["--wfcc"])


class TestSingleVariable:
    """Arms must differ from their baseline by added flags only (the module's
    own rule: an arm that changes two things cannot attribute its result)."""

    @pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
    def test_arm_is_baseline_plus_additions(self, arm):
        base = ARMS[ARM_BASELINES[arm]]
        assert ARMS[arm][: len(base)] == base
        assert len(ARMS[arm]) > len(base)

    @pytest.mark.parametrize("arm", sorted(_EXPECTED_DEST))
    def test_only_the_knob_under_test_changes(self, monkeypatch, arm):
        """The two parsed Namespaces differ in exactly the tested dest plus
        the dests that flag documents itself as implying."""
        implied = {"elo-mcts": {"lineage"}, "elo-alphabeta": {"lineage"}}.get(arm, set())
        base = vars(_parse(monkeypatch, ARMS[ARM_BASELINES[arm]]))
        test = vars(_parse(monkeypatch, ARMS[arm]))
        # `func` is the spy _parse binds, a fresh closure per call.
        changed = {k for k in base if k != "func" and base[k] != test[k]}
        assert changed - implied == {_EXPECTED_DEST[arm]}, changed
