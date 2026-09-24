"""Strata A0-A4 arms and pre-registered analysis in tools/bench_paired.py (§6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bench_paired import (  # noqa: E402
    ARM_BASELINES,
    ARMS,
    STRATA_ARMS,
    _parse,
    holm,
    strata_verdict,
)

from fuzzer_tool.cli import commands  # noqa: E402

_KNOB = {
    "strata-a1": {"confirm_novelty"},
    "strata-a2": {"strata"},
    "strata-a3": {"op_strata"},
    "strata-a4": {"strata", "op_strata"},
    "strata-a1-elo": {"elo", "mc_bandit"},
}


def _cli(monkeypatch, flags):
    captured = {}

    def _spy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    assert commands.main() == 0
    return vars(captured["args"])


class TestArms:
    def test_registered_with_baselines(self):
        for arm in STRATA_ARMS:
            assert arm in ARMS
        for arm in _KNOB:
            assert ARM_BASELINES[arm] in ARMS

    def test_control_is_identical_to_a0(self):
        assert ARMS["strata-a0-ctl"] == ARMS["strata-a0"]

    @pytest.mark.parametrize("arm", sorted(_KNOB))
    def test_single_variable_against_baseline(self, monkeypatch, arm):
        base = _cli(monkeypatch, ARMS[ARM_BASELINES[arm]])
        test = _cli(monkeypatch, ARMS[arm])
        changed = {k for k in base if k != "func" and base[k] != test[k]}
        assert changed == _KNOB[arm]


class TestHolm:
    def test_matches_hand_computation(self):
        # sorted p: .01, .03, .04 -> x3, x2, x1 -> .03, .06, .06 (monotone max)
        assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])

    def test_monotone_and_capped(self):
        assert holm([0.5, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])

    def test_single_and_empty(self):
        assert holm([0.02]) == [0.02]
        assert holm([]) == []

    def test_falsification_differs_from_bonferroni(self):
        # Bonferroni would give .06 for the middle one; Holm .04.
        assert holm([0.01, 0.02, 0.5])[1] == pytest.approx(0.04)


def _rows(arm, edges, eps=1000.0):
    return [
        {
            "arm": arm,
            "target": "t",
            "seed": i,
            "rep": 0,
            "edges": e,
            "eps": eps,
            "coverage_attached": True,
        }
        for i, e in enumerate(edges)
    ]


def _loaded(a0, ctl, a1, a2, a1e, a3, a4, eps1=1000.0):
    return {
        "strata-a0": _rows("strata-a0", a0),
        "strata-a0-ctl": _rows("strata-a0-ctl", ctl),
        "strata-a1": _rows("strata-a1", a1, eps1),
        "strata-a2": _rows("strata-a2", a2),
        "strata-a1-elo": _rows("strata-a1-elo", a1e),
        "strata-a3": _rows("strata-a3", a3),
        "strata-a4": _rows("strata-a4", a4),
    }


N = 20
BASE = [100] * N


class TestVerdict:
    def test_control_failure_blocks_everything(self):
        v = strata_verdict(_loaded(BASE, [110] * N, BASE, BASE, BASE, BASE, BASE))
        assert not v["control_ok"]
        assert v["comparisons"] == {}

    def test_null_matrix(self):
        v = strata_verdict(_loaded(BASE, BASE, BASE, BASE, BASE, BASE, BASE))
        assert v["control_ok"] and v["a1_noninferior"] and v["a1_eps_ok"]
        assert set(v["comparisons"]) == {"strata-a2", "strata-a3", "strata-a4"}
        assert all(c["holm_p"] == 1.0 for c in v["comparisons"].values())

    def test_a2_win_survives_holm(self):
        v = strata_verdict(_loaded(BASE, BASE, BASE, [101] * N, BASE, BASE, BASE))
        c = v["comparisons"]["strata-a2"]
        assert c["wins"] == N
        assert c["holm_p"] < 0.05

    def test_a3_pairs_against_elo_baseline(self):
        # a3 equals a1-elo, not a1: pairing against a1 would show a win.
        v = strata_verdict(_loaded(BASE, BASE, BASE, BASE, [120] * N, [120] * N, BASE))
        assert v["comparisons"]["strata-a3"]["wins"] == 0

    def test_a1_inferior_and_slow(self):
        v = strata_verdict(_loaded(BASE, BASE, [90] * N, BASE, BASE, BASE, BASE, eps1=970.0))
        assert not v["a1_noninferior"]
        assert not v["a1_eps_ok"]


class TestParseThroughput:
    def test_eps_parsed(self):
        row = _parse("  Edges discovered: 12\n  Avg throughput:   1234.5 execs/sec\n")
        assert row["eps"] == pytest.approx(1234.5)

    def test_eps_parsed_from_campaign_summary(self):
        assert _parse("  Avg eps:           57.7\n")["eps"] == pytest.approx(57.7)

    def test_eps_missing_is_zero(self):
        assert _parse("")["eps"] == 0.0
