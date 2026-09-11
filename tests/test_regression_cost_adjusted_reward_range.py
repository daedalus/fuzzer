"""Regression: per-op bandit rewards are in [0, 1], and cost means time saved.

``Fuzzer._cost_adjusted_weight`` scales each operator's reward by how much
the round cost. It used ``median_op_cost / this_op_cost`` with the ratio
clamped to [0.05, 20], where ``_op_time_ema`` times the operator call alone.
Two defects in one line:

- The execution every iteration pays was left out, so a 1us operator was
  paid ~10x a 10us one for a 9us difference against an execution of
  milliseconds. The "time" the ratio rewarded was noise next to the time
  the round actually took.
- The ratio was clamped at 20, not 1, so rewards reached 20 while ducb,
  swucb, kl_*, cusum_ucb, exp3 and the Beta posteriors all take them to be
  [0, 1] (ducb.py said ``_cost_adjusted_weight`` "keeps" them there).
  Measured on targets/test_target over 1500 rounds: 16 of 25 success
  rewards above 1, maximum 15.2, mean 4.4. KL-UCB's bound saturates at 1.0
  for any arm whose mean exceeds 1, so every such arm scored exactly 1.0 and
  ties went to list order; a Beta posterior counted a weight of 15 as
  fifteen discoveries.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
requires_test_target = pytest.mark.skipif(
    not Path(_TARGET).exists(), reason="targets/test_target not built"
)

_US = 1e-6


def _stub(t_exec_s: float):
    costs = {"cheap": 1 * _US, "mid_a": 10 * _US, "mid_b": 10 * _US, "slow": 1000 * _US}
    return SimpleNamespace(_op_time_ema=costs, _exec_time_tracker=SimpleNamespace(p50=t_exec_s))


def _w(stub, op, base=1.0):
    return Fuzzer._cost_adjusted_weight(stub, op, base)


class TestRatioIncludesTheExecution:
    def test_median_operator_is_unaffected(self):
        assert _w(_stub(9e-3), "mid_a") == pytest.approx(1.0)

    def test_cheap_operator_gains_only_the_time_it_saves(self):
        # 9us saved against a 9ms round: ~0.1%, not the 10x the op-only
        # ratio paid.
        assert _w(_stub(9e-3), "cheap") == pytest.approx((9e-3 + 10 * _US) / (9e-3 + _US))
        assert _w(_stub(9e-3), "cheap") < 1.01

    def test_expensive_operator_still_penalized(self):
        w = _w(_stub(9e-3), "slow")
        assert w == pytest.approx((9e-3 + 10 * _US) / (9e-3 + 1000 * _US))
        assert w < 0.95

    def test_fast_target_makes_op_cost_matter(self):
        """With an in-process target (tens of us), operator cost is a real
        share of the round again, and the ratio says so."""
        assert _w(_stub(20 * _US), "slow") < 0.05 + 1e-9  # clamped floor
        assert _w(_stub(20 * _US), "cheap") > 1.3

    def test_no_execution_timing_yet_falls_back_to_op_ratio(self):
        assert _w(_stub(0.0), "cheap") == pytest.approx(10.0)

    def test_unbounded_for_elo_proportions(self):
        """The Elo edge_counts caller passes an edge share, not a reward in
        [0, 1]; the method must not clamp it."""
        assert _w(_stub(9e-3), "mid_a", base=5.0) == pytest.approx(5.0)


@requires_test_target
def test_rewards_fed_to_bandits_are_bounded():
    tmp = tempfile.TemporaryDirectory()
    corpus, crashes = Path(tmp.name) / "c", Path(tmp.name) / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=_TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        use_coverage=True,
        kl_ducb=True,
    )
    # Old code, same run: a maximum of 14.29 within these 300 rounds.
    weights: list[float] = []
    real = f._kl_ducb.record

    def spy(name, ok, weight=1.0):
        weights.append(weight)
        return real(name, ok, weight=weight)

    f._kl_ducb.record = spy
    for i in range(300):
        f.fuzz_one(bytes([65 + i % 26]) * 32)
    assert any(w > 0 for w in weights), "premise: no rewarded round"
    assert max(weights) <= 1.0, f"reward {max(weights):.2f} outside [0, 1]"
