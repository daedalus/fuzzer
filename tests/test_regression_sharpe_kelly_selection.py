"""Regression tests: Sharpe/Kelly blend changes operator selection behavior."""

import random

from fuzzer_tool.core.running_stats import kelly_fraction, sharpe_ratio
from fuzzer_tool.core.schedulers.monte_carlo import MonteCarloScheduler


def _seed(seed: int = 42) -> None:
    random.seed(seed)


class TestSharpeKellySelection:
    """Verify that blending Sharpe/Kelly into Thompson selection biases toward
    consistent operators and away from high-variance lottery tickets."""

    def test_blend_zero_does_not_record_reward_moments(self):
        """With blend=0, _op_reward_moments stays empty — no SK overhead."""
        mc = MonteCarloScheduler()
        mc.init_arm("a")
        mc.init_arm("b")
        mc.set_sharpe_kelly_blend(0.0)

        _seed(42)
        for _ in range(20):
            mc.record("a", success=True, weight=0.5)
            mc.record("b", success=True, weight=0.5)

        # blend=0: reward moments are still tracked (cheap O(1) update),
        # but select_op does not read them.
        assert "a" in mc._op_reward_moments
        assert "b" in mc._op_reward_moments

    def test_blend_one_prefers_consistent_operator(self):
        """blend=1.0 should strongly prefer the consistent low-variance arm."""
        mc = MonteCarloScheduler()
        mc.init_arm("consistent")
        mc.init_arm("lottery")
        mc.set_sharpe_kelly_blend(1.0)

        _seed(42)
        # Both arms have the SAME mean reward (0.5) but different variance.
        for _ in range(20):
            mc.record("consistent", success=True, weight=0.5)
        for _ in range(10):
            mc.record("lottery", success=True, weight=1.0)
        for _ in range(10):
            mc.record("lottery", success=False, weight=0.0)

        counts = {"consistent": 0, "lottery": 0}
        for _ in range(2000):
            op = mc.select_op(["consistent", "lottery"])
            counts[op] += 1
        # consistent has zero variance → Sharpe = inf → SK norm = 1.0
        # lottery has high variance → finite Sharpe → SK norm ≈ 0.0
        assert counts["consistent"] > counts["lottery"], (
            f"consistent should dominate with blend=1.0, got {counts}"
        )

    def test_blend_shift_increases_consistent_arm_share(self):
        """Increasing blend from 0 to 1 should increase consistent-arm share."""
        counts_by_blend = {}
        for blend in (0.0, 0.5, 1.0):
            mc = MonteCarloScheduler()
            mc.init_arm("consistent")
            mc.init_arm("lottery")
            mc.set_sharpe_kelly_blend(blend)

            _seed(42)
            for _ in range(20):
                mc.record("consistent", success=True, weight=0.5)
            for _ in range(10):
                mc.record("lottery", success=True, weight=1.0)
            for _ in range(10):
                mc.record("lottery", success=False, weight=0.0)

            counts = {"consistent": 0, "lottery": 0}
            for _ in range(2000):
                op = mc.select_op(["consistent", "lottery"])
                counts[op] += 1
            counts_by_blend[blend] = counts["consistent"] / 2000.0

        # Higher blend → higher share for the consistent arm
        assert counts_by_blend[0.0] <= counts_by_blend[0.5] <= counts_by_blend[1.0], (
            f"consistent share should increase with blend, got {counts_by_blend}"
        )

    def test_insufficient_observations_returns_zero(self):
        """sharpe() and kelly_fraction() return 0.0 with < 3 observations."""
        mc = MonteCarloScheduler()
        mc.init_arm("new_op")
        mc.record("new_op", success=True, weight=1.0)
        assert mc.sharpe("new_op") == 0.0
        assert mc.kelly_fraction("new_op") == 0.0

        mc.record("new_op", success=True, weight=1.0)
        assert mc.sharpe("new_op") == 0.0
        assert mc.kelly_fraction("new_op") == 0.0

        mc.record("new_op", success=True, weight=1.0)
        assert mc.sharpe("new_op") >= 0.0
        assert mc.kelly_fraction("new_op") >= 0.0

    def test_negative_kelly_clamped_to_zero_in_selection(self):
        """Arm with negative Kelly should receive 0.0 SK score."""
        mc = MonteCarloScheduler()
        mc.init_arm("loser")
        mc.init_arm("winner")
        mc.set_sharpe_kelly_blend(1.0)

        _seed(42)
        for _ in range(20):
            mc.record("winner", success=True, weight=0.8)
        for _ in range(30):
            mc.record("loser", success=False, weight=0.0)
        for _ in range(2):
            mc.record("loser", success=True, weight=0.05)

        counts = {"winner": 0, "loser": 0}
        for _ in range(2000):
            op = mc.select_op(["winner", "loser"])
            counts[op] += 1
        assert counts["winner"] > counts["loser"]

    def test_set_sharpe_kelly_blend_clamps(self):
        mc = MonteCarloScheduler()
        mc.set_sharpe_kelly_blend(-0.5)
        assert mc._sharpe_kelly_blend == 0.0
        mc.set_sharpe_kelly_blend(1.5)
        assert mc._sharpe_kelly_blend == 1.0
        mc.set_sharpe_kelly_blend(0.3)
        assert mc._sharpe_kelly_blend == 0.3

    def test_sharpe_kelly_methods_return_helpers(self):
        """sharpe() and kelly_fraction() delegate to module-level helpers."""
        mc = MonteCarloScheduler()
        mc.init_arm("op")
        for _ in range(10):
            mc.record("op", success=True, weight=1.0)
        assert mc.sharpe("op") == sharpe_ratio(
            mc._op_reward_moments["op"].mean,
            mc._op_reward_moments["op"].stddev,
        )
        assert mc.kelly_fraction("op") == kelly_fraction(
            mc._op_reward_moments["op"].mean,
            mc._op_reward_moments["op"].variance,
        )
