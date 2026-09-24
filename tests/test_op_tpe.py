"""Tests for OpTPEScheduler (BO-3): categorical Tree-structured Parzen Estimator.

Expected values are derived from the l/g definition in the module docstring,
not echoed from the implementation.
"""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_tpe import GAMMA, N_CANDIDATES, OpTPEScheduler
from tests.support.scripted_rng import ScriptedRng

OPS = ["a", "b", "c"]


def _sched(rng=None, **kw):
    s = OpTPEScheduler(rng=rng or RandPool(seed=1), **kw)
    for op in OPS:
        s.init_arm(op)
    return s


class TestContract:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpTPEScheduler(rng=None)

    def test_supports_priors_declared(self):
        assert OpTPEScheduler.supports_priors is True

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_gamma_bounds(self, bad):
        with pytest.raises(ValueError):
            OpTPEScheduler(rng=RandPool(), gamma=bad)

    def test_non_positive_prior_rejected(self):
        s = OpTPEScheduler(rng=RandPool())
        with pytest.raises(ValueError):
            s.init_arm("x", 0.0, 1.0)

    def test_empty_and_single(self):
        s = _sched()
        assert s.select_op([]) == ""
        assert s.select_op(["b"]) == "b"


class TestSplit:
    def test_good_set_is_top_gamma_positive(self):
        s = _sched(window=8)
        # 8 outcomes: a wins with 0.9, 0.8, b wins with 0.5, rest failures.
        for op, ok, w in [
            ("a", True, 0.9),
            ("b", True, 0.5),
            ("a", True, 0.8),
            ("c", False, 1.0),
            ("c", False, 1.0),
            ("b", False, 1.0),
            ("c", False, 1.0),
            ("a", False, 1.0),
        ]:
            s.record(op, ok, weight=w)
        n_good = math.ceil(GAMMA * 8)  # 2 -> the two a wins
        good, bad = s.split_counts()
        assert sum(good.values()) == n_good
        assert good == {"a": 2}
        assert bad == {"a": 1, "b": 2, "c": 3}

    def test_failures_never_good(self):
        s = _sched(window=4)
        for _ in range(4):
            s.record("a", False)
        good, _ = s.split_counts()
        assert good == {}

    def test_window_evicts_oldest(self):
        s = _sched(window=2)
        s.record("a", True, weight=1.0)
        s.record("b", False)
        s.record("c", False)
        good, bad = s.split_counts()
        assert good == {}
        assert bad == {"b": 1, "c": 1}


class TestRatio:
    def test_ratio_matches_definition(self):
        s = _sched(window=4)
        s.init_arm("a", 3.0, 1.0)
        s.record("a", True, weight=1.0)
        s.record("b", False)
        s.record("b", False)
        s.record("c", False)
        # good={a:1}, bad={b:2, c:1}; l ∝ prior_a + good, g ∝ prior_b + bad
        r = s.ratios(OPS)
        assert r["a"] == pytest.approx((3.0 + 1) / (1.0 + 0))
        assert r["b"] == pytest.approx((1.0 + 0) / (1.0 + 2))
        assert r["c"] == pytest.approx((1.0 + 0) / (1.0 + 1))

    def test_unregistered_op_uses_default_prior(self):
        s = _sched()
        assert s.ratios(["zz"]) == {"zz": 1.0}


class TestSelect:
    def test_picks_best_ratio_among_candidates(self):
        s = _sched(window=4)
        s.record("b", True, weight=1.0)
        s.record("a", False)
        s.record("a", False)
        s.record("c", False)
        # l weights: a=1, b=2, c=1 (total 4). Script candidates hitting a, c, b.
        draws = [0.0, 0.99, 0.5] + [0.0] * (N_CANDIDATES - 3)
        s._rng = ScriptedRng(randoms=draws)
        assert s.select_op(OPS) == "b"

    def test_candidates_limited_to_l_draws(self):
        """A best-ratio op never drawn from l is not picked (hyperopt semantics)."""
        s = _sched(window=4)
        s.record("b", True, weight=1.0)
        s.record("a", False)
        s.record("c", False)
        s.record("c", False)
        # l: a=1, b=2, c=1 → cum 1,3,4; r*4 < 1 → a only.
        s._rng = ScriptedRng(randoms=[0.0] * N_CANDIDATES)
        assert s.select_op(OPS) == "a"

    def test_draw_count_is_exact(self):
        """Exactly N_CANDIDATES draws: one more raises StopIteration."""
        s = _sched()
        s._rng = ScriptedRng(randoms=[0.1] * (N_CANDIDATES - 1))
        with pytest.raises(StopIteration):
            s.select_op(OPS)

    def test_control_same_seed_same_picks(self):
        """Control (Rule 46): the scheduler against itself is identical."""

        def run():
            s = _sched(rng=RandPool(seed=7))
            picks = []
            for i in range(200):
                op = s.select_op(OPS)
                picks.append(op)
                s.record(op, op == "c" and i % 2 == 0, weight=1.0)
            return picks

        assert run() == run()


class TestFalsification:
    def test_rewarded_op_dominates(self):
        """Falsification: an op that always wins must end up picked most."""
        s = _sched(rng=RandPool(seed=3))
        counts = dict.fromkeys(OPS, 0)
        for _ in range(600):
            op = s.select_op(OPS)
            counts[op] += 1
            s.record(op, op == "b", weight=1.0)
        assert counts["b"] > counts["a"] + counts["c"]


class TestAdversarial:
    def test_nan_and_negative_weights_are_not_good(self):
        s = _sched(window=4)
        s.record("a", True, weight=float("nan"))
        s.record("b", True, weight=-5.0)
        s.record("c", True, weight=float("inf"))
        good, bad = s.split_counts()
        assert "a" not in good and "b" not in good
        assert bad["a"] == 1 and bad["b"] == 1
        assert s.select_op(OPS) in OPS

    def test_ops_outside_registry_and_duplicates(self):
        s = _sched()
        for _ in range(50):
            assert s.select_op(["x", "x", "y"]) in ("x", "y")

    def test_stats_keys(self):
        s = _sched()
        s.record("a", True)
        st = s.bandit_stats()
        assert st["tpe_pulls"] == 1
        assert st["operators_tracked"] >= len(OPS)


class TestWiring:
    """op_tpe reaches the Elo ballot, gets elected, and the CLI forwards it."""

    def _fuzzer(self, **flags):
        from tests.support.operator_env import make_minimal_fuzzer

        f = make_minimal_fuzzer(seed=3)
        f._use_op_tpe = True
        f._op_tpe = _sched(rng=RandPool(seed=3))
        for name, value in flags.items():
            setattr(f, name, value)
        return f

    def test_on_elo_ballot(self):
        from fuzzer_tool.services.operators import operator_strategy_pool

        assert "op_tpe" in operator_strategy_pool(self._fuzzer())

    def test_elo_elects_op_tpe(self):
        from fuzzer_tool.services.operators import OperatorEngine

        class _Elo:
            def select_strategy(self, available):
                assert "op_tpe" in available
                return "op_tpe"

        f = self._fuzzer(_use_elo=True, _elo=_Elo())
        assert OperatorEngine(f).select_op(OPS) in OPS
        assert f._op_selector == "op_tpe"

    def test_cli_flag_forwarded(self, monkeypatch):
        import sys

        from fuzzer_tool.cli import commands

        captured = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--op-tpe"])
        commands.main()
        assert captured["a"].op_tpe is True
        assert "op_tpe" in commands._HAIL_MARY_FLAGS
