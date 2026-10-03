"""Tests for OpGoodTuringScheduler: per-operator Good-Turing discovery probability.

Expected values are derived by hand from ``M_op = (Q1 + K*M_global) / (T + K)``
in the module docstring, not echoed from the implementation.
"""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_good_turing import (
    MIN_OBSERVATIONS,
    MIN_WEIGHT,
    OpGoodTuringScheduler,
)
from tests.support.scripted_rng import ScriptedRng

OPS = ["flip", "havoc", "splice"]


def _sched(rng=None, **kw):
    kw.setdefault("min_observations", 0)
    s = OpGoodTuringScheduler(rng=rng or RandPool(seed=1), **kw)
    for op in OPS:
        s.init_arm(op)
    return s


class TestContract:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpGoodTuringScheduler(rng=None)

    def test_negative_prior_rejected(self):
        with pytest.raises(ValueError):
            OpGoodTuringScheduler(rng=RandPool(seed=1), prior_strength=-1.0)

    def test_record_feeds_bandit_stats(self):
        s = _sched()
        s.record("flip", True)
        assert s.bandit_stats()["flip"] == (1.0, 0.0)

    def test_empty_ops_returns_empty(self):
        assert _sched().select_op([]) == ""

    def test_single_op_returned_without_draw(self):
        assert _sched(rng=ScriptedRng()).select_op(["flip"]) == "flip"


class TestEstimate:
    def test_unseen_op_scores_global_rate(self):
        s = _sched()
        s.observe(["havoc"], {1})
        s.observe(["havoc"], {1})
        # global: T=2, edge 1 hit twice -> Q1=0 -> M0=0
        assert s.discovery_probability("flip") == pytest.approx(s.residual_risk())

    def test_all_singletons_closed_form(self):
        s = _sched(prior_strength=20.0)
        for i in range(4):
            s.observe(["havoc"], {i})
        # global T=4, Q1=4 -> M0=1.0; havoc T=4, Q1=4
        # (4 + 20*1.0) / (4 + 20) = 1.0
        assert s.residual_risk() == pytest.approx(1.0)
        assert s.discovery_probability("havoc") == pytest.approx(1.0)

    def test_shrinkage_closed_form(self):
        s = _sched(prior_strength=20.0)
        for i in range(10):
            s.observe(["havoc"], {i})  # 10 singletons
        for _ in range(10):
            s.observe(["flip"], {100})  # one edge, hit 10x -> Q1=0 for flip
        # global: T=20, edges 0..9 once, edge 100 ten times -> Q1=10, M0=0.5
        assert s.residual_risk() == pytest.approx(0.5)
        # flip: Q1 contribution 1 at its first hit then 0 -> Q1=0, T=10
        # (0 + 20*0.5) / (10 + 20) = 1/3
        assert s.discovery_probability("flip") == pytest.approx(1.0 / 3.0)
        # havoc: Q1=10, T=10 -> (10 + 10) / 30 = 2/3
        assert s.discovery_probability("havoc") == pytest.approx(2.0 / 3.0)


class TestFalsification:
    def test_retreading_op_ranks_below_discovering_op(self):
        s = _sched()
        for i in range(30):
            s.observe(["havoc"], {i, i + 1000})
            s.observe(["flip"], {7})
        scores = s.scores(OPS)
        by_op = dict(zip(OPS, scores, strict=True))
        assert by_op["havoc"] > by_op["flip"]

    def test_stacked_ops_share_the_execution(self):
        # Documented smear: every op in the stack is credited with the same
        # edges, so a lone productive op and its stack-mate score alike.
        s = _sched()
        for i in range(30):
            s.observe(["havoc", "flip"], {i})
        assert s.discovery_probability("havoc") == pytest.approx(s.discovery_probability("flip"))

    def test_duplicate_op_in_stack_counts_once(self):
        s = _sched()
        s.observe(["havoc", "havoc"], {1})
        assert s.executions("havoc") == 1

    def test_saturated_op_loses_to_fresh_one_under_scripted_draw(self):
        s = _sched(rng=ScriptedRng(randoms=[0.999]))
        for i in range(40):
            s.observe(["havoc"], {i})
            s.observe(["flip"], {-1})
        # r=0.999 lands on the last positively weighted op by mass; the
        # saturated op must not own the top of the cumulative range.
        assert s.select_op(["flip", "havoc"]) == "havoc"


class TestAdversarial:
    def test_all_zero_scores_do_not_raise(self):
        s = _sched(rng=ScriptedRng(randoms=[0.5]))
        for _ in range(60):
            s.observe(["flip", "havoc"], {1})
        assert s.residual_risk() == 0.0
        assert s.select_op(["flip", "havoc"]) in ("flip", "havoc")

    def test_scores_floored(self):
        s = _sched()
        for _ in range(60):
            s.observe(["flip"], {1})
        assert min(s.scores(["flip"])) >= MIN_WEIGHT

    def test_empty_edge_set_counts_as_an_execution(self):
        s = _sched()
        s.observe(["flip"], set())
        assert s.executions("flip") == 1
        assert s.singletons("flip") == 0

    def test_generator_edges_consumed_once(self):
        s = _sched()
        s.observe(["flip"], (e for e in (1, 2, 3)))
        assert s.singletons("flip") == 3
        assert s.stats()["q1"] == 3

    def test_observe_with_no_ops_is_inert(self):
        s = _sched()
        s.observe([], {1, 2})
        assert s.stats()["observed"] == 0

    def test_cold_arm_falls_back_to_uniform_draw(self):
        s = _sched(rng=ScriptedRng(choice_idxs=[2]), min_observations=MIN_OBSERVATIONS)
        assert not s.ready
        assert s.select_op(OPS) == "splice"

    def test_op_table_is_bounded(self):
        s = _sched(op_cap=4)
        for i in range(50):
            s.observe([f"op{i}"], {i})
        assert s.stats()["ops"] <= 4

    def test_evicted_op_reverts_to_global_rate(self):
        s = _sched(op_cap=2)
        s.observe(["a"], {1})
        s.observe(["b"], {2})
        s.observe(["c"], {3})
        assert s.executions("a") == 0


class TestWiring:
    """op_good_turing reaches the Elo ballot, gets elected, and the CLI forwards it."""

    def _fuzzer(self, **flags):
        from tests.support.operator_env import make_minimal_fuzzer

        f = make_minimal_fuzzer(seed=3)
        f._use_op_good_turing = True
        f._op_good_turing = _sched(rng=RandPool(seed=3))
        for name, value in flags.items():
            setattr(f, name, value)
        return f

    def test_on_elo_ballot(self):
        from fuzzer_tool.services.operators import operator_strategy_pool

        assert "op_good_turing" in operator_strategy_pool(self._fuzzer())

    def test_elo_elects_op_good_turing(self):
        from fuzzer_tool.services.operators import OperatorEngine

        class _Elo:
            def select_strategy(self, available):
                assert "op_good_turing" in available
                return "op_good_turing"

        f = self._fuzzer(_use_elo=True, _elo=_Elo())
        assert OperatorEngine(f).select_op(OPS) in OPS
        assert f._op_selector == "op_good_turing"

    def test_cli_flag_forwarded(self, monkeypatch):
        import sys

        from fuzzer_tool.cli import commands

        captured = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
        monkeypatch.setattr(
            sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--op-good-turing"]
        )
        commands.main()
        assert captured["a"].op_good_turing is True
        assert "op_good_turing" in commands._HAIL_MARY_FLAGS


class TestRoundFeed:
    """FuzzRound hands the stack and the mutant's edges to the arm."""

    def _round(self, ops, edges, strategy):
        from types import SimpleNamespace

        from fuzzer_tool.services.fuzz_round import FuzzRound

        fr = FuzzRound.__new__(FuzzRound)
        fr._f = SimpleNamespace(
            _op_good_turing=strategy,
            _last_ops_used=ops,
            _current_edges_cache=edges,
        )
        return fr

    def test_feed_credits_stack_with_edges(self):
        s = _sched()
        self._round(["flip", "havoc"], {1, 2}, s)._feed_op_good_turing()
        assert s.executions("flip") == 1
        assert s.singletons("havoc") == 2

    def test_feed_off_is_inert(self):
        self._round(["flip"], {1}, None)._feed_op_good_turing()

    def test_non_set_edges_count_as_empty(self):
        s = _sched()
        self._round(["flip"], [1, 2, 3], s)._feed_op_good_turing()
        assert s.executions("flip") == 1
        assert s.singletons("flip") == 0
