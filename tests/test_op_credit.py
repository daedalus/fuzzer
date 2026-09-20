"""Tests for OpCreditScheduler (core/schedulers/op_credit.py)."""

import pytest

from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_credit import DECAY, OpCreditScheduler


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    """coverage_trust returns early for a target-less run, so the gate tests need one."""
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")


class _Tracker:
    def __init__(self, profiles):
        self.seed_hit_counts = profiles
        self.seed_edges = {k: set(v) for k, v in profiles.items()}
        self.cumulative_edges = set().union(*self.seed_edges.values())


class _Rng:
    def __init__(self, randoms=(), betas=(), randranges=(), choices=()):
        self._r, self._b, self._rr, self._c = map(iter, (randoms, betas, randranges, choices))
        self.beta_args = []

    def random(self):
        return next(self._r)

    def betavariate(self, a, b):
        self.beta_args.append((a, b))
        return next(self._b)

    def randrange(self, n):
        return next(self._rr)

    def choice(self, seq):
        return seq[next(self._c)]


# Edges 10-12 are one straight-line chain (identical columns); 20 and 30 stand alone.
PROFILES = {"a": {10: 1, 11: 1, 12: 1}, "b": {10: 1, 11: 1, 12: 1, 20: 4}, "c": {20: 4, 30: 1}}


def _sub():
    sub = MatrixSubstrate(target="/x")
    sub.maybe_refit(_Tracker(PROFILES), 0)
    return sub


class TestCredit:
    def test_a_chain_pays_once(self):
        s = OpCreditScheduler(RandPool(seed=1), _sub())
        s.observe_new_edges("flip", {10, 11, 12})
        s.observe_new_edges("splice", {20})
        assert s.credit("flip") == 1 and s.credit("splice") == 1

    def test_credit_is_a_set_so_repeats_do_not_accrue(self):
        s = OpCreditScheduler(RandPool(seed=1), _sub())
        for _ in range(50):
            s.observe_new_edges("flip", {20})
        assert s.credit("flip") == 1  # fatigue by construction

    def test_resettles_when_the_partition_changes(self):
        sub = MatrixSubstrate()
        s = OpCreditScheduler(RandPool(seed=1), sub)
        s.observe_new_edges("flip", {10, 11, 12})
        assert s.credit("flip") == 3  # no fold yet: three unknown edges
        sub.maybe_refit(_Tracker(PROFILES), 0)
        assert s.credit("flip") == 1

    def test_shaped_weight_is_the_independent_fraction(self):
        s = OpCreditScheduler(RandPool(seed=1), _sub())
        assert s.shaped_weight({10, 11, 12}) == pytest.approx(1 / 3)
        assert s.shaped_weight({10, 20}) == 1.0
        assert s.shaped_weight(set()) == 0.0


class TestSelection:
    def test_thompson_posterior_is_credit_and_pulls(self):
        rng = _Rng(randoms=[0.99], betas=[0.2, 0.9])
        s = OpCreditScheduler(rng, _sub())
        s.observe_new_edges("a_op", {20})
        for _ in range(4):
            s.record("a_op", True)
        s.record("b_op", False)
        assert s.select_op(["a_op", "b_op"]) == "b_op"
        # credit 1, pulls 4 -> Beta(2, 4); credit 0, pulls 1 -> Beta(1, 2)
        assert rng.beta_args == [(2.0, 4.0), (1.0, 2.0)]

    def test_credit_above_pulls_clamps_failures_at_one(self):
        rng = _Rng(randoms=[0.99], betas=[0.5, 0.5])
        s = OpCreditScheduler(rng, _sub())
        s.observe_new_edges("a_op", {20, 30})
        s.select_op(["a_op", "b_op"])
        assert rng.beta_args[0] == (3.0, 1.0)

    def test_saturation_decays_stale_failures_not_credit(self):
        sub = _sub()
        sub._history.clear()
        sub._history.extend([(100, 40.0), (100, 35.0), (100, 30.0), (100, 20.0)])
        assert sub.saturation_signal() == 1.0
        rng = _Rng(randoms=[0.99], betas=[0.5, 0.5])
        s = OpCreditScheduler(rng, sub)
        s.observe_new_edges("a_op", {20})
        for _ in range(5):
            s.record("a_op", False)
        s.select_op(["a_op", "b_op"])
        # credit stays 1; stale failures (5 - 1) shrink by DECAY at full saturation.
        assert rng.beta_args[0] == (2.0, 1.0 + 4.0 * (1.0 - DECAY))

    def test_saturation_widens_exploration(self):
        sub = _sub()
        sub._history.clear()
        sub._history.extend([(100, 40.0), (100, 35.0), (100, 30.0), (100, 20.0)])
        # 0.07 is above EXPLORE_BASE (0.05) but below the widened 0.10.
        s = OpCreditScheduler(_Rng(randoms=[0.07], randranges=[1]), sub)
        assert s.select_op(["x", "y"]) == "y"

    def test_explore_draw_is_uniform(self):
        s = OpCreditScheduler(_Rng(randoms=[0.0], randranges=[1]), _sub())
        assert s.select_op(["x", "y"]) == "y"

    def test_closed_gate_selects_uniformly_and_reports_unavailable(self):
        sub = _sub()
        sub.set_stability(0.0)
        s = OpCreditScheduler(_Rng(choices=[1]), sub)
        assert not s.available() and s.select_op(["x", "y"]) == "y"

    def test_trivial_lists(self):
        s = OpCreditScheduler(RandPool(seed=1), _sub())
        assert s.select_op([]) == "" and s.select_op(["only"]) == "only"

    def test_high_credit_op_wins_more_often(self):
        s = OpCreditScheduler(RandPool(seed=3), _sub())
        s.observe_new_edges("good", {10, 20, 30})
        for _ in range(10):
            s.record("good", True)
            s.record("bad", False)
        picks = [s.select_op(["good", "bad"]) for _ in range(400)]
        assert picks.count("good") > picks.count("bad")


class TestContract:
    def test_requires_rng_and_substrate(self):
        with pytest.raises(ValueError):
            OpCreditScheduler(rng=None, substrate=MatrixSubstrate())
        with pytest.raises(ValueError):
            OpCreditScheduler(rng=RandPool(seed=1), substrate=None)

    def test_init_arm_stats_and_priors(self):
        s = OpCreditScheduler(RandPool(seed=1), _sub())
        s.init_arm("x")
        st = s.bandit_stats()
        assert st["op_credit_pulls"]["x"] == 0.0 and st["op_credit_gate"] == "unverified"
        assert OpCreditScheduler.supports_priors is False
