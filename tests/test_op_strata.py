"""Tests for OpStrataScheduler (core/schedulers/op_strata.py)."""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_strata import POOL_K, OpStrataScheduler

OPS = ["flip", "insert", "splice"]


class _Rng:
    """Records the Beta parameters and returns scripted draws."""

    def __init__(self, draws):
        self._draws = iter(draws)
        self.params: list[tuple[list[float], list[float]]] = []

    def betavariate_array(self, alphas, betas):
        self.params.append((list(np.asarray(alphas)), list(np.asarray(betas))))
        return np.asarray(next(self._draws), dtype=float)


def _sched(draws=()):
    rng = _Rng(draws)
    s = OpStrataScheduler(rng=rng)
    for op in OPS:
        s.init_arm(op)
    return s, rng


def _expected(s_op, n_op, s_cell, f_cell, k=POOL_K):
    p = (s_op + 1.0) / (n_op + 2.0)
    return k * p + s_cell, k * (1.0 - p) + f_cell


class TestSelect:
    def test_argmax_of_draws(self):
        s, _ = _sched([[0.2, 0.7, 0.1]])
        assert s.select_op(OPS) == "insert"

    def test_cold_prior_is_pooled_half(self):
        s, rng = _sched([[0.1, 0.1, 0.9]])
        s.select_op(OPS)
        a, b = _expected(0, 0, 0, 0)
        assert rng.params[0] == ([a] * 3, [b] * 3)

    def test_partial_pooling_parameters(self):
        s, rng = _sched([[0.9, 0.1, 0.1]] * 5)
        s.set_stratum(7)
        s.select_op(OPS)
        s.record("flip", True, 0.5)
        s.select_op(OPS)
        s.record("flip", False, 1.0)
        s.set_stratum(8)
        s.select_op(OPS)
        s.record("flip", True, 1.0)
        # Back in stratum 7: flip's pooled mean sees all three pulls, its cell two.
        s.set_stratum(7)
        s.select_op(OPS)
        a, b = _expected(1.5, 3, 0.5, 1.5)
        assert rng.params[3][0][0] == pytest.approx(a)
        assert rng.params[3][1][0] == pytest.approx(b)
        # insert/splice untouched: pooled cold prior.
        a0, b0 = _expected(0, 0, 0, 0)
        assert rng.params[3][0][1:] == pytest.approx([a0, a0])
        assert rng.params[3][1][1:] == pytest.approx([b0, b0])

    def test_no_stratum_uses_pooled_only(self):
        s, rng = _sched([[0.9, 0.1, 0.1]] * 3)
        s.set_stratum(None)
        s.select_op(OPS)
        s.record("flip", True, 1.0)
        s.select_op(OPS)
        a, b = _expected(1, 1, 0, 0)
        assert rng.params[1][0][0] == pytest.approx(a)
        assert rng.params[1][1][0] == pytest.approx(b)

    def test_single_and_empty(self):
        s, rng = _sched()
        assert s.select_op(["flip"]) == "flip"
        assert s.select_op([]) == ""
        assert rng.params == []

    def test_unregistered_op_accepted(self):
        s, _ = _sched([[0.1, 0.1, 0.1, 0.9]])
        assert s.select_op([*OPS, "new"]) == "new"


class TestRecord:
    @pytest.mark.parametrize("w", [float("nan"), float("inf"), -1.0])
    def test_bad_weight_is_failure(self, w):
        s, _ = _sched()
        s.record("flip", True, w)
        assert s.cell("flip", None) == (0.0, 1.0)

    def test_weight_clamped_to_one(self):
        s, _ = _sched()
        s.record("flip", True, 3.0)
        assert s.cell("flip", None) == (1.0, 0.0)

    def test_record_uses_stratum_at_record_time(self):
        s, _ = _sched()
        s.set_stratum(4)
        s.record("flip", True, 1.0)
        assert s.cell("flip", 4) == (1.0, 0.0)
        assert s.cell("flip", 5) == (0.0, 0.0)


class TestContract:
    def test_supports_priors_false(self):
        assert OpStrataScheduler.supports_priors is False

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpStrataScheduler(rng=None)

    def test_bandit_stats(self):
        s, _ = _sched()
        s.set_stratum(3)
        s.record("flip", True, 1.0)
        st = s.bandit_stats()
        assert st["strata_cells"] >= 1
        assert st["operators_tracked"] >= len(OPS)


class TestControl:
    def test_same_seed_same_picks(self):
        def run():
            s = OpStrataScheduler(rng=RandPool(seed=4))
            out = []
            for i in range(200):
                s.set_stratum(i % 3)
                op = s.select_op(OPS)
                s.record(op, op == "splice", 1.0)
                out.append(op)
            return out

        assert run() == run()

    def test_learns_stratum_specific_best(self):
        # Adversarial for pooling: the best op differs by stratum. Enough
        # evidence must override the pooled prior in each cell.
        s = OpStrataScheduler(rng=RandPool(seed=1))
        best = {0: "flip", 1: "splice"}
        for i in range(4000):
            st = i % 2
            s.set_stratum(st)
            op = s.select_op(OPS)
            s.record(op, op == best[st], 1.0)
        tail = {0: [], 1: []}
        for i in range(400):
            st = i % 2
            s.set_stratum(st)
            tail[st].append(s.select_op(OPS))
        assert tail[0].count("flip") > len(tail[0]) * 0.8
        assert tail[1].count("splice") > len(tail[1]) * 0.8
