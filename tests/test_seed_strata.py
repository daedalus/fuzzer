"""Tests for StrataSeedScheduler (core/schedulers/seed_strata.py)."""

import math

import numpy as np
import pytest

from fuzzer_tool.core.edge_ledger import EdgeLedger, Trust
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_strata import Guard, StrataSeedScheduler

SHIFT = 4


def _e(fam, tag):
    return (fam << SHIFT) | tag


class _Rng:
    """Scripted RandPool stand-in: Beta draws per call, weighted-choice indices."""

    def __init__(self, betas=(), picks=()):
        self._betas = iter(betas)
        self._picks = iter(picks)
        self.weights: list[list[float]] = []

    def betavariate_array(self, alphas, betas):
        draw = np.asarray(next(self._betas), dtype=float)
        assert draw.shape == np.shape(alphas)
        return draw

    def weighted_choice(self, seq, weights):
        self.weights.append(list(weights))
        return seq[next(self._picks)]


def _corpus_ledger():
    """Family 0 is prologue; families 1..3 frontier with known owners."""
    led = EdgeLedger(SHIFT)
    led.observe("a", frozenset({_e(0, 0), _e(1, 0), _e(1, 1)}))
    led.observe("b", frozenset({_e(0, 0), _e(1, 0), _e(2, 0)}))
    led.observe("c", frozenset({_e(0, 0), _e(3, 0)}))
    led.observe("d", frozenset({_e(0, 0), _e(3, 1)}))
    led.observe("e", frozenset({_e(0, 0)}))
    return led


LIVE = {"a", "b", "c", "d", "e"}


class TestSelect:
    def test_thompson_picks_argmax_family_then_weighted_seed(self):
        led = _corpus_ledger()
        assert led.frontier() == [1, 2, 3]
        rng = _Rng(betas=[[0.1, 0.2, 0.9]], picks=[1])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        assert s.select_key(LIVE) == "d"  # family 3 -> seeds [c, d] -> index 1
        assert s.last_phi == 3

    def test_seed_weight_is_mean_log1p_rarity(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.9, 0.1, 0.1]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)
        n = led.n_seeds
        # family 1: a owns e(1,0) [owner 2], e(1,1) [owner 1]; b owns e(1,0).
        wa = (math.log1p(n / 2) + math.log1p(n / 1)) / 2
        wb = math.log1p(n / 2)
        assert rng.weights[0] == pytest.approx([wa, wb])

    def test_family_resolution_is_uniform(self):
        led = _corpus_ledger()
        led.set_trust(Trust.UNSTABLE)
        rng = _Rng(betas=[[0.9, 0.1, 0.1]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)
        assert rng.weights[0] == [1.0, 1.0]

    def test_dead_seeds_skipped(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.1, 0.9]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        assert s.select_key({"d"}) == "d"
        assert rng.weights[0] == [pytest.approx(math.log1p(5))]


class TestRecord:
    def test_hit_increments_alpha_miss_increments_beta(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.9, 0.1]] * 3, picks=[0, 0, 0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)  # phi = 2
        s.credit(frozenset({2}), s.last_key)
        s.select_key(LIVE)  # finalises nothing (already credited); phi = 2
        s.select_key(LIVE)  # previous pick missed -> beta
        assert s.posterior(2) == (2.0, 2.0)
        assert s.stats()["hits"] == 1
        assert s.stats()["picks"] == 3

    def test_credit_other_family_is_not_a_hit(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.9, 0.1]] * 2, picks=[0, 0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)
        s.credit(frozenset({1, 3}), s.last_key)
        s.select_key(LIVE)
        assert s.posterior(2) == (1.0, 2.0)

    def test_credit_for_other_seed_ignored(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.9, 0.1]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        assert s.select_key(LIVE) == "b"
        s.credit(frozenset({2}), "a")
        assert s.posterior(2) == (1.0, 1.0)
        assert s.last_phi == 2

    def test_double_credit_counts_once(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.9, 0.1]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)
        s.credit(frozenset({2}), s.last_key)
        s.credit(frozenset({2}), s.last_key)
        assert s.posterior(2) == (2.0, 1.0)


class TestAbstain:
    @pytest.mark.parametrize("guard", [Guard.ABSENT, Guard.UNKNOWN])
    def test_guard_not_present(self, guard):
        s = StrataSeedScheduler(_Rng(), _corpus_ledger(), guard)
        assert not s.available()

    def test_empty_frontier(self):
        led = EdgeLedger(SHIFT)
        for k in "abc":
            led.observe(k, frozenset({_e(0, 0)}))
        s = StrataSeedScheduler(_Rng(), led, Guard.PRESENT)
        assert not s.available()
        assert s.select_key({"a"}) is None

    def test_no_live_seed_in_phi_abstains_without_pending(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.9, 0.1, 0.1]])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        assert s.select_key({"e"}) is None
        assert s.last_phi is None

    def test_empty_corpus(self):
        s = StrataSeedScheduler(_Rng(), EdgeLedger(SHIFT), Guard.PRESENT)
        assert not s.available()
        assert s.select_key(set()) is None

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            StrataSeedScheduler(None, EdgeLedger(SHIFT), Guard.PRESENT)


class TestControl:
    def test_same_seed_same_picks(self):
        # Rule 46 control: the scheduler against itself.
        def run():
            s = StrataSeedScheduler(RandPool(seed=9), _corpus_ledger(), Guard.PRESENT)
            out = []
            for i in range(50):
                out.append(s.select_key(LIVE))
                if i % 3 == 0:
                    s.credit(frozenset({s.last_phi}), s.last_key)
            return out, s.stats()

        assert run() == run()


class TestPersistence:
    def test_roundtrip(self):
        led = _corpus_ledger()
        rng = _Rng(betas=[[0.1, 0.9, 0.1]], picks=[0])
        s = StrataSeedScheduler(rng, led, Guard.PRESENT)
        s.select_key(LIVE)
        s.credit(frozenset({2}), s.last_key)
        back = StrataSeedScheduler.from_dict(s.to_dict(), _Rng(), led, Guard.PRESENT)
        assert back.posterior(2) == (2.0, 1.0)
        assert back.stats() == s.stats()
