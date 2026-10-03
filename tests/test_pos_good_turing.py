"""PositionGoodTuringScheduler: per-offset-bin Good-Turing discovery probability.

Covers core/schedulers/pos_good_turing.py. Draw order in ``propose``: one
``random()`` for the bin, one ``randint`` for the offset inside it.
"""

import pytest

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_good_turing import (
    MAX_BINS,
    MAX_SEEDS,
    MIN_WEIGHT,
    PositionGoodTuringScheduler,
)
from fuzzer_tool.core.schedulers.seed_good_turing import good_turing_m0, shrunk_m0

SEED = bytes(MAX_BINS)  # width 1 -> bin == offset
K = 4.0
READY = 1  # min_observations: ready after one round


class ScriptedRng:
    def __init__(self, randoms=(), lows=True):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0)

    def randint(self, a, b):
        return a


def _gt(*randoms, min_obs=READY, prior=K):
    return PositionGoodTuringScheduler(
        ScriptedRng(randoms), prior_strength=prior, min_observations=min_obs
    )


def _round(gt, offset, edges, data=SEED):
    gt.observe(edges)
    gt.record(data, [offset], Outcome.MISS, 1.0)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_gt(), PositionScheduler)

    def test_name_and_priors_flag(self):
        assert PositionGoodTuringScheduler.name == "good_turing"
        assert PositionGoodTuringScheduler.supports_priors is False

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            PositionGoodTuringScheduler(None)

    def test_rejects_negative_prior(self):
        with pytest.raises(ValueError):
            PositionGoodTuringScheduler(ScriptedRng(), prior_strength=-1.0)


class TestEstimate:
    def test_bin_score_matches_closed_form(self):
        gt = _gt()
        for e in (1, 2, 3):
            _round(gt, 5, {e})  # bin 5: T=3, Q1=3
        for _ in range(3):
            _round(gt, 9, {7})  # bin 9: T=3, Q1=0 (edge 7 also new once: see below)

        glob = good_turing_m0(*_global_q1_t(gt))
        seed_m = shrunk_m0(*_seed_q1_t(gt), glob, K)
        hot = shrunk_m0(3, 3, seed_m, K)
        assert gt.discovery_probability(SEED, 5) == pytest.approx(hot)

    def test_retread_bin_scores_below_fresh_bin(self):
        gt = _gt()
        for e in (1, 2, 3, 4):
            _round(gt, 5, {e})
        for _ in range(4):
            _round(gt, 9, {1})
        assert gt.discovery_probability(SEED, 5) > gt.discovery_probability(SEED, 9)

    def test_unseen_bin_scores_the_seed_rate(self):
        gt = _gt()
        _round(gt, 5, {1})
        _round(gt, 5, {2})
        glob = good_turing_m0(*_global_q1_t(gt))
        seed_m = shrunk_m0(*_seed_q1_t(gt), glob, K)
        assert gt.discovery_probability(SEED, MAX_BINS - 14) == pytest.approx(seed_m)

    def test_control_identical_feeds_score_identically(self):
        # Hard Rule 46: two bins fed the same edges must tie, else the oracle is broken.
        gt = _gt()
        for e in (1, 2, 3):
            _round(gt, 5, {e})
            _round(gt, 9, {e})
        assert gt.discovery_probability(SEED, 5) == pytest.approx(gt.discovery_probability(SEED, 9))

    def test_credit_smears_across_bins_in_one_round(self):
        gt = _gt()
        gt.observe({1})
        gt.record(SEED, [3, 4], Outcome.MISS, 1.0)
        assert gt.executions(SEED, 3) == 1
        assert gt.executions(SEED, 4) == 1

    def test_same_bin_twice_in_one_round_counts_once(self):
        gt = _gt()
        gt.observe({1})
        gt.record(SEED, [3, 3], Outcome.MISS, 1.0)
        assert gt.executions(SEED, 3) == 1


class TestPropose:
    def test_declines_until_ready(self):
        gt = _gt(0.5, min_obs=3)
        assert gt.propose(SEED, MAX_BINS) is None
        for e in (1, 2):
            _round(gt, 5, {e})
        assert gt.propose(SEED, MAX_BINS) is None
        _round(gt, 5, {3})
        assert gt.propose(SEED, MAX_BINS) is not None

    def test_draw_lands_in_the_productive_bin(self):
        gt = _gt()
        for e in (1, 2, 3, 4, 5, 6):
            _round(gt, 5, {e})
        for _ in range(6):
            _round(gt, 9, {1})
        w = gt.scores(SEED)
        # random() just above every bin before 5 selects bin 5 exactly.
        target = sum(w[:5]) / sum(w) + 1e-9
        gt._rng._randoms[:] = [target]
        assert gt.propose(SEED, MAX_BINS) == 5

    def test_clamped_to_live_buffer(self):
        gt = _gt()
        _round(gt, MAX_BINS - 4, {1})
        gt._rng._randoms[:] = [0.999999]
        pos = gt.propose(SEED, 10)
        assert pos is not None and 0 <= pos <= 9

    def test_empty_buffer_declines(self):
        gt = _gt()
        _round(gt, 5, {1})
        assert gt.propose(SEED, 0) is None
        assert gt.propose(b"", 0) is None

    def test_scores_never_zero(self):
        gt = _gt()
        for _ in range(10):
            _round(gt, 5, {1})  # saturated: Q1 -> 0 everywhere but first
        assert min(gt.scores(SEED)) >= MIN_WEIGHT


class TestRecord:
    def test_record_without_observe_is_noop(self):
        gt = _gt()
        gt.record(SEED, [5], Outcome.GAIN, 1.0)
        assert gt.executions(SEED, 5) == 0

    def test_pending_edges_are_consumed_once(self):
        gt = _gt()
        gt.observe({1})
        gt.record(SEED, [5], Outcome.MISS, 1.0)
        gt.record(SEED, [5], Outcome.MISS, 1.0)  # no fresh observe
        assert gt.executions(SEED, 5) == 1

    def test_empty_and_negative_offsets_ignored(self):
        gt = _gt()
        gt.observe({1})
        gt.record(SEED, [], Outcome.MISS, 1.0)
        gt.record(SEED, [-1, -7], Outcome.MISS, 1.0)
        assert gt.seed_count() == 0

    def test_outcome_does_not_change_credit(self):
        a, b = _gt(), _gt()
        a.observe({1, 2})
        a.record(SEED, [5], Outcome.GAIN, 1.0)
        b.observe({1, 2})
        b.record(SEED, [5], Outcome.MISS, 1.0)
        assert a.discovery_probability(SEED, 5) == b.discovery_probability(SEED, 5)


class TestBounds:
    def test_bins_bounded_for_huge_seed(self):
        gt = _gt()
        big = bytes(MAX_BINS * 50)
        _round(gt, len(big) - 1, {1}, data=big)
        assert len(gt.scores(big)) <= MAX_BINS

    def test_seed_lru_bounded(self):
        gt = _gt()
        for i in range(MAX_SEEDS + 20):
            _round(gt, 0, {1}, data=i.to_bytes(4, "big") + b"x")
        assert gt.seed_count() <= MAX_SEEDS

    def test_stats_shape(self):
        gt = _gt()
        _round(gt, 5, {1})
        st = gt.stats()
        assert {"observed", "seeds", "residual_risk"} <= set(st)


def _global_q1_t(gt):
    return gt._global.q1, gt._global.t


def _seed_q1_t(gt):
    inc = gt._seed_table(SEED)
    return inc.q1, inc.t


class TestRoundFeed:
    """FuzzRound hands the mutant's edges to the arm; settle credits the bins."""

    @staticmethod
    def _round(edges, strategy):
        from types import SimpleNamespace

        from fuzzer_tool.services.fuzz_round import FuzzRound

        fr = FuzzRound.__new__(FuzzRound)
        fr._f = SimpleNamespace(_pos_good_turing=strategy, _current_edges_cache=edges)
        return fr

    def test_feed_then_settle_credits_the_bin(self):
        gt = _gt()
        self._round({1, 2}, gt)._feed_pos_good_turing()
        gt.record(SEED, [5], Outcome.MISS, 1.0)
        assert gt.executions(SEED, 5) == 1
        assert gt._seed_table(SEED).q1 == 2

    def test_feed_off_is_inert(self):
        self._round({1}, None)._feed_pos_good_turing()

    def test_non_set_edges_count_as_empty(self):
        gt = _gt()
        self._round([1, 2, 3], gt)._feed_pos_good_turing()
        gt.record(SEED, [5], Outcome.MISS, 1.0)
        assert gt.executions(SEED, 5) == 1
        assert gt._seed_table(SEED).q1 == 0
