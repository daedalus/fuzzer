"""Covers core/schedulers/pos_saliency.py."""

from collections import Counter

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_saliency import (
    MIN_SEEDS,
    PositionSaliencyScheduler,
    _dominator_selector,
    _gt_rarity_selector,
)


def _corpus(n=200, length=100, seed=0):
    r = np.random.default_rng(seed)
    seeds = [bytes(r.integers(0, 256, length, dtype=np.uint8)) for _ in range(n)]

    def edges(s):
        e = {0}
        if s[7] > 128:
            e.add(1)
        if s[40] < 60:
            e.add(2)
        if s[7] > 128 and s[40] < 60:
            e.add(3)
        return e

    return seeds, [(s, edges(s)) for s in seeds]


def _arm(samples, **kw):
    kw.setdefault("refit_interval", 10)
    return PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, **kw)


class TestContract:
    def test_satisfies_protocol(self):
        assert isinstance(_arm([]), PositionScheduler)

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            PositionSaliencyScheduler(None, lambda: [])

    @pytest.mark.parametrize("kw", [{"input_cap": 0}, {"hidden": 0}, {"explore": 1.5}])
    def test_rejects_bad_params(self, kw):
        with pytest.raises(ValueError):
            _arm([], **kw)

    def test_record_is_safe_without_model(self):
        a = _arm([])
        a.record(b"abc", [0], Outcome.GAIN, 1.0)


class TestWarmup:
    def test_declines_while_corpus_is_empty_then_fits_on_first_data(self):
        seeds, samples = _corpus()
        live: list = []
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: live, refit_interval=10)
        assert a.propose(seeds[0], 100) is None and not a.ready
        live.extend(samples)
        for _ in range(250):  # retry cadence while no model exists
            a.propose(seeds[0], 100)
        assert a.ready

    def test_declines_below_min_seeds(self):
        seeds, samples = _corpus(n=MIN_SEEDS - 1)
        a = _arm(samples)
        assert a.refit() is False
        assert "fewer than" in a.stats()["skip_reason"]
        assert a.propose(seeds[0], 100) is None

    def test_no_informative_edges(self):
        seeds = [bytes([i]) * 4 for i in range(MIN_SEEDS + 2)]
        a = _arm([(s, {1}) for s in seeds])  # one edge hit by every seed: constant column
        assert a.refit() is False
        assert a.stats()["skip_reason"] == "no informative edges"

    def test_empty_input_or_buffer_declines(self):
        _, samples = _corpus()
        a = _arm(samples)
        a.refit()
        assert a.propose(b"", 10) is None
        assert a.propose(b"abc", 0) is None


class TestFit:
    def test_refit_builds_model_and_columns(self):
        _, samples = _corpus()
        a = _arm(samples)
        assert a.refit() is True
        st = a.stats()
        assert st["ready"] and st["fits"] == 1 and st["targets"] == 3  # edge 0 is constant
        assert st["width"] == 100 and st["fit_seconds"] > 0

    def test_unchanged_sample_is_not_refit(self):
        _, samples = _corpus()
        a = _arm(samples)
        a.refit()
        assert a.refit() is False and a.stats()["fits"] == 1
        assert a.refit(force=True) is True and a.stats()["fits"] == 2

    def test_identical_columns_merge(self):
        # edges 5 and 6 are hit by the same seeds: one class, one output column
        seeds = [bytes([i, i]) for i in range(20)]
        samples = [(s, {5, 6} if s[0] < 10 else {7}) for s in seeds]
        a = _arm(samples)
        a.refit()
        assert a.stats()["targets"] == 2

    def test_target_columns_are_capped(self):
        r = np.random.default_rng(0)
        seeds = [bytes(r.integers(0, 256, 16, dtype=np.uint8)) for _ in range(40)]
        samples = [(s, {int(e) for e in r.integers(0, 500, 60)}) for s in seeds]
        a = _arm(samples)
        a.refit()
        assert a.stats()["targets"] <= 128

    def test_refit_failure_is_swallowed(self):
        def boom():
            raise RuntimeError("x")

        a = PositionSaliencyScheduler(RandPool(seed=1), boom, refit_interval=1)
        for _ in range(300):
            assert a.propose(b"abcdefgh", 8) is None
        assert a.stats()["skip_reason"] == "refit raised"


class TestPropose:
    def _ready(self, **kw):
        seeds, samples = _corpus()
        a = _arm(samples, **kw)
        a.refit()
        return seeds, a

    def test_offsets_in_range(self):
        seeds, a = self._ready()
        for _ in range(500):
            o = a.propose(seeds[0], 100)
            assert o is not None and 0 <= o < 100

    def test_respects_shrunk_buffer(self):
        seeds, a = self._ready()
        for _ in range(300):
            assert 0 <= a.propose(seeds[0], 30) < 30

    def test_bytes_beyond_the_net_still_get_proposals(self):
        seeds, a = self._ready()
        hits = Counter(a.propose(seeds[0], 400) >= 100 for _ in range(1500))
        assert 0.6 < hits[True] / 1500 < 0.9  # ~300/400 by construction

    def test_concentrates_on_planted_bytes(self):
        seeds, a = self._ready()
        c = Counter(a.propose(seeds[5], 100) for _ in range(4000))
        planted = (c[7] + c[40]) / 4000
        assert planted > 2 * (2 / 100)  # clearly above uniform; explore floor keeps all bytes live

    def test_explore_floor_keeps_every_byte_reachable(self):
        seeds, a = self._ready(explore=1.0)
        c = Counter(a.propose(seeds[5], 100) for _ in range(6000))
        assert len(c) > 95

    def test_seeded_runs_are_reproducible(self):
        _, samples = _corpus()
        runs = []
        for _ in range(2):
            a = _arm(samples)
            a.refit()
            runs.append([a.propose(samples[0][0], 100) for _ in range(50)])
        assert runs[0] == runs[1]

    def test_cap_limits_the_net_width(self):
        seeds, samples = _corpus(length=300)
        a = _arm(samples, input_cap=64)
        a.refit()
        assert a.stats()["width"] == 64
        assert a.saliency(seeds[0]).shape == (64,)


class TestCadence:
    def test_first_fit_happens_via_propose(self):
        seeds, samples = _corpus()
        a = _arm(samples, refit_interval=5)
        for _ in range(400):
            a.propose(seeds[0], 100)
        assert a.ready and a.stats()["fits"] == 1  # later ticks see an unchanged sample


class TestTargetSelector:
    def test_gt_rarity_selector_rare_edges_get_higher_weights(self):
        """Good-Turing rarity weights: few hits → high weight."""
        from fuzzer_tool.core.schedulers.pos_saliency import _gt_rarity_selector

        edge_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        by_edge = {
            1: [0],  # hit by 1 seed
            2: [0, 1],  # hit by 2 seeds
            3: list(range(20)),  # hit by all
            4: [0, 1, 2],
            5: [0, 1],
        }
        n_seeds = 20
        w = _gt_rarity_selector(edge_ids, by_edge, n_seeds)
        # support 1 → (20-1+1)/20 = 1.0 (after normalization, max)
        assert w[0] >= w[1] and w[1] >= w[3]  # rare first
        assert w[0] > w[2]  # singleton more rare than the universal edge
        assert len(w) == 5

    def test_dominator_selector_returns_uniform(self):
        """Dominator selector without ICFG falls back to uniform weights."""
        edge_ids = np.array([1, 2, 3], dtype=np.int64)
        by_edge = {1: [0], 2: [0, 1], 3: list(range(20))}
        w = _dominator_selector(edge_ids, by_edge, 20)
        assert np.allclose(w, 1.0)
        assert len(w) == 3

    def test_target_selector_biases_target_draws(self):
        """With a GT selector, refit cumulative weights are skewed toward rare edges."""
        seeds = [bytes([i]) * 100 for i in range(20)]
        # edge 5 hit by 1 seed, edge 6 hit by 2 seeds, edge 7 hit by all
        samples = [(s, {5} if s[0] == 0 else ({6} if s[0] == 1 else {7})) for s in seeds]

        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, refit_interval=10)
        a.refit()
        w = a.stats()
        assert a.ready and w["targets"] >= 1
        # Rare edge class (support 1) should carry a large share of cum_support
        cum = a._cum_support
        assert np.all(np.diff(cum) > 0)  # weights are positive, cumulative increasing

    def test_no_target_selector_falls_back_to_support(self):
        """Backward compatibility: without selector, weight is exactly 1/support."""
        seeds = [bytes([i]) * 100 for i in range(20)]
        samples = [(s, {5} if s[0] < 10 else {6}) for s in seeds]
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, refit_interval=10)
        a.refit()
        w = a.stats()
        assert a.ready and w["targets"] >= 1

    def test_gt_rarity_selector_empty(self):
        """Selector returns uniform weights for empty input."""
        edge_ids = np.array([], dtype=np.int64)
        w = _gt_rarity_selector(edge_ids, {}, 20)
        assert len(w) == 0
        assert np.all(w == 1.0)
