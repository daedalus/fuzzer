"""Unit tests for core.occupation primitives (no wiring)."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.occupation import LongitudinalRarity, OccupationMeasure


class TestOccupationMeasure:
    def test_empty(self):
        m = OccupationMeasure.from_counts({})
        assert m.is_empty()
        assert m.total_visits == 0
        assert m.support_size() == 0
        assert m.dominant() is None
        assert m.shannon_entropy() == 0.0
        assert m.renyi_entropy(2.0) == 0.0
        assert m.top_k(5) == []

    def test_single_edge(self):
        m = OccupationMeasure.from_counts({7: 10})
        assert not m.is_empty()
        assert m.total_visits == 10
        assert m.support_size() == 1
        assert m.dominant() == 7
        assert m.mass_of(7) == pytest.approx(1.0)
        assert m.count_of(7) == 10
        assert m.shannon_entropy() == pytest.approx(0.0)
        assert m.renyi_entropy(0) == pytest.approx(0.0)  # log2(1)

    def test_uniform_two(self):
        m = OccupationMeasure.from_counts({1: 5, 2: 5})
        assert m.mass_of(1) == pytest.approx(0.5)
        assert m.shannon_entropy() == pytest.approx(1.0)
        assert m.renyi_entropy(2.0) == pytest.approx(1.0)

    def test_raw_not_normalised(self):
        m = OccupationMeasure.from_counts({3: 2, 4: 6}, normalised=False)
        assert m.mass_of(3) == pytest.approx(2.0)
        assert m.mass_of(4) == pytest.approx(6.0)
        assert m.shannon_entropy() == pytest.approx(
            OccupationMeasure.from_counts({3: 2, 4: 6}).shannon_entropy()
        )

    def test_drop_zero(self):
        m = OccupationMeasure.from_counts({1: 0, 2: 3, 3: 0})
        assert 1 not in m.counts
        assert m.counts == {2: 3}

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            OccupationMeasure.from_counts({1: -1})

    def test_from_iterable_pairs(self):
        m = OccupationMeasure.from_counts([(9, 1), (9, 2), (8, 3)])
        assert m.counts[9] == 3
        assert m.counts[8] == 3
        assert m.total_visits == 6

    def test_top_k_and_snapshot(self):
        m = OccupationMeasure.from_counts({1: 1, 2: 10, 3: 5, 4: 2})
        top = m.top_k(2)
        assert [e for e, _ in top] == [2, 3]
        snap = m.sparse_snapshot(max_edges=2)
        assert set(snap) == {2, 3}
        assert m.sparse_snapshot(max_edges=0) == {}
        assert m.sparse_snapshot() == m.counts

    def test_push_forward(self):
        m = OccupationMeasure.from_counts({10: 4, 11: 6, 20: 10})
        partition = {10: 100, 11: 100, 20: 200}
        mac = m.push_forward(partition)
        assert mac.counts == {100: 10, 200: 10}
        assert mac.mass_of(100) == pytest.approx(0.5)

    def test_push_forward_drops_unknown(self):
        m = OccupationMeasure.from_counts({1: 5, 2: 5})
        mac = m.push_forward({1: 9})
        assert mac.counts == {9: 5}
        assert mac.total_visits == 5

    def test_total_variation(self):
        a = OccupationMeasure.from_counts({1: 1, 2: 0})
        # drop_zero removes 2
        a = OccupationMeasure.from_counts({1: 10})
        b = OccupationMeasure.from_counts({1: 5, 2: 5})
        assert a.total_variation(a) == pytest.approx(0.0)
        tv = a.total_variation(b)
        assert tv == pytest.approx(0.5)
        empty = OccupationMeasure.from_counts({})
        assert empty.total_variation(empty) == 0.0

    def test_renyi_alpha_limits(self):
        m = OccupationMeasure.from_counts({1: 1, 2: 1, 3: 1, 4: 1})
        assert m.renyi_entropy(0) == pytest.approx(2.0)  # log2(4)
        assert m.renyi_entropy(1.0) == pytest.approx(m.shannon_entropy())


class TestLongitudinalRarity:
    def test_unobserved_is_max_rare(self):
        lr = LongitudinalRarity()
        assert lr.rarity(42) == 1.0
        assert lr.n_histories == 0

    def test_heavy_occupation_low_rarity(self):
        lr = LongitudinalRarity()
        for _ in range(5):
            lr.observe(OccupationMeasure.from_counts({1: 100}))
        assert lr.mean_mass(1) == pytest.approx(1.0)
        assert lr.rarity(1) <= 0.5

    def test_light_occupation_high_rarity(self):
        lr = LongitudinalRarity()
        # edge 1 appears with tiny mass among many edges
        for _ in range(5):
            counts = {1: 1, **{i: 20 for i in range(2, 12)}}
            lr.observe(OccupationMeasure.from_counts(counts))
        assert lr.mean_mass(1) < 0.05
        assert lr.rarity(1) > lr.rarity(2)

    def test_clear(self):
        lr = LongitudinalRarity()
        lr.observe(OccupationMeasure.from_counts({1: 1}))
        lr.clear()
        assert lr.n_histories == 0
        assert lr.rarity(1) == 1.0

    def test_rarity_map_keys(self):
        lr = LongitudinalRarity()
        lr.observe(OccupationMeasure.from_counts({3: 1, 4: 2}))
        rm = lr.rarity_map()
        assert set(rm) == {3, 4}
        assert all(0.0 <= v <= 1.0 for v in rm.values())
