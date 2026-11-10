"""Tests for tools/count_buckets.py: fixed AFL ladder vs exact 1-D k-means.

Replay runs through injected callables, so no target is built here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import count_buckets as cb  # noqa: E402
import edge_hit_spectrum as ehs  # noqa: E402

from fuzzer_tool.core.kmeans1d import Scale  # noqa: E402


def _freq(**by_count) -> np.ndarray:
    freq = np.zeros(cb.MAX_COUNT + 1, dtype=np.uint64)
    for c, n in by_count.items():
        freq[int(c[1:])] = n
    return freq


class TestHistogram:
    def test_add_counts_occurrences(self):
        h = cb.Histogram()
        h.add({1: 1, 2: 3, 3: 3})
        h.add({9: 3})
        f = h.freq()
        assert (f[1], f[2], f[3]) == (1, 0, 3)
        assert f.sum() == 4

    def test_saturates_at_byte(self):
        h = cb.Histogram()
        h.add({1: 300, 2: 255, 3: 256, 4: 1 << 20})
        f = h.freq()
        assert f[cb.MAX_COUNT] == 4
        assert f.size == cb.MAX_COUNT + 1

    def test_ignores_zero_and_empty(self):
        h = cb.Histogram()
        h.add({})
        h.add({1: 0})
        assert h.freq().sum() == 0

    def test_roundtrip(self, tmp_path):
        h = cb.Histogram()
        h.add({1: 1, 2: 7, 3: 200})
        path = tmp_path / "h.pkl"
        h.save(path)
        assert cb.Histogram.load(path).freq().tolist() == h.freq().tolist()


class TestSaveOverwrite:
    def test_second_save_replaces_first(self, tmp_path):
        # EdgePickleWriter is append-only and skips keys already on disk, so a
        # re-run with the same -o used to keep the first run's data.
        path = tmp_path / "h.pkl"
        old = cb.Histogram()
        old.save(path)  # all zeros, as an empty replay leaves it
        new = cb.Histogram()
        new.add({1: 5, 2: 5})
        new.save(path)
        assert cb.Histogram.load(path).freq().tolist() == new.freq().tolist()


class TestLadder:
    def test_starts_follow_afl_classes(self):
        counts = np.arange(1, 256)
        starts = cb.ladder_starts(counts)
        firsts = counts[starts].tolist()
        assert firsts == [1, 2, 3, 4, 8, 16, 32, 64, 128]

    def test_only_observed_values(self):
        # 5 and 6 share a class; 9 is the next.
        starts = cb.ladder_starts(np.array([1, 5, 6, 9]))
        assert starts.tolist() == [0, 1, 3]


class TestCompare:
    def test_dp_never_worse_than_ladder(self):
        rng = np.random.default_rng(7)
        freq = np.zeros(cb.MAX_COUNT + 1, dtype=np.uint64)
        freq[1:] = rng.integers(1, 1000, cb.MAX_COUNT)
        for scale in Scale:
            r = cb.compare(freq, scale)
            assert r.dp_cost <= r.ladder_cost + 1e-9
            assert r.k == len(r.ladder_starts)
            assert len(r.dp_starts) == r.k

    def test_ladder_optimal_when_data_is_the_ladder(self):
        # Ladder classes {1} {2} {64,65}: the DP at k=3 cannot beat it.
        r = cb.compare(_freq(c1=100, c2=100, c64=100, c65=100), Scale.LOG)
        assert r.k == 3
        assert r.dp_cost == pytest.approx(r.ladder_cost, abs=1e-9)
        assert r.k_match == 3

    def test_ladder_loses_when_it_merges_heavy_values(self):
        # 4 and 7 share a ladder class but carry the mass; 8, 9 are noise.
        r = cb.compare(_freq(c4=1000, c7=1000, c8=1, c9=1), Scale.LOG)
        assert r.k == 2
        assert r.ladder_cost > 100 * r.dp_cost
        assert r.dp_starts.tolist() == [0, 1]

    def test_k_match_is_smallest_k_that_ties(self):
        # Four distinct values, each its own ladder class: ladder cost 0,
        # and no 3-bucket partition reaches 0.
        r = cb.compare(_freq(c1=1000, c2=1000, c3=1000, c200=1000), Scale.LOG)
        assert (r.k, r.k_match) == (4, 4)
        assert r.ladder_cost == pytest.approx(0.0, abs=1e-9)

    def test_fewer_values_than_buckets(self):
        r = cb.compare(_freq(c1=5, c2=5), Scale.LOG)
        assert r.ladder_cost == 0.0 and r.dp_cost == 0.0
        assert r.k_match == 2

    def test_single_value_has_no_variance(self):
        r = cb.compare(_freq(c5=9), Scale.LOG)
        assert r.total_cost == 0.0
        assert r.unexplained(r.ladder_cost) == 0.0

    def test_control_same_input_same_result(self):
        freq = _freq(c1=10, c3=4, c9=2, c40=1, c130=8)
        a = cb.compare(freq, Scale.LINEAR)
        b = cb.compare(freq, Scale.LINEAR)
        assert (a.dp_cost, a.ladder_cost, a.k_match) == (b.dp_cost, b.ladder_cost, b.k_match)

    def test_empty_histogram_rejected(self):
        with pytest.raises(ValueError):
            cb.compare(np.zeros(cb.MAX_COUNT + 1, dtype=np.uint64), Scale.LOG)

    def test_k_below_one_rejected(self):
        with pytest.raises(ValueError):
            cb.compare(_freq(c1=1, c2=1), Scale.LOG, k=0)


class TestRanges:
    def test_labels(self):
        counts = np.array([1, 2, 5, 6, 9])
        assert cb.ranges(counts, np.array([0, 2, 4])) == ["1-2", "5-6", "9"]


class TestTap:
    def test_tap_sees_every_exec_and_returns_counts(self):
        seen = []
        counts = ehs.tapped(lambda: {1: 2}, seen.append)
        assert counts() == {1: 2}
        counts()
        assert seen == [{1: 2}, {1: 2}]

    def test_no_tap_returns_source_unchanged(self):
        src = lambda: {}  # noqa: E731
        assert ehs.tapped(src, None) is src
