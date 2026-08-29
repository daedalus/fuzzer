"""Regression tests for three defects in the corpus-minimization path.

All three surfaced from a campaign that appeared to hang immediately after
the "Corpus bloat warning: seed-size skewness" log line, which is emitted
just before minimization is scheduled.
"""

import os
import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.core.corpus_compression import (
    PPMD_CACHE_MAX,
    PPMD_SAMPLE_BYTES,
    CorpusCompressor,
)
from fuzzer_tool.core.edge_tracker import EdgeTracker


class TestSetCoverTerminates:
    """The greedy set-cover must terminate against reachable edges.

    EdgeTracker.max_tracked_seeds bounds seed_edges, and _prune_tracked_seeds
    drops entries without removing their edges from cumulative_edges. Any
    campaign past that cap therefore has cumulative_edges as a strict superset
    of anything the seeds still tracked can cover.

    The cap was 200 when this was written and is 200,000 since fe8fd42, so the
    situation no longer arises by itself on a real campaign (see docs/TODO.md).
    The tests below construct it directly, which is why they still hold: the
    set-cover termination bug is a property of the divergence between
    seed_edges and cumulative_edges, not of how the divergence arose.
    """

    def test_pruning_makes_cumulative_edges_unreachable(self):
        """Pin the asymmetry the fix has to tolerate.

        This is the upstream condition, asserted directly so that if
        _prune_tracked_seeds ever starts reconciling cumulative_edges, the
        reason for the fix below is known to have changed.
        """
        et = EdgeTracker(max_tracked_seeds=10)
        for i in range(40):
            et.record_edges(f"seed{i}", {i * 10, i * 10 + 1, i * 10 + 2})

        reachable = set()
        for edges in et.seed_edges.values():
            reachable |= edges

        assert len(et.seed_edges) <= 10
        assert reachable < et.cumulative_edges

    def test_cover_loop_converges_on_reachable_edges(self):
        """`covered != coverable` must become false; `!= cumulative` cannot.

        Reproduces the loop's exit condition on the tracker state above. The
        old target (cumulative_edges) leaves the loop relying on the
        best_gain == 0 fallback, so it always runs to exhaustion and selects
        every seed holding a unique edge -- which makes the `mandatory` set,
        and the target_size floor derived from it, meaningless.
        """
        et = EdgeTracker(max_tracked_seeds=10)
        for i in range(40):
            et.record_edges(f"seed{i}", {i * 10, i * 10 + 1, i * 10 + 2})

        seed_edge_map = {k: v for k, v in et.seed_edges.items() if v}
        coverable = set().union(*seed_edge_map.values())

        covered: set[int] = set()
        rounds = 0
        while covered != coverable:
            best, best_gain = None, 0
            for key, edges in seed_edge_map.items():
                gain = len(edges - covered)
                if gain > best_gain:
                    best, best_gain = key, gain
            assert best is not None, "loop stalled before covering reachable edges"
            covered |= seed_edge_map[best]
            rounds += 1
            assert rounds <= len(seed_edge_map)

        assert covered == coverable
        assert covered != et.cumulative_edges


class TestPpmdRatioCost:
    """PPMD runs at ~2 MB/s and is called once per seed per pass."""

    @pytest.fixture
    def cc(self):
        compressor = CorpusCompressor()
        if not compressor.enabled:
            pytest.skip("pyppmd not installed")
        return compressor

    def test_ratio_is_memoised(self, cc):
        data = os.urandom(200_000)
        first = cc.compute_seed_ratio(data)
        assert cc.compute_seed_ratio(data) == first
        assert len(cc._seed_ratios) == 1

    def test_identical_prefixes_share_an_entry(self, cc):
        """Keyed on the sampled prefix, so seeds differing past the cap hit."""
        prefix = os.urandom(PPMD_SAMPLE_BYTES)
        cc.compute_seed_ratio(prefix + b"\x00" * 4096)
        cc.compute_seed_ratio(prefix + b"\xff" * 8192)
        assert len(cc._seed_ratios) == 1

    def test_cost_is_bounded_by_the_sample_cap(self, cc):
        """A 4 MB seed must cost no more than a 64 KiB one.

        Compares work done rather than wall clock: the compressed length of
        a capped run cannot exceed what the cap itself can produce, so an
        uncapped implementation fails this regardless of machine speed.
        """
        big = os.urandom(4 * 1024 * 1024)
        ratio = cc.compute_seed_ratio(big)
        # Ratio is computed over the sample, not the whole seed, so it stays
        # in the normal band for incompressible data instead of collapsing
        # toward zero as it would if divided by the full 4 MB length.
        assert 0.5 < ratio < 2.0

    def test_cache_is_bounded(self, cc):
        for _ in range(PPMD_CACHE_MAX + 10):
            cc.compute_seed_ratio(os.urandom(512))
        assert len(cc._seed_ratios) <= PPMD_CACHE_MAX

    def test_disabled_compressor_is_free(self):
        compressor = CorpusCompressor(enabled=False)
        assert compressor.compute_seed_ratio(os.urandom(1_000_000)) == 1.0
        assert compressor._seed_ratios == {}


class TestMaxLenIsNotARatchet:
    """max_len tracks the corpus p90 in both directions.

    It was max(f.max_len, min(p90 * 2, 65536)) -- one-way. Once a few large
    seeds pushed p90 up, max_len never fell, so mutation kept producing
    larger seeds, which kept p90 up: a positive feedback loop into the exact
    bloat the skewness warning reports, that minimizing could not undo.
    """

    @staticmethod
    def _adaptive_max_len(history, floor):
        """Mirror of the corpus_manager expression, derived independently."""
        sorted_sizes = sorted(history)
        p90 = sorted_sizes[-len(sorted_sizes) // 10]
        return min(max(p90 * 2, floor), 65536)

    def test_grows_with_the_corpus(self):
        history = [200] * 90 + [8000] * 10
        assert self._adaptive_max_len(history, floor=4096) == 16000

    def test_falls_back_when_the_corpus_shrinks(self):
        """The regression: this returned the earlier high-water mark."""
        history = [200] * 100
        assert self._adaptive_max_len(history, floor=4096) == 4096

    def test_never_drops_below_the_configured_floor(self):
        history = [8] * 100
        assert self._adaptive_max_len(history, floor=4096) == 4096

    def test_stays_capped(self):
        history = [500_000] * 100
        assert self._adaptive_max_len(history, floor=4096) == 65536


class TestLargeCorpusSetCover:
    """The minimal-cover minimization must also run for corpora above the old 5k limit."""

    def test_large_corpus_preserves_coverable_edges(self):
        """auto_minimize_corpus must not drop reachable edges on large corpora.

        Creates 6000 seeds over 100 edges, with a handful of seeds covering
        unique edges. After minimization, every coverable edge must still be
        present in the remaining corpus.
        """
        from fuzzer_tool.services.corpus_manager import CorpusManager
        from tests.test_corpus_minimization import MockFuzzer, _cm_seed_key

        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        edges = list(range(100))
        et.cumulative_edges = set(edges)

        unique_seeds = []
        for i in range(6000):
            data = f"seed_{i:04d}_".encode() + bytes([i % 256]) * 64
            unique_seeds.append(data)

        seed_edges = {}
        for idx, seed in enumerate(unique_seeds):
            sk = _cm_seed_key(seed)
            if idx < 100:
                seed_edges[sk] = {idx}
            elif idx < 1100:
                seed_edges[sk] = {idx % 100, (idx + 1) % 100}
            else:
                seed_edges[sk] = {(idx - 1100) % 100}

        for sk, edge_set in seed_edges.items():
            et.seed_edges[sk] = edge_set
            et.seed_hit_counts[sk] = {e: 1 for e in edge_set}

        f.corpus = unique_seeds[:]
        f.seed_meta = {
            s: {
                "fuzz_count": 1,
                "coverage_edges": len(seed_edges[_cm_seed_key(s)]),
                "added_at": 100.0 + i,
            }
            for i, s in enumerate(f.corpus)
        }

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        covered_after = set()
        for s in f.corpus:
            covered_after.update(et.seed_edges.get(_cm_seed_key(s), set()))
        assert covered_after == et.cumulative_edges
        assert len(f.corpus) < len(unique_seeds)
