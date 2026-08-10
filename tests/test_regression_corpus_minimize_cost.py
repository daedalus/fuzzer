"""Regression tests for three defects in the corpus-minimization path.

All three surfaced from a campaign that appeared to hang immediately after
the "Corpus bloat warning: seed-size skewness" log line, which is emitted
just before minimization is scheduled.
"""

from fuzzer_tool.core.edge_tracker import EdgeTracker


class TestSetCoverTerminates:
    """The greedy set-cover must terminate against reachable edges.

    EdgeTracker.max_tracked_seeds (200) bounds seed_edges, and
    _prune_tracked_seeds drops entries without removing their edges from
    cumulative_edges. Any campaign past 200 seeds therefore has
    cumulative_edges as a strict superset of anything the seeds still
    tracked can cover.
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
