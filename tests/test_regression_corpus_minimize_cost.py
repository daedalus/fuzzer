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
