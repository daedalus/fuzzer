"""Tests for invasion_select (percolation handover Module 4).

operator_stats follows the (successes, failures) shape every scheduler's
bandit_stats() returns (see core/schedulers/monte_carlo.py:654).
"""

from fuzzer_tool.services.seed_picker import INVASION_STUCK_THRESHOLD, invasion_select


class TestInvasionSelect:
    def test_selects_lowest_resistance_operator(self):
        # havoc: 9/10 success (resistance 1/0.9), splice: 1/10 (resistance
        # 1/0.1) -- havoc is unambiguously the better bond to invade next.
        stats = {"havoc": (9.0, 1.0), "splice": (1.0, 9.0)}
        assert invasion_select(stats) == "havoc"

    def test_untried_arm_preferred_over_low_success(self):
        # No observations at all beats a poor observed success rate --
        # invasion explores the unknown bond before writing anything off.
        stats = {"tried": (1.0, 9.0), "untried": (0.0, 0.0)}
        assert invasion_select(stats) == "untried"

    def test_deterministic_tie_break_is_alphabetical(self):
        stats = {"zzz": (5.0, 5.0), "aaa": (5.0, 5.0)}
        assert invasion_select(stats) == "aaa"

    def test_stuck_when_every_operator_at_or_above_threshold(self):
        # Every operator resisting more than INVASION_STUCK_THRESHOLD ->
        # nothing left worth invading, caller should reseed/switch strategy.
        stats = {"a": (0.0, 10.0), "b": (0.0, 10.0)}
        assert invasion_select(stats) is None

    def test_not_stuck_just_below_threshold(self):
        # success_rate = 1/INVASION_STUCK_THRESHOLD gives resistance exactly
        # at the threshold from below (independent arithmetic, not an echo
        # of the implementation): pick a success_rate strictly above
        # 1/INVASION_STUCK_THRESHOLD so resistance is strictly below it.
        success_rate = 1.0 / INVASION_STUCK_THRESHOLD + 0.05
        successes = success_rate * 10.0
        failures = 10.0 - successes
        stats = {"a": (successes, failures)}
        assert invasion_select(stats) == "a"

    def test_empty_operator_stats_returns_none(self):
        assert invasion_select({}) is None

    def test_empty_frontier_returns_none_even_with_stats(self):
        stats = {"havoc": (9.0, 1.0)}
        assert invasion_select(stats, frontier_edges=set()) is None

    def test_none_frontier_does_not_short_circuit(self):
        stats = {"havoc": (9.0, 1.0)}
        assert invasion_select(stats, frontier_edges=None) == "havoc"

    def test_nonempty_frontier_does_not_short_circuit(self):
        stats = {"havoc": (9.0, 1.0)}
        assert invasion_select(stats, frontier_edges={1, 2, 3}) == "havoc"


class TestInvasionSelectAdversarial:
    def test_all_zero_successes_with_failures_is_stuck(self):
        # 0% success rate everywhere -> infinite resistance -> stuck.
        stats = {"a": (0.0, 5.0), "b": (0.0, 1.0)}
        assert invasion_select(stats) is None

    def test_single_untried_operator_is_selected_not_stuck(self):
        # The only operator has no data at all -- still selected (explore),
        # not reported as stuck.
        stats = {"only": (0.0, 0.0)}
        assert invasion_select(stats) == "only"


def test_nonempty_frontier_does_not_block_selection():
    """Production now passes discovery_frontier_edges(); a live set must not short-circuit."""
    stats = {"havoc": (9.0, 1.0), "splice": (1.0, 9.0)}
    assert invasion_select(stats, frontier_edges={1, 2, 3}) == "havoc"


def test_discovery_frontier_edges_shape():
    from fuzzer_tool.core.edge_tracker import EdgeTracker

    et = EdgeTracker.__new__(EdgeTracker)
    et._edge_first_seen = {}
    assert et.discovery_frontier_edges() is None
    et._edge_first_seen = {10: 1, 20: 100, 30: 101}
    frontier = et.discovery_frontier_edges()
    assert frontier is not None
    assert 10 not in frontier
    assert 20 in frontier and 30 in frontier


class TestDiscoveryFrontierMemo:
    """discovery_frontier_edges is called per operator selection (P2-3)."""

    def _tracker(self, first_seen):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        et = EdgeTracker()
        et._edge_first_seen = dict(first_seen)
        et._frontier_cache = None
        return et

    def test_memo_matches_uncached_result(self):
        import random

        from fuzzer_tool.core.edge_tracker import _FRONTIER_FRACTION

        rnd = random.Random(4)
        fs = {e: rnd.randrange(1, 5000) for e in range(500)}
        et = self._tracker(fs)

        clock = max(fs.values())
        cutoff = clock * (1.0 - _FRONTIER_FRACTION)
        expected = {e for e, t0 in fs.items() if t0 >= cutoff}

        assert et.discovery_frontier_edges() == expected
        # Second call is served from the memo and must agree.
        assert et.discovery_frontier_edges() == expected

    def test_memo_invalidates_when_an_edge_is_recorded(self):
        et = self._tracker({1: 100, 2: 200})
        before = set(et.discovery_frontier_edges())
        et._edge_first_seen[3] = 1000
        after = et.discovery_frontier_edges()
        assert after != before
        assert after == {3}

    def test_empty_and_zero_clock_return_none(self):
        assert self._tracker({}).discovery_frontier_edges() is None
        et = self._tracker({1: 0, 2: 0})
        assert et.discovery_frontier_edges() is None
        # Cached None is returned unchanged.
        assert et.discovery_frontier_edges() is None

    def test_from_dict_drops_a_stale_memo(self):
        et = self._tracker({1: 10, 2: 20, 3: 30})
        et.discovery_frontier_edges()
        # A restored map of the same size must not reuse the old answer.
        et.from_dict({"edge_first_seen": {"7": 10, "8": 20, "9": 30}})
        assert et.discovery_frontier_edges() == {9}

    def test_memo_is_not_serialized(self):
        et = self._tracker({1: 10, 2: 20})
        et.discovery_frontier_edges()
        assert "frontier_cache" not in et.to_dict()
        assert "_frontier_cache" not in et.to_dict()
