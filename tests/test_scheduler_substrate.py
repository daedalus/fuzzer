"""The substrate a scheduler reads before it scores anything.

Three things, each of them a consequence of
``docs/handover/handover_edge_id_axis_2026-09-18.md``: whether per-edge
statistics can be trusted at all (F1, F11), how many edges the execution
volume is actually spread over (F5), and which edges no input has yet told
apart (F10).
"""

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.scheduler_substrate import (
    EdgeCanonicalizer,
    coverage_trust,
    effective_edges,
)


class TestEffectiveEdges:
    def test_uniform_hits_give_back_the_edge_count(self):
        assert effective_edges({i: 10 for i in range(8)}) == pytest.approx(8.0)

    def test_one_edge_taking_everything_gives_one(self):
        assert effective_edges({1: 500, 2: 0, 3: 0}) == pytest.approx(1.0)

    def test_concentration_is_what_moves_it_not_edge_count(self):
        """Same 100 edges, all the volume on one: effectively a single edge."""
        spread = {i: 1 for i in range(100)}
        concentrated = {i: (1000 if i == 0 else 1) for i in range(100)}
        assert effective_edges(spread) == pytest.approx(100.0)
        assert effective_edges(concentrated) < 10.0

    def test_accepts_a_bare_iterable(self):
        assert effective_edges([4, 4, 4, 4]) == pytest.approx(4.0)

    def test_empty_and_all_zero_are_zero(self):
        assert effective_edges({}) == 0.0
        assert effective_edges({1: 0, 2: 0}) == 0.0

    def test_tracker_exposes_it(self):
        tracker = EdgeTracker()
        tracker._global_edge_hits.update({1: 5, 2: 5, 3: 5, 4: 5})
        assert tracker.effective_edges() == pytest.approx(4.0)

    def test_tracker_with_nothing_recorded(self):
        assert EdgeTracker().effective_edges() == 0.0


class TestCoverageTrust:
    def test_no_target_is_trusted(self):
        assert coverage_trust(None) == (True, None)

    def test_coverage_off_short_circuits(self):
        """The premise does not hold, so there is nothing to warn about."""
        assert coverage_trust("/nonexistent", use_coverage=False) == (True, None)

    def test_ptrace_needs_no_build_flags(self):
        assert coverage_trust("/nonexistent", ptrace=True) == (True, None)

    def test_unstable_ids_are_not_trusted(self):
        """Per-process ids make every edge a singleton owned by one seed."""
        trusted, reason = coverage_trust("/bin/sh", id_stability=0.007)
        assert not trusted
        assert "reproducible" in reason

    def test_perfect_stability_passes_that_check(self):
        trusted, _ = coverage_trust("/bin/sh", id_stability=1.0)
        assert trusted is True  # /bin/sh is stripped -> "unknown", not "absent"

    def test_stability_is_checked_before_the_binary(self):
        """A moving id space invalidates the statistics whatever the build."""
        trusted, reason = coverage_trust("/nonexistent/binary", id_stability=0.5)
        assert not trusted
        assert "reproducible" in reason


class TestEdgeCanonicalizer:
    def test_identical_profiles_collapse(self):
        seeds = {
            "a": {1: 3, 2: 3, 3: 1},
            "b": {1: 7, 2: 7, 3: 2},
        }
        canon = EdgeCanonicalizer()
        canon.refit(seeds)
        assert canon.class_of(1) == canon.class_of(2)
        assert canon.class_of(3) != canon.class_of(1)
        assert canon.multiplicity(1) == 2
        assert canon.multiplicity(3) == 1
        stats = canon.stats()
        assert (stats["edges"], stats["classes"], stats["duplicate_edges"]) == (3, 2, 1)
        assert stats["largest_class"] == 2

    def test_counts_matter_not_just_presence(self):
        """Two edges hit by the same seeds at different rates are distinct."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1, 2: 2}})
        assert canon.class_of(1) != canon.class_of(2)

    def test_a_split_is_picked_up_on_refit(self):
        """Classes must not be sticky: one input can tell members apart."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 4, 2: 4}})
        assert canon.multiplicity(1) == 2
        canon.refit({"a": {1: 4, 2: 4}, "b": {1: 1}})
        assert canon.multiplicity(1) == 1
        assert canon.stats()["duplicate_edges"] == 0

    def test_unseen_edges_are_their_own_class(self):
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1}})
        assert canon.class_of(99) == 99
        assert canon.multiplicity(99) == 1

    def test_empty_corpus(self):
        canon = EdgeCanonicalizer()
        canon.refit({})
        assert canon.stats() == {
            "edges": 0,
            "classes": 0,
            "duplicate_edges": 0,
            "duplicate_fraction": 0.0,
            "largest_class": 0,
        }

    def test_an_edge_missing_from_a_seed_reads_as_zero(self):
        """Absence is part of the profile, so it separates classes."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 2, 2: 2}, "b": {1: 2}})
        assert canon.class_of(1) != canon.class_of(2)
