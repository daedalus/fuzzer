"""Regression: the >=99% saturation gate must not latch on.

The gate replaces the per-seed discovery analyses (subsumption, hit-count
diversity, Wasserstein, coverage proximity) with neutral multipliers when the
estimated sample coverage is >= 0.99. Three properties were missing:

* the estimate was recomputed only when a NEW EDGE arrived, so the single
  event able to clear the gate was the one the gate makes less likely;
* Chao2 returns 1.0 for any plateau, so the gate switched on exactly when the
  discovery signals were most needed;
* ``_cached_weights`` is populated on absence only, so neutral tuples written
  while gated outlived the gate.
"""

import types

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.seed_picker import (
    SATURATION_REFRESH_EXECS,
    SATURATION_STALL_EXECS,
    SeedPicker,
)


class _Fuzzer:
    def __init__(self, tracker=None):
        self._edge_tracker = tracker or EdgeTracker()
        self.exec_count = 0
        self._last_new_edge_exec = 0
        self._cached_weights: dict = {}


def _picker(tracker=None):
    f = _Fuzzer(tracker)
    return f, SeedPicker(f)


def _tracker_with(saturation: float) -> EdgeTracker:
    t = EdgeTracker()
    t.good_turing_estimate = lambda: {"saturation": saturation}  # type: ignore[method-assign]
    return t


class TestGoodTuringReportsSaturationAtPlateau:
    """The premise of the fix, pinned so it is not taken on faith."""

    def test_closed_universe_reads_fully_saturated(self):
        t = EdgeTracker()
        universe = list(range(200))
        for i in range(30):
            t.record_edges(f"s{i}", set(universe))
        assert t.good_turing_estimate()["saturation"] == 1.0

    def test_single_starved_seed_also_reads_fully_saturated(self):
        t = EdgeTracker()
        t.record_edges("s0", {1, 2, 3, 4, 5})
        assert t.good_turing_estimate()["saturation"] == 1.0


class TestGateIsReEvaluated:
    def test_estimate_refreshes_without_a_new_edge(self):
        f, p = _picker(_tracker_with(1.0))
        assert p._saturation_gate() is True

        # Saturation drops, but no new edge is recorded, so the old
        # invalidation path never fires.
        f._edge_tracker.good_turing_estimate = lambda: {"saturation": 0.2}
        f.exec_count = SATURATION_REFRESH_EXECS - 1
        assert p._saturation_gate() is True  # still inside the refresh window

        f.exec_count = SATURATION_REFRESH_EXECS
        f._last_new_edge_exec = f.exec_count
        assert p._saturation_gate() is False

    def test_stall_forces_the_gate_off(self):
        f, p = _picker(_tracker_with(1.0))
        assert p._saturation_gate() is True
        f.exec_count = SATURATION_STALL_EXECS
        f._last_new_edge_exec = 0
        assert p._saturation_gate() is False

    def test_recent_discovery_keeps_the_gate_on(self):
        f, p = _picker(_tracker_with(1.0))
        f.exec_count = SATURATION_STALL_EXECS * 4
        f._last_new_edge_exec = f.exec_count - 10
        assert p._saturation_gate() is True

    def test_new_edge_invalidation_still_works(self):
        f, p = _picker(_tracker_with(1.0))
        assert p._saturation_gate() is True
        f._edge_tracker.good_turing_estimate = lambda: {"saturation": 0.1}
        f._saturation = None  # what Fuzzer does when a new edge arrives
        assert p._saturation_gate() is False


class TestCachedWeightsAreFlushedOnFlip:
    def test_neutral_entries_do_not_outlive_the_gate(self):
        f, p = _picker(_tracker_with(1.0))
        assert p._saturation_gate() is True
        f._cached_weights["seedkey"] = (1.0, 1.0, 1.0, 0.5)

        f._edge_tracker.good_turing_estimate = lambda: {"saturation": 0.0}
        f._saturation = None
        assert p._saturation_gate() is False
        assert "seedkey" not in f._cached_weights

    def test_no_flush_while_the_gate_holds(self):
        f, p = _picker(_tracker_with(1.0))
        p._saturation_gate()
        f._cached_weights["seedkey"] = (1.0, 1.0, 1.0, 0.5)
        p._saturation_gate()
        assert "seedkey" in f._cached_weights

    def test_full_analyses_are_dropped_when_the_gate_turns_on(self):
        f, p = _picker(_tracker_with(0.0))
        assert p._saturation_gate() is False
        f._cached_weights["seedkey"] = (0.3, 2.0, 1.4, 0.9)
        f._edge_tracker.good_turing_estimate = lambda: {"saturation": 1.0}
        f._saturation = None
        assert p._saturation_gate() is True
        assert "seedkey" not in f._cached_weights


class TestGateConsumerReadsTheFlag:
    def test_weight_helper_follows_the_gate_not_the_raw_estimate(self):
        """_weight_secretary_and_cached used to test f._saturation >= 0.99
        directly, which bypasses both the staleness refresh and the stall
        override."""
        t = _tracker_with(1.0)
        f = _Fuzzer(t)
        f._secretary = None
        f._seed_secretary = {}
        f.exec_count = SATURATION_STALL_EXECS
        f._last_new_edge_exec = 0
        f._edge_tracker.seed_edges = {"sk": {1, 2, 3}}
        calls = {"n": 0}

        def _sub(key):
            calls["n"] += 1
            return 0.5

        f._edge_tracker.compute_subsumption_weight = _sub
        f._edge_tracker.compute_hitcount_diversity_weight = lambda k: 1.0
        f._edge_tracker.compute_wasserstein_weight = lambda k: 1.0
        f._edge_tracker.compute_coverage_proximity = lambda k: 0.5
        p = SeedPicker(f)

        assert p._saturation_gate() is False  # stalled: gate off
        p._weight_secretary_and_cached("sk", 1.0, {}, f)
        assert calls["n"] == 1  # the real analysis ran


class TestSeedPickerConstruction:
    def test_picker_accepts_a_bare_namespace(self):
        f = types.SimpleNamespace(
            _edge_tracker=_tracker_with(0.5),
            exec_count=0,
            _last_new_edge_exec=0,
            _cached_weights={},
        )
        assert SeedPicker(f)._saturation_gate() is False
