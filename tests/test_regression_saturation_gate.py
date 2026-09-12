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
    SATURATION_MAX_GATED_EXECS,
    SATURATION_MIN_UNGATED_EXECS,
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


class TestGateDutyCycleIsBounded:
    """The gate sits on a positive feedback path; its on-time must be capped.

    The three earlier mechanisms are all keyed on the signal the gate
    suppresses: the refresh reads an estimate that stays high because the
    analyses are off, and the stall override waits 20,000 execs for an edge
    the gate makes less likely. On a plateau -- which Chao2 reports as 1.0
    unconditionally, pinned by
    ``TestGoodTuringReportsSaturationAtPlateau`` above -- that is a 20,000
    exec latch. The on-time cap plus off-time floor bound the duty cycle
    outright, independently of loop gain.

    Deliberately *not* ordinary hysteresis: a release threshold below the
    engage threshold makes the engaged state stickier, which is backwards
    for a loop whose engaged state raises the measured variable.
    """

    def _sweep(self, saturation: float, n_execs: int, step: int = 250):
        """Drive the gate over a campaign; return per-exec gated history."""
        f, p = _picker(_tracker_with(saturation))
        history = []
        for e in range(0, n_execs, step):
            f.exec_count = e
            # No new coverage ever arrives: a plateau, the case the gate is
            # worst at. Held below SATURATION_STALL_EXECS so the stall
            # override is not what produces the release.
            f._last_new_edge_exec = max(0, e - (SATURATION_STALL_EXECS - 1))
            history.append((e, p._saturation_gate()))
        return history

    def test_gate_releases_before_the_stall_override_on_a_plateau(self):
        history = self._sweep(1.0, SATURATION_STALL_EXECS)
        released = [e for e, g in history if not g]
        assert released, "gate never released on a plateau"
        first_release = min(e for e, g in history if not g and e > 0)
        assert first_release <= SATURATION_MAX_GATED_EXECS + 250
        # The point of the fix: release arrives long before the 20k override.
        assert first_release < SATURATION_STALL_EXECS

    def test_duty_cycle_is_bounded_over_a_long_plateau(self):
        history = self._sweep(1.0, 8 * SATURATION_MAX_GATED_EXECS)
        duty = sum(1 for _, g in history if g) / len(history)
        ceiling = SATURATION_MAX_GATED_EXECS / (
            SATURATION_MAX_GATED_EXECS + SATURATION_MIN_UNGATED_EXECS
        )
        # Sampling granularity costs a little slack in either direction.
        assert duty <= ceiling + 0.1, f"duty {duty:.2%} exceeds ceiling {ceiling:.2%}"
        assert duty > 0.3, f"duty {duty:.2%} -- gate is not engaging at all"

    def test_analyses_run_periodically_however_high_the_estimate(self):
        """The guarantee that actually matters downstream."""
        history = self._sweep(1.0, 4 * SATURATION_MAX_GATED_EXECS)
        window = SATURATION_MAX_GATED_EXECS + SATURATION_MIN_UNGATED_EXECS
        for start in range(0, 3 * SATURATION_MAX_GATED_EXECS, window):
            inside = [g for e, g in history if start <= e < start + 2 * window]
            assert any(not g for g in inside), (
                f"no ungated sample in execs [{start}, {start + 2 * window})"
            )

    def test_off_time_floor_blocks_immediate_re_engagement(self):
        f, p = _picker(_tracker_with(1.0))
        f.exec_count = 0
        assert p._saturation_gate() is True
        f.exec_count = SATURATION_MAX_GATED_EXECS
        assert p._saturation_gate() is False  # forced release
        # Estimate has not moved; without the floor this re-engages at once.
        f.exec_count = SATURATION_MAX_GATED_EXECS + 1
        assert p._saturation_gate() is False
        f.exec_count = SATURATION_MAX_GATED_EXECS + SATURATION_MIN_UNGATED_EXECS
        assert p._saturation_gate() is True

    def test_first_engagement_is_not_delayed_by_the_floor(self):
        """`_saturation_gate_exec` starts unset, not 0, for exactly this."""
        f, p = _picker(_tracker_with(1.0))
        f.exec_count = 0
        assert p._saturation_gate() is True

    def test_a_genuinely_unsaturated_estimate_still_keeps_the_gate_off(self):
        history = self._sweep(0.5, 4 * SATURATION_MAX_GATED_EXECS)
        assert not any(g for _, g in history)
