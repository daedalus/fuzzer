"""Regression for HIGH finding #17 from docs/bugreport_2026-08-21_merged.md.

Fails against the pre-fix source.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.crash_eta import CrashMITracker


class TestCrashMITrackerNonCrashObservations:
    """#17 core/crash_eta.py:66-135 -- CrashMITracker.record only tracked
    positions for crashing inputs, so byte_total == joint_crash always;
    MI degenerated to position frequency x log2(1/p_crash)."""

    def test_non_crash_inputs_feed_byte_total(self):
        tracker = CrashMITracker(min_observations=1)
        tracker.record(b"AAAAAAAAAA", is_crash=False)
        assert tracker.total_execs == 1
        assert len(tracker.position_counts) == 10
        assert len(tracker.joint_crash) == 0

    def test_byte_total_diverges_from_joint_crash(self):
        tracker = CrashMITracker(min_observations=1)
        for _ in range(20):
            tracker.record(b"AAAAAAAAAA", is_crash=False)
        tracker.record(b"BBBBBBBBBB", is_crash=True)

        # Pre-fix, byte_total == joint_crash for every tracked position.
        assert tracker.byte_total != tracker.joint_crash
        # Position 0 saw both the non-crashing 'A' and the crashing 'B'.
        assert tracker.byte_total[0][ord("A")] == 20
        assert tracker.byte_total[0][ord("B")] == 1
        assert tracker.joint_crash[0].get(ord("A"), 0) == 0
        assert tracker.joint_crash[0][ord("B")] == 1

    def test_mi_is_not_driven_purely_by_position_frequency(self):
        """With non-crash data present, a byte value seen equally often in
        crashing and non-crashing runs should show near-zero MI, not the
        max-possible MI the pre-fix degenerate formula produced."""
        tracker = CrashMITracker(min_observations=1)
        rng = random.Random(0)
        # Small alphabet with many observations per value, so this isn't
        # confounded by the separate high-cardinality small-sample noise
        # that afflicts MI/TE estimators generally (out of scope for #17).
        for _ in range(400):
            is_crash = rng.random() < 0.5
            # Byte value at position 0 is independent of crash outcome.
            tracker.record(bytes([rng.randint(0, 3)]), is_crash=is_crash)

        assert tracker.mi(0) < 0.2
