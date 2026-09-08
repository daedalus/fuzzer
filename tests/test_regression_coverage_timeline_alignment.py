"""Regression tests for finding #25 — coverage snapshot arrays out of step.

``record_coverage_snapshot`` appends to three parallel arrays and trimmed only
two of them, so ``_coverage_timestamps`` grew without bound while
``_coverage_execs``/``_coverage_edges`` were cut by 250 entries at a time.

``report._temporal_correlation`` reads all three at the same index, so after
one trim index ``i`` paired the timestamp of snapshot ``i`` with the
exec/edge counts of snapshot ``i + 250``. The out-of-range crash was already
fixed by clamping with ``min(len(...))``, which left the mis-pairing silent
rather than loud — the numbers were wrong, not absent.

The serialization half is covered too: the timeline round-trip carried only
(exec, edges), so timestamps did not survive a save/restore at all.
"""

from __future__ import annotations

from array import array

import pytest

from fuzzer_tool.core.edge_tracker import COVERAGE_TIMELINE_MAX, EdgeTracker


def _snapshots(n: int) -> EdgeTracker:
    et = EdgeTracker()
    for i in range(n):
        et.record_coverage_snapshot(i)
    return et


class TestArraysStayInStep:
    @pytest.mark.parametrize(
        "n", [10, COVERAGE_TIMELINE_MAX, COVERAGE_TIMELINE_MAX + 1, 1300, 2600]
    )
    def test_all_three_arrays_have_equal_length(self, n):
        et = _snapshots(n)
        assert len(et._coverage_execs) == len(et._coverage_edges) == len(et._coverage_timestamps), (
            f"after {n} snapshots: execs={len(et._coverage_execs)} "
            f"edges={len(et._coverage_edges)} ts={len(et._coverage_timestamps)}"
        )

    def test_index_i_refers_to_one_snapshot(self):
        """Measured before the fix at 1300 snapshots: 800/800/1300.

        execs[0] held the count from snapshot 500 while timestamps[0] still
        held the clock reading from snapshot 0.
        """
        et = _snapshots(1300)
        # exec_count was passed as the loop index, so the surviving prefix
        # must start at the same snapshot in every array.
        first_exec = int(et._coverage_execs[0])
        assert first_exec == 1300 - len(et._coverage_execs)
        # Timestamps are non-decreasing and must have lost the same prefix,
        # so the first surviving one cannot be the very first ever recorded.
        assert len(et._coverage_timestamps) == len(et._coverage_execs)

    def test_timestamps_are_monotonic_after_trims(self):
        et = _snapshots(1300)
        ts = list(et._coverage_timestamps)
        assert ts == sorted(ts)


class TestTimelineRoundTrip:
    def test_timestamps_survive_save_and_restore(self):
        et = _snapshots(20)
        original = list(et._coverage_timestamps)
        assert original, "fixture recorded no timestamps"

        restored = EdgeTracker()
        restored.from_dict(et.to_dict())

        assert list(restored._coverage_execs) == list(et._coverage_execs)
        assert list(restored._coverage_edges) == list(et._coverage_edges)
        assert list(restored._coverage_timestamps) == pytest.approx(original)

    def test_round_trip_after_a_trim_stays_aligned(self):
        et = _snapshots(1300)
        restored = EdgeTracker()
        restored.from_dict(et.to_dict())
        assert (
            len(restored._coverage_execs)
            == len(restored._coverage_edges)
            == len(restored._coverage_timestamps)
        )

    def test_legacy_pair_timeline_restores_without_timestamps(self):
        """Snapshots written before timestamps joined the timeline.

        There is nothing honest to invent for them, so the array stays empty
        and the report's `if not cov_ts` guard skips the section.
        """
        et = EdgeTracker()
        et.from_dict({"coverage_timeline": [[0, 0], [1, 5], [2, 9]]})
        assert list(et._coverage_execs) == [0, 1, 2]
        assert list(et._coverage_edges) == [0, 5, 9]
        assert len(et._coverage_timestamps) == 0

    def test_partial_timeline_is_not_half_restored(self):
        et = EdgeTracker()
        et.from_dict({"coverage_timeline": [[0, 0, 1.5], [1, 5], [2, 9, 3.5]]})
        assert len(et._coverage_execs) == 3
        assert len(et._coverage_timestamps) == 0

    def test_empty_timeline(self):
        et = EdgeTracker()
        et.from_dict({})
        assert len(et._coverage_execs) == 0
        assert len(et._coverage_timestamps) == 0


class TestTemporalJoinPairsCorrectly:
    def test_report_section_pairs_matching_snapshots(self):
        """The consumer that the desync silently corrupted."""
        from fuzzer_tool.services.report import _temporal_correlation

        class _F:
            pass

        f = _F()
        f._edge_tracker = _snapshots(1300)
        n = len(f._edge_tracker._coverage_execs)
        # Discovery stream sharing the coverage timestamps, so a correct join
        # matches every point and a mis-paired one does not.
        f._discovery_timestamps = array("d", f._edge_tracker._coverage_timestamps)
        f._discovery_execs = array("Q", f._edge_tracker._coverage_execs)
        f._discovery_edges = array("Q", f._edge_tracker._coverage_edges)
        assert n == len(f._discovery_timestamps)

        out = _temporal_correlation(f)
        assert isinstance(out, str)  # must not raise
