"""Tests for tools/novelty_rate.py — the offline campaign-level novelty rate
(docs/port-backlog.md §C2): the fraction of generated inputs that found at
least one previously-unseen edge, computed from an already-recorded
coverage log, not the hot loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from novelty_rate import cumulative_novelty_rate, windowed_novelty_rate  # noqa: E402


def _row(elapsed, execs, edges, corpus, crashes, novel=None):
    row = {
        "elapsed": elapsed,
        "exec_count": execs,
        "cumulative_edges": edges,
        "corpus_size": corpus,
        "crash_count": crashes,
    }
    if novel is not None:
        row["novel_input_count"] = novel
    return row


class TestCumulativeNoveltyRate:
    def test_empty_log_is_none(self):
        assert cumulative_novelty_rate([]) is None

    def test_missing_column_is_none(self):
        """A log written before the counter existed must not be silently
        reported as a rate of zero -- that is a different, false claim."""
        rows = [_row(1.0, 100, 5, 2, 0)]
        assert cumulative_novelty_rate(rows) is None

    def test_basic_fraction(self):
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=5),
            _row(2.0, 400, 12, 5, 0, novel=40),
        ]
        assert cumulative_novelty_rate(rows) == 40 / 400

    def test_zero_execs_is_none_not_zero_division(self):
        rows = [_row(0.0, 0, 0, 0, 0, novel=0)]
        assert cumulative_novelty_rate(rows) is None


class TestWindowedNoveltyRate:
    def test_two_ticks_give_one_window(self):
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=10),
            _row(2.0, 300, 9, 3, 0, novel=30),
        ]
        windows = windowed_novelty_rate(rows)
        assert len(windows) == 1
        elapsed, rate = windows[0]
        assert elapsed == 2.0
        assert rate == (30 - 10) / (300 - 100)

    def test_a_flat_interval_is_a_zero_rate_not_skipped(self):
        """No new novel inputs in a window is a real, reportable zero --
        distinct from a window that couldn't be computed at all."""
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=10),
            _row(2.0, 200, 5, 2, 0, novel=10),
        ]
        windows = windowed_novelty_rate(rows)
        assert windows == [(2.0, 0.0)]

    def test_stalled_exec_count_is_skipped_not_a_div_by_zero(self):
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=10),
            _row(2.0, 100, 5, 2, 0, novel=10),
        ]
        assert windowed_novelty_rate(rows) == []

    def test_gap_missing_column_breaks_the_window_pairing(self):
        """A row with no counter (e.g. a log upgraded mid-run) must not be
        diffed against a neighbour as if it had novel_input_count=0 --
        that would fabricate a spurious rate for the straddling window."""
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=10),
            _row(2.0, 200, 8, 3, 0),  # no novel_input_count
            _row(3.0, 300, 12, 4, 0, novel=25),
        ]
        windows = windowed_novelty_rate(rows)
        # Only ticks with the column on both sides of a pair count.
        assert windows == []

    def test_three_ticks_two_windows(self):
        rows = [
            _row(1.0, 100, 5, 2, 0, novel=10),
            _row(2.0, 200, 8, 3, 0, novel=15),
            _row(3.0, 300, 12, 4, 0, novel=25),
        ]
        windows = windowed_novelty_rate(rows)
        assert windows == [
            (2.0, (15 - 10) / (200 - 100)),
            (3.0, (25 - 15) / (300 - 200)),
        ]
