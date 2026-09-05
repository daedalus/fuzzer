"""Regression tests for the dense CrashMITracker histograms.

record() used to walk every byte of every executed input updating three
nested defaultdicts, and a periodic _prune() kept only the 32 most
frequent byte values per position. The store is now a pair of dense
(positions, 256) arrays. These tests pin the counting semantics, the
dict views the rest of the codebase reads, the save/load round trip, and
the fact that nothing is discarded any more.
"""

from __future__ import annotations

import os
import random

import pytest

from fuzzer_tool.core.crash_eta import CrashMITracker


# ---------------------------------------------------------------------------
# Oracle: the old dict-of-dicts counting, without the prune.
# ---------------------------------------------------------------------------


def _reference_counts(inputs, max_positions):
    position_counts: dict[int, int] = {}
    byte_total: dict[int, dict[int, int]] = {}
    joint_crash: dict[int, dict[int, int]] = {}
    for data, is_crash in inputs:
        n = min(len(data), max_positions)
        for pos in range(n):
            bv = data[pos]
            position_counts[pos] = position_counts.get(pos, 0) + 1
            byte_total.setdefault(pos, {})
            byte_total[pos][bv] = byte_total[pos].get(bv, 0) + 1
            if is_crash:
                joint_crash.setdefault(pos, {})
                joint_crash[pos][bv] = joint_crash[pos].get(bv, 0) + 1
    return position_counts, byte_total, joint_crash


# ---------------------------------------------------------------------------


def test_counts_match_the_dict_of_dicts_oracle():
    rnd = random.Random(0)
    inputs = [
        (os.urandom(rnd.randrange(1, 600)), rnd.random() < 0.05) for _ in range(400)
    ]
    tracker = CrashMITracker(max_positions=512)
    for data, is_crash in inputs:
        tracker.record(data, is_crash)

    pc, bt, jc = _reference_counts(inputs, 512)
    assert tracker.position_counts == pc
    assert tracker.byte_total == bt
    assert tracker.joint_crash == jc


def test_nothing_is_discarded_after_the_old_prune_interval():
    """The old _prune() ran every 500 execs and kept 32 values per position.

    With 256 distinct values seen at position 0, the dict store dropped
    224 of them; the dense store keeps all of them.
    """
    tracker = CrashMITracker()
    for _ in range(4):
        for v in range(256):
            tracker.record(bytes([v, v, v, v]), is_crash=False)
    assert tracker.total_execs == 1024  # well past the old prune interval
    assert len(tracker.byte_total[0]) == 256
    assert sum(tracker.byte_total[0].values()) == 1024


def test_max_positions_still_caps_tracking():
    tracker = CrashMITracker(max_positions=2)
    tracker.record(b"abcdef", is_crash=False)
    assert set(tracker.position_counts) == {0, 1}
    assert 2 not in tracker.byte_total


def test_non_crash_inputs_do_not_reach_joint_crash():
    tracker = CrashMITracker(min_observations=1)
    tracker.record(b"AAAAAAAAAA", is_crash=False)
    assert tracker.joint_crash == {}
    assert len(tracker.position_counts) == 10
    tracker.record(b"BBBBBBBBBB", is_crash=True)
    assert tracker.byte_total != tracker.joint_crash
    assert tracker.byte_total[0][ord("A")] == 1
    assert tracker.joint_crash[0].get(ord("A"), 0) == 0
    assert tracker.joint_crash[0][ord("B")] == 1


def test_empty_input_is_a_no_op_beyond_the_exec_counter():
    tracker = CrashMITracker()
    tracker.record(b"", is_crash=False)
    assert tracker.total_execs == 1
    assert tracker.position_counts == {}
    assert tracker.byte_total == {}


def test_rows_grow_lazily_with_the_longest_input():
    """A 4096-position tracker must not allocate 8 MiB for short inputs."""
    tracker = CrashMITracker(max_positions=4096)
    assert tracker._rows == 0
    tracker.record(b"x" * 10, is_crash=False)
    assert 0 < tracker._rows < 4096
    small = tracker._rows
    tracker.record(b"x" * 3000, is_crash=False)
    assert tracker._rows > small
    assert tracker._rows <= 4096


def test_growth_preserves_earlier_counts():
    tracker = CrashMITracker(max_positions=4096)
    tracker.record(bytes([7] * 8), is_crash=True)
    tracker.record(bytes([9] * 2000), is_crash=False)
    assert tracker.byte_total[0][7] == 1
    assert tracker.byte_total[0][9] == 1
    assert tracker.joint_crash[0][7] == 1
    assert tracker.position_counts[0] == 2
    assert tracker.position_counts[1999] == 1


def test_save_load_round_trip_preserves_every_count():
    rnd = random.Random(7)
    tracker = CrashMITracker(max_positions=256, min_observations=2)
    for _ in range(200):
        tracker.record(os.urandom(rnd.randrange(1, 300)), rnd.random() < 0.1)
    blob = tracker.save()

    restored = CrashMITracker()
    restored.load(blob)
    assert restored.position_counts == tracker.position_counts
    assert restored.byte_total == tracker.byte_total
    assert restored.joint_crash == tracker.joint_crash
    assert restored.total_execs == tracker.total_execs
    assert restored.total_crashes == tracker.total_crashes
    assert restored.all_mi() == tracker.all_mi()


def test_load_accepts_state_written_before_the_dense_store():
    """The on-disk shape did not change; old state must still restore."""
    restored = CrashMITracker()
    restored.load(
        {
            "max_positions": 4096,
            "min_observations": 1,
            "total_execs": 4,
            "total_crashes": 1,
            "position_counts": {"0": 4, "1": 4},
            "byte_total": {"0": {"65": 3, "66": 1}, "1": {"65": 4}},
            "joint_crash": {"0": {"66": 1}},
        }
    )
    assert restored.position_counts == {0: 4, 1: 4}
    assert restored.byte_total == {0: {65: 3, 66: 1}, 1: {65: 4}}
    assert restored.joint_crash == {0: {66: 1}}
    assert restored.mi(0) > 0.0


def test_mi_matches_a_direct_computation():
    """MI read off the dense rows equals MI over the dict view."""
    import math

    rnd = random.Random(1)
    tracker = CrashMITracker(min_observations=1)
    for _ in range(300):
        data = bytes(rnd.randrange(4) for _ in range(6))
        tracker.record(data, is_crash=(data[0] == 0))

    n = tracker.total_execs
    p_crash = tracker.total_crashes / n
    p_no = 1.0 - p_crash
    for pos in range(6):
        expected = 0.0
        for bv, bc in tracker.byte_total[pos].items():
            p_x = bc / n
            cc = tracker.joint_crash.get(pos, {}).get(bv, 0)
            nc = bc - cc
            if cc > 0:
                expected += (cc / n) * math.log2((cc / n) / (p_x * p_crash))
            if nc > 0:
                expected += (nc / n) * math.log2((nc / n) / (p_x * p_no))
        assert tracker.mi(pos) == pytest.approx(max(0.0, expected))


def test_top_values_ranks_by_crash_count():
    tracker = CrashMITracker(min_observations=1)
    for _ in range(5):
        tracker.record(bytes([10]), is_crash=True)
    for _ in range(3):
        tracker.record(bytes([20]), is_crash=True)
    tracker.record(bytes([30]), is_crash=True)
    tracker.record(bytes([40]), is_crash=False)
    assert tracker.top_values(0, k=3) == [10, 20, 30]
    assert tracker.top_values(0, k=1) == [10]
    assert tracker.top_values(99, k=3) == []
