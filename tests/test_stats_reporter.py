"""Tests for services/stats_reporter.py — statistics and crash replay."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services.stats_reporter import (
    discovery_rate,
    format_elapsed,
    record_discovery_snapshot,
    run_crash_replays,
)


class TestFormatElapsed:
    def test_zero(self):
        assert format_elapsed(time.time()) == "00:00:00"

    def test_seconds(self):
        assert format_elapsed(time.time() - 45) == "00:00:45"

    def test_minutes(self):
        assert format_elapsed(time.time() - 125) == "00:02:05"

    def test_hours(self):
        assert format_elapsed(time.time() - 3661) == "01:01:01"

    def test_large(self):
        assert format_elapsed(time.time() - 36000) == "10:00:00"


class TestRecordDiscoverySnapshot:
    def test_with_shm_cov(self):
        history = []
        shm = SimpleNamespace(cumulative_edges=100)
        record_discovery_snapshot(500, shm, None, history)
        assert history == [(500, 100)]

    def test_with_ptrace_cov(self):
        history = []
        ptrace = SimpleNamespace(cumulative_edges=50)
        record_discovery_snapshot(300, None, ptrace, history)
        assert history == [(300, 50)]

    def test_shm_takes_priority(self):
        history = []
        shm = SimpleNamespace(cumulative_edges=100)
        ptrace = SimpleNamespace(cumulative_edges=50)
        record_discovery_snapshot(500, shm, ptrace, history)
        assert history == [(500, 100)]

    def test_both_none(self):
        history = []
        record_discovery_snapshot(500, None, None, history)
        assert history == [(500, 0)]

    def test_trims_old_entries(self):
        history = [(i, i * 10) for i in range(600)]
        record_discovery_snapshot(700, SimpleNamespace(cumulative_edges=7000), None, history)
        assert len(history) == 351  # 600 + 1 - 250 trimmed


class TestDiscoveryRate:
    def test_empty_history(self):
        assert discovery_rate([]) == 0.0

    def test_single_entry(self):
        assert discovery_rate([(100, 10)]) == 0.0

    def test_basic_rate(self):
        history = [
            (0, 0), (100, 10), (200, 20), (300, 30), (400, 40),
        ]
        # Window: last 5 → (0,0) to (400,40) → 40 edges / 400 execs * 1000 = 100
        assert discovery_rate(history) == pytest.approx(100.0)

    def test_zero_exec_delta(self):
        history = [(100, 10), (100, 20)]
        assert discovery_rate(history) == 0.0

    def test_two_entries(self):
        history = [(0, 0), (100, 50)]
        # 50 edges / 100 execs * 1000 = 500
        assert discovery_rate(history) == pytest.approx(500.0)

    def test_sliding_window(self):
        # More than 5 entries → only last 5 used
        history = [
            (0, 0), (100, 100),  # old
            (200, 100), (300, 100), (400, 100), (500, 100), (600, 150),
        ]
        # Window: last 5 = [(200,100), (300,100), (400,100), (500,100), (600,150)]
        # 50 edges / 400 execs * 1000 = 125
        assert discovery_rate(history) == pytest.approx(125.0)


class TestRunCrashReplays:
    def test_no_replay_needed(self):
        run_crash_replays(
            crashes_dir=Path("/nonexistent"),
            target="target",
            timeout=5.0,
            crash_replays={},
            replay_n=3,
            seed_key_fn=lambda d: "sig",
        )
        # No crash, no error

    def test_replay_n_zero(self):
        run_crash_replays(
            crashes_dir=Path("/nonexistent"),
            target="target",
            timeout=5.0,
            crash_replays={"sig": [1, 2]},
            replay_n=0,
            seed_key_fn=lambda d: "sig",
        )
        # No replay when replay_n=0

    def test_already_full_replays(self):
        replays = {"sig": [1, 2, 3]}
        run_crash_replays(
            crashes_dir=Path("/nonexistent"),
            target="target",
            timeout=5.0,
            crash_replays=replays,
            replay_n=3,
            seed_key_fn=lambda d: "sig",
        )
        assert replays["sig"] == [1, 2, 3]  # unchanged

    def test_crash_not_found_appends_negative(self, tmp_path):
        replays = {"unknown_sig": []}
        run_crash_replays(
            crashes_dir=tmp_path,
            target="target",
            timeout=5.0,
            crash_replays=replays,
            replay_n=2,
            seed_key_fn=lambda d: "other",
        )
        assert replays["unknown_sig"] == [-3]

    def test_budget_exceeded(self, tmp_path):
        """Budget of 0ms means no replays should happen."""
        replays = {"sig": []}
        run_crash_replays(
            crashes_dir=tmp_path,
            target="target",
            timeout=5.0,
            crash_replays=replays,
            replay_n=3,
            seed_key_fn=lambda d: "sig",
            budget_ms=0,
        )
        assert replays["sig"] == []
