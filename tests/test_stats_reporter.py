"""Tests for services/stats_reporter.py — statistics and crash replay."""

import time
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.schedulers import MOptScheduler
from fuzzer_tool.services.stats import StatsReporter
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
        execs, edges = array("Q"), array("Q")
        shm = SimpleNamespace(cumulative_edges=100)
        record_discovery_snapshot(500, shm, None, (execs, edges))
        assert execs.tolist() == [500]
        assert edges.tolist() == [100]

    def test_with_ptrace_cov(self):
        execs, edges = array("Q"), array("Q")
        ptrace = SimpleNamespace(cumulative_edges=50)
        record_discovery_snapshot(300, None, ptrace, (execs, edges))
        assert execs.tolist() == [300]
        assert edges.tolist() == [50]

    def test_shm_takes_priority(self):
        execs, edges = array("Q"), array("Q")
        shm = SimpleNamespace(cumulative_edges=100)
        ptrace = SimpleNamespace(cumulative_edges=50)
        record_discovery_snapshot(500, shm, ptrace, (execs, edges))
        assert execs.tolist() == [500]
        assert edges.tolist() == [100]

    def test_both_none(self):
        execs, edges = array("Q"), array("Q")
        record_discovery_snapshot(500, None, None, (execs, edges))
        assert execs.tolist() == [500]
        assert edges.tolist() == [0]

    def test_trims_old_entries(self):
        execs = array("Q", (i for i in range(600)))
        edges = array("Q", (i * 10 for i in range(600)))
        record_discovery_snapshot(700, SimpleNamespace(cumulative_edges=7000), None, (execs, edges))
        assert len(execs) == 351  # 600 + 1 - 250 trimmed
        assert len(edges) == 351


def _pairs(*samples):
    """Build a paired (execs, edges) tuple of arrays from (exec, edge) samples."""
    return (
        array("Q", (e for e, _ in samples)),
        array("Q", (c for _, c in samples)),
    )


class TestDiscoveryRate:
    def test_empty_history(self):
        assert discovery_rate((array("Q"), array("Q"))) == 0.0

    def test_single_entry(self):
        assert discovery_rate(_pairs((100, 10))) == 0.0

    def test_basic_rate(self):
        history = _pairs((0, 0), (100, 10), (200, 20), (300, 30), (400, 40))
        # Window: last 5 → (0,0) to (400,40) → 40 edges / 400 execs * 1000 = 100
        assert discovery_rate(history) == pytest.approx(100.0)

    def test_zero_exec_delta(self):
        assert discovery_rate(_pairs((100, 10), (100, 20))) == 0.0

    def test_two_entries(self):
        # 50 edges / 100 execs * 1000 = 500
        assert discovery_rate(_pairs((0, 0), (100, 50))) == pytest.approx(500.0)

    def test_sliding_window(self):
        # More than 5 entries → only last 5 used
        history = _pairs(
            (0, 0),
            (100, 100),  # old
            (200, 100),
            (300, 100),
            (400, 100),
            (500, 100),
            (600, 150),
        )
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


# ── helper: minimal mock fuzzer for print_stats tests ──────────────────────


def _mock_fuzzer(**overrides) -> MagicMock:
    """Build a MagicMock fuzzer with sensible defaults for print_stats()."""
    et = MagicMock()
    et.seed_hit_counts = {"a": 1, "b": 2}
    et._global_edge_hits = {}
    et.compute_corpus_diversity.return_value = 0.5
    et.compute_average_jaccard.return_value = 0.3
    et.shannon_entropy_global.return_value = 4.2
    et.simpson_diversity_global.return_value = 0.8
    et.bitmap_density.return_value = 0.15
    et.birthday_collision_risk.return_value = 0.02
    et.coverage_growth_model.return_value = {
        "confidence": 0.0,
        "current_rate": 0.0,
        "projected_total": 0,
        "time_to_plateau": 0,
    }
    et.bayesian_coverage_growth_model.return_value = {"p_stalled": 0.0}
    et.good_turing_estimate.return_value = {
        "n": 0,
        "estimated_undiscovered": 0,
        "saturation": 0.0,
        "confidence": 0.0,
    }

    csd = MagicMock()
    csd.observe.return_value = None
    csd.is_approaching_transition.return_value = (False, "")

    defaults = {
        "start_time": time.time(),
        "exec_count": 1000,
        "_resume_baseline_exec": 0,
        "_last_eps_count": 0,
        "_last_eps_time": 0.0,
        # print_stats appends one avg-eps sample per tick and trims to the
        # window; needs a real array + max for the window bookkeeping.
        "_eps_history": array("d"),
        "_eps_history_max": 10,
        "dictionary": None,
        "markov_trained": False,
        "markov_generate": False,
        "_cmplog": None,
        "_smt_solver": None,
        "shm_cov": None,
        "ptrace_cov": None,
        "multi_targets": False,
        "_target_shm_covs": {},
        "_distance": None,
        "_dist_min_observed": None,
        "_dist_max_observed": None,
        "mc": None,
        "mc_bandit": False,
        "mc_cem": False,
        "crash_sigs": {},
        "timeout_count": 0,
        "_peak_rss": 0,
        "_last_ops_used": [],
        "_edge_tracker": et,
        "_crash_replays": {},
        "_csd": csd,
        "replay_n": 3,
        "_format_learner": None,
        "_perf_counters": None,
        "honggfuzz": False,
        "_entropy_execs": array("Q"),
        "_entropy_vals": array("d"),
        "_exec_time_tracker": SimpleNamespace(count=0),
    }
    defaults.update(overrides)
    fuzzer = MagicMock()
    fuzzer.configure_mock(**defaults)
    return fuzzer


class TestPrintStats:
    """Tests for StatsReporter.print_stats() — live stats line."""

    def test_shows_path_hash_hex_when_shm_available(self):
        """print_stats() includes ph: 0x<hex> when shm_cov is present."""
        shm = MagicMock()
        shm.read_path_hash.return_value = 0xABCD1234
        shm.read_distance_tail.return_value = (0, 0)
        shm.cumulative_edges = 100
        fuzzer = _mock_fuzzer(shm_cov=shm)
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "ph: 0xabcd1234" in line, f"expected hex path_hash in: {line[:200]}"

    def test_omits_path_hash_without_shm(self):
        """print_stats() omits ph: when no shm_cov present."""
        fuzzer = _mock_fuzzer()
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "ph:" not in line, f"unexpected ph: in: {line[:200]}"

    def test_path_hash_shows_zero_hex(self):
        """When path_hash is 0, shows ph: 0x0."""
        shm = MagicMock()
        shm.read_path_hash.return_value = 0
        shm.read_distance_tail.return_value = (0, 0)
        shm.cumulative_edges = 100
        fuzzer = _mock_fuzzer(shm_cov=shm)
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "ph: 0x0" in line, f"expected 0x0 in: {line[:200]}"

    def test_path_hash_shows_large_value(self):
        """Large path_hash (64-bit) renders as hex correctly."""
        shm = MagicMock()
        shm.read_path_hash.return_value = 0xDEADBEEFCAFE
        shm.read_distance_tail.return_value = (0, 0)
        shm.cumulative_edges = 100
        fuzzer = _mock_fuzzer(shm_cov=shm)
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "ph: 0xdeadbeefcafe" in line, f"expected large hex in: {line[:200]}"

    def test_dist_stats_when_directed(self):
        """print_stats() includes dist: avg/min/max in directed mode."""
        shm = MagicMock()
        shm.read_path_hash.return_value = 0
        shm.read_distance_tail.return_value = (25000, 100)  # 25000/100/100 = 2.5
        shm.cumulative_edges = 100
        fuzzer = _mock_fuzzer(
            shm_cov=shm,
            _distance=object(),
            _dist_min_observed=2.0,
            _dist_max_observed=9.5,
        )
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "dist: avg:2.5 min:2.0 max:9.5" in line, f"expected dist stats in: {line[:300]}"

    def test_dist_stats_no_data_when_directed(self):
        """Directed mode with no distance signal reports dist: no-data."""
        fuzzer = _mock_fuzzer(
            shm_cov=None,
            _distance=object(),
        )
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
            assert "dist: no-data" in line, f"expected dist: no-data in: {line[:300]}"

    def test_regression_mopt_particle_count(self):
        """print_stats() reports the real MOpt particle count, not 0p.

        Regression: the status line read the nonexistent ``_particles``
        attribute, so ``mopt: Np`` always rendered as ``mopt: 0p`` even
        though the scheduler held ``n_particles`` particles.
        """
        mopt = MOptScheduler(n_particles=5, window_size=200)
        mopt.init_arm("flip1")
        mopt.init_arm("flip2")
        fuzzer = _mock_fuzzer(_use_mopt=True, _mopt=mopt)
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
        assert "mopt: 5p" in line, f"expected mopt: 5p in: {line[:300]}"

    def test_eps_is_session_local_after_resume(self):
        """Regression: resumed runs must not divide cumulative exec_count by
        fresh wall time.

        ``--resume`` loads exec_count from state while start_time is the new
        process start; the old ``exec_count / elapsed`` showed absurd rates
        (e.g. 2.3M execs over 1s).  The session baseline subtracts the
        loaded count, so the displayed eps reflects only this process.
        """
        fuzzer = _mock_fuzzer(
            exec_count=1_000_000,
            _resume_baseline_exec=999_900,  # 100 execs this session
            start_time=time.time() - 10.0,  # 10s of fresh wall time
        )
        reporter = StatsReporter(fuzzer)
        with patch("builtins.print") as mock_print:
            reporter.print_stats()
            line = mock_print.call_args[0][0]
        # (1_000_000 - 999_900) / 10 = 10 eps, not 100_000
        assert "eps: 10" in line, f"expected session-local eps: 10 in: {line[:300]}"
