"""Tests for report.py — explainability report generation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from fuzzer_tool.services.report import generate_report


def _make_mock_fuzzer(**overrides):
    """Create a minimal mock Fuzzer for report testing."""
    f = MagicMock()
    f.target = "targets/png_read_afl.so"
    f.exec_count = 1000
    f.crash_count = 5
    f.timeout_count = 2
    f.corpus = [b"seed1", b"seed2"]
    f.max_len = 4096
    f.timeout = 5.0
    f.use_coverage = True
    f._inprocess_runner = MagicMock()
    f._inprocess_runner.direct = False
    f._inprocess_runner._persistent = None
    f._inprocess_runner._bitmap_out = None
    f._inprocess_runner.coverage_env_id = "12345"
    f._inprocess_runner.shm_size = 65536
    f._inprocess_runner.read_bitmap.return_value = None
    f.crash_sigs = {"sig1": 3, "sig2": 2}
    f.seed_meta = {b"seed1": {"lineage_depth": 1}, b"seed2": {"lineage_depth": 0}}
    f.op_counts = {"harm": 100, "bit_flip": 50}
    f.op_success = {"harm": 10, "bit_flip": 5}
    f._corpus_size_history = [100, 200, 300]
    f._discovery_history = [(100, 10), (200, 15), (300, 18)]
    f._duplicate_reject_count = 3
    f._total_corpus_attempts = 10
    f._crash_replays = {}
    f.replay_n = 0
    f._peak_rss = 29000
    f.start_time = 0.0
    f.markov = MagicMock()
    f.markov.is_trained.return_value = True
    f.markov.codelength_ratio.return_value = 3.5
    f.mc = None
    f.shm_cov = MagicMock()
    f.shm_cov.cumulative_edges = 18
    f.shm_cov.size = 65536
    seen = bytearray(65536)
    for i in [100, 101, 102, 200]:
        seen[i] = 1
    f.shm_cov._seen = seen
    f.ptrace_cov = None
    f._edge_tracker = MagicMock()
    f._edge_tracker.cumulative_edges = set(range(18))
    f._edge_tracker._global_edge_hits = {i: 2 for i in range(18)}
    f._edge_tracker.map_size = 65536
    f._edge_tracker.seed_hit_counts = {"a": {1: 5, 2: 3}, "b": {3: 4}}
    f._edge_tracker.seed_edges = {"a": {1, 2}, "b": {3}}
    f._edge_tracker.good_turing_estimate.return_value = {
        "n": 18, "n1": 5, "n2": 2, "estimated_undiscovered": 6,
        "saturation": 0.75, "confidence": "medium",
    }
    f._edge_tracker.bitmap_density.return_value = 0.0003
    f._edge_tracker.compute_corpus_diversity.return_value = 100.0
    f._exec_time_tracker = MagicMock()
    f._exec_time_tracker.count = 100
    f._exec_time_tracker.mean_crps.return_value = 0.005
    f._exec_time_tracker.crps_trend.return_value = 0.0001
    f._exec_time_tracker.p50 = 0.008
    f._exec_time_tracker.p99 = 0.015
    f._exec_time_tracker.suggested_timeout.return_value = 0.07
    f.discovery_rate = MagicMock(return_value=50.0)
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


class TestReportSections:
    def test_generate_report_returns_string(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert isinstance(report, str)
        assert "FUZZING REPORT" in report

    def test_run_summary_present(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Run Summary" in report
        assert "1,000" in report

    def test_good_turing_present(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Good-Turing" in report
        assert "18" in report

    def test_mutation_effectiveness(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Mutation Effectiveness" in report

    def test_mdl_codelength_present(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "MDL Codelength" in report

    def test_corpus_health_present(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Corpus Health" in report
        assert "Lineage depth" in report

    def test_execution_time_present(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Execution Time Analysis" in report

    def test_edge_map_analysis(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Edge Map Regions" in report

    def test_crash_analysis_empty_dir(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        # No crash files — section should be empty or absent
        assert "Crash Analysis" not in report

    def test_disk_footprint(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            # Create a dummy corpus file
            Path(td).joinpath("seed.bin").write_bytes(b"hello")
            report = generate_report(f, td, td)
        assert "Disk Footprint" in report

    def test_bandit_calibration_absent_when_no_mc(self):
        f = _make_mock_fuzzer(mc=None)
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Bandit Calibration" not in report

    def test_crash_reproducibility_absent_when_empty(self):
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Crash Reproducibility" not in report

    def test_no_shm_cov_no_crash(self):
        f = _make_mock_fuzzer(shm_cov=None, ptrace_cov=None)
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Coverage Analysis" not in report

    def test_empty_edge_tracker_no_good_turing(self):
        f = _make_mock_fuzzer()
        f._edge_tracker.good_turing_estimate.return_value = {
            "n": 0, "n1": 0, "n2": 0, "estimated_undiscovered": 0,
            "saturation": 0.0, "confidence": "low",
        }
        with tempfile.TemporaryDirectory() as td:
            report = generate_report(f, td, td)
        assert "Good-Turing" not in report
