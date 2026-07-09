"""Tests for report.py — explainability report generation."""

import json
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
    f.map_size = 65536
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
    f._peak_eps = 33.7
    f._pruned_count = 0
    f._crash_rate_history = [(100, 5), (200, 12), (300, 18)]
    f.start_time = 0.0
    f.markov = MagicMock()
    f.markov.is_trained.return_value = True
    f.markov.codelength_ratio.return_value = 3.5
    f.markov.corpus_perplexity.return_value = {
        "mean": 12.0, "median": 10.0, "p10": 5.0, "p90": 20.0,
        "low_surprise_count": 1, "high_surprise_count": 0,
    }
    f.mc = None
    f._mopt = None
    f._replicator = None
    f.markov_trained = True
    f._use_mi = False
    f._mi = None
    f._use_transfer_entropy = False
    f._te = None
    f._secretary = False
    f._corpus_secretary = None
    f._anneal_budget = 0
    f._anneal_progress = 0.0
    f.grammar = None
    f.dictionary = [b"\x89PNG"]
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
    f._edge_tracker.edge_rarity_stats.return_value = {
        "total": 18, "singleton": 3, "cold": 5, "warm": 7, "hot": 3,
        "avg_seeds_per_edge": 1.4,
    }
    f._edge_tracker.seed_uniqueness.return_value = {"a": 2, "b": 1}
    f._edge_tracker.edge_cooccurrence.return_value = [(1, 2, 0.8)]
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
        assert "Markov Model Quality" in report
        assert "Perplexity" in report

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


class TestReportBranchCoverage:
    def test_coverage_growth_timeline(self):
        """Lines 97-111: edge_tracker.json with cumulative_edges."""
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            f.corpus_dir = td
            # Create edge_tracker.json
            et = {"cumulative_edges": list(range(100)), "seed_edges": {}}
            Path(td, "edge_tracker.json").write_text(json.dumps(et))
            report = generate_report(f, td, td)
        assert "Coverage growth" in report

    def test_mutation_effectiveness_empty(self):
        """Line 120: empty op_counts returns empty."""
        from fuzzer_tool.services.report import _mutation_effectiveness
        f = _make_mock_fuzzer(op_counts={}, op_success={})
        result = _mutation_effectiveness(f)
        assert result == ""

    def test_mdl_not_trained(self):
        """Lines 144: markov not trained → empty."""
        from fuzzer_tool.services.report import _mdl_codelength
        f = _make_mock_fuzzer()
        f.markov.is_trained.return_value = False
        result = _mdl_codelength(f)
        assert result == ""

    def test_mdl_empty_corpus(self):
        """Line 146: empty corpus → empty."""
        from fuzzer_tool.services.report import _mdl_codelength
        f = _make_mock_fuzzer(corpus=[])
        result = _mdl_codelength(f)
        assert result == ""

    def test_seed_contribution_with_coverage(self):
        """Lines 201-217: seed contribution with coverage data."""
        from fuzzer_tool.services.report import _seed_contribution
        f = _make_mock_fuzzer()
        f.seed_meta = {
            b"alpha": {"coverage_edges": 10, "fuzz_count": 5},
            b"beta": {"coverage_edges": 3, "fuzz_count": 10},
        }
        result = _seed_contribution(f)
        assert "Seed Contribution" in result
        assert "10" in result

    def test_seed_contribution_no_coverage(self):
        """Line 194-196: seed with zero coverage → not ranked."""
        from fuzzer_tool.services.report import _seed_contribution
        f = _make_mock_fuzzer()
        f.seed_meta = {b"seed": {"coverage_edges": 0, "fuzz_count": 10}}
        result = _seed_contribution(f)
        assert result == ""

    def test_corpus_overview_empty_dir(self):
        """Line 223: non-existent dir → empty."""
        from fuzzer_tool.services.report import _corpus_overview
        f = _make_mock_fuzzer()
        result = _corpus_overview(f, "/nonexistent/dir")
        assert result == ""

    def test_corpus_overview_various_sizes(self):
        """Lines 248-255: size distribution buckets."""
        from fuzzer_tool.services.report import _corpus_overview
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "tiny.bin").write_bytes(b"x" * 50)
            Path(td, "small.bin").write_bytes(b"x" * 500)
            Path(td, "medium.bin").write_bytes(b"x" * 5000)
            Path(td, "large.bin").write_bytes(b"x" * 50000)
            Path(td, "huge.bin").write_bytes(b"x" * 200000)
            Path(td, "skip.json").write_text("{}")
            result = _corpus_overview(f, td)
        assert "<100B" in result
        assert "100B-1KB" in result
        assert "1KB-10KB" in result
        assert "10KB-100KB" in result
        assert ">100KB" in result

    def test_crash_analysis_with_crashes(self):
        """Lines 268-294: crash analysis with files."""
        from fuzzer_tool.services.report import _crash_analysis
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "crash1.bin").write_bytes(b"A" * 100)
            Path(td, "crash2.bin").write_bytes(b"B" * 100)
            Path(td, "crash3.bin").write_bytes(b"C" * 50)
            result = _crash_analysis(f, td)
        assert "Crash Analysis" in result
        assert "3" in result

    def test_crash_analysis_empty(self):
        """Line 268: empty crashes dir."""
        from fuzzer_tool.services.report import _crash_analysis
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            result = _crash_analysis(f, td)
        assert result == ""

    def test_crash_reproducibility_with_data(self):
        """Lines 322-334: crash reproducibility with replays."""
        from fuzzer_tool.services.report import _crash_reproducibility
        f = _make_mock_fuzzer()
        f.replay_n = 3
        f._crash_replays = {"sig1": [0, 0, 0], "sig2": [0, 0, -1]}
        result = _crash_reproducibility(f)
        assert "Reproducibility" in result
        assert "100%" in result

    def test_disk_footprint_empty(self):
        """Line 340: empty corpus dir."""
        from fuzzer_tool.services.report import _disk_footprint
        with tempfile.TemporaryDirectory() as td:
            result = _disk_footprint(td)
        assert result == ""

    def test_disk_footprint_with_small_files(self):
        """Lines 354-356: small/large file split."""
        from fuzzer_tool.services.report import _disk_footprint
        with tempfile.TemporaryDirectory() as td:
            Path(td, "tiny.bin").write_bytes(b"x" * 50)
            Path(td, "big.bin").write_bytes(b"x" * 500)
            result = _disk_footprint(td)
        assert "Small" in result
        assert "Large" in result

    def test_bandit_calibration_with_data(self):
        """Lines 363-377: bandit calibration with Brier score."""
        from fuzzer_tool.services.report import _bandit_calibration
        f = _make_mock_fuzzer()
        f.mc = MagicMock()
        f.mc_bandit = True
        f.mc.brier_score.return_value = 0.15
        f.mc.calibration_report.return_value = {"50-60%": (0.55, 0.60)}
        result = _bandit_calibration(f)
        assert "Bandit Calibration" in result
        assert "0.15" in result

    def test_bandit_calibration_zero_brier(self):
        """Line 364: zero Brier score → empty."""
        from fuzzer_tool.services.report import _bandit_calibration
        f = _make_mock_fuzzer()
        f.mc = MagicMock()
        f.mc_bandit = True
        f.mc.brier_score.return_value = 0.0
        result = _bandit_calibration(f)
        assert result == ""

    def test_execution_time_warning(self):
        """Line 395: CRPS rising → warning."""
        from fuzzer_tool.services.report import _execution_time_analysis
        f = _make_mock_fuzzer()
        f._exec_time_tracker.crps_trend.return_value = 0.01
        result = _execution_time_analysis(f)
        assert "WARNING" in result

    def test_execution_time_few_observations(self):
        """Line 383: fewer than 10 observations → empty."""
        from fuzzer_tool.services.report import _execution_time_analysis
        f = _make_mock_fuzzer()
        f._exec_time_tracker.count = 5
        result = _execution_time_analysis(f)
        assert result == ""

    def test_crash_exploitability_with_metadata(self):
        """Lines 442-460: crash exploitability from JSON metadata."""
        from fuzzer_tool.services.report import _crash_exploitability
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "crash1.json").write_text(json.dumps({"exploitability": "HIGH"}))
            Path(td, "crash2.json").write_text(json.dumps({"exploitability": "LOW"}))
            result = _crash_exploitability(f, td)
        assert "Exploitability" in result
        assert "HIGH" in result

    def test_crash_exploitability_corrupt_json(self):
        """Line 450: corrupt JSON → skip."""
        from fuzzer_tool.services.report import _crash_exploitability
        f = _make_mock_fuzzer()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "bad.json").write_text("not json {{{")
            result = _crash_exploitability(f, td)
        assert result == ""

    def test_crash_exploitability_nonexistent_dir(self):
        """Line 442: non-existent crashes dir → empty."""
        from fuzzer_tool.services.report import _crash_exploitability
        f = _make_mock_fuzzer()
        result = _crash_exploitability(f, "/nonexistent")
        assert result == ""

    def test_edge_map_empty_seen(self):
        """Line 469: no edges seen → empty."""
        from fuzzer_tool.services.report import _edge_map_analysis
        f = _make_mock_fuzzer()
        f.shm_cov._seen = bytearray(65536)
        result = _edge_map_analysis(f)
        assert result == ""

    def test_edge_map_no_shm_cov(self):
        """Line 464: no shm_cov → empty."""
        from fuzzer_tool.services.report import _edge_map_analysis
        f = _make_mock_fuzzer(shm_cov=None)
        result = _edge_map_analysis(f)
        assert result == ""

    def test_edge_map_end_of_sequence(self):
        """Line 483: edge runs to end of map."""
        from fuzzer_tool.services.report import _edge_map_analysis
        f = _make_mock_fuzzer()
        seen = bytearray(65536)
        seen[65530] = 1
        seen[65535] = 1
        f.shm_cov._seen = seen
        result = _edge_map_analysis(f)
        assert "Edge Map Regions" in result

    def test_human_size_mb(self):
        """Lines 505-508: MB path."""
        from fuzzer_tool.services.report import _human_size
        assert _human_size(2 * 1024 * 1024) == "2.0MB"
