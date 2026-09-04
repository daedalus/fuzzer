"""Tests for differential.py — diff_run, DifferentialTracker KS/KL."""

from collections import Counter
from unittest.mock import patch

from fuzzer_tool.services.differential import DifferentialTracker, diff_run


class TestDiffRun:
    @patch("fuzzer_tool.services.differential.run_target_stdin")
    def test_identical_outputs(self, mock_run):
        mock_run.return_value = (0, "", 111)
        diverged, desc = diff_run("target_a", "target_b", b"test")
        assert not diverged
        assert desc == "identical"

    @patch("fuzzer_tool.services.differential.run_target_stdin")
    def test_different_returncodes(self, mock_run):
        mock_run.side_effect = [(0, "", 111), (1, "", 222)]
        diverged, desc = diff_run("target_a", "target_b", b"test")
        assert diverged
        assert "returncode" in desc

    @patch("fuzzer_tool.services.differential.run_target_stdin")
    def test_different_stderr(self, mock_run):
        # This test previously asserted `not diverged` with the comment
        # "stderr differs but not diverged" -- written by observing what
        # diff_run returned rather than what this module says it should
        # return. The module docstring states the per-input contract as an
        # exact match on returncode, sanitizer report AND stderr, so a pure
        # stderr mismatch IS a divergence; the code appended the reason and
        # left the boolean False. Same shape as the `test_hex_escape` case
        # recorded in docs/TODO.md: an assertion strong enough to pass review
        # that pins the defect in place.
        mock_run.side_effect = [(0, "error A", 111), (0, "error B", 222)]
        diverged, desc = diff_run("target_a", "target_b", b"test")
        assert diverged
        assert "different stderr" in desc

    @patch("fuzzer_tool.services.differential.run_target_stdin")
    def test_asan_only_in_a(self, mock_run):
        asan_a = "ERROR: AddressSanitizer: heap-buffer-overflow\nABORTING"
        mock_run.side_effect = [(1, asan_a, 111), (0, "", 222)]
        diverged, desc = diff_run("a", "b", b"test")
        assert diverged
        assert "crashes" in desc

    @patch("fuzzer_tool.services.differential.run_target_file")
    def test_file_mode(self, mock_run):
        mock_run.return_value = (0, "", 111)
        diverged, desc = diff_run("a", "b", b"test", file_mode=True)
        assert not diverged

    @patch("fuzzer_tool.services.differential.run_target_stdin")
    def test_with_target_args(self, mock_run):
        mock_run.return_value = (0, "", 111)
        diverged, desc = diff_run("a", "b", b"test", target_args=["--flag"])
        assert not diverged


class TestDifferentialTrackerExtended:
    def test_get_report(self):
        dt = DifferentialTracker()
        report = dt.get_report()
        assert report["total_inputs"] == 0
        assert not report["drift_detected"]
        assert report["drift_description"] == ""

    def test_kl_divergence_empty(self):
        dt = DifferentialTracker()
        kl = dt._kl_divergence(Counter(), Counter())
        assert kl == 0.0

    def test_kl_divergence_identical(self):
        dt = DifferentialTracker()
        p = Counter({1: 10, 2: 20})
        kl = dt._kl_divergence(p, p)
        assert kl < 0.01  # very close to 0

    def test_kl_divergence_different(self):
        dt = DifferentialTracker()
        p = Counter({1: 100})
        q = Counter({2: 100})
        kl = dt._kl_divergence(p, q)
        assert kl > 0.0

    def test_median_empty(self):
        assert DifferentialTracker._median([]) == 0.0

    def test_median_odd(self):
        assert DifferentialTracker._median([1, 3, 5]) == 3

    def test_median_even(self):
        assert DifferentialTracker._median([1, 2, 3, 4]) == 2.5

    def test_extract_signature_asan(self):
        asan_msg = "ERROR: AddressSanitizer: heap-buffer-overflow\nABORTING"
        sig = DifferentialTracker._extract_signature(asan_msg, 1)
        assert "asan:" in sig

    def test_extract_signature_signal(self):
        sig = DifferentialTracker._extract_signature("", -11)
        assert sig == "signal:11"

    def test_extract_signature_exit(self):
        sig = DifferentialTracker._extract_signature("", 1)
        assert sig == "exit:1"

    def test_extract_signature_clean(self):
        sig = DifferentialTracker._extract_signature("", 0)
        assert sig == "clean"

    def test_check_drift_too_few(self):
        dt = DifferentialTracker()
        dt._check_drift()  # total_inputs=0 < 20, should return
        assert not dt.drift_detected

    def test_check_drift_with_enough_data(self):
        dt = DifferentialTracker(drift_threshold=0.001)
        dt.total_inputs = 30
        for _ in range(30):
            dt.rc_counts_a[0] += 1
            dt.rc_counts_b[1] += 1
        dt._check_drift()
        assert dt.drift_detected
        assert "returncode" in dt.drift_description

    def test_check_drift_no_drift(self):
        dt = DifferentialTracker(drift_threshold=0.5)
        dt.total_inputs = 30
        for _ in range(30):
            dt.rc_counts_a[0] += 1
            dt.rc_counts_b[0] += 1
        dt._check_drift()
        assert not dt.drift_detected

    def test_report_with_report_data(self):
        dt = DifferentialTracker()
        dt.total_inputs = 100
        dt.drift_detected = True
        dt.drift_description = "test drift"
        dt.last_kl_returncode = 0.1
        dt.last_kl_signature = 0.05
        report = dt.get_report()
        assert report["total_inputs"] == 100
        assert report["drift_detected"]
        assert report["drift_description"] == "test drift"
        assert report["kl_returncode"] == 0.1
