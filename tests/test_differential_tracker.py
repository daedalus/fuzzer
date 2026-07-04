"""Tests for DifferentialTracker: KL divergence drift detection."""

from fuzzer_tool.services.differential import DifferentialTracker


class TestDifferentialTracker:
    def test_init(self):
        dt = DifferentialTracker()
        assert dt.total_inputs == 0
        assert not dt.drift_detected
        assert dt.drift_threshold == 0.05

    def test_init_custom_threshold(self):
        dt = DifferentialTracker(drift_threshold=0.1)
        assert dt.drift_threshold == 0.1

    def test_record_accumulates(self):
        dt = DifferentialTracker()
        dt.record(0, "", 0, "")
        dt.record(0, "", 0, "")
        assert dt.total_inputs == 2
        assert dt.rc_counts_a[0] == 2
        assert dt.rc_counts_b[0] == 2

    def test_no_drift_identical_targets(self):
        dt = DifferentialTracker()
        for _ in range(30):
            dt.record(0, "", 0, "")
        assert not dt.drift_detected
        assert dt.last_kl_returncode == 0.0

    def test_drift_detected(self):
        dt = DifferentialTracker(drift_threshold=0.01)
        # Target A always returns 0
        for _ in range(50):
            dt.record(0, "", 0, "")
        # Now make B return 1 frequently
        for _ in range(50):
            dt.record(0, "", 1, "")
        dt._check_drift()
        assert dt.drift_detected
        assert dt.last_kl_returncode > 0.0

    def test_signature_drift(self):
        dt = DifferentialTracker(drift_threshold=0.01)
        # A always clean, B always crashes
        for _ in range(50):
            dt.record(0, "", -11, "SIGSEGV")
        dt._check_drift()
        assert dt.drift_detected
        assert "signature" in dt.drift_description.lower() or dt.last_kl_signature > 0.0

    def test_extract_signature_asan(self):
        sig = DifferentialTracker._extract_signature(
            "AddressSanitizer: heap-buffer-overflow", 0
        )
        assert sig.startswith("asan:")

    def test_extract_signature_signal(self):
        sig = DifferentialTracker._extract_signature("", -11)
        assert sig == "signal:11"

    def test_extract_signature_exit(self):
        sig = DifferentialTracker._extract_signature("", 1)
        assert sig == "exit:1"

    def test_extract_signature_clean(self):
        sig = DifferentialTracker._extract_signature("", 0)
        assert sig == "clean"

    def test_kl_divergence_identical(self):
        from collections import Counter
        p = Counter({0: 10, 1: 10})
        assert DifferentialTracker._kl_divergence(p, p) == 0.0

    def test_kl_divergence_different(self):
        from collections import Counter
        p = Counter({0: 10})
        q = Counter({1: 10})
        kl = DifferentialTracker._kl_divergence(p, q)
        assert kl > 0.0

    def test_kl_smoothing_prevents_zero(self):
        from collections import Counter
        p = Counter({0: 10})
        q = Counter({99: 10})
        # Category 0 in p but not in q — smoothing prevents log(0)
        kl = DifferentialTracker._kl_divergence(p, q, smoothing=0.5)
        assert kl > 0.0
        assert kl < float("inf")

    def test_report(self):
        dt = DifferentialTracker()
        dt.record(0, "", 0, "")
        report = dt.get_report()
        assert report["total_inputs"] == 1
        assert not report["drift_detected"]
        assert "returncode_dist_a" in report

    def test_no_drift_below_minimum_inputs(self):
        dt = DifferentialTracker(drift_threshold=0.0)
        dt.record(0, "", 1, "")
        dt._check_drift()
        # Less than 20 inputs → no drift check
        assert not dt.drift_detected
