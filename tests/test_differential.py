"""Tests for DifferentialTracker — KS + KL drift detection."""

from fuzzer_tool.services.differential import DifferentialTracker


class TestDifferentialTracker:
    def test_init(self):
        dt = DifferentialTracker()
        assert dt.total_inputs == 0
        assert not dt.drift_detected

    def test_record_increments(self):
        dt = DifferentialTracker()
        dt.record(0, "", 0, "")
        assert dt.total_inputs == 1

    def test_drift_not_detected_few_samples(self):
        dt = DifferentialTracker(drift_threshold=0.01)
        for _ in range(15):
            dt.record(0, "", 0, "")
        assert not dt.drift_detected

    def test_drift_detected_categorical(self):
        dt = DifferentialTracker(drift_threshold=0.001)
        # Target A always returns 0, target B always returns 1
        for _ in range(30):
            dt.record(0, "", 1, "")
        assert dt.drift_detected
        assert "returncode" in dt.drift_description

    def test_no_drift_same_behavior(self):
        dt = DifferentialTracker(drift_threshold=0.1)
        for _ in range(30):
            dt.record(0, "ok", 0, "ok")
        assert not dt.drift_detected

    def test_signature_drift(self):
        dt = DifferentialTracker(drift_threshold=0.001)
        for _ in range(30):
            dt.record(1, "ASAN:heap-buffer-overflow", 0, "")
        assert dt.drift_detected
        assert "signature" in dt.drift_description

    def test_exec_time_ks_no_drift(self):
        dt = DifferentialTracker(drift_threshold=0.1)
        # Same timing distribution
        for _ in range(30):
            dt.record(0, "", 0, "", time_a=0.01, time_b=0.01)
        assert not dt.drift_detected

    def test_exec_time_ks_drift(self):
        dt = DifferentialTracker(drift_threshold=0.5)
        # Very different timing: A is fast, B is slow
        for _ in range(30):
            dt.record(0, "", 0, "", time_a=0.001, time_b=0.1)
        assert dt.drift_detected
        assert "exec_time" in dt.drift_description

    def test_exec_time_ks_stored(self):
        dt = DifferentialTracker(drift_threshold=0.5)
        for _ in range(30):
            dt.record(0, "", 0, "", time_a=0.001, time_b=0.1)
        assert dt.last_ks_exec_time > 0.0
        assert dt.last_ks_exec_time_p < 0.05

    def test_extract_signature(self):
        dt = DifferentialTracker()
        assert dt._extract_signature("", 0) == "clean"
        assert dt._extract_signature("", -11) == "signal:11"
        assert dt._extract_signature("", 1) == "exit:1"

    def test_drift_description_empty_when_no_drift(self):
        dt = DifferentialTracker(drift_threshold=0.1)
        for _ in range(30):
            dt.record(0, "", 0, "")
        assert dt.drift_description == ""

    def test_exec_times_capped(self):
        dt = DifferentialTracker()
        for i in range(1500):
            dt.record(0, "", 0, "", time_a=float(i), time_b=float(i))
        assert len(dt.exec_times_a) <= 1000
        assert len(dt.exec_times_b) <= 1000
