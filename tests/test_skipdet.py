"""Tests for core/skipdet.py — skip deterministic stages for low-info seeds."""

import pytest

from fuzzer_tool.core.skipdet import (
    MAX_INF_EXECS,
    MAX_QUICK_EFF_EXECS,
    MINIMAL_BLOCK_SIZE,
    THRESHOLD_DEC_TIME_MS,
    SkipDetector,
)


class TestSkipDetectorInit:
    def test_default_map_size(self):
        sd = SkipDetector()
        assert sd.map_size == 65536
        assert len(sd.virgin_det_bits) == 65536

    def test_custom_map_size(self):
        sd = SkipDetector(map_size=1024)
        assert sd.map_size == 1024
        assert len(sd.virgin_det_bits) == 1024

    def test_initial_threshold_zero(self):
        sd = SkipDetector()
        assert sd.undet_bits_threshold == 0.0


class TestShouldDetFuzz:
    def test_not_favored_returns_false(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)  # 64 bits
        assert sd.should_det_fuzz(trace, seed_favored=False, seed_passed_det=False, current_time_ms=0) is False

    def test_already_passed_det_returns_false(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)
        assert sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=True, current_time_ms=0) is False

    def test_none_trace_returns_false(self):
        sd = SkipDetector(map_size=64)
        assert sd.should_det_fuzz(None, seed_favored=True, seed_passed_det=False, current_time_ms=0) is False

    def test_first_seed_with_new_bits_accepted(self):
        sd = SkipDetector(map_size=64)
        # Trace with some set bits
        trace = bytearray(8)
        trace[0] = 0x0F  # bits 0-3 set
        result = sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        assert result is True

    def test_threshold_initialized_from_first_seed(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)
        trace[0] = 0x0F  # 4 new bits
        sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # threshold = max(1.0, 4 * 0.05) = 1.0
        assert sd.undet_bits_threshold == 1.0

    def test_threshold_initialized_larger(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)
        trace[0] = 0xFF  # 8 new bits
        sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # threshold = max(1.0, 8 * 0.05) = 1.0
        assert sd.undet_bits_threshold == 1.0

    def test_threshold_from_many_bits(self):
        sd = SkipDetector(map_size=1024)
        trace = bytearray(128)  # 1024 bits
        for i in range(100):
            trace[i >> 3] |= 1 << (i & 7)
        sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # threshold = max(1.0, 100 * 0.05) = 5.0
        assert sd.undet_bits_threshold == 5.0

    def test_subsequent_seed_below_threshold_rejected(self):
        sd = SkipDetector(map_size=64)
        # First seed: 4 bits → threshold = 1.0
        trace1 = bytearray(8)
        trace1[0] = 0x0F
        sd.should_det_fuzz(trace1, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # Second seed: same bits → 0 new bits (already in virgin_det_bits)
        result = sd.should_det_fuzz(trace1, seed_favored=True, seed_passed_det=False, current_time_ms=1000)
        assert result is False

    def test_subsequent_seed_above_threshold_accepted(self):
        sd = SkipDetector(map_size=64)
        # First seed: 4 bits
        trace1 = bytearray(8)
        trace1[0] = 0x0F
        sd.should_det_fuzz(trace1, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # Second seed: different 4 bits → 4 new bits
        trace2 = bytearray(8)
        trace2[1] = 0xF0
        result = sd.should_det_fuzz(trace2, seed_favored=True, seed_passed_det=False, current_time_ms=1000)
        assert result is True

    def test_virgin_bits_marked_after_acceptance(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)
        trace[0] = 0x0F  # bits 0-3
        sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # Check bits 0-3 are now marked
        for i in range(4):
            assert sd.virgin_det_bits[i] == 1

    def test_threshold_decay(self):
        sd = SkipDetector(map_size=256)
        # First seed at t=1000: 40 bits → threshold = 2.0, _last_cov_undet_time=1000
        trace1 = bytearray(32)
        for i in range(40):
            trace1[i >> 3] |= 1 << (i & 7)
        sd.should_det_fuzz(trace1, seed_favored=True, seed_passed_det=False, current_time_ms=1000)
        old_threshold = sd.undet_bits_threshold
        assert old_threshold == 2.0
        # Advance time past decay threshold (20 min = 1,200,000 ms)
        sd.should_det_fuzz(
            bytearray(32),  # empty trace → 0 new bits
            seed_favored=True,
            seed_passed_det=False,
            current_time_ms=1000 + THRESHOLD_DEC_TIME_MS + 1,
        )
        assert sd.undet_bits_threshold < old_threshold

    def test_threshold_no_decay_if_too_low(self):
        sd = SkipDetector(map_size=256)
        trace1 = bytearray(32)
        trace1[0] = 0x03  # 2 bits
        sd.should_det_fuzz(trace1, seed_favored=True, seed_passed_det=False, current_time_ms=0)
        # threshold = max(1.0, 2 * 0.05) = 1.0 (< 2, so no decay)
        assert sd.undet_bits_threshold == 1.0
        old_threshold = sd.undet_bits_threshold
        sd.should_det_fuzz(
            bytearray(32),
            seed_favored=True,
            seed_passed_det=False,
            current_time_ms=THRESHOLD_DEC_TIME_MS + 1,
        )
        assert sd.undet_bits_threshold == old_threshold  # no change


class TestBuildSkipEffMap:
    def test_empty_data(self):
        sd = SkipDetector()
        result = sd.build_skip_eff_map(b"", lambda d: 0)
        assert result == bytearray()

    def test_all_bytes_effective(self):
        """If every byte flip changes the checksum, all are effective."""
        sd = SkipDetector()
        data = b"ABCDEFGH"
        # Make every single-byte flip produce a different checksum
        call_count = [0]
        def exec_fn(d):
            call_count[0] += 1
            return sum(d)  # different input → different checksum
        eff_map = sd.build_skip_eff_map(data, exec_fn, max_execs=200)
        assert all(b == 1 for b in eff_map)

    def test_no_bytes_effective(self):
        """If no flip changes checksum, all are ineffective."""
        sd = SkipDetector()
        data = b"ABCDEFGH"
        eff_map = sd.build_skip_eff_map(data, lambda d: 42, max_execs=200)
        assert all(b == 0 for b in eff_map)

    def test_mixed_effective(self):
        """Some bytes effective, some not."""
        sd = SkipDetector()
        data = bytes(128)
        # Only byte 0 affects the checksum
        def exec_fn(d):
            return d[0]
        eff_map = sd.build_skip_eff_map(data, exec_fn, max_execs=500)
        assert eff_map[0] == 1
        # Others may or may not be effective depending on block flipping

    def test_respects_max_execs(self):
        sd = SkipDetector()
        data = bytes(256)
        exec_count = [0]
        def exec_fn(d):
            exec_count[0] += 1
            return sum(d)
        sd.build_skip_eff_map(data, exec_fn, max_execs=10)
        assert exec_count[0] <= 10 + 1  # +1 for baseline

    def test_large_data_with_block_flips(self):
        """Block flipping should find effective regions in large data."""
        sd = SkipDetector()
        # 256 bytes, only first 64 are effective
        data = bytes(256)
        def exec_fn(d):
            # Only the first 64 bytes affect checksum
            return sum(d[:64])
        eff_map = sd.build_skip_eff_map(data, exec_fn, max_execs=1000)
        # First 64 bytes should be marked effective
        assert sum(eff_map[:64]) > 0


class TestInference:
    def test_short_data_returns_all_zeros(self):
        sd = SkipDetector()
        # Short data: length < MINIMAL_BLOCK_SIZE * 8 = 512
        # Returns bytearray(length) — all zeros (no inference performed)
        data = bytes(256)
        result = sd.inference(data, lambda d: 0)
        assert len(result) == 256
        assert all(b == 0 for b in result)

    def test_empty_data_returns_empty(self):
        sd = SkipDetector()
        result = sd.inference(b"", lambda d: 0)
        assert len(result) == 0

    def test_all_ineffective(self):
        """If no flip changes checksum, eff_map stays all zeros."""
        sd = SkipDetector()
        data = bytes(MINIMAL_BLOCK_SIZE * 16)
        eff_map = sd.inference(data, lambda d: 42, max_execs=1000)
        assert all(b == 0 for b in eff_map)

    def test_respects_max_execs(self):
        sd = SkipDetector()
        data = bytes(MINIMAL_BLOCK_SIZE * 16)
        exec_count = [0]
        def exec_fn(d):
            exec_count[0] += 1
            return sum(d)
        sd.inference(data, exec_fn, max_execs=50)
        assert exec_count[0] <= 50 + 1

    def test_returns_correct_length(self):
        """inference returns bytearray of length len(data)."""
        sd = SkipDetector()
        data = bytes(MINIMAL_BLOCK_SIZE * 16)
        eff_map = sd.inference(data, lambda d: 42, max_execs=100)
        assert len(eff_map) == len(data)

    def test_exec_fn_called_for_baseline(self):
        """Baseline checksum is computed (1 exec before loop)."""
        sd = SkipDetector()
        data = bytes(MINIMAL_BLOCK_SIZE * 16)
        calls = []
        def exec_fn(d):
            calls.append(d)
            return 42
        sd.inference(data, exec_fn, max_execs=100)
        assert len(calls) >= 1
        assert calls[0] == data


class TestConstants:
    def test_minimal_block_size(self):
        assert MINIMAL_BLOCK_SIZE == 64

    def test_max_inf_execs(self):
        assert MAX_INF_EXECS == 16 * 1024

    def test_max_quick_eff_execs(self):
        assert MAX_QUICK_EFF_EXECS == 64 * 1024

    def test_threshold_decay_time(self):
        assert THRESHOLD_DEC_TIME_MS == 20 * 60 * 1000
