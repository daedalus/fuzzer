"""Tests for core/skipdet.py — skip deterministic stages for low-info seeds."""

from fuzzer_tool.core.skipdet import (
    MAX_QUICK_EFF_EXECS,
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
        assert (
            sd.should_det_fuzz(trace, seed_favored=False, seed_passed_det=False, current_time_ms=0)
            is False
        )

    def test_already_passed_det_returns_false(self):
        sd = SkipDetector(map_size=64)
        trace = bytearray(8)
        assert (
            sd.should_det_fuzz(trace, seed_favored=True, seed_passed_det=True, current_time_ms=0)
            is False
        )

    def test_none_trace_returns_false(self):
        sd = SkipDetector(map_size=64)
        assert (
            sd.should_det_fuzz(None, seed_favored=True, seed_passed_det=False, current_time_ms=0)
            is False
        )

    def test_first_seed_with_new_bits_accepted(self):
        sd = SkipDetector(map_size=64)
        # Trace with some set bits
        trace = bytearray(8)
        trace[0] = 0x0F  # bits 0-3 set
        result = sd.should_det_fuzz(
            trace, seed_favored=True, seed_passed_det=False, current_time_ms=0
        )
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
        result = sd.should_det_fuzz(
            trace1, seed_favored=True, seed_passed_det=False, current_time_ms=1000
        )
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
        result = sd.should_det_fuzz(
            trace2, seed_favored=True, seed_passed_det=False, current_time_ms=1000
        )
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


class TestConstants:
    def test_max_quick_eff_execs(self):
        assert MAX_QUICK_EFF_EXECS == 64 * 1024

    def test_threshold_decay_time(self):
        assert THRESHOLD_DEC_TIME_MS == 20 * 60 * 1000


class TestEffectorMapHalfIsRetired:
    """The block-flip effector search is gone on purpose, not by accident.

    ``build_skip_eff_map`` and ``inference`` were both unreachable from
    ``src/``, and ``inference`` never wrote its output map at all -- it
    returned all-zeros ("every byte ineffective") on every call, with a debug
    line reporting a count that was zero by construction. The effector map is
    now built from the byteflip 8/8 pass the deterministic schedule already
    runs, for no extra executions, so reinstating either of these would be
    paying ``O(len)`` probes for something already free.
    """

    def test_block_flip_effector_search_is_not_reinstated(self):
        sd = SkipDetector()
        assert not hasattr(sd, "build_skip_eff_map")
        assert not hasattr(sd, "inference")

    def test_the_gate_is_the_only_public_surface(self):
        public = {n for n in vars(SkipDetector) if not n.startswith("_")}
        assert public == {"should_det_fuzz"}

    def test_effector_map_lives_in_the_deterministic_stream(self):
        from fuzzer_tool.services.operators import DeterministicEffectorMap

        assert hasattr(DeterministicEffectorMap(4), "eff")
