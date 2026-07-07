"""Tests for core/count_class.py — logarithmic hit-count classification."""

from fuzzer_tool.core.count_class import (
    LOOKUP_U16,
    _build_u16_table,
    _classify_byte,
    classify_counts,
    classify_single,
    new_bits,
)


class TestClassifyByte:
    def test_zero(self):
        assert _classify_byte(0) == 0

    def test_one(self):
        assert _classify_byte(1) == 1

    def test_two(self):
        assert _classify_byte(2) == 2

    def test_three(self):
        assert _classify_byte(3) == 3

    def test_range_4_7(self):
        for v in range(4, 8):
            assert _classify_byte(v) == 4

    def test_range_8_15(self):
        for v in range(8, 16):
            assert _classify_byte(v) == 8

    def test_range_16_31(self):
        for v in range(16, 32):
            assert _classify_byte(v) == 16

    def test_range_32_127(self):
        for v in range(32, 128):
            assert _classify_byte(v) == 32

    def test_range_128_255(self):
        for v in range(128, 256):
            assert _classify_byte(v) == 128

    def test_max_byte(self):
        assert _classify_byte(255) == 128

    def test_all_classes_present(self):
        classes = {_classify_byte(v) for v in range(256)}
        assert classes == {0, 1, 2, 3, 4, 8, 16, 32, 128}


class TestBuildU16Table:
    def test_table_size(self):
        table = _build_u16_table()
        assert len(table) == 65536

    def test_all_zeros(self):
        table = _build_u16_table()
        assert table[0] == 0

    def test_lo_and_hi_independent(self):
        table = _build_u16_table()
        # lo=1, hi=0 -> classify(1) | (classify(0) << 8) = 1 | 0 = 1
        assert table[1] == 1
        # lo=0, hi=1 -> classify(0) | (classify(1) << 8) = 0 | 256 = 256
        assert table[256] == 256

    def test_two_bytes_combined(self):
        table = _build_u16_table()
        # lo=255 (class 128), hi=255 (class 128)
        # 128 | (128 << 8) = 128 | 32768 = 32896
        assert table[0xFFFF] == 128 | (128 << 8)

    def test_boundary_lo_3_hi_0(self):
        table = _build_u16_table()
        # lo=3 (class 3), hi=0 (class 0) -> 3 | 0 = 3
        assert table[3] == 3

    def test_boundary_lo_0_hi_3(self):
        table = _build_u16_table()
        # lo=0 (class 0), hi=3 (class 3) -> 0 | (3 << 8) = 768
        assert table[0x300] == 3 << 8


class TestLookupU16:
    def test_global_table_exists(self):
        assert len(LOOKUP_U16) == 65536

    def test_global_table_matches_build(self):
        assert LOOKUP_U16 == _build_u16_table()


class TestClassifySingle:
    def test_delegates_to_classify_byte(self):
        for v in (0, 1, 2, 3, 4, 8, 16, 32, 128, 255):
            assert classify_single(v) == _classify_byte(v)


class TestClassifyCounts:
    def test_empty_buffer(self):
        assert classify_counts(b"") == bytearray(b"")

    def test_single_zero(self):
        result = classify_counts(b"\x00")
        assert result == bytearray(b"\x00")

    def test_single_byte(self):
        result = classify_counts(bytes([128]))
        assert result == bytearray(bytes([128]))

    def test_two_bytes(self):
        # lo=1 (class 1), hi=128 (class 128)
        result = classify_counts(bytes([1, 128]))
        assert result == bytearray(bytes([1, 128]))

    def test_preserves_all_zeroes(self):
        buf = b"\x00" * 100
        result = classify_counts(buf)
        assert result == bytearray(b"\x00" * 100)

    def test_odd_length(self):
        buf = bytes([0, 1, 2])
        result = classify_counts(buf)
        assert len(result) == 3
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_even_length(self):
        buf = bytes([0, 1, 2, 3])
        result = classify_counts(buf)
        assert len(result) == 4

    def test_does_not_mutate_input(self):
        buf = bytearray([5, 10, 20, 50])
        original = bytes(buf)
        classify_counts(buf)
        assert bytes(buf) == original

    def test_classifies_correctly_pairwise(self):
        # 4 -> class 4, 10 -> class 8
        result = classify_counts(bytes([4, 10]))
        assert result[0] == 4
        assert result[1] == 8

    def test_all_same_value(self):
        buf = bytes([50] * 10)
        result = classify_counts(buf)
        for b in result:
            assert b == 32  # 50 is in range 32-127


class TestNewBits:
    def test_all_zero_trace_and_virgin(self):
        assert new_bits(b"\x00\x00\x00", b"\x00\x00\x00") == 0

    def test_new_count_in_existing(self):
        # trace has nonzero where virgin is also nonzero -> overlap (1)
        trace = bytes([5, 0, 0])
        virgin = bytes([1, 0, 0])
        assert new_bits(trace, virgin) == 1

    def test_new_coverage(self):
        # trace nonzero where virgin is 0 -> new edge (2)
        trace = bytes([1, 0, 0])
        virgin = bytes([0, 0, 0])
        assert new_bits(trace, virgin) == 2

    def test_all_zero_virgin(self):
        trace = bytes([0, 0, 1])
        virgin = bytes([0, 0, 0])
        assert new_bits(trace, virgin) == 2

    def test_mixed(self):
        # Byte 0: both nonzero -> overlap (1)
        # Byte 1: trace nonzero, virgin 0 -> new coverage (2)
        trace = bytes([1, 5, 0])
        virgin = bytes([1, 0, 0])
        assert new_bits(trace, virgin) == 2

    def test_different_lengths_uses_min(self):
        trace = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9])
        virgin = bytes([1, 2, 3])
        # min length = 3; both have overlap -> returns 1
        assert new_bits(trace, virgin) == 1

    def test_empty_buffers(self):
        assert new_bits(b"", b"") == 0

    def test_8byte_boundary_overlap(self):
        trace = bytes([1] * 8)
        virgin = bytes([1] * 8)
        # t & v nonzero -> overlap -> returns 1
        assert new_bits(trace, virgin) == 1

    def test_8byte_new_coverage(self):
        trace = bytes([1] * 8)
        virgin = bytes([0] * 8)
        assert new_bits(trace, virgin) == 2

    def test_virgin_ff_count_changed(self):
        trace = bytes([0xFF] * 8)
        virgin = bytes([0xFF] * 8)
        # t & v = 0xFF (truthy) -> result = 1
        # t & ~v = 0 -> no return 2
        assert new_bits(trace, virgin) == 1

    def test_trace_zero_virgin_nonzero(self):
        trace = bytes([0, 0])
        virgin = bytes([5, 5])
        assert new_bits(trace, virgin) == 0

    def test_single_byte_new_coverage(self):
        assert new_bits(bytes([1]), bytes([0])) == 2

    def test_single_byte_overlap(self):
        # trace nonzero, virgin nonzero -> overlap -> returns 1
        assert new_bits(bytes([1]), bytes([1])) == 1

    def test_single_byte_count_changed(self):
        assert new_bits(bytes([5]), bytes([1])) == 1

    def test_with_bytearray_input(self):
        assert new_bits(bytearray([1]), bytearray([0])) == 2

    def test_odd_trailing_byte(self):
        # 9 bytes: 8 processed in word loop, 1 trailing
        trace = bytes([0] * 8 + [1])
        virgin = bytes([0] * 8 + [0])
        assert new_bits(trace, virgin) == 2

    def test_trace_larger_than_virgin(self):
        trace = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        virgin = bytes([1, 2])
        # min=2, both overlap -> 1
        assert new_bits(trace, virgin) == 1

    def test_virgin_larger_than_trace(self):
        trace = bytes([1, 2])
        virgin = bytes([1, 2, 3, 4, 5])
        # min=2, both overlap -> 1
        assert new_bits(trace, virgin) == 1
