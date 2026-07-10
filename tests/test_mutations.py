"""Tests for mutations module."""

import tempfile
from pathlib import Path

from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    MUTATIONS,
    _divisor_sizes,
    bit_transpose,
    load_dictionary,
    minimize_bytes,
    parse_dict_line,
    splice,
    transpose_bytes,
)


class TestConstants:
    def test_interesting_8_values(self):
        assert 0 in INTERESTING_8
        assert 127 in INTERESTING_8
        assert -128 in INTERESTING_8
        assert len(INTERESTING_8) == 9

    def test_interesting_16_values(self):
        assert 32767 in INTERESTING_16
        assert -32768 in INTERESTING_16
        assert len(INTERESTING_16) == 10

    def test_interesting_32_values(self):
        assert 2147483647 in INTERESTING_32
        assert -2147483648 in INTERESTING_32
        assert len(INTERESTING_32) == 9

    def test_mutations_list(self):
        assert "bit_flip" in MUTATIONS
        assert "havoc" in MUTATIONS
        assert "splice" in MUTATIONS
        assert "arithmetic" in MUTATIONS
        assert "transpose_16" in MUTATIONS
        assert "transpose_32" in MUTATIONS
        assert "transpose_64" in MUTATIONS
        assert "bit_transpose_8" in MUTATIONS
        assert "bit_transpose_16" in MUTATIONS
        assert "bit_transpose_32" in MUTATIONS
        assert "bit_transpose_64" in MUTATIONS
        assert len(MUTATIONS) >= 19

    def test_dict_mutations_list(self):
        assert "dict_insert" in DICT_MUTATIONS
        assert "dict_replace" in DICT_MUTATIONS


class TestParseDictLine:
    def test_empty_line(self):
        assert parse_dict_line("") is None

    def test_comment_line(self):
        assert parse_dict_line("# comment") is None

    def test_name_value(self):
        result = parse_dict_line("STR=hello")
        assert result is not None
        assert isinstance(result, bytes)

    def test_raw_bytes(self):
        result = parse_dict_line("\\x00\\xff")
        assert result is not None


class TestLoadDictionary:
    def test_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\n")
            f.write("STR=hello\n")
            f.write("NUM=\\x00\\x01\\x02\n")
            f.write("\n")
            path = f.name
        try:
            tokens = load_dictionary(path)
            assert len(tokens) >= 2
        finally:
            Path(path).unlink()

    def test_file_not_found(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_dictionary("/nonexistent/path.txt")


class TestSplice:
    def test_basic_splice(self):
        a = b"AAAA"
        b = b"BBBB"
        result = splice(a, b)
        assert isinstance(result, bytes)
        assert len(result) >= 2

    def test_result_prefix_from_a_suffix_from_b(self):
        a = b"AAAA"
        b = b"BBBB"
        found_valid = False
        for _ in range(200):
            result = splice(a, b)
            if result.startswith(b"A") and result.endswith(b"B"):
                found_valid = True
                break
        assert found_valid

    def test_short_a_returns_a(self):
        assert splice(b"A", b"BBBB") == b"A"
        assert splice(b"", b"BBBB") == b""

    def test_short_b_returns_a(self):
        assert splice(b"AAAA", b"B") == b"AAAA"

    def test_both_short_returns_a(self):
        assert splice(b"A", b"B") == b"A"

    def test_both_two_bytes(self):
        result = splice(b"AB", b"CD")
        assert isinstance(result, bytes)
        assert 2 <= len(result) <= 2

    def test_result_is_combination(self):
        a = b"AABB"
        b = b"CCDD"
        for _ in range(200):
            result = splice(a, b)
            assert len(result) >= 2
            assert result[:1] in (b"A", b"C") or result[-1:] in (b"B", b"D")


class TestDivisorSizes:
    def test_basic(self):
        sizes = _divisor_sizes(16)
        assert 8 in sizes
        assert 4 in sizes
        assert 2 in sizes
        assert 1 in sizes
        assert sizes == sorted(sizes, reverse=True)

    def test_small_input(self):
        sizes = _divisor_sizes(2)
        assert 1 in sizes

    def test_one(self):
        sizes = _divisor_sizes(1)
        assert sizes == [1]


class TestMinimizeBytes:
    def test_trivial_minimize(self):
        data = b"AAAA"
        result = minimize_bytes(data, lambda x: True, max_stages=10)
        assert isinstance(result, bytes)
        assert len(result) <= len(data)

    def test_preserves_minimum(self):
        data = b"ABCD"
        result = minimize_bytes(data, lambda x: len(x) >= 4, max_stages=10)
        assert len(result) == 4

    def test_reduces_when_possible(self):
        data = b"A" * 100
        result = minimize_bytes(data, lambda x: len(x) >= 1, max_stages=10)
        assert len(result) <= len(data)

    def test_empty_input(self):
        result = minimize_bytes(b"", lambda x: True)
        assert result == b""

    def test_uninteresting_input(self):
        data = b"AAAA"
        result = minimize_bytes(data, lambda x: False)
        assert result == data

    def test_max_stages_limits(self):
        data = b"A" * 1000
        result = minimize_bytes(data, lambda x: True, max_stages=1)
        assert len(result) <= len(data)


class TestTransposeBytes:
    def test_preserves_length(self):
        data = bytes(range(32))
        for width in (2, 4, 8):
            result = transpose_bytes(data, width)
            assert len(result) == len(data)

    def test_short_input_returns_original(self):
        data = b"\x01\x02"
        assert transpose_bytes(data, 4) == data
        assert transpose_bytes(data, 8) == data

    def test_permutes_bytes(self):
        data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        changed = False
        for _ in range(50):
            result = transpose_bytes(data, 4)
            assert sorted(result) == sorted(data)
            if result != data:
                changed = True
        assert changed

    def test_transpose_16_swaps_pair(self):
        data = b"\xAA\xBB\xCC\xDD"
        found = False
        for _ in range(50):
            result = transpose_bytes(data, 2)
            # Must be a valid permutation of the original bytes
            assert sorted(result) == sorted(data)
            if result != data:
                found = True
        assert found

    def test_transpose_64(self):
        data = bytes(range(16))
        found = False
        for _ in range(50):
            result = transpose_bytes(data, 8)
            assert len(result) == len(data)
            # Each 8-byte block should be a permutation of the original block
            for off in range(0, len(data), 8):
                assert sorted(result[off : off + 8]) == sorted(data[off : off + 8])
            if result != data:
                found = True
        assert found


class TestBitTranspose:
    def test_preserves_length(self):
        data = bytes(range(16))
        for width in (1, 2, 4, 8):
            result = bit_transpose(data, width)
            assert len(result) == len(data)

    def test_short_input_returns_original(self):
        data = b"\x01"
        assert bit_transpose(data, 2) == data
        assert bit_transpose(data, 4) == data
        assert bit_transpose(data, 8) == data

    def test_bit_transpose_8_permutes_bits(self):
        data = bytes([0b10101010])
        found = False
        for _ in range(50):
            result = bit_transpose(data, 1)
            assert len(result) == 1
            # Popcount must be preserved
            assert bin(result[0]).count("1") == 4
            if result != data:
                found = True
        assert found

    def test_bit_transpose_preserves_popcount(self):
        data = bytes([0b10101010, 0b11001100, 0b11110000, 0b00001111])
        expected_popcount = sum(bin(b).count("1") for b in data)
        for width in (2, 4):
            for _ in range(20):
                result = bit_transpose(data, width)
                result_popcount = sum(bin(b).count("1") for b in result)
                assert result_popcount == expected_popcount

    def test_bit_transpose_64(self):
        data = bytes(range(16))
        result = bit_transpose(data, 8)
        assert len(result) == len(data)
        expected_popcount = sum(bin(b).count("1") for b in data)
        result_popcount = sum(bin(b).count("1") for b in result)
        assert result_popcount == expected_popcount
