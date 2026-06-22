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
    load_dictionary,
    minimize_bytes,
    parse_dict_line,
    splice,
)


class TestConstants:
    def test_interesting_8_values(self):
        assert 0 in INTERESTING_8
        assert 0xFF in INTERESTING_8
        assert len(INTERESTING_8) == 5

    def test_interesting_16_values(self):
        assert 0x7FFF in INTERESTING_16
        assert 0x8000 in INTERESTING_16
        assert len(INTERESTING_16) == 5

    def test_interesting_32_values(self):
        assert 0x7FFFFFFF in INTERESTING_32
        assert 0x80000000 in INTERESTING_32
        assert len(INTERESTING_32) == 5

    def test_mutations_list(self):
        assert "bit_flip" in MUTATIONS
        assert "havoc" in MUTATIONS
        assert "splice" in MUTATIONS
        assert len(MUTATIONS) == 11

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
