"""Tests for mutations module."""

import tempfile
from pathlib import Path

from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    MUTATIONS,
    load_dictionary,
    parse_dict_line,
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
        assert len(MUTATIONS) == 10

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
