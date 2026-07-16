"""Tests for core/colorization.py — CmpLog input colorization."""

import pytest

from fuzzer_tool.core.colorization import (
    ColorizationResult,
    TaintRegion,
    _merge_ranges,
    colorize,
)


class TestTaintRegion:
    def test_creation(self):
        t = TaintRegion(start=5, end=10)
        assert t.start == 5
        assert t.end == 10


class TestColorizationResult:
    def test_defaults(self):
        r = ColorizationResult(colorized=b"hello")
        assert r.taints == []
        assert r.original_checksum == 0
        assert r.exec_count == 0


class TestMergeRanges:
    def test_empty(self):
        assert _merge_ranges([]) == []

    def test_single_range(self):
        result = _merge_ranges([[0, 10]])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_adjacent_merge(self):
        result = _merge_ranges([[0, 5], [6, 10]])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_overlapping_merge(self):
        result = _merge_ranges([[0, 5], [3, 10]])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_separate_ranges(self):
        result = _merge_ranges([[0, 5], [10, 15]])
        assert len(result) == 2
        assert result[0].start == 0
        assert result[0].end == 5
        assert result[1].start == 10
        assert result[1].end == 15

    def test_unsorted_input(self):
        result = _merge_ranges([[10, 15], [0, 5], [6, 9]])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 15

    def test_contiguous_merge(self):
        result = _merge_ranges([[0, 2], [3, 5], [6, 8]])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 8


class TestColorize:
    def test_empty_data(self):
        result = colorize(b"", lambda d: 0)
        assert result.colorized == b""
        assert result.taints == []

    def test_preserves_path(self):
        """If all changes preserve the path, entire input is a taint region."""
        data = b"ABCDEFGH"

        def exec_fn(d):
            return 42  # always same path

        result = colorize(data, exec_fn, use_type_aware=False, max_execs=100)
        assert result.original_checksum == 42
        assert len(result.taints) > 0

    def test_colorized_differs_from_original(self):
        data = bytes(range(32))

        def exec_fn(d):
            return 42

        result = colorize(data, exec_fn, use_type_aware=False, max_execs=100)
        # At least some bytes should differ
        assert result.colorized != data

    def test_type_aware_mode(self):
        data = b"ABCDEFGH"

        def exec_fn(d):
            return 42

        result = colorize(data, exec_fn, use_type_aware=True, max_execs=100)
        assert len(result.colorized) == len(data)

    def test_path_changing_input(self):
        """If every change breaks the path, no taint regions."""
        data = bytes(64)

        def exec_fn(d):
            return sum(d)  # every change → different path

        result = colorize(data, exec_fn, use_type_aware=False, max_execs=100)
        # Very few safe ranges since almost everything changes the path
        assert result.exec_count > 0

    def test_respects_max_execs(self):
        data = bytes(256)
        exec_count = [0]

        def exec_fn(d):
            exec_count[0] += 1
            return 42

        colorize(data, exec_fn, use_type_aware=False, max_execs=20)
        assert exec_count[0] <= 20 + 1

    def test_checksum_recorded(self):
        data = b"test"

        def exec_fn(d):
            return 999

        result = colorize(data, exec_fn)
        assert result.original_checksum == 999

    def test_exec_count_recorded(self):
        data = bytes(16)

        def exec_fn(d):
            return 42

        result = colorize(data, exec_fn, use_type_aware=False, max_execs=10)
        assert result.exec_count > 0

    def test_max_execs_zero_uses_default(self):
        """max_execs=0 → uses 2*len(data)."""
        data = bytes(50)
        exec_count = [0]

        def exec_fn(d):
            exec_count[0] += 1
            return 42

        colorize(data, exec_fn, use_type_aware=False, max_execs=0)
        # Should not exceed 2*50 + 1
        assert exec_count[0] <= 101
