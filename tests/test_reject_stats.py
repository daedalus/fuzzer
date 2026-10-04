"""Rejection rate per operator and per mutation position.

The validity channel records *that* a run was rejected; it never says which
operator or which part of the input caused it. On a linear instruction
input a mid-stream insert reinterprets everything after it (the "havoc
effect"), so the interesting number is the reject rate by where the mutant
first diverges from its parent -- and by operator.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.reject_stats import (
    MAX_OPS,
    NUM_BINS,
    RejectStats,
    first_diff,
)
from fuzzer_tool.core.validity import Validity


class TestFirstDiff:
    @pytest.mark.parametrize(
        ("a", "b", "want"),
        [
            (b"", b"", 0),
            (b"abc", b"abc", 3),
            (b"abc", b"abcd", 3),  # pure append -> parent length
            (b"abcd", b"abc", 3),  # truncation
            (b"abcd", b"abXd", 2),
            (b"abcd", b"Xbcd", 0),
            (b"abcd", b"abcX", 3),
            (b"abcd", b"abXcd", 2),  # mid insert
        ],
    )
    def test_cases(self, a, b, want):
        assert first_diff(a, b) == want

    def test_matches_naive_oracle(self):
        # Oracle derived independently: first index where zip pairs differ.
        def naive(a, b):
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                if x != y:
                    return i
            return min(len(a), len(b))

        cases = [
            (bytes(range(50)), bytes(range(25)) + b"\xff" + bytes(range(26, 50))),
            (b"\x00" * 100, b"\x00" * 99 + b"\x01"),
            (b"\x00" * 100, b"\x01" + b"\x00" * 99),
            (b"a" * 7, b"a" * 7 + b"b" * 9),
        ]
        for a, b in cases:
            assert first_diff(a, b) == naive(a, b)


def _rec(stats, ops, parent, mutant, validity):
    stats.record(ops, parent, mutant, validity)


class TestRecord:
    def test_per_op_counts_and_rate(self):
        s = RejectStats()
        _rec(s, ["a"], b"xxxx", b"xxxxy", Validity.INVALID)
        _rec(s, ["a"], b"xxxx", b"xxxxy", Validity.INVALID)
        _rec(s, ["a"], b"xxxx", b"xxxxy", Validity.VALID)
        rows = {r.name: r for r in s.op_rows()}
        assert (rows["a"].valid, rows["a"].invalid) == (1, 2)
        assert rows["a"].reject_rate == pytest.approx(2 / 3)

    def test_unknown_is_not_a_verdict(self):
        s = RejectStats()
        _rec(s, ["a"], b"xx", b"xy", Validity.UNKNOWN)
        assert s.op_rows() == []
        assert s.bin_rows() == []

    def test_each_op_counted_once_per_round(self):
        s = RejectStats()
        _rec(s, ["a", "a", "b"], b"xx", b"xy", Validity.INVALID)
        rows = {r.name: r for r in s.op_rows()}
        assert rows["a"].invalid == 1
        assert rows["b"].invalid == 1

    def test_append_lands_in_last_bin_and_front_edit_in_first(self):
        s = RejectStats()
        parent = bytes(64)
        _rec(s, ["x"], parent, parent + b"\x01", Validity.VALID)
        _rec(s, ["x"], parent, b"\x01" + parent[1:], Validity.INVALID)
        bins = {r.bin: r for r in s.bin_rows()}
        assert bins[NUM_BINS - 1].valid == 1
        assert bins[0].invalid == 1

    def test_mid_insert_vs_append_separate_rates(self):
        # Falsification: if position were ignored both would share one bin.
        s = RejectStats()
        parent = bytes(range(64))
        for _ in range(9):
            _rec(s, ["ins"], parent, parent[:32] + b"\xff" + parent[32:], Validity.INVALID)
            _rec(s, ["app"], parent, parent + b"\xff", Validity.VALID)
        bins = {r.bin: r for r in s.bin_rows()}
        assert bins[NUM_BINS // 2].reject_rate == 1.0
        assert bins[NUM_BINS - 1].reject_rate == 0.0

    def test_control_identical_streams_identical_rows(self):
        a, b = RejectStats(), RejectStats()
        for st in (a, b):
            _rec(st, ["p", "q"], b"abcdef", b"abXcdef", Validity.INVALID)
            _rec(st, ["p"], b"abcdef", b"abcdefg", Validity.VALID)
        assert a.op_rows() == b.op_rows()
        assert a.bin_rows() == b.bin_rows()


class TestAdversarial:
    def test_op_table_is_bounded(self):
        s = RejectStats()
        for i in range(MAX_OPS * 3):
            _rec(s, [f"op{i}"], b"ab", b"ac", Validity.INVALID)
        assert len(s.op_rows()) <= MAX_OPS

    def test_empty_inputs(self):
        s = RejectStats()
        _rec(s, ["a"], b"", b"", Validity.INVALID)
        _rec(s, ["a"], b"", b"z", Validity.VALID)
        assert sum(r.valid + r.invalid for r in s.bin_rows()) == 2

    def test_identical_mutant_lands_in_last_bin(self):
        s = RejectStats()
        _rec(s, ["a"], b"abcd", b"abcd", Validity.VALID)
        assert {r.bin for r in s.bin_rows()} == {NUM_BINS - 1}

    def test_no_ops_still_counts_position(self):
        s = RejectStats()
        _rec(s, [], b"abcd", b"Xbcd", Validity.INVALID)
        assert s.op_rows() == []
        assert s.bin_rows()[0].invalid == 1
