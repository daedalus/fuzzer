"""Tests for core/field_spec.py -- declared fixed-offset fields as a FIC adapter."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import failure_inducing as fi
from fuzzer_tool.core import field_spec as fs


class TestParse:
    def test_basic_and_options(self):
        f = fs.parse_spec("magic@0:4, ver@0x4:1, len@6:2le=0|1|0xFFFF")
        assert [(x.name, x.offset, x.size, x.little) for x in f] == [
            ("magic", 0, 4, False),
            ("ver", 4, 1, False),
            ("len", 6, 2, True),
        ]
        assert f[2].values == (0, 1, 0xFFFF)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "a@0",
            "a@0:0",
            "a@0:9",
            "a@0:1,a@1:1",  # duplicate name
            "a@0:2,b@1:1",  # overlap
            "a@0:1=256",  # out of range
            "a@0:1=x",
            "1a@0:1",
        ],
    )
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            fs.parse_spec(bad)

    def test_default_values_in_range_and_include_edges(self):
        for size in (1, 2, 4, 8):
            vs = fs.default_values(size)
            mx = (1 << (8 * size)) - 1
            assert {0, 1, mx, mx >> 1, (mx >> 1) + 1} <= set(vs)
            assert all(0 <= v <= mx for v in vs)


class TestRoundTrip:
    def test_read_apply_inverse_both_endians(self):
        f = fs.parse_spec("a@1:2,b@4:2le,c@7:1")
        data = bytes(range(10))
        row = fs.read_row(data, f)
        assert row == (0x0102, 0x0504, 7)
        assert fs.apply_row(data, f, row) == data
        out = fs.apply_row(data, f, (0xAABB, 0x1122, 0xFF))
        assert out[1:3] == b"\xaa\xbb" and out[4:6] == b"\x22\x11" and out[7] == 0xFF
        assert out[0] == 0 and out[3] == 3 and out[8:] == data[8:]

    def test_short_input_is_none(self):
        assert fs.read_row(b"abc", fs.parse_spec("a@2:2")) is None

    def test_baseline_value_added_to_domain(self):
        f = fs.parse_spec("a@0:1=1|2")
        assert fs.value_sets(f, b"\x09") == [(1, 2, 9)]
        assert fs.value_sets(f, b"\x01") == [(1, 2)]
        assert fs.value_sets(f, b"") == [(1, 2)]  # baseline too short: ignored


class TestIsolate:
    F = fs.parse_spec("a@0:1,b@1:2,c@3:2le,d@5:1")

    def _crash(self, data: bytes) -> bool:
        # fails iff b == 0x1234 and d == 7; a, c and trailing bytes irrelevant
        return data[1:3] == b"\x12\x34" and data[5] == 7

    def test_isolates_exact_pair(self):
        data = bytes([9, 0x12, 0x34, 0x55, 0x66, 7, 0xEE])
        schema = fs.isolate_fields_failure(
            data,
            self._crash,
            self.F,
            baseline=bytes([1, 0, 0, 0, 0, 0, 0]),
            rng=random.Random(2),
            verify_samples=8,
        )
        assert schema is not None and schema.status == fi.ISOLATED
        assert schema.params == {1: 0x1234, 3: 7}
        assert schema.verified is True
        assert "b=4660 & d=7" in fs.format_fields_schema(schema, self.F)

    def test_oracle_only_sees_full_length_inputs(self):
        seen = []
        data = bytes(range(7))
        fs.isolate_fields_failure(
            data, lambda d: seen.append(d) or True, self.F, rng=random.Random(0)
        )
        assert seen and all(len(d) == len(data) and d[6:] == data[6:] for d in seen)

    def test_too_short_returns_none(self):
        assert fs.isolate_fields_failure(b"ab", lambda d: True, self.F) is None
