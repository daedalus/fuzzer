"""Tests for core/colorizer.py — Redqueen-style input colorization."""

import random

from fuzzer_tool.core.colorizer import (
    COLORABLE,
    CmplogColorizer,
    Colorizer,
    FIXED,
    UNKNOWN,
)


class TestColorizer:
    def test_init(self):
        c = Colorizer(10, lambda lo, hi: True)
        assert len(c.color_info) == 10
        assert all(v == UNKNOWN for v in c.color_info)
        assert len(c.unknown_ranges) == 1

    def test_init_zero_length(self):
        c = Colorizer(0, lambda lo, hi: True)
        assert len(c.color_info) == 0
        assert c.unknown_ranges == set()

    def test_all_colorable(self):
        interesting = set(range(2, 6))

        def checker(lo, hi):
            return all(i in interesting for i in range(lo, hi))

        c = Colorizer(10, checker)
        c.classify_all()
        assert c.fraction_classified() == 1.0
        assert c.color_info[3] == COLORABLE
        assert c.color_info[0] == FIXED
        assert c.color_info[7] == FIXED

    def test_all_interesting(self):
        c = Colorizer(5, lambda lo, hi: True)
        c.classify_all()
        assert c.fraction_classified() == 1.0
        assert all(v == COLORABLE for v in c.color_info)

    def test_none_interesting(self):
        c = Colorizer(5, lambda lo, hi: False)
        c.classify_all()
        assert c.fraction_classified() == 1.0
        assert all(v == FIXED for v in c.color_info)

    def test_colorable_bytes(self):
        c = Colorizer(10, lambda lo, hi: 2 <= lo < hi <= 6)
        c.classify_all()
        colored = c.colorable_bytes()
        for i in colored:
            assert c.color_info[i] == COLORABLE

    def test_fixed_bytes(self):
        c = Colorizer(10, lambda lo, hi: False)
        c.classify_all()
        fixed = c.fixed_bytes()
        assert len(fixed) == 10

    def test_color_mask(self):
        c = Colorizer(10, lambda lo, hi: lo >= 2 and hi <= 6)
        c.classify_all()
        mask = c.color_mask()
        assert len(mask) == 10
        assert mask[3] == 0xFF
        assert mask[0] == 0x00

    def test_step_iterator(self):
        c = Colorizer(5, lambda lo, hi: True)
        steps = 0
        while c.step():
            steps += 1
        assert c.fraction_classified() == 1.0
        assert steps >= 0

    def test_max_steps(self):
        c = Colorizer(100, lambda lo, hi: False)
        c.classify_all(max_steps=10)
        assert c.fraction_classified() < 1.0

    def test_larger_input(self):
        c = Colorizer(64, lambda lo, hi: random.random() < 0.3)
        c.classify_all(max_steps=50)
        assert c.fraction_classified() > 0


class TestCmplogColorizer:
    def test_empty_mask_on_no_pairs(self):
        c = CmplogColorizer()
        assert c.color_mask() == b""

    def test_empty_input(self):
        c = CmplogColorizer()
        mask = c.colorize_from_cmplog(b"", [(b"a", b"b")])
        assert mask == b""

    def test_colorizes_from_cmplog(self):
        c = CmplogColorizer()
        input_data = b"hello world foo bar"
        pairs = [(b"world", b"WORLD"), (b"foo", b"BAR")]
        mask = c.colorize_from_cmplog(input_data, pairs)
        assert mask[6] == 0xFF
        assert mask[12] == 0xFF
        assert sum(1 for i in range(0, 6) if mask[i] == 0xFF) <= 1

    def test_mask_length_matches_input(self):
        c = CmplogColorizer()
        input_data = b"test data here"
        mask = c.colorize_from_cmplog(input_data, [(b"data", b"DATA")])
        assert len(mask) == len(input_data)

    def test_repeated_calls(self):
        c = CmplogColorizer()
        input_data = b"hello world"
        mask = c.colorize_from_cmplog(input_data, [(b"hello", b"HELLO"), (b"world", b"WORLD")])
        assert mask[0] == 0xFF
        assert mask[6] == 0xFF
