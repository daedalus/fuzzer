"""``div_trap``: plant the signed-division overflow pair (MIN, -1).

Hacker's Delight ch. 9/10: two's-complement ``MIN / -1`` (and ``MIN % -1``)
does not fit the signed range, so x86 ``idiv`` raises SIGFPE and Rust/Swift
panic, at every width the language does not promote.  Both values are already
in the interesting-value tables, but singly: the trap needs them in
*neighbouring* fields (dividend then divisor, or the reverse for a parser that
reads the divisor first), which no single-field operator can produce.

The load-bearing falsification test is ``test_pair_decodes_to_the_trap``: if the
operator ever writes anything but ``(MIN, -1)`` in a field pair, it is not
doing what it claims.  Its sensitivity control checks that a near miss
``(MIN + 1, -1)`` fits the signed range, so the assertion distinguishes the
trap from an ordinary division.
"""

import pytest

import fuzzer_tool.core.mutations.structured as S
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

SEEDS = tuple(range(40))


def _signed(b: bytes, endian: str) -> int:
    return int.from_bytes(b, endian, signed=True)


def _fits(value: int, width: int) -> bool:
    return -(1 << (8 * width - 1)) <= value < (1 << (8 * width - 1))


class TestRegistration:
    def test_registered_in_regularity_band(self):
        assert "div_trap" in REGISTRY.names()
        assert REGISTRY.category_of("div_trap") == "regularity"
        assert "div_trap" in OPERATOR_CATEGORIES["regularity"]

    def test_handler_preserves_length(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(64))
        result = dispatch["div_trap"](bytearray(data), 0, data)
        assert result is not None
        assert len(result) == len(data)


class TestExactOutput:
    # Draw order: choice(fitting widths), randint(offset), choice(endian), choice(order).

    def test_width4_little_dividend_first(self):
        rng = ScriptedRng(randints=[0], choice_idxs=[2, 0, 0])
        out = S.div_trap(b"\x11" * 20, rng)
        assert out == b"\x00\x00\x00\x80" + b"\xff\xff\xff\xff" + b"\x11" * 12

    def test_width2_big_divisor_first_at_offset(self):
        rng = ScriptedRng(randints=[3], choice_idxs=[1, 1, 1])
        out = S.div_trap(b"\x11" * 10, rng)
        assert out == b"\x11" * 3 + b"\xff\xff" + b"\x80\x00" + b"\x11" * 3

    def test_width8_little_dividend_first(self):
        rng = ScriptedRng(randints=[0], choice_idxs=[3, 0, 0])
        out = S.div_trap(b"\x11" * 16, rng)
        assert out == b"\x00" * 7 + b"\x80" + b"\xff" * 8

    def test_only_widths_that_fit_are_drawn(self):
        # 5 bytes: widths 1 and 2 fit (2w <= 5); index 1 -> width 2.
        rng = ScriptedRng(randints=[0], choice_idxs=[1, 0, 0])
        out = S.div_trap(b"\x11" * 5, rng)
        assert out == b"\x00\x80\xff\xff\x11"

    def test_too_short_is_untouched_and_draws_nothing(self):
        assert S.div_trap(b"\x11", ScriptedRng()) == b"\x11"
        assert S.div_trap(b"", ScriptedRng()) == b""


class TestInvariant:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_pair_decodes_to_the_trap(self, seed):
        data = bytes((i * 37 + seed) & 0xFF for i in range(64))
        out = S.div_trap(data, RandPool(seed=seed))
        assert len(out) == len(data)
        diff = [i for i in range(len(data)) if out[i] != data[i]]
        assert diff, "operator must change something on this input"
        found = False
        for width in (1, 2, 4, 8):
            for endian in ("little", "big"):
                for off in range(len(data) - 2 * width + 1):
                    lo, hi = out[off : off + width], out[off + width : off + 2 * width]
                    lo_v, hi_v = _signed(lo, endian), _signed(hi, endian)
                    mn = -(1 << (8 * width - 1))
                    if (lo_v, hi_v) in ((mn, -1), (-1, mn)):
                        outside = out[:off] + out[off + 2 * width :]
                        orig = data[:off] + data[off + 2 * width :]
                        if outside == orig:
                            found = True
        assert found

    @pytest.mark.parametrize("width", (1, 2, 4, 8))
    def test_control_near_miss_fits_the_signed_range(self, width):
        mn = -(1 << (8 * width - 1))
        assert not _fits(mn // -1, width)  # the trap: quotient overflows
        assert _fits((mn + 1) // -1, width)  # control: no trap
