"""``overflow_pair``: plant an operand pair that wraps +, -, *, or << in adjacent fields.

Each pair is the failure input of a branch-free or bounds-check idiom
(awesome-bits audit): ``(x+y)>>1`` and ``off+len`` wrap on add, ``(a-b)>>31``
min/max and ``len-hdr`` on sub, ``count*size`` on mul, ``1 << bits`` is UB.
Single fields are already in the interesting-value tables; the overflow needs
both operands side by side, which no single-field operator produces.

The falsification test is ``test_every_draw_overflows``: an independent oracle
applies each kind's operation to the decoded pair and asserts the result leaves
the field's range.  ``test_control_near_miss_fits`` runs the same oracle on
near-miss pairs and asserts it does *not* fire, so the oracle can fail.
"""

import pytest

import fuzzer_tool.core.mutations.structured as S
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

WIDTHS = (1, 2, 4, 8)
ENDIANS = ("little", "big")
KINDS = ("add_signed", "add_unsigned", "sub_signed", "sub_unsigned", "mul", "shift")
FILL = 0x11
SEEDS = tuple(range(40))


def _signed(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) else v


def _s_fits(v: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= v < (1 << (bits - 1))


def _u_fits(v: int, bits: int) -> bool:
    return 0 <= v < (1 << bits)


def _overflows(kind: str, a: int, b: int, bits: int) -> bool:
    """Oracle: does ``a <kind> b`` leave the *bits*-wide range?"""
    sa, sb = _signed(a, bits), _signed(b, bits)
    checks = {
        "add_signed": lambda: not _s_fits(sa + sb, bits),
        "add_unsigned": lambda: not _u_fits(a + b, bits),
        "sub_signed": lambda: not _s_fits(sa - sb, bits),
        "sub_unsigned": lambda: not _u_fits(a - b, bits),
        "mul": lambda: not _u_fits(a * b, bits),
        "shift": lambda: b >= bits,
    }
    return checks[kind]()


def _draw(width: int, length: int, offset: int, endian: str, kind: str, swap: int) -> ScriptedRng:
    # Draw order: choice(fitting widths), randint(offset), choice(endian),
    # choice(kind), choice(order).
    fitting = [w for w in WIDTHS if 2 * w <= length]
    return ScriptedRng(
        randints=[offset],
        choice_idxs=[fitting.index(width), ENDIANS.index(endian), KINDS.index(kind), swap],
    )


class TestRegistration:
    def test_registered_in_regularity_band(self):
        assert "overflow_pair" in REGISTRY.names()
        assert REGISTRY.category_of("overflow_pair") == "regularity"
        assert "overflow_pair" in OPERATOR_CATEGORIES["regularity"]

    def test_handler_preserves_length(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(64))
        result = dispatch["overflow_pair"](bytearray(data), 0, data)
        assert result is not None
        assert len(result) == len(data)


class TestExactOutput:
    def test_width4_little_add_signed(self):
        out = S.overflow_pair(bytes([FILL]) * 12, _draw(4, 12, 0, "little", "add_signed", 0))
        assert out == b"\xff\xff\xff\x7f" + b"\x01\x00\x00\x00" + bytes([FILL]) * 4

    def test_width2_big_shift_swapped_at_offset(self):
        out = S.overflow_pair(bytes([FILL]) * 8, _draw(2, 8, 3, "big", "shift", 1))
        assert out == bytes([FILL]) * 3 + b"\x00\x10" + b"\x00\x01" + bytes([FILL])

    def test_width1_mul(self):
        out = S.overflow_pair(b"\x11\x11", _draw(1, 2, 0, "little", "mul", 0))
        assert out == b"\x10\x10"


class TestFalsification:
    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("endian", ENDIANS)
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("swap", (0, 1))
    def test_every_draw_overflows(self, width, endian, kind, swap):
        length, offset = 2 * width + 3, 2
        data = bytes([FILL]) * length
        out = S.overflow_pair(data, _draw(width, length, offset, endian, kind, swap))
        first = int.from_bytes(out[offset : offset + width], endian)
        second = int.from_bytes(out[offset + width : offset + 2 * width], endian)
        a, b = (second, first) if swap else (first, second)
        assert _overflows(kind, a, b, 8 * width)
        assert out[:offset] + out[offset + 2 * width :] == data[: length - 2 * width]

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_near_miss_fits(self, width):
        bits = 8 * width
        half, smax, umax = 1 << (bits // 2), (1 << (bits - 1)) - 1, (1 << bits) - 1
        near = {
            "add_signed": (smax - 1, 1),
            "add_unsigned": (umax - 1, 1),
            "sub_signed": ((1 << (bits - 1)) + 1, 1),
            "sub_unsigned": (1, 1),
            "mul": (half - 1, half),
            "shift": (1, bits - 1),
        }
        for kind, (a, b) in near.items():
            assert not _overflows(kind, a, b, bits), kind


class TestAdversarial:
    @pytest.mark.parametrize("data", (b"", b"\x11"))
    def test_too_short_is_untouched_and_draws_nothing(self, data):
        assert S.overflow_pair(data, ScriptedRng()) == data

    def test_only_widths_that_fit_are_drawn(self):
        # 5 bytes: widths 1 and 2 fit; index 1 -> width 2, add_unsigned.
        out = S.overflow_pair(b"\x11" * 5, _draw(2, 5, 1, "little", "add_unsigned", 0))
        assert out == b"\x11" + b"\xff\xff" + b"\x01\x00"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_randpool_changes_at_most_one_pair_window(self, seed):
        data = bytes((i * 37 + seed) & 0xFF for i in range(64))
        out = S.overflow_pair(data, RandPool(seed=seed))
        assert len(out) == len(data)
        diff = [i for i in range(len(data)) if out[i] != data[i]]
        assert not diff or diff[-1] - diff[0] < 2 * max(WIDTHS)
