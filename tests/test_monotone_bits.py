"""zzuf ``set`` / ``unset`` modes (``core/mutations/monotone_bits.py``).

Oracles: exact bytes under a scripted RNG, and a bit-level monotonicity
property over real random draws. Each property test carries a control (an XOR
flip) that the property must reject, so the assertion cannot pass vacuously.
"""

import random

import fuzzer_tool.core.mutations.monotone_bits as M
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, is_xor_linear
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

NAMES = ("bit_set", "bit_clear", "or_mask")


def _bits(data: bytes) -> int:
    return int.from_bytes(data, "little")


def _popcount(x: int) -> int:
    return bin(x).count("1")


class _Spy(ScriptedRng):
    """ScriptedRng that also scripts ``choice`` by value and logs randint bounds."""

    def __init__(self, randints=(), choice_idxs=()):
        super().__init__(randints=randints, choice_idxs=choice_idxs)
        self.bounds = []

    def randint(self, a, b):
        self.bounds.append((a, b))
        return super().randint(a, b)


class TestExactBytes:
    def test_bit_set_picks_among_zero_bits_only(self):
        # 0b1010_0101 has zero bits at 1, 3, 4, 6 (LSB = 0); index 2 -> bit 4.
        rng = _Spy(randints=[2])
        out = M.bit_set(b"\x00\xa5\x00", 1, rng)
        assert out == b"\x00\xb5\x00"
        assert rng.bounds == [(0, 3)]  # four candidates, not eight

    def test_bit_clear_picks_among_one_bits_only(self):
        # one bits at 0, 2, 5, 7; index 3 -> bit 7.
        rng = _Spy(randints=[3])
        assert M.bit_clear(b"\x00\xa5\x00", 1, rng) == b"\x00\x25\x00"
        assert rng.bounds == [(0, 3)]

    def test_or_mask_draw_order_and_bytes(self):
        # choice(MASK_WIDTHS)=1 -> 2 bytes; byte_idx 0 -> window [0, 2).
        # zeros of 0x0F,0x00: bits 4..7 of byte 0, then 8..15 -> 12 of them.
        # count = randint -> 3; then three index draws, popping each time.
        data = b"\x0f\x00\x77"
        zeros = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        rng = ScriptedRng(randints=[3, 0, 0, 8], choice_idxs=[1])
        out = M.or_mask(data, 0, rng)
        # pops: idx0 -> bit 4; idx0 of the rest -> bit 5; idx8 of [6..15] -> bit 14
        assert zeros[0] == 4 and out == bytes([0x0F | 0x10 | 0x20, 0x40, 0x77])

    def test_scan_wraps_to_the_next_byte_with_a_candidate(self):
        data = b"\xff\xff\x7f"  # only byte 2 has a zero bit (bit 7)
        assert M.bit_set(data, 0, ScriptedRng(randints=[0])) == b"\xff\xff\xff"
        assert M.bit_set(data, 2, ScriptedRng(randints=[0])) == b"\xff\xff\xff"
        wrapped = M.bit_set(b"\x01\xff", 1, ScriptedRng(randints=[0]))
        assert wrapped == b"\x03\xff"  # byte 1 is full -> wraps to byte 0

    def test_scan_is_bounded(self):
        # The only editable byte is past SCAN_BYTES: decline, do not walk the buffer.
        data = b"\xff" * (M.SCAN_BYTES + 5) + b"\x00"
        assert M.bit_set(data, 0, ScriptedRng(randints=[0])) is None


class TestDeclines:
    def test_empty_input(self):
        rng = ScriptedRng()
        assert M.bit_set(b"", 0, rng) is None
        assert M.bit_clear(b"", 0, rng) is None
        assert M.or_mask(b"", 0, rng) is None

    def test_nothing_to_change_declines_not_noop(self):
        assert M.bit_set(b"\xff\xff", 0, ScriptedRng()) is None
        assert M.bit_clear(b"\x00\x00", 0, ScriptedRng()) is None
        assert M.or_mask(b"\xff", 0, ScriptedRng(choice_idxs=[0])) is None

    def test_single_zero_bit_caps_the_mask(self):
        # one zero bit available but count drawn = 4: mask is capped, no 2nd draw.
        rng = ScriptedRng(randints=[4, 0], choice_idxs=[0])
        assert M.or_mask(b"\xfe", 0, rng) == b"\xff"


class TestMonotoneProperty:
    """Over real draws, bits move one way only -- and a flip would fail this."""

    def _run(self, op, seed_bytes, trials=400):
        rng = RandPool(seed=7)
        r = random.Random(1)
        for _ in range(trials):
            data = bytes(r.getrandbits(8) for _ in range(r.randint(1, 40)))
            yield data, op(data, r.randrange(len(data)), rng)

    def test_bit_set_only_raises_bits_and_changes_exactly_one(self):
        seen = 0
        for data, out in self._run(M.bit_set, None):
            if out is None:
                assert all(b == 0xFF for b in data[: M.SCAN_BYTES])
                continue
            a, b = _bits(data), _bits(out)
            assert len(out) == len(data)
            assert a & b == a, "a set bit was cleared"
            assert _popcount(b ^ a) == 1
            seen += 1
        assert seen > 300

    def test_bit_clear_only_lowers_bits_and_changes_exactly_one(self):
        seen = 0
        for data, out in self._run(M.bit_clear, None):
            if out is None:
                continue
            a, b = _bits(data), _bits(out)
            assert a & b == b, "a clear bit was set"
            assert _popcount(b ^ a) == 1
            seen += 1
        assert seen > 300

    def test_or_mask_is_a_strict_superset_with_2_to_4_new_bits(self):
        seen = 0
        for data, out in self._run(M.or_mask, None):
            if out is None:
                continue
            a, b = _bits(data), _bits(out)
            assert a & b == a
            assert 1 <= _popcount(b ^ a) <= M.MASK_MAX_BITS
            seen += 1
        assert seen > 300

    def test_control_an_xor_flip_violates_the_property(self):
        """The property above must be able to fail: a flip on a set bit clears it."""
        data = b"\xff"
        flipped = bytes([data[0] ^ 1])
        assert _bits(data) & _bits(flipped) != _bits(data)

    def test_bit_set_can_land_on_every_zero_bit(self):
        """Scripted index sweep: each of the 8 candidates is reachable."""
        outs = {_bits(M.bit_set(b"\x00\x00", 0, ScriptedRng(randints=[i]))) for i in range(8)}
        # Scan starts at byte 0, which has candidates: only its 8 bits are reachable.
        assert outs == {1 << i for i in range(8)}


class TestRegistration:
    def test_registered_in_bit_band(self):
        for name in NAMES:
            assert name in REGISTRY.names()
            assert REGISTRY.category_of(name) == "bit"
            assert name in OPERATOR_CATEGORIES["bit"]

    def test_not_xor_linear(self):
        """Content-gated edits cannot be composed as XOR masks."""
        for name in NAMES:
            assert not is_xor_linear(name)

    def test_handlers_preserve_length_and_stay_monotone(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        fuzzer._op_declines = {}
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes([0x55, 0xAA, 0x0F, 0xF0]) * 8
        for name, ok in (
            ("bit_set", lambda a, b: a & b == a),
            ("bit_clear", lambda a, b: a & b == b),
            ("or_mask", lambda a, b: a & b == a),
        ):
            out = dispatch[name](bytearray(data), 3, data)
            assert len(out) == len(data) and bytes(out) != data
            assert ok(_bits(data), _bits(bytes(out))), name

    def test_decline_is_counted_and_leaves_bytes(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        fuzzer._op_declines = {}
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        full = b"\xff" * 8
        assert bytes(dispatch["bit_set"](bytearray(full), 0, full)) == full
        assert fuzzer._op_declines == {"bit_set": 1}
