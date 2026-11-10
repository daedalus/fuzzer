"""Known-bits abstract domain: barrel-shifter shl vs exhaustive oracle."""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.core.knownbits import KnownBits

MAX_EXHAUSTIVE_W = 4
MAX_AMOUNT_W = 5
WIDE = 64
MAX_UNKNOWN_ENUM = 8


def _all(width: int):
    full = (1 << width) - 1
    for ones in range(full + 1):
        for unk in range(full + 1):
            if ones & unk == 0:
                yield KnownBits(ones, unk, width)


def _concrete(kb: KnownBits):
    """Every integer the abstract value contains (bounded by unknown count)."""
    bits = [i for i in range(kb.width) if kb.unknowns >> i & 1]
    assert len(bits) <= MAX_UNKNOWN_ENUM
    for combo in itertools.product((0, 1), repeat=len(bits)):
        yield kb.ones | sum(c << b for c, b in zip(combo, bits, strict=True))


def _optimal(values, width: int) -> KnownBits:
    """Most precise KnownBits containing all values."""
    it = iter(values)
    acc = KnownBits.const(next(it), width)
    for v in it:
        acc = acc.union(KnownBits.const(v, width))
    return acc


def _oracle(val: KnownBits, amt: KnownBits) -> KnownBits:
    mask = (1 << val.width) - 1
    outs = [(v << a) & mask for v in _concrete(val) for a in _concrete(amt)]
    return _optimal(outs, val.width)


def test_post_example():
    val = KnownBits.from_str("0000?101")
    amt = KnownBits.from_str("?0")
    assert str(val.shl(amt)) == "00???10?"


def test_shl_exhaustive_optimal():
    for vw in range(1, MAX_EXHAUSTIVE_W + 1):
        for aw in range(1, MAX_AMOUNT_W + 1):
            for val in _all(vw):
                for amt in _all(aw):
                    assert val.shl(amt) == _oracle(val, amt), (val, amt)


def test_control_oracle_vs_itself():
    # Rule 46: oracle must agree with itself before it judges the code.
    for val in _all(3):
        for amt in _all(3):
            assert _oracle(val, amt) == _oracle(val, amt)


def test_falsify_missing_range_check_detected():
    # A shl without the out-of-range -> 0 union must fail the oracle.
    def broken(val: KnownBits, amt: KnownBits) -> KnownBits:
        out = val
        for i in range((val.width - 1).bit_length()):
            bit = amt.bit(i)
            out = (
                out.shl_const(1 << i).union(out)
                if bit.unknowns
                else (out.shl_const(1 << i) if bit.ones else out)
            )
        return out

    val = KnownBits.from_str("111")
    amt = KnownBits.from_str("1000")  # always >= width -> 0
    assert broken(val, amt) != _oracle(val, amt)
    assert val.shl(amt) == _oracle(val, amt)


def test_shl_const_amount_ge_width_is_zero():
    val = KnownBits.from_str("1?1")
    assert val.shl(KnownBits.const(3, 8)) == KnownBits.const(0, 3)
    assert val.shl(KnownBits.const(1 << 63, 64)) == KnownBits.const(0, 3)


def test_shl_all_unknown_wide():
    val = KnownBits.all_unknown(WIDE)
    amt = KnownBits.all_unknown(WIDE)
    assert val.shl(amt) == KnownBits.all_unknown(WIDE)


def test_shl_wide_matches_oracle():
    val = KnownBits(0xDEADBEEF00000001, 0xF0, WIDE)
    amt = KnownBits(1 << 6, 0b101111, WIDE)  # may be >= 64
    assert val.shl(amt) == _oracle(val, amt)


def test_shl_amount_high_bits_unknown():
    val = KnownBits.from_str("1011")
    amt = KnownBits(0, 0xFF00 | 0b11, 16)
    # amounts 0..3 and >=4 (zero) are all possible
    assert val.shl(amt) == _optimal([(0b1011 << a) & 0xF for a in range(4)] + [0], 4)


def test_width_one():
    val = KnownBits.from_str("?")
    amt = KnownBits.from_str("?")
    assert val.shl(amt) == KnownBits.from_str("?")
    assert KnownBits.from_str("1").shl(KnownBits.from_str("1")) == KnownBits.const(0, 1)


def test_malformed_rejected():
    with pytest.raises(ValueError):
        KnownBits(1, 1, 4)  # bit both one and unknown
    with pytest.raises(ValueError):
        KnownBits(16, 0, 4)  # exceeds width
    with pytest.raises(ValueError):
        KnownBits(0, 0, 0)
    with pytest.raises(ValueError):
        KnownBits.from_str("01x")


def test_str_roundtrip():
    for s in ("0", "?", "10?1", "????", "0000?101"):
        assert str(KnownBits.from_str(s)) == s
