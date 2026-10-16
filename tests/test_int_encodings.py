"""Tests for the numeric re-encodings in core/mutations/int_encodings.py.

Randomness is scripted: every function draws only through ``rng.randrange(n)``.
"""

import random

import pytest

from fuzzer_tool.core.mutations import int_encodings as ie
from fuzzer_tool.core.mutations.int_encodings import (
    EBML,
    LEB128,
    QUIC,
    VARINT_DECODERS,
    VARINT_ENCODERS,
    VARINT_MAX,
    VLQ,
    varint_reencode,
)


class S:
    def __init__(self, *values):
        self._v = list(values)

    def randrange(self, n):
        v = self._v.pop(0)
        assert 0 <= v < n, f"scripted draw {v} outside [0, {n})"
        return v


# draws: format, offset, padding - 1.
def test_regression_ebml_reencode_keeps_odd_values():
    # 0x85 is the 1-byte vint for 5; the old `& ((1 << 56) - 2)` cleared bit 0 -> 40 04.
    assert varint_reencode(b"\x85", S(EBML, 0, 0)) == b"\x40\x05"


@pytest.mark.parametrize("value", [1, 3, 5, 0x7D, 0x7F, 0xFF, 0x3FFF, 0x12345677])
def test_ebml_reencode_preserves_every_odd_value(value):
    raw = VARINT_ENCODERS[EBML](value, 0)
    out = varint_reencode(raw, S(EBML, 0, 0))
    decoded = VARINT_DECODERS[EBML](out, 0)
    assert decoded is not None
    assert decoded[0] == value


# Falsification: the identity must hold for all four formats, odd and even.
@pytest.mark.parametrize("fmt", [LEB128, VLQ, QUIC, EBML])
@pytest.mark.parametrize("value", [0, 1, 2, 5, 0x7F, 0x80, 0x3FFF, 0x4001, 0xFFFFFFFF])
@pytest.mark.parametrize("extra", [0, 1, 2])
def test_encode_decode_round_trip(fmt, value, extra):
    raw = VARINT_ENCODERS[fmt](value, extra)
    decoded = VARINT_DECODERS[fmt](raw, 0)
    assert decoded == (value, len(raw))


@pytest.mark.parametrize("fmt", [LEB128, VLQ, QUIC, EBML])
def test_reencode_preserves_value_at_each_padding(fmt):
    value = 0x2B  # odd, one group in every format
    raw = VARINT_ENCODERS[fmt](value, 0)
    for pad in range(ie.VARINT_MAX_EXTRA[fmt]):
        out = varint_reencode(b"\xaa" + raw + b"\xbb", S(fmt, 1, pad))
        assert out[0:1] == b"\xaa" and out[-1:] == b"\xbb"
        decoded = VARINT_DECODERS[fmt](out, 1)
        assert decoded is not None and decoded[0] == value
        assert decoded[1] > len(raw)


# Adversarial: boundaries of the EBML range and the reserved all-ones payload.
def test_ebml_max_value_round_trips_unchanged():
    top = VARINT_MAX[EBML]
    raw = VARINT_ENCODERS[EBML](top, 0)
    assert VARINT_DECODERS[EBML](raw, 0) == (top, 8)


def test_ebml_never_emits_the_reserved_unknown_size():
    # 0xFF is the 1-byte "unknown size" (all-ones payload, value 127).  Re-encoding must
    # yield a known size of 127, never the reserved pattern at any width.
    out = varint_reencode(b"\xff", S(EBML, 0, 0))
    decoded = VARINT_DECODERS[EBML](out, 0)
    assert decoded is not None
    value, span = decoded
    assert value == 0x7F
    assert value != (1 << (7 * span)) - 1


def test_ebml_eight_byte_unknown_size_is_not_returned_as_reserved():
    unknown = b"\x01" + b"\xff" * 7
    out = varint_reencode(unknown, S(EBML, 0, 0))
    decoded = VARINT_DECODERS[EBML](out, 0)
    assert decoded is not None
    value, span = decoded
    assert value != (1 << (7 * span)) - 1 or out == unknown


def test_ebml_zero_lead_byte_falls_back_to_insertion():
    # 0x00 does not decode as a vint; an interesting value is inserted instead.
    out = varint_reencode(b"\x00", S(EBML, 0, 0, 0))
    assert out != b"\x00"
    assert out.endswith(b"\x00")


def test_varint_reencode_empty_declines():
    assert varint_reencode(b"", S()) == b""


@pytest.mark.parametrize("fmt", [LEB128, VLQ, QUIC, EBML])
def test_varint_reencode_never_crashes_on_noise(fmt):
    rnd = random.Random(fmt)
    for _ in range(300):
        data = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 24)))
        out = varint_reencode(data, rnd)
        assert isinstance(out, bytes)


# --- sign_extend: draws width, offset, order, bits-1, force ---------------------
def test_sign_extend_widens_low_bits_as_signed():
    # 16-bit BE field at 0, k=8, forced: 00 80 -> ff 80.
    assert ie.sign_extend(b"\x00\x80", S(0, 0, 0, 7, 1)) == b"\xff\x80"


def test_sign_extend_clear_top_bit_zero_extends():
    assert ie.sign_extend(b"\xff\x7f", S(0, 0, 0, 7, 0)) == b"\x00\x7f"


def test_sign_extend_declines_on_one_byte_input():
    assert ie.sign_extend(b"\x80", S()) == b"\x80"


# --- negabinary: involution under the two directions ----------------------------
def test_negabinary_known_value_and_round_trip():
    assert ie.negabinary(b"\xff", S(0, 0, 0, 1)) == b"\x03"  # width 1, to base -2
    for x in range(256):
        fwd = ie.negabinary(bytes([x]), S(0, 0, 0, 1))
        assert ie.negabinary(fwd, S(0, 0, 0, 0)) == bytes([x])


# --- pext / pdep ---------------------------------------------------------------
def test_pext_pdep_oracle():
    assert ie._pext(0xB7, 0x0F) == 0x07
    assert ie._pdep(0x07, 0x0F) == 0x07
    assert ie._pdep(0b11, 0b1010) == 0b1010
    for mask in (0x55, 0xF0, 0x81, 0xFE):
        for x in range(256):
            assert ie._pdep(ie._pext(x, mask), mask) == x & mask


def _bits_ref(x: int, mask: int, nbits: int) -> tuple[list[int], list[int]]:
    """Positions of the mask's set bits, and x's bit at each one (LSB first)."""
    pos = [i for i in range(nbits) if mask >> i & 1]
    return pos, [x >> i & 1 for i in pos]


def _pext_ref(x: int, mask: int, nbits: int) -> int:
    _, bits = _bits_ref(x, mask, nbits)
    return sum(b << k for k, b in enumerate(bits))


def _pdep_ref(x: int, mask: int, nbits: int) -> int:
    pos, _ = _bits_ref(0, mask, nbits)
    return sum((x >> k & 1) << p for k, p in enumerate(pos))


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_pext_pdep_bytes_match_reference(width):
    """Table pext/pdep over a repeated mask byte equals the bit-by-bit reference."""
    rng = random.Random(width)
    nbits = 8 * width
    for byte in range(256):
        mask = int.from_bytes(bytes([byte]) * width, "big")
        for _ in range(8):
            x = rng.getrandbits(nbits)
            assert ie._pext_bytes(x, byte, width) == _pext_ref(x, mask, nbits)
            assert ie._pdep_bytes(x, byte, width) == _pdep_ref(x, mask, nbits)


def test_pext_pdep_bytes_known_values():
    """Falsification: a byte-order or bit-order slip changes these."""
    assert ie._pext_bytes(0xB7, 0x0F, 1) == 0x07
    assert ie._pext_bytes(0xABCD, 0x0F, 2) == 0xBD
    assert ie._pext_bytes(0xABCD, 0xF0, 2) == 0xAC
    assert ie._pdep_bytes(0xBD, 0x0F, 2) == 0x0B0D
    assert ie._pdep_bytes(0b0110, 0x81, 2) == 0x0180


def test_pext_pdep_bytes_edge_masks():
    """Adversarial: empty mask, full mask, bits outside the field."""
    top = (1 << 64) - 1
    assert ie._pext_bytes(top, 0x00, 8) == 0
    assert ie._pdep_bytes(top, 0x00, 8) == 0
    assert ie._pext_bytes(top, 0xFF, 8) == top
    assert ie._pdep_bytes(0x1234, 0xFF, 2) == 0x1234
    # pdep input wider than popcount * width: excess high bits are dropped.
    assert ie._pdep_bytes(0xFFFF, 0x01, 2) == 0x0101


def _bit_compress_run(monkeypatch, path: str, seed: int) -> list[bytes]:
    """bit_compress over fixed inputs; path 'loop' is the pre-table full-mask code."""
    if path == "loop":

        def ext(x, byte, width):
            return ie._pext(x, int.from_bytes(bytes([byte]) * width, "big"))

        def dep(x, byte, width):
            return ie._pdep(x, int.from_bytes(bytes([byte]) * width, "big"))

        monkeypatch.setattr(ie, "_pext_bytes", ext)
        monkeypatch.setattr(ie, "_pdep_bytes", dep)

    data_rng = random.Random(seed)
    inputs = [data_rng.randbytes(data_rng.randrange(1, 40)) for _ in range(300)]
    rng = random.Random(seed + 1)
    out = [ie.bit_compress(d, rng) for d in inputs]
    monkeypatch.undo()
    return out


def test_regression_bit_compress_table_matches_loop(monkeypatch):
    """Table path reproduces the bit-loop path draw for draw."""
    # Control: the oracle against itself must agree first.
    assert _bit_compress_run(monkeypatch, "loop", 7) == _bit_compress_run(monkeypatch, "loop", 7)
    assert _bit_compress_run(monkeypatch, "table", 7) == _bit_compress_run(monkeypatch, "loop", 7)


def test_pext_tables_bounded():
    """Memory stays bounded: one table pair per mask byte, at most 256."""
    for byte in range(256):
        ie._pext_bytes(0, byte, 1)
    assert len(ie._BIT_TABLES) <= 256


# --- bit_matrix_transpose: involution -------------------------------------------
def test_transpose_8x8_example_and_involution():
    row = b"\xff" + b"\x00" * 7
    once = ie.bit_matrix_transpose(row, S(0, 0))
    assert once == b"\x80" * 8
    assert ie.bit_matrix_transpose(once, S(0, 0)) == row


def test_transpose_symmetric_block_and_short_input_decline():
    assert ie.bit_matrix_transpose(b"\x00" * 8, S(0, 0)) == b"\x00" * 8
    assert ie.bit_matrix_transpose(b"\x01" * 7, S()) == b"\x01" * 7


# --- bcd_pack ----------------------------------------------------------------------
def test_bcd_digit_run_shrinks():
    assert ie.bcd_pack(b"id=1234;", S(0, 0)) == b"id=\x12\x34;"


def test_bcd_invalid_nibble_is_never_a_decimal_digit():
    out = ie.bcd_pack(b"\x12\x34", S(2, 0, 0, 0))
    assert out != b"\x12\x34"
    assert (out[0] >> 4) > 9


# --- timestamp_specials -----------------------------------------------------------
def test_timestamp_table_values_fit_their_width():
    for value, width in ie.TIMESTAMPS:
        assert 0 <= value < 1 << (8 * width)


def test_timestamp_declines_when_no_value_fits():
    assert ie.timestamp_specials(b"\x01", S()) == b"\x01"
