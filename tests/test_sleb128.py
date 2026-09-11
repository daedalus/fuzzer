"""Tests for the SLEB128 encoder and the sleb128_encode mutator.

Closes the gap alongside leb128_encode (ULEB128): signed varint decoders
(protobuf sint32/64, DWARF, WASM signed ints) sign-extend their last
continuation byte based on bit 0x40, a code path an unsigned-only mutator
can never reach.
"""

import random

from fuzzer_tool.core.mutations.generic import encode_sleb128, sleb128_encode
from fuzzer_tool.core.rand_pool import RandPool


def _decode_sleb128(data: bytes, idx: int = 0) -> int:
    """Reference decoder, independent of the encoder under test."""
    result = 0
    shift = 0
    byte = 0
    while True:
        byte = data[idx]
        idx += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    if byte & 0x40:
        result |= -(1 << shift)
    return result


class TestEncodeSleb128:
    def test_round_trip_small_values(self):
        for value in range(-128, 128):
            encoded = encode_sleb128(value)
            assert _decode_sleb128(encoded) == value

    def test_round_trip_multi_byte_values(self):
        for value in [
            300,
            -300,
            65535,
            -65536,
            2**31 - 1,
            -(2**31),
            2**32,
            -(2**32),
        ]:
            encoded = encode_sleb128(value)
            assert _decode_sleb128(encoded) == value

    def test_zero_is_single_byte(self):
        assert encode_sleb128(0) == b"\x00"

    def test_minus_one_is_single_byte(self):
        # 0x7f: low 7 bits all set, continuation bit clear, sign bit (0x40) set.
        assert encode_sleb128(-1) == b"\x7f"

    def test_negative_values_are_shorter_than_uleb128_would_be(self):
        # -64 fits in one SLEB128 byte; encoding it unsigned would take
        # several bytes. This is the gap: an unsigned-only mutator can
        # never produce this byte pattern from a negative candidate.
        assert len(encode_sleb128(-64)) == 1
        assert len(encode_sleb128(64)) == 2


class TestSleb128EncodeMutator:
    def test_empty_input_returns_unchanged(self):
        assert sleb128_encode(b"", RandPool(seed=1)) == b""

    def test_mutates_and_stays_decodable(self):
        data = bytes([200, 0, 0, 0, 0, 0xAA, 0xBB])
        # Some seeds land on a candidate whose SLEB128 re-encoding happens
        # to be a no-op (e.g. a zero byte re-encoding to 0x00); across a
        # spread of seeds at least some must actually change the buffer.
        assert any(sleb128_encode(data, RandPool(seed=s)) != data for s in range(20))

    def test_no_leftover_byte_duplication(self):
        """The unsigned sibling (leb128_encode) has a pre-existing quirk
        where only width-1 bytes of the candidate are overwritten, leaving
        the candidate's last byte duplicated in the output. sleb128_encode
        replaces the full candidate width, so no byte from inside the
        replaced span should survive unchanged and duplicated."""
        data = bytes([0xAA, 0xBB]) * 4
        for seed in range(50):
            rng = RandPool(seed=seed)
            out = sleb128_encode(data, rng)
            # A duplication artifact would manifest as the same trailing
            # byte pair appearing twice in a row past the original length.
            assert out.count(b"\xbb\xaa\xbb") <= data.count(b"\xbb\xaa\xbb")

    def test_respects_max_len(self):
        # Input already fits within max_len (the invariant callers rely
        # on); the mutator must not grow it past the budget.
        data = bytes(range(12))
        for seed in range(20):
            rng = RandPool(seed=seed)
            out = sleb128_encode(data, rng, max_len=16)
            assert len(out) <= 16

    def test_negative_candidate_round_trips_through_output(self):
        # Force a negative 1-byte candidate (0x80 as a signed byte is -128)
        # and confirm the SLEB128 replacement decodes back correctly.
        data = bytes([0x80])
        rng = random.Random(1)

        class _Wrap:
            def randint(self, a, b):
                return rng.randint(a, b)

        out = sleb128_encode(data, _Wrap())
        assert _decode_sleb128(out) == -128
