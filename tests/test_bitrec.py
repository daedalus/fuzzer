"""Tests for the bit-record helpers (`mutations/bitrec.py`)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.bitrec import BitReader, Field, Kind, Tail, pack, ue_bits

# H.264 Table 9-2 / 9-3: codeNum -> bit string, se(v) mapping.
UE_TABLE = [(0, "1"), (1, "010"), (2, "011"), (3, "00100"), (6, "00111"), (7, "0001000")]
SE_TABLE = [(0, "1"), (1, "010"), (-1, "011"), (2, "00100"), (-2, "00101")]


def _bytes(bits: str) -> bytes:
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


class TestExpGolomb:
    @pytest.mark.parametrize(("value", "bits"), UE_TABLE)
    def test_ue_encode(self, value, bits):
        assert ue_bits(value) == bits

    @pytest.mark.parametrize(("value", "bits"), UE_TABLE)
    def test_ue_decode(self, value, bits):
        assert BitReader(_bytes(bits + "1")).ue() == value

    @pytest.mark.parametrize(("value", "bits"), SE_TABLE)
    def test_se_round_trip(self, value, bits):
        assert BitReader(_bytes(bits + "1")).se() == value
        assert pack([Field("x", Kind.SE, 0, value)], "", Tail.NONE) == _bytes(bits)

    # Adversarial: 32+ leading zeros is not a legal codeword.
    def test_ue_too_many_zeros(self):
        with pytest.raises(ValueError):
            BitReader(bytes(5) + b"\xff").ue()

    def test_read_past_end(self):
        with pytest.raises(ValueError):
            BitReader(b"\x00").read(9)


class TestPack:
    def test_rbsp_tail_restores_stop_bit(self):
        # u3=5, rest "01", stop bit, pad: 101 01 1 00
        fields = [Field("a", Kind.FIXED, 3, 5)]
        assert pack(fields, "01", Tail.RBSP) == bytes([0b1010_1100])

    def test_rest_rbsp_strips_stop_and_padding(self):
        r = BitReader(bytes([0b1010_1100]))
        r.read(3)
        assert r.rest(Tail.RBSP) == "01"

    def test_rest_plain_keeps_padding(self):
        r = BitReader(bytes([0b1010_1100]))
        r.read(3)
        assert r.rest(Tail.NONE) == "01100"

    # Falsification: a fixed field wider than its value's bits must not grow.
    def test_fixed_value_masked(self):
        assert pack([Field("a", Kind.FIXED, 4, 0x1F)], "", Tail.NONE) == bytes([0xF0])
