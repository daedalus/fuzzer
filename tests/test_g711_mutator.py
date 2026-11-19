"""Tests for the G.711 A-law/µ-law sample mutator (`g711_mutate`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.g711 import (
    GAINS,
    Law,
    Op,
    decode,
    encode,
    find_payload,
    mutate_g711,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

WAV_ALAW, WAV_ULAW = 6, 7  # WAVE fmt tags (RFC 2361)
AU_ULAW, AU_ALAW = 1, 27  # Sun .snd encodings
AU_SIZE_UNKNOWN = 0xFFFFFFFF


def _wav(tag: int, payload: bytes, extra: bytes = b"") -> bytes:
    fmt = struct.pack("<HHIIHH", tag, 1, 8000, 8000, 1, 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _au(enc: int, payload: bytes, size: int | None = None) -> bytes:
    size = len(payload) if size is None else size
    return b".snd" + struct.pack(">IIIII", 24, size, enc, 8000, 1) + payload


def _scripted(op: Op, lo: int, hi: int, *extra_ints, choice_idxs=(), randbytes=()):
    return ScriptedRng(
        randints=[int(op), lo, hi, *extra_ints], choice_idxs=choice_idxs, randbytes=randbytes
    )


class TestCodec:
    # Canonical G.711 values (ITU-T table 1/2, FFmpeg pcm_tablegen.h).
    @pytest.mark.parametrize(
        ("code", "lin"), [(0xD5, 8), (0x55, -8), (0xAA, 32256), (0x2A, -32256)]
    )
    def test_alaw_decode_known(self, code, lin):
        assert decode(bytes([code]), Law.ALAW).tolist() == [lin]

    @pytest.mark.parametrize(("code", "lin"), [(0xFF, 0), (0x00, -32124), (0x80, 32124)])
    def test_ulaw_decode_known(self, code, lin):
        assert decode(bytes([code]), Law.ULAW).tolist() == [lin]

    def test_alaw_round_trip_every_code(self):
        codes = bytes(range(256))
        assert encode(decode(codes, Law.ALAW), Law.ALAW) == codes

    def test_ulaw_round_trip_canonicalizes_negative_zero(self):
        codes = bytes(range(256))
        expected = bytearray(codes)
        expected[0x7F] = 0xFF  # -0 and +0 both decode to 0; +0 is canonical
        assert encode(decode(codes, Law.ULAW), Law.ULAW) == bytes(expected)

    @pytest.mark.parametrize(
        ("law", "lin", "code"),
        [
            (Law.ULAW, 0, 0xFF),
            (Law.ULAW, 32767, 0x80),
            (Law.ULAW, -32768, 0x00),
            (Law.ALAW, 9, 0xD5),
            (Law.ALAW, -1, 0x55),
        ],
    )
    def test_encode_nearest(self, law, lin, code):
        assert encode([lin], law) == bytes([code])

    @pytest.mark.parametrize("law", list(Law))
    def test_quantizer_monotone(self, law):
        # Falsification: a wrong table would map a larger sample to a smaller level.
        lin = list(range(-32768, 32768, 7))
        levels = decode(encode(lin, law), law).tolist()
        assert all(a <= b for a, b in zip(levels, levels[1:], strict=False))

    def test_encode_clips_out_of_range(self):
        assert encode([10**6, -(10**6)], Law.ULAW) == bytes([0x80, 0x00])


class TestFindPayload:
    def test_wav_alaw(self):
        data = _wav(WAV_ALAW, b"\x01\x02\x03")
        lo, hi, law = find_payload(data)
        assert (data[lo:hi], law) == (b"\x01\x02\x03", Law.ALAW)

    def test_wav_ulaw_after_odd_chunk(self):
        data = _wav(WAV_ULAW, b"\x10\x20", extra=b"JUNK" + struct.pack("<I", 1) + b"\x00\x00")
        lo, hi, law = find_payload(data)
        assert (data[lo:hi], law) == (b"\x10\x20", Law.ULAW)

    def test_au_alaw(self):
        data = _au(AU_ALAW, b"\x05\x06")
        lo, hi, law = find_payload(data)
        assert (data[lo:hi], law) == (b"\x05\x06", Law.ALAW)

    def test_au_unknown_size_runs_to_end(self):
        data = _au(AU_ULAW, b"\x05\x06\x07", size=AU_SIZE_UNKNOWN)
        lo, hi, _ = find_payload(data)
        assert data[lo:hi] == b"\x05\x06\x07"

    def test_pcm_wav_declined(self):
        assert find_payload(_wav(1, b"\x00\x00")) is None

    def test_other_au_encoding_declined(self):
        assert find_payload(_au(3, b"\x00\x00")) is None

    def test_truncated_data_chunk_clamped(self):
        data = _wav(WAV_ULAW, b"\x01\x02\x03\x04")[:-2]
        lo, hi, _ = find_payload(data)
        assert data[lo:hi] == b"\x01\x02"

    # Adversarial: hostile header fields must not index past the buffer or loop.
    def test_au_offset_past_end(self):
        data = b".snd" + struct.pack(">IIIII", 10**9, 4, AU_ULAW, 8000, 1)
        assert find_payload(data) is None

    def test_wav_huge_chunk_size_terminates(self):
        bad = b"RIFF" + bytes(4) + b"WAVE" + b"JUNK" + struct.pack("<I", 0xFFFFFFFF) + bytes(8)
        assert find_payload(bad) is None

    def test_empty_payload_declined(self):
        assert find_payload(_wav(WAV_ALAW, b"")) is None

    def test_short_inputs(self):
        for n in range(16):
            assert find_payload(_wav(WAV_ALAW, b"\x01")[:n]) is None


class TestMutate:
    PAYLOAD = bytes([0x80, 0x90, 0xA0, 0x10, 0x20])  # µ-law, mixed signs

    def _run(self, rng, payload=PAYLOAD, law_tag=WAV_ULAW):
        data = _wav(law_tag, payload)
        lo, hi, _ = find_payload(data)
        out = mutate_g711(data, max_len=len(data), rng=rng)
        assert out[:lo] == data[:lo] and len(out) == len(data)
        return out[lo:hi]

    def test_silence(self):
        assert self._run(_scripted(Op.SILENCE, 1, 3)) == bytes([0x80, 0xFF, 0xFF, 0x10, 0x20])

    def test_reverse(self):
        assert self._run(_scripted(Op.REVERSE, 0, 5)) == self.PAYLOAD[::-1]

    def test_invert_flips_ulaw_sign_bit(self):
        # µ-law stores sign in bit 7, so negation is byte ^ 0x80 (independent of the codec).
        out = self._run(_scripted(Op.GAIN, 0, 5, choice_idxs=[GAINS.index(-1)]))
        assert out == bytes(b ^ 0x80 for b in self.PAYLOAD)

    def test_saturate_alternates_extremes(self):
        out = self._run(_scripted(Op.SATURATE, 0, 4, 1))
        assert out == bytes([0x80, 0x00, 0x80, 0x00, 0x20])

    def test_offset(self):
        # +0 offset re-encodes to the same codes: the codec round-trip is exact here.
        assert self._run(_scripted(Op.OFFSET, 0, 5, 0)) == self.PAYLOAD

    def test_noise_zero_bytes_is_identity(self):
        assert self._run(_scripted(Op.NOISE, 0, 5, randbytes=[bytes(5)])) == self.PAYLOAD

    def test_alaw_silence(self):
        out = self._run(_scripted(Op.SILENCE, 0, 2), payload=b"\xaa\x2a\x00", law_tag=WAV_ALAW)
        assert out == bytes([0xD5, 0xD5, 0x00])  # A-law +0 level is 0xD5 (+8)

    def test_non_g711_declined(self):
        assert mutate_g711(_wav(1, b"\x00\x00"), max_len=64, rng=ScriptedRng()) is None

    def test_max_len_truncates(self):
        data = _wav(WAV_ULAW, self.PAYLOAD)
        out = mutate_g711(data, max_len=10, rng=_scripted(Op.REVERSE, 0, 5))
        assert len(out) == 10


class TestRegistry:
    def test_sniffer_matches_wav_and_au(self):
        assert format_gate_matches("g711_mutate", _wav(WAV_ALAW, b"\x01"))
        assert format_gate_matches("g711_mutate", _au(AU_ULAW, b"\x01"))

    def test_sniffer_rejects_pcm(self):
        assert not format_gate_matches("g711_mutate", _wav(1, b"\x01\x00"))
