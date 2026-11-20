"""Tests for the FLAC frame mutator (`flac_frame_mutate`)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.flac_frame import (
    NUMBER_EDGES,
    Op,
    decode_number,
    encode_number,
    mutate_flac_frame,
    parse_frames,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")
STREAMINFO_LEN = 34


def _crc(data: bytes, width: int, poly: int) -> int:
    """Bitwise MSB-first CRC, init 0: independent oracle for CRC-8 / CRC-16."""
    top, mask = 1 << (width - 1), (1 << width) - 1
    reg = 0
    for byte in data:
        reg ^= byte << (width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ poly) if reg & top else reg << 1
            reg &= mask
    return reg


def crc8(data: bytes) -> int:
    return _crc(data, 8, 0x07)


def crc16(data: bytes) -> int:
    return _crc(data, 16, 0x8005)


def _frame(number: int, body: bytes, bs_code: int = 0xC, sr_code: int = 0x9) -> bytes:
    """Fixed-blocksize frame: 4096 samples (code 12), 44.1 kHz (code 9), mono, 16-bit."""
    head = bytes([0xFF, 0xF8, (bs_code << 4) | sr_code, 0x08]) + bytes([number])
    head += bytes([crc8(head)])
    frame = head + body
    return frame + crc16(frame).to_bytes(2, "big")


def _native(frames: list[bytes]) -> bytes:
    streaminfo = bytes(STREAMINFO_LEN)
    return (
        b"fLaC" + bytes([0x80]) + STREAMINFO_LEN.to_bytes(3, "big") + streaminfo + b"".join(frames)
    )


F0, F1, F2 = _frame(0, b"\x10" * 20), _frame(1, b"\x20" * 20), _frame(2, b"\x30" * 20)
STREAM = _native([F0, F1, F2])
AUDIO = STREAM.index(F0)


def _frame_crc_ok(frame: bytes, header_len: int) -> bool:
    return crc8(frame[:header_len]) == 0 and crc16(frame) == 0


class TestCodedNumber:
    @pytest.mark.parametrize(
        ("value", "raw"),
        [(0, b"\x00"), (0x7F, b"\x7f"), (0x80, b"\xc2\x80"), (0x7FF, b"\xdf\xbf")],
    )
    def test_utf8_like(self, value, raw):
        assert encode_number(value) == raw
        assert decode_number(raw + b"\xaa", 0) == (value, len(raw))

    def test_36_bit(self):
        raw = encode_number(2**36 - 1)
        assert raw == b"\xfe" + b"\xbf" * 6
        assert decode_number(raw, 0) == (2**36 - 1, 7)

    # Adversarial: bad lead byte, bad continuation, truncation.
    @pytest.mark.parametrize("raw", [b"\x80", b"\xff", b"\xc2\x00", b"\xe0\x80"])
    def test_invalid(self, raw):
        assert decode_number(raw, 0) is None


class TestParse:
    def test_native_stream(self):
        start, frames = parse_frames(STREAM)
        assert start == AUDIO
        assert frames == [F0, F1, F2]

    def test_raw_frames(self):
        assert parse_frames(F0 + F1) == (0, [F0, F1])

    def test_sync_lookalike_with_bad_crc8_is_not_a_boundary(self):
        fake = bytearray(_frame(9, b"")[:6])
        fake[5] ^= 0xFF
        f = _frame(0, b"\x11" + bytes(fake) + b"\x22")
        assert parse_frames(f + F1)[1] == [f, F1]

    # Falsification: metadata without any frame.
    def test_no_frames(self):
        assert parse_frames(_native([])) is None

    def test_not_flac(self):
        assert parse_frames(bytes(64)) is None


class TestMutate:
    def test_block_size_code_keeps_crcs(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.BLOCK_SIZE, 0, 0x1]))
        frame = out[AUDIO : AUDIO + len(F0)]
        assert frame[2] >> 4 == 0x1
        assert _frame_crc_ok(frame, 6)
        assert out[AUDIO + len(F0) :] == F1 + F2

    def test_block_size_code_7_adds_16_bit_field(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.BLOCK_SIZE, 1, 0x7]))
        frame = out[AUDIO + len(F0) : AUDIO + len(F0) + len(F1) + 2]
        assert frame[5:7] == b"\xff\xff"
        assert _frame_crc_ok(frame, 8)

    def test_sample_rate_code_13_adds_16_bit_field(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.SAMPLE_RATE, 0, 13]))
        assert len(out) == len(STREAM) + 2
        frame = out[AUDIO : AUDIO + len(F0) + 2]
        assert frame[2] & 0xF == 13
        assert _frame_crc_ok(frame, 8)

    def test_channels(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.CHANNELS, 2, 0xF]))
        assert out[-len(F2) + 3] >> 4 == 0xF

    def test_sample_size(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.SAMPLE_SIZE, 0, 3]))
        assert (out[AUDIO + 3] >> 1) & 7 == 3

    def test_strategy_flip(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.STRATEGY, 0]))
        assert out[AUDIO + 1] == 0xF9

    def test_number_resizes_header(self):
        rng = ScriptedRng(randints=[Op.NUMBER, 0], choice_idxs=[len(NUMBER_EDGES) - 1])
        out = mutate_flac_frame(STREAM, rng=rng)
        frame = out[AUDIO : AUDIO + len(F0) + 6]
        assert decode_number(frame, 4) == (NUMBER_EDGES[-1], 7)
        assert _frame_crc_ok(frame, 12)

    def test_payload_flip(self):
        bit = 8 * 5 + 1
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.PAYLOAD, 1, bit]))
        frame = out[AUDIO + len(F0) : AUDIO + len(F0) + len(F1)]
        assert frame[6 + 5] == F1[6 + 5] ^ 0x40
        assert _frame_crc_ok(frame, 6)

    def test_swap(self):
        out = mutate_flac_frame(STREAM, rng=ScriptedRng(randints=[Op.SWAP, 0, 1]))
        assert out[AUDIO:] == F2 + F1 + F0
        assert out[:AUDIO] == STREAM[:AUDIO]

    def test_drop_single_declines(self):
        assert mutate_flac_frame(_native([F0]), rng=ScriptedRng(randints=[Op.DROP, 0])) is None

    def test_payload_on_empty_body_declines(self):
        data = _native([_frame(0, b"")])
        assert mutate_flac_frame(data, rng=ScriptedRng(randints=[Op.PAYLOAD, 0, 0])) is None

    # Adversarial: every truncation, every op, never raises.
    def test_truncations_never_raise(self):
        for cut in range(AUDIO, len(STREAM) + 1, 3):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0], choice_idxs=[0])
                out = mutate_flac_frame(STREAM[:cut], rng=rng)
                assert out is None or out[:4] == b"fLaC"


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("flac_frame_mutate", STREAM)
        assert format_gate_matches("flac_frame_mutate", F0)
        assert not format_gate_matches("flac_frame_mutate", _native([]))


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
class TestFfmpeg:
    def _encode(self, tmp_path) -> bytes:
        path = tmp_path / "in.flac"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=d=0.5", "-c:a", "flac", str(path)]
        subprocess.run(cmd, check=True)
        return path.read_bytes()

    def _errors(self, tmp_path, data: bytes) -> str:
        path = tmp_path / "out.flac"
        path.write_bytes(data)
        cmd = [FFMPEG, "-v", "error", "-err_detect", "crccheck", "-i", str(path), "-f", "null", "-"]
        return subprocess.run(cmd, capture_output=True, text=True).stderr

    def test_parse_covers_audio(self, tmp_path):
        data = self._encode(tmp_path)
        start, frames = parse_frames(data)
        assert b"".join(frames) == data[start:]
        assert len(frames) > 1
        assert all(crc16(f) == 0 for f in frames)

    def test_payload_flip_passes_crccheck(self, tmp_path):
        data = self._encode(tmp_path)
        n = len(parse_frames(data)[1][1])
        out = mutate_flac_frame(data, rng=ScriptedRng(randints=[Op.PAYLOAD, 1, n * 4]))
        assert "CRC" not in self._errors(tmp_path, out)

    # Control: the same flip unrepaired trips the frame CRC.
    def test_unrepaired_flip_fails_crccheck(self, tmp_path):
        data = bytearray(self._encode(tmp_path))
        start, frames = parse_frames(bytes(data))
        data[start + len(frames[0]) + len(frames[1]) // 2] ^= 0x08
        assert "CRC" in self._errors(tmp_path, bytes(data))
