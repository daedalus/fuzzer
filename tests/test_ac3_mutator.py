"""Tests for the AC-3 / E-AC-3 frame mutator (`ac3_frame_mutate`)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.ac3 import (
    BSID_EDGES,
    EAC3_FRMSIZ_EDGES,
    FRMSIZECOD_EDGES,
    Op,
    frame_size,
    mutate_ac3,
    parse_ac3,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

CRC16_POLY = 0x8005  # FFmpeg AV_CRC_16_ANSI, MSB-first, init 0
AC3_BSID, EAC3_BSID = 8, 16
FFMPEG = shutil.which("ffmpeg")


def _crc16(data: bytes) -> int:
    """Bitwise oracle, independent of the table-driven code under test."""
    reg = 0
    for byte in data:
        reg ^= byte << 8
        for _ in range(8):
            reg = ((reg << 1) ^ CRC16_POLY) if reg & 0x8000 else reg << 1
            reg &= 0xFFFF
    return reg


def _crc_ok(frame: bytes) -> bool:
    """FFmpeg's check: CRC over everything after the sync word is zero."""
    return _crc16(frame[2:]) == 0


def _bits(fields: list[tuple[int, int]]) -> bytes:
    """Pack (value, width) pairs MSB-first, zero-padded to a byte."""
    acc, n = 0, 0
    for value, width in fields:
        acc, n = (acc << width) | value, n + width
    pad = -n % 8
    return (acc << pad).to_bytes((n + pad) // 8, "big")


def _seal(frame: bytearray) -> bytes:
    frame[-2:] = _crc16(bytes(frame[2:-2])).to_bytes(2, "big")
    return bytes(frame)


def _ac3(fscod: int, frmsizecod: int, size: int, acmod: int = 2) -> bytes:
    hdr = _bits([(0x0B77, 16), (0, 16), (fscod, 2), (frmsizecod, 6), (AC3_BSID, 5), (0, 3)])
    hdr += _bits([(acmod, 3), (0, 5)])
    return _seal(bytearray(hdr + bytes(size - len(hdr))))


def _eac3(size: int) -> bytes:
    fields = [(0x0B77, 16), (0, 2), (0, 3), (size // 2 - 1, 11), (0, 2), (3, 2), (2, 3), (0, 1)]
    hdr = _bits([*fields, (EAC3_BSID, 5), (0, 3)])
    return _seal(bytearray(hdr + bytes(size - len(hdr))))


# ATSC A/52 Table 5.18: 48 kHz code 0 = 64 words, 44.1 kHz code 1 = 70, 32 kHz code 0 = 96.
F48 = _ac3(0, 0, 128)
F44 = _ac3(1, 1, 140)
F32 = _ac3(2, 0, 192)
STREAM = F48 + F48 + F48


class TestFrameSize:
    @pytest.mark.parametrize(("frame", "size"), [(F48, 128), (F44, 140), (F32, 192)])
    def test_ac3_table(self, frame, size):
        assert frame_size(frame, 0) == size

    def test_eac3_frmsiz(self):
        assert frame_size(_eac3(200), 0) == 200

    @pytest.mark.parametrize(("fscod", "code"), [(3, 0), (0, 38), (0, 63)])
    def test_reserved_codes(self, fscod, code):
        assert (
            frame_size(_ac3(0, 0, 128)[:4] + _bits([(fscod, 2), (code, 6)]) + bytes(8), 0) is None
        )

    def test_bsid_past_eac3(self):
        frame = bytearray(F48)
        frame[5] = (17 << 3) | (frame[5] & 7)
        assert frame_size(frame, 0) is None


class TestParse:
    def test_chain(self):
        start, frames, tail = parse_ac3(STREAM + b"\x01\x02")
        assert (start, frames, tail) == (0, [F48] * 3, b"\x01\x02")

    def test_mixed_ac3_eac3(self):
        e = _eac3(200)
        assert parse_ac3(F48 + e)[1] == [F48, e]

    def test_prefix_needs_second_frame(self):
        assert parse_ac3(b"ID3junk" + F48 + F48)[0] == 7
        # Falsification: one frame after a prefix is too weak a signal.
        assert parse_ac3(b"ID3junk" + F48) is None

    # Adversarial: a sync word with a plausible header in random-looking data.
    def test_lone_sync_in_noise_rejected(self):
        assert parse_ac3(bytes(100) + F48[:10] + bytes(300)) is None

    def test_truncated_frame(self):
        assert parse_ac3(F48[:100]) is None


class TestMutate:
    def test_frmsizecod(self):
        out = mutate_ac3(STREAM, rng=ScriptedRng(randints=[Op.SIZE_CODE, 1], choice_idxs=[3]))
        assert out[128 + 4] & 0x3F == FRMSIZECOD_EDGES[3]
        assert out[:128] == F48
        assert _crc_ok(out[128:256])

    def test_eac3_frmsiz(self):
        e = _eac3(200)
        out = mutate_ac3(e + e, rng=ScriptedRng(randints=[Op.SIZE_CODE, 0], choice_idxs=[1]))
        assert int.from_bytes(out[2:4], "big") & 0x7FF == EAC3_FRMSIZ_EDGES[1]

    def test_bsid(self):
        out = mutate_ac3(STREAM, rng=ScriptedRng(randints=[Op.BSID, 2], choice_idxs=[6]))
        assert out[256 + 5] >> 3 == BSID_EDGES[6]
        assert _crc_ok(out[256:])

    def test_acmod_ac3(self):
        out = mutate_ac3(STREAM, rng=ScriptedRng(randints=[Op.ACMOD, 0, 7]))
        assert out[6] >> 5 == 7

    def test_acmod_eac3(self):
        e = _eac3(200)
        out = mutate_ac3(e + e, rng=ScriptedRng(randints=[Op.ACMOD, 1, 5]))
        assert (out[200 + 4] >> 1) & 7 == 5

    def test_lfeon_after_mix_levels(self):
        # acmod 7 carries cmixlev and surmixlev: lfeon is bit 51 + 2 + 2 = 55.
        frame = _ac3(0, 0, 128, acmod=7)
        out = mutate_ac3(frame + frame, rng=ScriptedRng(randints=[Op.LFEON, 0]))
        assert out[6] & 1 == 1
        assert _crc_ok(out[:128])

    def test_payload_flip_keeps_crc(self):
        bit = 8 * 60 + 3
        out = mutate_ac3(STREAM, rng=ScriptedRng(randints=[Op.PAYLOAD, 1, bit]))
        assert out[128 + 60] == F48[60] ^ (0x80 >> 3)
        assert all(_crc_ok(out[k : k + 128]) for k in range(0, 384, 128))

    def test_drop(self):
        out = mutate_ac3(STREAM, rng=ScriptedRng(randints=[Op.DROP, 0]))
        assert out == F48 + F48

    # Falsification: a lone frame cannot be dropped.
    def test_drop_single_declines(self):
        assert mutate_ac3(F48, rng=ScriptedRng(randints=[Op.DROP, 0])) is None

    def test_non_ac3_declined(self):
        assert mutate_ac3(b"\x00" * 256, rng=ScriptedRng()) is None

    # Adversarial: every op on every truncation never raises.
    def test_truncations_never_raise(self):
        for cut in range(0, len(STREAM) + 1, 7):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0, 48], choice_idxs=[0])
                out = mutate_ac3(STREAM[:cut], rng=rng)
                assert out is None or isinstance(out, bytes)


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
class TestFfmpeg:
    """Real encoder output: framing and CRC repair as FFmpeg sees them."""

    def _encode(self, tmp_path, codec: str) -> bytes:
        path = tmp_path / f"in.{codec}"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=d=0.2", "-c:a", codec, str(path)]
        subprocess.run(cmd, check=True)
        return path.read_bytes()

    def _crc_errors(self, tmp_path, data: bytes) -> str:
        path = tmp_path / "out.ac3"
        path.write_bytes(data)
        cmd = [FFMPEG, "-v", "error", "-err_detect", "crccheck", "-i", str(path), "-f", "null", "-"]
        return subprocess.run(cmd, capture_output=True, text=True).stderr

    @pytest.mark.parametrize("codec", ["ac3", "eac3"])
    def test_parse_covers_file(self, tmp_path, codec):
        data = self._encode(tmp_path, codec)
        start, frames, tail = parse_ac3(data)
        assert (start, tail) == (0, b"")
        assert b"".join(frames) == data
        assert all(_crc_ok(f) for f in frames)

    def test_payload_flip_passes_crccheck(self, tmp_path):
        data = self._encode(tmp_path, "ac3")
        n = len(parse_ac3(data)[1][0])
        out = mutate_ac3(data, rng=ScriptedRng(randints=[Op.PAYLOAD, 1, n * 4]))
        assert "CRC" not in self._crc_errors(tmp_path, out)

    # Control: the same flip without repair must trip the check.
    def test_unrepaired_flip_fails_crccheck(self, tmp_path):
        data = bytearray(self._encode(tmp_path, "ac3"))
        n = len(parse_ac3(bytes(data))[1][0])
        data[n + n // 2] ^= 0x10
        assert "CRC" in self._crc_errors(tmp_path, bytes(data))


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("ac3_frame_mutate", STREAM)
        assert not format_gate_matches("ac3_frame_mutate", F48[:64])
