"""Tests for the DTS core frame mutator (`dca_frame_mutate`)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.dca import (
    AMODE_EDGES,
    FSIZE_EDGES,
    NBLKS_EDGES,
    Op,
    frame_size,
    mutate_dca,
    parse_dca,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")
FRAME_LEN = 128


def _bits(fields: list[tuple[int, int]]) -> bytes:
    acc, n = 0, 0
    for value, width in fields:
        acc, n = (acc << width) | value, n + width
    pad = -n % 8
    return (acc << pad).to_bytes((n + pad) // 8, "big")


def _get(data: bytes, pos: int, width: int) -> int:
    """Test-side bit reader (MSB-first), independent of frame_stream."""
    return (int.from_bytes(data, "big") >> (len(data) * 8 - pos - width)) & ((1 << width) - 1)


def _dts(size: int = FRAME_LEN, cpf: int = 0) -> bytes:
    """Core header per ETSI TS 102 114 §5.3.1, zero audio after it."""
    fields = [(0x7FFE8001, 32), (1, 1), (31, 5), (cpf, 1), (15, 7), (size - 1, 14)]
    fields += [(2, 6), (13, 4), (15, 5), (0, 1), (0, 4), (0, 3), (0, 1), (0, 1), (0, 2), (0, 1)]
    fields += [(0, 16)] * cpf + [(0, 1), (7, 4), (0, 2), (0, 3)]
    hdr = _bits(fields)
    return hdr + bytes(size - len(hdr))


F = _dts()
STREAM = F + F + F


class TestFrameSize:
    def test_fsize_plus_one(self):
        assert frame_size(F, 0) == FRAME_LEN

    # Falsification: FSIZE below 95 is invalid per the spec.
    def test_short_fsize_rejected(self):
        assert frame_size(_dts(95), 0) is None
        assert frame_size(_dts(96), 0) == 96

    def test_wrong_sync(self):
        assert frame_size(b"\x7f\xfe\x80\x00" + F[4:], 0) is None


class TestParse:
    def test_chain_and_tail(self):
        assert parse_dca(STREAM + b"\xaa") == (0, [F] * 3, b"\xaa")

    def test_prefix_needs_second_frame(self):
        assert parse_dca(b"xx" + F + F)[0] == 2
        assert parse_dca(b"xx" + F) is None

    # Adversarial: truncated mid-frame.
    def test_truncated(self):
        assert parse_dca(F[:-1]) is None


class TestMutate:
    def _field(self, out: bytes, i: int, pos: int, width: int) -> int:
        return _get(out[i * FRAME_LEN : (i + 1) * FRAME_LEN], pos, width)

    def test_ftype(self):
        out = mutate_dca(STREAM, rng=ScriptedRng(randints=[Op.FTYPE, 1]))
        assert self._field(out, 1, 32, 1) == 0
        assert out[:FRAME_LEN] == F

    def test_nblks(self):
        out = mutate_dca(STREAM, rng=ScriptedRng(randints=[Op.NBLKS, 0], choice_idxs=[2]))
        assert self._field(out, 0, 39, 7) == NBLKS_EDGES[2]

    def test_fsize(self):
        out = mutate_dca(STREAM, rng=ScriptedRng(randints=[Op.FSIZE, 2], choice_idxs=[0]))
        assert self._field(out, 2, 46, 14) == FSIZE_EDGES[0]

    def test_amode(self):
        out = mutate_dca(STREAM, rng=ScriptedRng(randints=[Op.AMODE, 0], choice_idxs=[3]))
        assert self._field(out, 0, 60, 6) == AMODE_EDGES[3]

    @pytest.mark.parametrize(
        ("op", "pos", "width", "value"),
        [(Op.SFREQ, 66, 4, 15), (Op.LFF, 85, 2, 3), (Op.EXT, 80, 4, 9)],
    )
    def test_plain_fields(self, op, pos, width, value):
        out = mutate_dca(STREAM, rng=ScriptedRng(randints=[op, 1, value]))
        assert self._field(out, 1, pos, width) == value

    @pytest.mark.parametrize(("cpf", "pos"), [(0, 95), (1, 111)])
    def test_pcmr_follows_header_crc(self, cpf, pos):
        frame = _dts(cpf=cpf)
        out = mutate_dca(frame + frame, rng=ScriptedRng(randints=[Op.PCMR, 0, 5]))
        assert _get(out[:FRAME_LEN], pos, 3) == 5

    def test_dup(self):
        assert mutate_dca(F + F, rng=ScriptedRng(randints=[Op.DUP, 0])) == F * 3

    def test_swap_single_declines(self):
        assert mutate_dca(F, rng=ScriptedRng(randints=[Op.SWAP, 0, 0])) is None

    def test_non_dts_declined(self):
        assert mutate_dca(bytes(512), rng=ScriptedRng()) is None

    # Adversarial: every op on every truncation never raises.
    def test_truncations_never_raise(self):
        for cut in range(0, len(STREAM) + 1, 5):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0], choice_idxs=[0])
                out = mutate_dca(STREAM[:cut], rng=rng)
                assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("dca_frame_mutate", STREAM)
        assert not format_gate_matches("dca_frame_mutate", b"\x7f\xfe\x80\x01" + bytes(20))


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
def test_parse_covers_ffmpeg_output(tmp_path):
    path = tmp_path / "in.dts"
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=d=0.2", "-ac", "2"]
    subprocess.run([*cmd, "-strict", "-2", "-c:a", "dca", str(path)], check=True)
    data = path.read_bytes()
    start, frames, tail = parse_dca(data)
    assert (start, tail) == (0, b"")
    assert b"".join(frames) == data
