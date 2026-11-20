"""Tests for the IVF container mutator (`ivf_chunk_mutate`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.ivf import (
    DIM_EDGES,
    FOURCCS,
    PTS_EDGES,
    U32_EDGES,
    Op,
    mutate_ivf,
    parse_ivf,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

HDR = 32
FRAME_HDR = 12


def _ivf(payloads: list[bytes], fourcc: bytes = b"VP80", tail: bytes = b"") -> bytes:
    head = b"DKIF" + struct.pack("<HH4sHHIII", 0, HDR, fourcc, 64, 48, 30, 1, len(payloads))
    head += bytes(4)
    body = b"".join(struct.pack("<IQ", len(p), i) + p for i, p in enumerate(payloads))
    return head + body + tail


SAMPLE = _ivf([b"\x10\x11\x12", b"\x20\x21", b"\x30"])


def _frame(data: bytes, i: int) -> tuple[int, int, bytes]:
    """(size, pts, payload) of frame *i* by walking the 12-byte headers."""
    pos = HDR
    for _ in range(i):
        pos += FRAME_HDR + struct.unpack_from("<I", data, pos)[0]
    size, pts = struct.unpack_from("<IQ", data, pos)
    return size, pts, data[pos + FRAME_HDR : pos + FRAME_HDR + size]


class TestParse:
    def test_frames_and_tail(self):
        frames, tail = parse_ivf(_ivf([b"ab", b"c"], tail=b"\x05\x00"))
        assert frames == [struct.pack("<IQ", 2, 0) + b"ab", struct.pack("<IQ", 1, 1) + b"c"]
        assert tail == b"\x05\x00"

    def test_header_only(self):
        assert parse_ivf(_ivf([])) == ([], b"")

    def test_not_ivf(self):
        assert parse_ivf(b"RIFF" + bytes(40)) is None

    # Adversarial: a frame size past EOF stops the walk; bytes go to the tail.
    def test_oversized_frame_goes_to_tail(self):
        data = _ivf([b"ok"]) + struct.pack("<IQ", 0xFFFFFFFF, 9) + b"xy"
        frames, tail = parse_ivf(data)
        assert len(frames) == 1
        assert tail == struct.pack("<IQ", 0xFFFFFFFF, 9) + b"xy"

    def test_short_header(self):
        assert parse_ivf(b"DKIF" + bytes(20)) is None


class TestMutate:
    def test_dims_height(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.DIMS, 1], choice_idxs=[4]))
        assert struct.unpack_from("<HH", out, 12) == (64, DIM_EDGES[4])
        assert out[16:] == SAMPLE[16:]

    def test_fourcc(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.FOURCC], choice_idxs=[2]))
        assert out[8:12] == FOURCCS[2]

    def test_timebase_scale(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.TIMEBASE, 1], choice_idxs=[0]))
        assert struct.unpack_from("<II", out, 16) == (30, U32_EDGES[0])

    def test_nframes(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.NFRAMES], choice_idxs=[4]))
        assert struct.unpack_from("<I", out, 24)[0] == U32_EDGES[4]

    def test_frame_size_desyncs_header_only(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.FRAME_SIZE, 1], choice_idxs=[1]))
        pos = HDR + FRAME_HDR + 3
        assert struct.unpack_from("<I", out, pos)[0] == U32_EDGES[1]
        assert len(out) == len(SAMPLE)

    def test_pts(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.PTS, 2], choice_idxs=[2]))
        assert _frame(out, 2)[1] == PTS_EDGES[2]

    def test_swap_frames(self):
        out = mutate_ivf(SAMPLE, rng=ScriptedRng(randints=[Op.SWAP, 0, 1]))
        assert [_frame(out, i)[2] for i in range(3)] == [b"\x30", b"\x20\x21", b"\x10\x11\x12"]

    def test_tail_preserved(self):
        data = _ivf([b"ab"], tail=b"junk")
        out = mutate_ivf(data, rng=ScriptedRng(randints=[Op.DUP, 0]))
        assert out.endswith(b"junk")
        assert [_frame(out, i)[2] for i in range(2)] == [b"ab", b"ab"]

    # Falsification: frame ops on a header-only file decline.
    @pytest.mark.parametrize("op", [Op.FRAME_SIZE, Op.PTS, Op.DUP, Op.DROP, Op.SWAP])
    def test_frame_ops_decline_without_frames(self, op):
        assert mutate_ivf(_ivf([]), rng=ScriptedRng(randints=[op, 0, 0])) is None

    def test_non_ivf_declined(self):
        assert mutate_ivf(b"\x00" * 64, rng=ScriptedRng()) is None

    def test_max_len_truncates(self):
        out = mutate_ivf(SAMPLE, max_len=40, rng=ScriptedRng(randints=[Op.FOURCC], choice_idxs=[0]))
        assert len(out) == 40

    # Adversarial: every op on every truncation of a valid file never raises.
    def test_truncations_never_raise(self):
        for cut in range(HDR, len(SAMPLE) + 1):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0, 0], choice_idxs=[0, 0])
                out = mutate_ivf(SAMPLE[:cut], rng=rng)
                assert out is None or out[:4] == b"DKIF"


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("ivf_chunk_mutate", SAMPLE)
        assert not format_gate_matches("ivf_chunk_mutate", b"DKIF" + bytes(8))
