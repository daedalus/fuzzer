"""Tests for the CAF container mutator (`caf_chunk_mutate`)."""

from __future__ import annotations

import math
import struct

import pytest

from fuzzer_tool.core.mutations.caf import (
    CHUNK_TYPES,
    CODECS,
    I64_EDGES,
    RATE_EDGES,
    SIZE_EDGES,
    U32_EDGES,
    VARINT_EDGES,
    VERSION_EDGES,
    Op,
    mutate_caf,
    parse_caf,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FILE_HDR = 8
CHUNK_HDR = 12


def _chunk(tag: bytes, payload: bytes, size: int | None = None) -> bytes:
    return tag + struct.pack(">q", len(payload) if size is None else size) + payload


DESC = struct.pack(">d", 44100.0) + b"lpcm" + struct.pack(">IIIII", 0, 2, 1, 1, 16)
PAKT = struct.pack(">qqII", 2, 4, 0, 0) + b"\x02\x81\x00"
SAMPLE = (
    b"caff\x00\x01\x00\x00"
    + _chunk(b"desc", DESC)
    + _chunk(b"pakt", PAKT)
    + _chunk(b"data", bytes(4) + b"abcd", size=-1)
)

DESC_OFF = FILE_HDR + CHUNK_HDR
PAKT_OFF = DESC_OFF + len(DESC) + CHUNK_HDR


class TestParse:
    def test_chunks(self):
        chunks, tail = parse_caf(SAMPLE)
        assert [c[:4] for c in chunks] == [b"desc", b"pakt", b"data"]
        assert tail == b""

    # data with size -1 runs to EOF.
    def test_unsized_data_to_eof(self):
        chunks, _ = parse_caf(SAMPLE)
        assert chunks[2].endswith(b"abcd")

    def test_not_caf(self):
        assert parse_caf(b"RIFF" + bytes(40)) is None

    # Adversarial: a chunk size past EOF ends the walk; bytes go to the tail.
    def test_oversized_chunk_goes_to_tail(self):
        bad = _chunk(b"free", b"xy", size=1 << 40)
        chunks, tail = parse_caf(b"caff\x00\x01\x00\x00" + _chunk(b"desc", DESC) + bad)
        assert len(chunks) == 1
        assert tail == bad


class TestMutate:
    def test_version(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.VERSION], choice_idxs=[0]))
        assert struct.unpack_from(">H", out, 4)[0] == VERSION_EDGES[0]
        assert out[6:] == SAMPLE[6:]

    def test_chunk_size(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.CHUNK_SIZE, 1], choice_idxs=[2]))
        assert struct.unpack_from(">Q", out, PAKT_OFF - 8)[0] == SIZE_EDGES[2]
        assert len(out) == len(SAMPLE)

    def test_chunk_type(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.CHUNK_TYPE, 2], choice_idxs=[4]))
        assert parse_caf(out)[0][2][:4] == CHUNK_TYPES[4]

    def test_rate(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.RATE], choice_idxs=[2]))
        assert math.isnan(struct.unpack_from(">d", out, DESC_OFF)[0]) == math.isnan(RATE_EDGES[2])

    def test_codec(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.CODEC], choice_idxs=[3]))
        assert out[DESC_OFF + 8 : DESC_OFF + 12] == CODECS[3]

    def test_desc_u32_channels(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.DESC_U32, 3], choice_idxs=[5]))
        assert struct.unpack_from(">I", out, DESC_OFF + 12 + 4 * 3)[0] == U32_EDGES[5]

    # The 9.0.2-only overflow: nb_frames near INT64_MAX, priming > 0.
    def test_pakt_nb_frames(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.PAKT_COUNT, 1], choice_idxs=[2]))
        assert struct.unpack_from(">Q", out, PAKT_OFF + 8)[0] == I64_EDGES[2]

    def test_pakt_priming(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.PAKT_EDGE, 0], choice_idxs=[1]))
        assert struct.unpack_from(">I", out, PAKT_OFF + 16)[0] == U32_EDGES[1]

    def test_pakt_varint_length_preserving(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.PAKT_VARINT, 1], choice_idxs=[2]))
        table = PAKT_OFF + 24
        assert out[table + 1 : table + 3] == VARINT_EDGES[2][:2]
        assert len(out) == len(SAMPLE)

    # desc stays first: chunk-order ops act on chunks[1:].
    def test_swap_keeps_desc_first(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.SWAP, 0, 0]))
        data = _chunk(b"data", bytes(4) + b"abcd", size=-1)
        assert out == SAMPLE[: PAKT_OFF - CHUNK_HDR] + data + _chunk(b"pakt", PAKT)

    def test_drop(self):
        out = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.DROP, 0]))
        assert [c[:4] for c in parse_caf(out)[0]] == [b"desc", b"data"]

    # Falsification: pakt ops decline without a pakt chunk.
    @pytest.mark.parametrize("op", [Op.PAKT_COUNT, Op.PAKT_EDGE, Op.PAKT_VARINT])
    def test_pakt_ops_decline_without_pakt(self, op):
        data = b"caff\x00\x01\x00\x00" + _chunk(b"desc", DESC)
        assert mutate_caf(data, rng=ScriptedRng(randints=[op, 0, 0])) is None

    def test_non_caf_declined(self):
        assert mutate_caf(b"\x00" * 64, rng=ScriptedRng()) is None

    def test_max_len_truncates(self):
        full = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.VERSION], choice_idxs=[0]))
        again = mutate_caf(SAMPLE, rng=ScriptedRng(randints=[Op.VERSION], choice_idxs=[0]))
        assert again == full  # control: the reference reproduces itself
        out = mutate_caf(
            SAMPLE, max_len=20, rng=ScriptedRng(randints=[Op.VERSION], choice_idxs=[0])
        )
        assert out == full[:20]
        assert out != SAMPLE[:20]

    # Adversarial: every op on every truncation never raises.
    def test_truncations_never_raise(self):
        for cut in range(FILE_HDR, len(SAMPLE) + 1):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0, 0], choice_idxs=[0, 0])
                out = mutate_caf(SAMPLE[:cut], rng=rng)
                assert out is None or out[:4] == b"caff"


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("caf_chunk_mutate", SAMPLE)
        assert not format_gate_matches("caf_chunk_mutate", b"caff")
