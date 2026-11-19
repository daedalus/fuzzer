"""Tests for the structure-aware MPEG-PS mutator (`mpegps_chunk_mutate`)."""

from __future__ import annotations

import random
import struct
from types import SimpleNamespace

from fuzzer_tool.core.mutations.mpegps import (
    PACK_CODE,
    PRIVATE_1,
    PROGRAM_END,
    STREAM_IDS,
    SUBSTREAM_IDS,
    MpegpsMutator,
    parse_ps_units,
    serialize_ps_units,
)
from fuzzer_tool.core.operator_registry import format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool

from .support.scripted_rng import ScriptedRng

START = b"\x00\x00\x01"
# MPEG-2 pack header body: SCR 0, mux rate, no stuffing.
PACK_BODY = bytes.fromhex("4400040004010189c3f8")
PTS_ZERO = bytes.fromhex("2100010001")
OP_PES_LENGTH, OP_STREAM_ID, OP_SUBSTREAM = 0, 1, 5


def _pes(stream_id: int, payload: bytes) -> bytes:
    """MPEG-2 PES with a PTS: flags 0x81 0x80, header_data_length 5."""
    content = b"\x81\x80\x05" + PTS_ZERO + payload
    return START + bytes([stream_id]) + struct.pack(">H", len(content)) + content


def _stream() -> bytes:
    return (
        START
        + bytes([PACK_CODE])
        + PACK_BODY
        + _pes(0xE0, b"\x00\x00\x01\xb3" + bytes(12))
        + _pes(PRIVATE_1, b"\x80\x01\x00\x01\x0b\x77" + bytes(10))
        + START
        + bytes([PROGRAM_END])
    )


def _run(op: int, randints=(), choice_idxs=()) -> bytes:
    rng = ScriptedRng(randints=[op, *randints], choice_idxs=list(choice_idxs))
    return MpegpsMutator().mutate(_stream(), max_len=1 << 16, rng=rng)


class TestSniffer:
    def test_pack_header_detected(self):
        assert format_gate_matches("mpegps_chunk_mutate", _stream())

    def test_es_not_detected(self):
        # Falsification: an MPEG-2 video ES starts with a sequence header, not a pack.
        assert not format_gate_matches("mpegps_chunk_mutate", START + b"\xb3" + bytes(16))

    def test_ps_not_nal(self):
        assert not format_gate_matches("nal_chunk_mutate", _stream())


class TestParse:
    def test_units(self):
        units = parse_ps_units(_stream())
        assert [u.code for u in units] == [PACK_CODE, 0xE0, PRIVATE_1, PROGRAM_END]

    def test_round_trip(self):
        data = _stream()
        assert serialize_ps_units(parse_ps_units(data)) == data

    def test_start_code_inside_pes_not_split(self):
        # The video PES payload carries 00 00 01 B3; its length field must win.
        assert len(parse_ps_units(_stream())) == 4

    def test_not_ps(self):
        assert parse_ps_units(START + b"\xb3" + bytes(16)) is None

    def test_adversarial_round_trips(self):
        # Length past EOF, truncated pack header, junk between units, bare codes.
        long_pes = START + b"\xe0\xff\xff" + bytes(8)
        cases = (
            START + bytes([PACK_CODE]) + PACK_BODY + long_pes,
            START + bytes([PACK_CODE]) + PACK_BODY[:3],
            START + bytes([PACK_CODE]) + PACK_BODY + b"junk\x00\x00" + _pes(0xC0, b"a"),
            START + bytes([PACK_CODE]) + PACK_BODY + START + b"\xe0",
            START + bytes([PACK_CODE]),
        )
        for data in cases:
            units = parse_ps_units(data)
            assert units is not None
            assert serialize_ps_units(units) == data


class TestOps:
    def test_pes_length(self):
        # choice 0 = video PES; value index 4 = 0xFFFF.
        out = _run(OP_PES_LENGTH, choice_idxs=[0, 4])
        video = parse_ps_units(out)[1]
        assert video.body[:2] == b"\xff\xff"

    def test_stream_id(self):
        value_idx = 3
        out = _run(OP_STREAM_ID, choice_idxs=[1, value_idx])
        assert parse_ps_units(out)[2].code == STREAM_IDS[value_idx]

    def test_substream(self):
        value_idx = 2
        out = _run(OP_SUBSTREAM, choice_idxs=[0, value_idx])
        private = parse_ps_units(out)[2]
        assert private.body[2 + 3 + 5] == SUBSTREAM_IDS[value_idx]

    def test_pes_ops_skip_streams_without_pes(self):
        # Falsification: pack header + end code only, PES ops are no-ops.
        data = START + bytes([PACK_CODE]) + PACK_BODY + START + bytes([PROGRAM_END])
        for op in (OP_PES_LENGTH, OP_STREAM_ID, OP_SUBSTREAM):
            rng = ScriptedRng(randints=[op], choice_idxs=[0, 0])
            assert MpegpsMutator().mutate(data, max_len=1 << 16, rng=rng) == data

    def test_max_len(self):
        mut = MpegpsMutator()
        for seed in range(30):
            out = mut.mutate(_stream(), max_len=40, rng=random.Random(seed))
            assert len(out) <= 40


class TestGenerator:
    def test_generated_stream_parses(self):
        out = MpegpsMutator()._generate_random_ps(rng=RandPool(seed=3))
        units = parse_ps_units(out)
        assert units[0].code == PACK_CODE
        assert units[-1].code == PROGRAM_END
        assert format_gate_matches("mpegps_chunk_mutate", out)

    def test_non_ps_input_generates(self):
        out = MpegpsMutator().mutate(b"not a program stream", rng=RandPool(seed=1))
        assert parse_ps_units(out)[0].code == PACK_CODE


class TestHandler:
    def test_handler_mutates_seed(self):
        from fuzzer_tool.services.operators import OperatorEngine

        engine = object.__new__(OperatorEngine)
        rng = ScriptedRng(randints=[OP_STREAM_ID], choice_idxs=[1, 0])
        engine._ctx_cache = SimpleNamespace(_rng=rng, max_len=1 << 16)
        out = engine._op_mpegps_chunk_mutate(bytearray(_stream()), 0, b"")
        assert parse_ps_units(bytes(out))[2].code == STREAM_IDS[0]
