"""Tests for the FLV payload-field mutator (`flv_field_mutate`)."""

from __future__ import annotations

import math
import struct

from fuzzer_tool.core.mutations.flv_fields import (
    AMF_TYPES,
    CTS_EDGES,
    DOUBLE_EDGES,
    HEAD_FLAGS,
    VIDEO_FOURCCS,
    mutate_flv_fields,
)
from fuzzer_tool.core.operator_registry import format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool

from .support.scripted_rng import ScriptedRng

FLV_HDR = 9
TAG_HDR = 11
PREV = 4


def _amf_key(name: bytes) -> bytes:
    return struct.pack(">H", len(name)) + name


def _num(name: bytes, value: float) -> bytes:
    return _amf_key(name) + b"\x00" + struct.pack(">d", value)


META = (
    b"\x02"
    + _amf_key(b"onMetaData")
    + b"\x08"
    + struct.pack(">I", 2)
    + _num(b"duration", 1.5)
    + _num(b"width", 64.0)
    + b"\x00\x00\x09"
)
AUDIO = b"\xaf\x00\x12\x10"  # AAC, sequence header, ASC
VIDEO = b"\x17\x01\x00\x00\x10" + b"vid!"  # AVC key frame, NALU, cts 16
ENH_VIDEO = b"\x91hvc1\x00\x00\x00" + b"vid!"  # ExHeader key, CodedFrames, hvc1, cts


def _flv(*tags: tuple[int, bytes]) -> tuple[bytes, list[int]]:
    """FLV file plus each tag's absolute payload offset."""
    out = bytearray(b"FLV\x01\x05" + struct.pack(">I", FLV_HDR))
    offs = []
    prev = 0
    for kind, body in tags:
        out += struct.pack(">I", prev) + bytes([kind]) + len(body).to_bytes(3, "big") + bytes(7)
        offs.append(len(out))
        out += body
        prev = TAG_HDR + len(body)
    out += struct.pack(">I", prev)
    return bytes(out), offs


SAMPLE, OFFS = _flv((18, META), (8, AUDIO), (9, VIDEO))
META_OFF, AUDIO_OFF, VIDEO_OFF = OFFS


class TestMutate:
    # sites: header, script, audio, video (file order after the header)
    def test_header_flags(self):
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[0, 0], choice_idxs=[0]))
        assert out[4] == HEAD_FLAGS[0]
        assert out[5:] == SAMPLE[5:]

    def test_ecma_count(self):
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[1, 0], choice_idxs=[3]))
        at = META_OFF + META.index(b"\x08\x00") + 1
        assert struct.unpack_from(">I", out, at)[0] == 0x80000000

    def test_duration_nan(self):
        idx = next(i for i, e in enumerate(DOUBLE_EDGES) if math.isnan(struct.unpack(">d", e)[0]))
        # script fields: ecma count, duration type, duration value, width type, width value
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[1, 2], choice_idxs=[idx]))
        at = META_OFF + META.index(b"duration") + len(b"duration") + 1
        assert math.isnan(struct.unpack_from(">d", out, at)[0])

    def test_width_type_retag(self):
        idx = AMF_TYPES.index(12)
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[1, 3], choice_idxs=[idx]))
        assert out[META_OFF + META.index(b"width") + len(b"width")] == 12

    def test_aac_packet_type(self):
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[2, 1], choice_idxs=[1]))
        assert out[AUDIO_OFF + 1] == 1

    def test_avc_cts(self):
        idx = CTS_EDGES.index(0x800000)
        out = mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[3, 2], choice_idxs=[idx]))
        assert out[VIDEO_OFF + 2 : VIDEO_OFF + 5] == b"\x80\x00\x00"

    def test_enhanced_fourcc_and_cts(self):
        data, (off,) = _flv((9, ENH_VIDEO))
        out = mutate_flv_fields(data, rng=ScriptedRng(randints=[1, 1], choice_idxs=[2]))
        assert out[off + 1 : off + 5] == VIDEO_FOURCCS[2]
        idx = CTS_EDGES.index(0xFFFFFF)
        out = mutate_flv_fields(data, rng=ScriptedRng(randints=[1, 2], choice_idxs=[idx]))
        assert out[off + 5 : off + 8] == b"\xff\xff\xff"

    # Falsification: writing the value already present declines.
    def test_unchanged_declines(self):
        idx = AMF_TYPES.index(0)
        assert (
            mutate_flv_fields(SAMPLE, rng=ScriptedRng(randints=[1, 1], choice_idxs=[idx])) is None
        )

    def test_not_flv(self):
        assert mutate_flv_fields(b"RIFF" + bytes(40), rng=ScriptedRng()) is None

    # Adversarial: every truncation never raises and keeps length (seeded draws).
    def test_truncations_never_raise(self):
        for cut in range(len(SAMPLE) + 1):
            out = mutate_flv_fields(SAMPLE[:cut], rng=RandPool(seed=cut))
            assert out is None or len(out) == cut

    # Adversarial: a key whose value runs past the tag end exposes no double.
    def test_key_at_tag_end(self):
        body = b"\x02" + _amf_key(b"onMetaData") + b"\x03" + _amf_key(b"duration") + b"\x00\x40"
        data, (off,) = _flv((18, body))
        # fields: duration type only
        out = mutate_flv_fields(data, rng=ScriptedRng(randints=[1, 0], choice_idxs=[1]))
        assert out[off + body.index(b"duration") + len(b"duration")] == AMF_TYPES[1]


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("flv_field_mutate", SAMPLE)
        assert not format_gate_matches("flv_field_mutate", b"FLV\x01")
