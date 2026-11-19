"""NAL emulation-prevention op (`nal.py`) and AVCC length op (`isobmff.py`)."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.isobmff import (
    EDGE_U32,
    Box,
    IsobmffMutator,
    avcc_length_size,
    serialize_boxes,
)
from fuzzer_tool.core.mutations.nal import EP_FOLLOW, NalMutator, NalUnit, serialize_nal_units

from .support.scripted_rng import ScriptedRng

SC4 = b"\x00\x00\x00\x01"
EP_OP = 9
AVCC_OP = 15
INSERT, STRIP = 0, 1


def _units(payload: bytes) -> list[NalUnit]:
    return [
        NalUnit(start_code=SC4, header=b"\x67", payload=b"\x42\x00\x1f"),
        NalUnit(start_code=SC4, header=b"\x65", payload=payload),
    ]


def _ep(units, randints, choice_idxs):
    mut = NalMutator()
    mut._rng = ScriptedRng(randints=randints, choice_idxs=choice_idxs)
    return mut._mutate_emulation(units, 1 << 16)


class TestEmulationPrevention:
    def test_insert(self):
        units = _units(b"\x11\x22\x33\x44")
        off, follow_idx = 2, 3
        _ep(units, [INSERT, off], [1, follow_idx])
        ep = b"\x00\x00\x03" + bytes([EP_FOLLOW[follow_idx]])
        assert units[1].payload == b"\x11\x22" + ep + b"\x33\x44"

    def test_strip_exposes_start_code(self):
        units = _units(b"\x11\x00\x00\x03\x01\x22")
        _ep(units, [STRIP], [1, 0])
        assert units[1].payload == b"\x11\x00\x00\x01\x22"

    def test_strip_without_sites_inserts(self):
        # No 00 00 03 to strip: falls back to an insert at the drawn offset.
        units = _units(b"\x11\x22")
        _ep(units, [STRIP, 0], [1, 0])
        assert units[1].payload == b"\x00\x00\x03" + bytes([EP_FOLLOW[0]]) + b"\x11\x22"

    def test_dispatched_from_mutate(self):
        data = serialize_nal_units(_units(b"\x11\x22"))
        rng = ScriptedRng(randints=[EP_OP, INSERT, 2], choice_idxs=[1, 0])
        out = NalMutator().mutate(data, max_len=1 << 16, rng=rng)
        assert out.endswith(b"\x11\x22\x00\x00\x03" + bytes([EP_FOLLOW[0]]))

    def test_no_payload_unchanged(self):
        # Falsification: units without payload bytes are not candidates.
        units = [NalUnit(start_code=SC4, header=b"\x67", payload=b"")]
        _ep(units, [], [])
        assert units[0].payload == b""


def _stsd(config_tag: bytes, config: bytes) -> Box:
    entry = b"avc1" + bytes(78) + struct.pack(">I", 8 + len(config)) + config_tag + config
    body = b"\x00\x00\x00\x00" + struct.pack(">I", 1) + struct.pack(">I", 4 + len(entry)) + entry
    return Box(box_type=b"stsd", size_orig=8 + len(body), data=body)


def _avcc(length_size: int) -> bytes:
    return bytes([1, 0x42, 0, 0x1F, 0xFC | (length_size - 1)]) + bytes(4)


def _mdat(nals: list[bytes], width: int = 4) -> Box:
    body = b"".join(len(n).to_bytes(width, "big") + n for n in nals)
    return Box(box_type=b"mdat", size_orig=8 + len(body), data=body)


def _avcc_op(boxes, choice_idxs):
    mut = IsobmffMutator()
    mut._rng = ScriptedRng(choice_idxs=choice_idxs)
    return mut._mutate_nal_length(boxes, 1 << 16)


class TestAvccLengthSize:
    def test_avcc(self):
        assert avcc_length_size([_stsd(b"avcC", _avcc(2))]) == 2

    def test_hvcc(self):
        config = bytes(21) + bytes([0xFC | 1]) + bytes(4)
        assert avcc_length_size([_stsd(b"hvcC", config)]) == 2

    def test_default(self):
        assert avcc_length_size([]) == 4

    def test_truncated_config_defaults(self):
        # Adversarial: fourcc at the very end of the stsd payload.
        stsd = Box(box_type=b"stsd", size_orig=16, data=b"\x00" * 4 + b"avcC")
        assert avcc_length_size([stsd]) == 4


class TestNalLength:
    def test_second_unit_length(self):
        mdat = _mdat([b"\x65" + bytes(9), b"\x41" + bytes(5)])
        value_idx = 4
        _avcc_op([mdat], [0, 1, value_idx])
        second = 4 + 10
        assert mdat.data[second : second + 4] == EDGE_U32[value_idx].to_bytes(4, "big")

    def test_two_byte_lengths(self):
        mdat = _mdat([b"\x65" + bytes(3)], width=2)
        _avcc_op([_stsd(b"avcC", _avcc(2)), mdat], [0, 0, 1])
        assert mdat.data[:2] == b"\x00\x01"

    def test_value_masked_to_width(self):
        # 0xFFFFFFFF into a 2-byte field becomes 0xFFFF, not an overflow.
        mdat = _mdat([b"\x65" + bytes(3)], width=2)
        _avcc_op([_stsd(b"avcC", _avcc(2)), mdat], [0, 0, 4])
        assert mdat.data[:2] == b"\xff\xff"

    def test_walk_stops_at_inconsistent_length(self):
        # Adversarial: second length overruns the box; only the first is a target.
        body = struct.pack(">I", 2) + b"\x65\x00" + struct.pack(">I", 999) + b"\x41"
        mdat = Box(box_type=b"mdat", size_orig=8 + len(body), data=body)
        rng_idxs = [0, 0, 0]
        _avcc_op([mdat], rng_idxs)
        assert mdat.data[:4] == b"\x00\x00\x00\x00"
        assert mdat.data[4:] == body[4:]

    def test_no_mdat_unchanged(self):
        # Falsification: nothing length-prefixed, nothing changes.
        boxes = [Box(box_type=b"free", size_orig=16, data=bytes(8))]
        before = serialize_boxes(boxes)
        _avcc_op(boxes, [])
        assert serialize_boxes(boxes) == before

    def test_dispatched_from_mutate(self):
        mdat = _mdat([b"\x65" + bytes(9)])
        rng = ScriptedRng(randints=[AVCC_OP], choice_idxs=[0, 0, 1])
        out = IsobmffMutator().mutate(serialize_boxes([mdat]), max_len=1 << 16, rng=rng)
        assert out[8:12] == b"\x00\x00\x00\x01"
