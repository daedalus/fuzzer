"""ISOBMFF sample-table value mutation (`IsobmffMutator._mutate_sample_table`)."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.isobmff import (
    EDGE_U32,
    EDGE_U64,
    Box,
    IsobmffMutator,
    parse_boxes,
    serialize_boxes,
)

from .support.scripted_rng import ScriptedRng

FULL_HEADER = b"\x00\x00\x00\x00"  # version 0, flags 0
COUNT_BRANCH, ENTRY_BRANCH = 0, 1
SAMPLE_TABLE_OP = 14


class _RecordingRng(ScriptedRng):
    """ScriptedRng that also records each randint range."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ranges: list[tuple[int, int]] = []

    def randint(self, a, b):
        self.ranges.append((a, b))
        return super().randint(a, b)


def _table(kind: bytes, body: bytes) -> Box:
    return Box(box_type=kind, size_orig=8 + len(body), data=body)


def _stbl(*tables: Box) -> list[Box]:
    stbl_size = 8 + sum(t.size_orig for t in tables)
    stbl = Box(box_type=b"stbl", size_orig=stbl_size, children=list(tables))
    return [Box(box_type=b"moov", size_orig=8 + stbl_size, children=[stbl])]


def _run(boxes: list[Box], rng: ScriptedRng) -> list[Box]:
    mut = IsobmffMutator()
    mut._rng = rng
    return mut._mutate_sample_table(boxes, 1 << 16)


def _stts(n: int) -> Box:
    entries = b"".join(struct.pack(">II", 1, 512) for _ in range(n))
    return _table(b"stts", FULL_HEADER + struct.pack(">I", n) + entries)


class TestSampleTable:
    def test_stts_entry_count(self):
        boxes = _stbl(_stts(2))
        value_idx = 4
        _run(boxes, ScriptedRng(randints=[COUNT_BRANCH], choice_idxs=[0, value_idx]))
        stts = boxes[0].children[0].children[0]
        assert stts.data[4:8] == EDGE_U32[value_idx].to_bytes(4, "big")

    def test_stco_entry_offset(self):
        offsets = struct.pack(">III", 100, 200, 300)
        stco = _table(b"stco", FULL_HEADER + struct.pack(">I", 3) + offsets)
        entry, value_idx = 2, 0
        _run(
            _stbl(stco), ScriptedRng(randints=[ENTRY_BRANCH, entry, 0], choice_idxs=[0, value_idx])
        )
        assert stco.data[8 + 4 * entry : 12 + 4 * entry] == EDGE_U32[value_idx].to_bytes(4, "big")
        assert stco.data[8:16] == offsets[:8]  # other entries untouched

    def test_co64_uses_64bit_fields(self):
        co64 = _table(b"co64", FULL_HEADER + struct.pack(">IQ", 1, 4096))
        value_idx = 3
        _run(_stbl(co64), ScriptedRng(randints=[ENTRY_BRANCH, 0, 0], choice_idxs=[0, value_idx]))
        assert co64.data[8:16] == EDGE_U64[value_idx].to_bytes(8, "big")

    def test_elst_v1_layout(self):
        # v1 entry: segment_duration u64, media_time s64, rate 4 -> field 1 at +8.
        body = b"\x01\x00\x00\x00" + struct.pack(">I", 1) + struct.pack(">QqI", 10, 0, 1 << 16)
        elst = _table(b"elst", body)
        value_idx = 1
        _run(_stbl(elst), ScriptedRng(randints=[ENTRY_BRANCH, 0, 1], choice_idxs=[0, value_idx]))
        assert elst.data[16:24] == EDGE_U64[value_idx].to_bytes(8, "big")

    def test_stsz_sample_count_offset(self):
        # stsz: version/flags, sample_size, sample_count, then sizes.
        stsz = _table(b"stsz", FULL_HEADER + struct.pack(">III", 0, 1, 77))
        value_idx = 2
        _run(_stbl(stsz), ScriptedRng(randints=[COUNT_BRANCH], choice_idxs=[0, value_idx]))
        assert stsz.data[8:12] == EDGE_U32[value_idx].to_bytes(4, "big")
        assert stsz.data[4:8] == b"\x00\x00\x00\x00"

    def test_count_only_when_no_entries(self):
        # An empty table has no entry to pick: the count is the only target.
        boxes = _stbl(_stts(0))
        value_idx = 1
        _run(boxes, ScriptedRng(choice_idxs=[0, value_idx]))
        stts = boxes[0].children[0].children[0]
        assert stts.data[4:8] == EDGE_U32[value_idx].to_bytes(4, "big")

    def test_dispatched_from_mutate(self):
        data = serialize_boxes(_stbl(_stts(1)))
        value_idx = 4
        rng = ScriptedRng(randints=[SAMPLE_TABLE_OP, COUNT_BRANCH], choice_idxs=[0, value_idx])
        out = IsobmffMutator().mutate(data, max_len=1 << 16, rng=rng)
        stts = parse_boxes(out)[0].children[0].children[0]
        assert stts.data[4:8] == EDGE_U32[value_idx].to_bytes(4, "big")


class TestSampleTableFalsification:
    def test_no_tables_unchanged(self):
        boxes = [_table(b"free", bytes(16))]
        before = serialize_boxes(boxes)
        _run(boxes, ScriptedRng())
        assert serialize_boxes(boxes) == before


class TestSampleTableAdversarial:
    def test_truncated_tables_untouched(self):
        # Too short for a count field: skipped, no raise.
        boxes = _stbl(_table(b"stts", b"\x00\x00"), _table(b"stsz", FULL_HEADER + b"\x00" * 5))
        before = serialize_boxes(boxes)
        _run(boxes, ScriptedRng(randints=[COUNT_BRANCH], choice_idxs=[0, 0]))
        assert serialize_boxes(boxes) == before

    def test_entry_index_bounded_by_bytes_not_count(self):
        # entry_count claims 1000 entries, one is present: only it can be hit.
        body = FULL_HEADER + struct.pack(">I", 1000) + struct.pack(">II", 1, 512) + b"\xaa"
        stts = _table(b"stts", body)
        rng = _RecordingRng(randints=[ENTRY_BRANCH, 0, 1], choice_idxs=[0, 0])
        _run(_stbl(stts), rng)
        assert rng.ranges[1] == (0, 0)  # entry index drawn from the one present entry
        assert stts.data[12:16] == EDGE_U32[0].to_bytes(4, "big")
        assert stts.data[16:] == b"\xaa"
