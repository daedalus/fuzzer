"""Tests for the stsd sample-entry mutator (`mp4_stsd_mutate`)."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.mp4_path import find_boxes
from fuzzer_tool.core.mutations.mp4_stsd import (
    AUDIO_FIELDS,
    VISUAL_FIELDS,
    Op,
    count_edges,
    mutate_mp4_stsd,
    parse_stsd,
)
from fuzzer_tool.core.operator_registry import format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool

from .support.mp4_build import avcc, box, esds, mp4, stco
from .support.scripted_rng import ScriptedRng

SPS = b"\x67\x42\xc0\x1e\xed\x08\xf2"
PPS = b"\x68\xce\x0f\xc8"
FIX, KEEP = 1, 0
HDR = 8
STSD_HEAD = 8  # version/flags + entry_count
VIS_FIXED, AUD_FIXED = 78, 28
FIRST_HIT = 0
MAX_LEN = 1 << 16
HOSTILE_SEEDS = range(24)


def _entry(typ: bytes, child: bytes = b"") -> bytes:
    fixed = AUD_FIXED if typ == b"mp4a" else VIS_FIXED
    return box(typ, bytes(fixed) + child)


def _stsd(*entries: bytes) -> bytes:
    return box(b"stsd", bytes(4) + struct.pack(">I", len(entries)) + b"".join(entries))


AVC1 = _entry(b"avc1", avcc(SPS, PPS))
AVC3 = _entry(b"avc3", avcc(SPS, PPS))
MP4A = _entry(b"mp4a", esds(b"\x12\x10"))
ONE = mp4(_stsd(AVC1))
TWO = mp4(_stsd(AVC1, AVC3))
AUDIO = mp4(_stsd(MP4A))


def _payload(data: bytes) -> bytes:
    (hit,) = find_boxes(data, frozenset({b"stsd"}))
    return data[hit.start + HDR : hit.start + hit.size]


def _count(data: bytes) -> int:
    return struct.unpack_from(">I", _payload(data), 4)[0]


def _tree_ok(data: bytes, pos: int = 0, end: int | None = None) -> bool:
    """Top-level boxes tile the file; moov..stbl nest exactly."""
    end = len(data) if end is None else end
    while pos < end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < HDR or pos + size > end:
            return False
        typ = data[pos + 4 : pos + 8]
        nested = typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl")
        if nested and not _tree_ok(data, pos + HDR, pos + size):
            return False
        pos += size
    return pos == end


def _run(data: bytes, op: Op, *draws: int, **kw) -> bytes | None:
    rng = ScriptedRng(randints=[FIRST_HIT, int(op), *draws], **kw)
    return mutate_mp4_stsd(data, MAX_LEN, rng=rng)


class TestParse:
    def test_two_entries(self):
        st = parse_stsd(_payload(TWO))
        assert st.count == 2
        assert st.entries == [AVC1, AVC3]
        assert st.tail == b""

    def test_lying_count_stops_at_data(self):
        payload = bytearray(_payload(ONE))
        struct.pack_into(">I", payload, 4, 0xFFFFFFFF)
        st = parse_stsd(bytes(payload))
        assert st.entries == [AVC1]
        assert st.count == 0xFFFFFFFF

    def test_lying_entry_size_goes_to_tail(self):
        payload = bytearray(_payload(ONE))
        struct.pack_into(">I", payload, STSD_HEAD, 0x7FFFFFFF)
        st = parse_stsd(bytes(payload))
        assert st.entries == []
        assert st.tail == bytes(payload[STSD_HEAD:])


class TestCount:
    def test_edge_written_in_place(self):
        n = 1
        out = _run(ONE, Op.COUNT, choice_idxs=[count_edges(n).index(0xFFFFFFFF)])
        assert _count(out) == 0xFFFFFFFF
        assert len(out) == len(ONE)
        assert _tree_ok(out)

    def test_changes_bytes(self):
        out = _run(ONE, Op.COUNT, choice_idxs=[count_edges(1).index(0)])
        assert out != ONE


class TestField:
    def test_audio_samplerate(self):
        fld = [f for f in AUDIO_FIELDS if f.name == "samplerate"][0]
        idx = AUDIO_FIELDS.index(fld)
        out = _run(AUDIO, Op.FIELD, 0, idx, choice_idxs=[0])
        entry = parse_stsd(_payload(out)).entries[0]
        assert entry[HDR + fld.off : HDR + fld.off + fld.width] == bytes(fld.width)
        assert len(out) == len(AUDIO)

    def test_visual_width_edge(self):
        fld = [f for f in VISUAL_FIELDS if f.name == "width"][0]
        idx = VISUAL_FIELDS.index(fld)
        out = _run(ONE, Op.FIELD, 0, idx, choice_idxs=[2])
        entry = parse_stsd(_payload(out)).entries[0]
        got = struct.unpack_from(">H", entry, HDR + fld.off)[0]
        assert got != 0
        assert _tree_ok(out)


class TestEntries:
    def test_dup_fixes_count_and_ancestors(self):
        out = _run(ONE, Op.DUP, 0, FIX)
        assert _count(out) == 2
        assert parse_stsd(_payload(out)).entries == [AVC1, AVC1]
        assert len(out) == len(ONE) + len(AVC1)
        assert _tree_ok(out)

    def test_dup_shifts_chunk_offsets(self):
        data = mp4(_stsd(AVC1))
        out = _run(data, Op.DUP, 0, FIX)
        old = struct.unpack_from(">I", data, data.index(b"stco") + 12)[0]
        new = struct.unpack_from(">I", out, out.index(b"stco") + 12)[0]
        assert new == old + len(AVC1)

    def test_dup_keep_leaves_count_stale(self):
        out = _run(ONE, Op.DUP, 0, KEEP)
        assert _count(out) == 1
        assert len(parse_stsd(_payload(out)).entries) == 2
        assert _tree_ok(out)

    def test_drop_fixes_count(self):
        out = _run(TWO, Op.DROP, 0, FIX)
        assert _count(out) == 1
        assert parse_stsd(_payload(out)).entries == [AVC3]
        assert _tree_ok(out)

    def test_drop_keep_leaves_count_stale(self):
        out = _run(TWO, Op.DROP, 1, KEEP)
        assert _count(out) == 2
        assert parse_stsd(_payload(out)).entries == [AVC1]

    def test_drop_last_entry_leaves_header(self):
        out = _run(ONE, Op.DROP, 0, FIX)
        assert _payload(out) == bytes(4) + struct.pack(">I", 0)
        assert _tree_ok(out)

    def test_swap(self):
        out = _run(TWO, Op.SWAP, 0, 1)
        assert parse_stsd(_payload(out)).entries == [AVC3, AVC1]
        assert len(out) == len(TWO)

    def test_swap_declines_single_entry(self):
        assert _run(ONE, Op.SWAP, 0, 0) is None


class TestChild:
    def test_drop_child_fixes_entry_size(self):
        out = _run(ONE, Op.CHILD_DROP, 0, 0, FIX)
        entry = parse_stsd(_payload(out)).entries[0]
        assert len(entry) == HDR + VIS_FIXED
        assert struct.unpack_from(">I", entry)[0] == len(entry)
        assert _tree_ok(out)

    def test_drop_child_keep_leaves_entry_size(self):
        out = _run(ONE, Op.CHILD_DROP, 0, 0, KEEP)
        payload = _payload(out)
        assert struct.unpack_from(">I", payload, STSD_HEAD)[0] == len(AVC1)
        assert len(payload) < STSD_HEAD + len(AVC1)
        assert _tree_ok(out)

    def test_dup_child(self):
        out = _run(ONE, Op.CHILD_DUP, 0, 0, FIX)
        entry = parse_stsd(_payload(out)).entries[0]
        assert entry.count(b"avcC") == 2
        assert struct.unpack_from(">I", entry)[0] == len(entry)

    def test_add_child_lands_at_position(self):
        out = _run(ONE, Op.CHILD_ADD, 0, 3, 0, FIX, choice_idxs=[0], randbytes=[b"\xaa\xbb\xcc"])
        entry = parse_stsd(_payload(out)).entries[0]
        assert entry[HDR + VIS_FIXED + 4 : HDR + VIS_FIXED + 8] != b"avcC"
        assert b"\xaa\xbb\xcc" in entry
        assert struct.unpack_from(">I", entry)[0] == len(entry)
        assert _tree_ok(out)

    def test_declines_unknown_entry_kind(self):
        odd = mp4(_stsd(box(b"tmcd", bytes(16))))
        assert _run(odd, Op.CHILD_DROP, 0, 0, FIX) is None


class TestGate:
    def test_matches_stsd_holder(self):
        assert format_gate_matches("mp4_stsd_mutate", ONE)
        assert format_gate_matches("mp4_stsd_mutate", AUDIO)

    def test_rejects_non_mp4(self):
        assert not format_gate_matches("mp4_stsd_mutate", b"\x00" * 16)
        assert not format_gate_matches("mp4_stsd_mutate", mp4(stco([0])))


class TestAdversarial:
    def test_no_stsd_declines(self):
        rng = ScriptedRng(randints=[])
        assert mutate_mp4_stsd(mp4(stco([0])), MAX_LEN, rng=rng) is None

    def test_max_len_caps(self):
        rng = RandPool(seed=3)
        out = mutate_mp4_stsd(TWO, 40, rng=rng)
        assert out is None or len(out) <= 40

    def test_hostile_tables_never_raise(self):
        hostile = []
        for count in (0, 1, 0xFFFFFFFF):
            for size in (0, 7, 8, 0x7FFFFFFF):
                body = bytes(4) + struct.pack(">II", count, size) + b"avc1" + bytes(20)
                hostile.append(mp4(box(b"stsd", body)))
        hostile.append(mp4(box(b"stsd", b"")))
        hostile.append(mp4(box(b"stsd", bytes(4))))
        for seed in HOSTILE_SEEDS:
            rng = RandPool(seed=seed)
            for data in [*hostile, TWO, AUDIO]:
                out = mutate_mp4_stsd(data, MAX_LEN, rng=rng)
                assert out is None or len(out) <= MAX_LEN

    def test_same_seed_same_output(self):
        a = mutate_mp4_stsd(TWO, MAX_LEN, rng=RandPool(seed=9))
        b = mutate_mp4_stsd(TWO, MAX_LEN, rng=RandPool(seed=9))
        assert a == b
