"""Tests for the NTFS volume-image mutator (`mutate_ntfs`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.ntfs import (
    ATTR_COMMON,
    ATTR_NONRES,
    ATTR_RESIDENT,
    BPB_FIELDS,
    INDX_FIELDS,
    REC_FIELDS,
    RUN_EDGES,
    Op,
    RunPos,
    available_ops,
    mutate_ntfs,
    sniff_ntfs,
)
from fuzzer_tool.core.rand_pool import RandPool

from .support.ntfs_build import (
    FIRST_ATTR,
    RECORD,
    SECTOR,
    USN,
    build_image,
    data_record,
    file_name_value,
    indx_record,
    mft_record,
    resident_attr,
)
from .support.scripted_rng import ScriptedRng

IMAGE = build_image()
MFT_OFF = SECTOR  # MFT LCN 1, 512-byte clusters
DATA_OFF = MFT_OFF + RECORD

# Attribute offsets inside the $MFT record, from the builder's attribute sizes.
STD_LEN = len(resident_attr(0x10, bytes(48)))
FN_LEN = len(resident_attr(0x30, file_name_value("$MFT")))
DATA_ATTR = FIRST_ATTR + STD_LEN + FN_LEN
MAPPING = DATA_ATTR + 0x40


def _rng(op: Op, data: bytes, randints=(), choices=()) -> ScriptedRng:
    """Script the op choice (its index in available_ops) then the op's draws."""
    idx = available_ops(data).index(op)
    return ScriptedRng(randints=randints, choice_idxs=[idx, *choices])


def _tails_ok(rec: bytes) -> bool:
    """Independent fixup oracle: every sector tail holds the USN."""
    usa_off, count = struct.unpack_from("<HH", rec, 4)
    usn = rec[usa_off : usa_off + 2]
    return all(rec[k * SECTOR + SECTOR - 2 : (k + 1) * SECTOR] == usn for k in range(count - 1))


def _defix(rec: bytes) -> bytes:
    """Logical record: restore each sector tail from the USA."""
    out = bytearray(rec)
    usa_off, count = struct.unpack_from("<HH", rec, 4)
    for k in range(count - 1):
        tail = k * SECTOR + SECTOR - 2
        out[tail : tail + 2] = rec[usa_off + 2 + 2 * k : usa_off + 4 + 2 * k]
    return bytes(out)


def _diff(a: bytes, b: bytes) -> list[int]:
    return [i for i in range(len(a)) if a[i] != b[i]]


class TestSniff:
    def test_boot_image(self):
        assert sniff_ntfs(IMAGE)

    def test_mft_fragment(self):
        assert sniff_ntfs(mft_record())

    def test_fragment_at_aligned_offset(self):
        assert sniff_ntfs(bytes(SECTOR) + data_record())

    def test_indx_fragment(self):
        assert sniff_ntfs(indx_record())

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"NTFS",
            bytes(1024),
            IMAGE[:510],  # no 0x55AA
            b"FILE" + bytes(60),  # USA offset 0, count 0
            b"FILE" + struct.pack("<HH", 0x30, 0x400) + bytes(1016),  # absurd count
            b"\xeb\x52\x90FAT32   " + bytes(498) + b"\x55\xaa",
        ],
    )
    def test_rejects(self, data):
        assert not sniff_ntfs(data)


class TestAvailable:
    def test_image_has_all_but_indx(self):
        assert available_ops(IMAGE) == (Op.BOOT, Op.REC_HEADER, Op.ATTR, Op.RUNLIST, Op.FIXUP)

    def test_indx_only(self):
        assert available_ops(indx_record()) == (Op.FIXUP, Op.INDX)

    def test_resident_only_record_has_no_runlist(self):
        assert Op.RUNLIST not in available_ops(data_record())


class TestBoot:
    @pytest.mark.parametrize("field", range(len(BPB_FIELDS)))
    def test_each_bpb_field(self, field):
        off, width, edges = BPB_FIELDS[field]
        value = edges[-1]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.BOOT, IMAGE, [field], [len(edges) - 1]))
        got = int.from_bytes(out[off : off + width], "little")
        assert got == value & ((1 << (8 * width)) - 1)
        assert _diff(IMAGE, out) and all(off <= i < off + width for i in _diff(IMAGE, out))

    def test_signed_clusters_per_record_edge(self):
        field = next(i for i, f in enumerate(BPB_FIELDS) if f[0] == 0x40)
        edges = BPB_FIELDS[field][2]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.BOOT, IMAGE, [field], [edges.index(0xF6)]))
        assert struct.unpack_from("<b", out, 0x40)[0] == -10  # 2^10 = 1024-byte records

    def test_equal_value_still_changes(self):
        # sectors/cluster is already 1; picking edge 1 must still change the byte.
        field = next(i for i, f in enumerate(BPB_FIELDS) if f[0] == 0x0D)
        edges = BPB_FIELDS[field][2]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.BOOT, IMAGE, [field], [edges.index(1)]))
        assert out != IMAGE and out[0x0D] != 1


class TestRecHeader:
    @pytest.mark.parametrize("field", range(len(REC_FIELDS)))
    def test_each_field_refixed(self, field):
        off, width, edges = REC_FIELDS[field]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.REC_HEADER, IMAGE, [1, field], [0]))
        rec_old = IMAGE[DATA_OFF : DATA_OFF + RECORD]
        rec_new = out[DATA_OFF : DATA_OFF + RECORD]
        changed = _diff(_defix(rec_old), _defix_with(rec_new, rec_old))
        assert changed and all(off <= i < off + width for i in changed)
        assert _tails_ok(rec_new)
        assert out[:DATA_OFF] == IMAGE[:DATA_OFF]

    def test_bytes_used_beyond_allocated(self):
        field = next(i for i, f in enumerate(REC_FIELDS) if f[0] == 0x18)
        edges = REC_FIELDS[field][2]
        out = mutate_ntfs(
            IMAGE, rng=_rng(Op.REC_HEADER, IMAGE, [0, field], [edges.index(0xFFFFFFFF)])
        )
        used, alloc = struct.unpack_from("<II", out, MFT_OFF + 0x18)
        assert used == 0xFFFFFFFF and used > alloc


def _defix_with(rec: bytes, layout: bytes) -> bytes:
    """De-fix *rec* using the USA layout of *layout* (edits may move the USA fields)."""
    hdr = bytearray(rec)
    hdr[4:8] = layout[4:8]
    out = bytearray(_defix(bytes(hdr)))
    out[4:8] = rec[4:8]
    return bytes(out)


class TestAttr:
    def test_resident_value_length_overrun(self):
        fields = ATTR_COMMON + ATTR_RESIDENT
        field = next(i for i, f in enumerate(fields) if f[0] == 0x10)
        edges = fields[field][2]
        rng = _rng(Op.ATTR, IMAGE, [1, 2, field], [edges.index(0xFFFFFFFF)])
        out = mutate_ntfs(IMAGE, rng=rng)
        attr = DATA_OFF + FIRST_ATTR + STD_LEN + len(resident_attr(0x30, file_name_value("a")))
        assert struct.unpack_from("<I", out, attr + 0x10)[0] == 0xFFFFFFFF
        assert _tails_ok(out[DATA_OFF : DATA_OFF + RECORD])

    def test_nonres_size_inversion(self):
        fields = ATTR_COMMON + ATTR_NONRES
        field = next(i for i, f in enumerate(fields) if f[0] == 0x38)  # initialized size
        edges = fields[field][2]
        rng = _rng(Op.ATTR, IMAGE, [0, 2, field], [edges.index(0xFFFFFFFFFFFFFFFF)])
        out = mutate_ntfs(IMAGE, rng=rng)
        alloc, size, init = struct.unpack_from("<QQQ", out, MFT_OFF + DATA_ATTR + 0x28)
        assert init > size and init > alloc

    def test_type_code(self):
        field = 0
        edges = ATTR_COMMON[0][2]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.ATTR, IMAGE, [0, 0, field], [len(edges) - 1]))
        assert struct.unpack_from("<I", out, MFT_OFF + FIRST_ATTR)[0] == edges[-1]

    def test_nonres_flag_flip(self):
        field = next(i for i, f in enumerate(ATTR_COMMON) if f[0] == 0x08)
        edges = ATTR_COMMON[field][2]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.ATTR, IMAGE, [0, 2, field], [edges.index(0)]))
        assert out[MFT_OFF + DATA_ATTR + 8] == 0


class TestRunlist:
    def test_header_oversized_nibbles(self):
        out = mutate_ntfs(IMAGE, rng=_rng(Op.RUNLIST, IMAGE, [0, RunPos.HEADER], [0]))
        assert out[MFT_OFF + MAPPING] == RUN_EDGES[RunPos.HEADER][0]
        assert _tails_ok(out[MFT_OFF : MFT_OFF + RECORD])

    def test_length_byte(self):
        edges = RUN_EDGES[RunPos.LENGTH]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.RUNLIST, IMAGE, [0, RunPos.LENGTH], [len(edges) - 1]))
        assert out[MFT_OFF + MAPPING + 1] == edges[-1]

    def test_negative_offset_delta(self):
        edges = RUN_EDGES[RunPos.OFFSET]
        out = mutate_ntfs(
            IMAGE, rng=_rng(Op.RUNLIST, IMAGE, [0, RunPos.OFFSET], [edges.index(0xFF)])
        )
        # header 0x11: 1 length byte, 1 offset byte -> offset at +2, now signed -1.
        assert struct.unpack_from("<b", out, MFT_OFF + MAPPING + 2)[0] == -1

    def test_missing_terminator(self):
        edges = RUN_EDGES[RunPos.TERM]
        out = mutate_ntfs(IMAGE, rng=_rng(Op.RUNLIST, IMAGE, [0, RunPos.TERM], [0]))
        assert IMAGE[MFT_OFF + MAPPING + 3] == 0
        assert out[MFT_OFF + MAPPING + 3] == edges[0] != 0


class TestFixup:
    def test_break_detected_by_oracle(self):
        out = mutate_ntfs(IMAGE, rng=_rng(Op.FIXUP, IMAGE, [1, 1]))
        rec = out[DATA_OFF : DATA_OFF + RECORD]
        assert not _tails_ok(rec)
        tail = struct.unpack_from("<H", rec, SECTOR + SECTOR - 2)[0]
        assert tail == USN ^ 0xFFFF
        assert _diff(IMAGE, out) == [DATA_OFF + 2 * SECTOR - 2, DATA_OFF + 2 * SECTOR - 1]

    # Hard Rule 46 control: the oracle passes on every untouched record.
    def test_oracle_control(self):
        for off in (MFT_OFF, DATA_OFF):
            assert _tails_ok(IMAGE[off : off + RECORD])
        assert _tails_ok(indx_record())

    # Falsification: a default edit that skipped re-fixing would leave the
    # logical tail (0xABCD) in the USA but a stale USN; the oracle catches it.
    def test_default_edit_preserves_logical_tail(self):
        out = mutate_ntfs(IMAGE, rng=_rng(Op.REC_HEADER, IMAGE, [0, 2], [1]))
        rec = out[MFT_OFF : MFT_OFF + RECORD]
        assert _tails_ok(rec)
        assert _defix(rec)[SECTOR - 2 : SECTOR] == b"\xab\xcd"

    def test_planted_bad_tail_is_rejected(self):
        bad = bytearray(mft_record())
        bad[SECTOR - 1] ^= 0x01
        assert not _tails_ok(bytes(bad))


class TestIndx:
    @pytest.mark.parametrize("field", range(len(INDX_FIELDS)))
    def test_each_field(self, field):
        data = indx_record()
        base, off, width, edges = INDX_FIELDS[field]
        out = mutate_ntfs(data, rng=_rng(Op.INDX, data, [0, field], [len(edges) - 1]))
        pos = off if base == 0 else 0x18 + struct.unpack_from("<I", data, 0x18)[0] + off
        got = int.from_bytes(out[pos : pos + width], "little")
        assert got == edges[-1] & ((1 << (8 * width)) - 1)
        assert _tails_ok(out)


class TestDecline:
    @pytest.mark.parametrize("data", [b"", b"PK\x03\x04" + bytes(600), bytes(4096)])
    def test_not_ntfs(self, data):
        assert mutate_ntfs(data, rng=RandPool(seed=1)) is None

    def test_max_len_truncates(self):
        out = mutate_ntfs(IMAGE, 100, rng=_rng(Op.BOOT, IMAGE, [0], [0]))
        assert len(out) == 100


class TestAdversarial:
    def test_random_never_raises(self):
        rng = RandPool(seed=7)
        seeds = [IMAGE, mft_record(), indx_record(), data_record()]
        for i in range(600):
            base = bytearray(seeds[i % len(seeds)])
            for _ in range(rng.randint(1, 40)):
                base[rng.randint(0, len(base) - 1)] = rng.randint(0, 255)
            cut = rng.randint(0, len(base))
            data = bytes(base[:cut]) if i % 3 == 0 else bytes(base)
            out = mutate_ntfs(data, 2048, rng=rng)
            assert out is None or (len(out) <= 2048 and out != data[:2048] or len(data) > 2048)

    def test_hostile_header_fields(self):
        rec = bytearray(mft_record())
        struct.pack_into("<HH", rec, 4, 0xFFFE, 0xFFFF)  # USA past the record
        struct.pack_into("<H", rec, 0x14, 0x3FF)  # first attribute at the very end
        rng = RandPool(seed=3)
        for _ in range(200):
            out = mutate_ntfs(bytes(rec), rng=rng)
            assert out is None or len(out) == len(rec)

    def test_real_rng_always_changes(self):
        rng = RandPool(seed=11)
        for data in (IMAGE, mft_record(), indx_record()):
            for _ in range(300):
                assert mutate_ntfs(data, rng=rng) != data
