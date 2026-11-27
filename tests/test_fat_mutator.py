"""Tests for the FAT12/16/32 filesystem-image mutator (core/mutations/fat.py)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.fat import (
    ATTR_EDGES,
    BPB_FIELDS,
    CLUSTER_THRESHOLDS,
    EOC,
    LABELS,
    LFN_SEQ_EDGES,
    NAME_EDITS,
    SIZE_EDGES,
    FatType,
    Op,
    mutate_fat,
    parse_fat,
    sniff_fat,
)
from fuzzer_tool.core.rand_pool import RandPool

from .support.fat_build import DOTDOT, SHORT_NAME, Kind, build_fat
from .support.scripted_rng import ScriptedRng

F12 = build_fat(Kind.FAT12)
F32 = build_fat(Kind.FAT32)
BIG = 1 << 24  # max_len that never truncates the 2 MiB FAT16 fixture

# FAT12 fixture: 1 rsvd + 2 FAT sectors, root at sector 3, data at 4 (cluster 2).
F12_FAT = (512, 1024)
F12_ROOT = 3 * 512
F12_FILE = F12_ROOT + 32  # LFN slot is root entry 0, the 8.3 file entry 1
F12_SUB = F12_ROOT + 64
F12_DOT = 5 * 512  # cluster 3 = subdir: '.' then '..'
# FAT32 fixture: boot, FSInfo, FAT1, FAT2, cluster 2 = root.
F32_FAT = (1024, 1536)
F32_ROOT = 4 * 512
FSI = 512


# --- independent oracles ---------------------------------------------------------


def _chk(name: bytes) -> int:
    """LFN checksum: rotate right by one, add the byte (written apart from the code)."""
    s = 0
    for c in name:
        s = (s >> 1 | (s & 1) << 7) + c
        s %= 256
    return s


def _fat12(data: bytes, base: int, c: int) -> int:
    pair = int.from_bytes(data[base + c * 3 // 2 : base + c * 3 // 2 + 2], "little")
    return pair >> 4 if c % 2 else pair & 0xFFF


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _clusters(data: bytes) -> int:
    """fatgen103 CountofClusters from the BPB, independent of parse_fat."""
    bps, spc, rsvd, nfats, root_ent, tot16 = struct.unpack_from("<HBHBHH", data, 11)
    fatsz = _u16(data, 22) or _u32(data, 36)
    tot = tot16 or _u32(data, 32)
    root_secs = (root_ent * 32 + bps - 1) // bps
    return (tot - rsvd - nfats * fatsz - root_secs) // spc


def _field_idx(kind: FatType, name: str) -> int:
    return [f.name for f in BPB_FIELDS[kind]].index(name)


def _mut(data: bytes, randints=(), choices=(), max_len: int = 65536) -> bytes | None:
    return mutate_fat(data, max_len, rng=ScriptedRng(randints=randints, choice_idxs=choices))


# --- sniff / parse ---------------------------------------------------------------


class TestSniff:
    @pytest.mark.parametrize("kind", list(Kind))
    def test_fixtures(self, kind):
        assert sniff_fat(build_fat(kind))

    def test_oracle_control(self):
        # Hard Rule 46: the oracle agrees with the fixture's own checksum and type.
        assert _chk(SHORT_NAME) == F12[F12_ROOT + 13]
        assert _clusters(F12) == 4
        assert _clusters(build_fat(Kind.FAT16)) == 4085

    @pytest.mark.parametrize(
        ("off", "blob"),
        [
            (510, b"\x55\xab"),  # boot signature
            (0, b"\x00"),  # no jump
            (11, struct.pack("<H", 513)),  # bytes/sector
            (13, b"\x03"),  # sectors/cluster not a power of two
            (13, b"\x00"),
        ],
    )
    def test_rejects_bad_header(self, off, blob):
        bad = bytearray(F12)
        bad[off : off + len(blob)] = blob
        assert not sniff_fat(bytes(bad))

    def test_label_optional_when_geometry_sane(self):
        bad = bytearray(F12)
        bad[54:62] = b"NOLABEL "
        assert sniff_fat(bytes(bad))
        bad[16] = 0  # no FATs: NTFS-like boot sector
        assert not sniff_fat(bytes(bad))

    @pytest.mark.parametrize("data", [b"", b"\xeb\x3c\x90", bytes(512), F12[:511]])
    def test_short_or_empty(self, data):
        assert not sniff_fat(data)

    def test_e9_jump(self):
        alt = b"\xe9\x00\x00" + F12[3:]
        assert sniff_fat(alt)


class TestParse:
    def test_fat12_geometry(self):
        g = parse_fat(F12)
        assert g.kind is FatType.FAT12
        assert (g.fat_off, g.fat_bytes, g.root_off, g.clusters) == (512, 512, F12_ROOT, 4)
        assert g.data_off == 4 * 512

    def test_fat16_type_from_cluster_count(self):
        assert parse_fat(build_fat(Kind.FAT16)).kind is FatType.FAT16

    def test_fat32_type_from_zero_fatsz16(self):
        g = parse_fat(F32)
        assert g.kind is FatType.FAT32
        assert (g.root_clus, g.data_off, g.clusters) == (2, F32_ROOT, 4)

    def test_not_fat(self):
        assert parse_fat(bytes(4096)) is None


# --- ops -------------------------------------------------------------------------


class TestBpb:
    def test_field_edge(self):
        i = _field_idx(FatType.FAT12, "RootEntCnt")
        edges = BPB_FIELDS[FatType.FAT12][i].edges
        out = _mut(F12, [Op.BPB], [i, edges.index(0xFFFF)])
        assert _u16(out, 17) == 0xFFFF
        assert out[:17] == F12[:17] and out[19:] == F12[19:]

    def test_noop_edge_differs(self):
        # 512 is already the value: the write must still change the field.
        i = _field_idx(FatType.FAT12, "BytsPerSec")
        edges = BPB_FIELDS[FatType.FAT12][i].edges
        out = _mut(F12, [Op.BPB], [i, edges.index(512)])
        assert _u16(out, 11) == 513

    def test_fat32_root_clus(self):
        i = _field_idx(FatType.FAT32, "RootClus")
        edges = BPB_FIELDS[FatType.FAT32][i].edges
        out = _mut(F32, [Op.BPB], [i, len(edges) - 1])
        assert _u32(out, 44) == edges[-1]

    def test_fat32_has_no_fat16_only_fields(self):
        names = {f.name for f in BPB_FIELDS[FatType.FAT32]}
        assert {"RootClus", "FSInfo", "BkBootSec", "FATSz32", "ExtFlags"} <= names
        assert "RootClus" not in {f.name for f in BPB_FIELDS[FatType.FAT12]}


class TestLabelAndType:
    def test_label_fat12_at_54(self):
        out = _mut(F12, [Op.LABEL], [LABELS.index(b"FAT32   ")])
        assert out[54:62] == b"FAT32   "

    def test_label_fat32_at_82(self):
        out = _mut(F32, [Op.LABEL], [LABELS.index(b"FAT12   ")])
        assert out[82:90] == b"FAT12   "

    def test_type_confuse_tot16(self):
        out = _mut(F12, [Op.TYPE_CONFUSE], [CLUSTER_THRESHOLDS.index(4085)])
        assert _clusters(out) == 4085
        assert _u32(out, 32) == 0

    def test_type_confuse_tot32_when_too_big(self):
        f16 = build_fat(Kind.FAT16)
        out = _mut(f16, [Op.TYPE_CONFUSE], [CLUSTER_THRESHOLDS.index(65525)], BIG)
        assert _u16(out, 19) == 0
        assert _clusters(out) == 65525


class TestFatTable:
    def test_loop_synced(self):
        # candidates = allocated clusters [2, 3]; pick 3, point it at 2.
        out = _mut(F12, [Op.FAT_LOOP, 2], [1])
        for base in F12_FAT:
            assert _fat12(out, base, 3) == 2
            assert _fat12(out, base, 2) == 0xFFF  # nibble neighbour intact
            assert _fat12(out, base, 4) == 0

    def test_self_loop_noop_still_changes(self):
        out = _mut(F12, [Op.FAT_LOOP, 3], [1])
        assert _fat12(out, F12_FAT[0], 3) == 3

    def test_free_in_chain(self):
        out = _mut(F12, [Op.FAT_FREE], [0])
        assert [_fat12(out, b, 2) for b in F12_FAT] == [0, 0]

    def test_bad_fat16(self):
        f16 = build_fat(Kind.FAT16)
        out = _mut(f16, [Op.FAT_BAD], [1], BIG)
        assert [_u16(out, 512 + 16 * 512 * k + 3 * 2) for k in range(2)] == [0xFFF7] * 2

    def test_eoc_fat32(self):
        i = EOC[FatType.FAT32].index(0x0FFFFFF8)
        out = _mut(F32, [Op.FAT_EOC], [2, i])
        assert [_u32(out, b + 4 * 4) for b in F32_FAT] == [0x0FFFFFF8] * 2

    def test_range_past_last_cluster(self):
        out = _mut(F12, [Op.FAT_RANGE], [0, 2])  # edge 2 = clusters + 2
        assert _fat12(out, F12_FAT[1], 2) == 4 + 2

    def test_desync_one_copy(self):
        out = _mut(F12, [Op.FAT_DESYNC, 1, 0x123], [0])
        assert _fat12(out, F12_FAT[1], 2) == 0x123
        assert _fat12(out, F12_FAT[0], 2) == 0xFFF

    # Adversarial: the chosen copy is cut off the image -> decline, never a no-op.
    def test_desync_declines_truncated_copy(self):
        cut = F12[: F12_FAT[1] + 3]  # copy 1's entry 2 spans bytes +3..+4
        assert _mut(cut, [Op.FAT_DESYNC, 1, 0x123], [0]) is None

    def test_desync_declines_single_fat(self):
        one = bytearray(F12)
        one[16] = 1
        assert _mut(bytes(one), [Op.FAT_DESYNC, 0, 5], [0]) is None


class TestDirEntries:
    def test_cluster_parent_edge(self):
        # Short entries: file, SUB, '.', '..'; edge -1 is the parent dir cluster.
        out = _mut(F12, [Op.DIR_CLUSTER], [1, -1])
        assert _u16(out, F12_SUB + 26) == 0

    def test_dot_self_loop_noop_differs(self):
        out = _mut(F12, [Op.DIR_CLUSTER], [2, -1])
        assert _u16(out, F12_DOT + 26) == 3 ^ 1

    def test_fat32_cluster_hi_lo(self):
        out = _mut(F32, [Op.DIR_CLUSTER], [0, 5])  # edge 5 = mask 0x0FFFFFFF
        hi, lo = _u16(out, F32_ROOT + 32 + 20), _u16(out, F32_ROOT + 32 + 26)
        assert hi << 16 | lo == 0x0FFFFFFF

    def test_size(self):
        out = _mut(F12, [Op.DIR_SIZE], [0, SIZE_EDGES.index(0xFFFFFFFF)])
        assert _u32(out, F12_FILE + 28) == 0xFFFFFFFF

    def test_attr_label_dir(self):
        out = _mut(F12, [Op.DIR_ATTR], [0, ATTR_EDGES.index(0x18)])
        assert out[F12_FILE + 11] == 0x18

    def test_name_repairs_lfn_checksum(self):
        out = _mut(F12, [Op.DIR_NAME], [0, NAME_EDITS.index(DOTDOT)])
        assert out[F12_FILE : F12_FILE + 11] == DOTDOT
        assert out[F12_ROOT + 13] == _chk(DOTDOT) != _chk(SHORT_NAME)

    # Falsification: a planted wrong checksum is caught, and DIR_NAME repairs it.
    def test_planted_bad_checksum_repaired(self):
        bad = bytearray(F12)
        bad[F12_ROOT + 13] ^= 0x5A
        assert bad[F12_ROOT + 13] != _chk(SHORT_NAME)
        out = _mut(bytes(bad), [Op.DIR_NAME], [0, NAME_EDITS.index(b"\xe5")])
        assert out[F12_FILE] == 0xE5
        assert out[F12_ROOT + 13] == _chk(out[F12_FILE : F12_FILE + 11])

    def test_dotdot_entry_reachable(self):
        out = _mut(F12, [Op.DIR_SIZE], [3, SIZE_EDGES.index(1)])
        assert _u32(out, F12_DOT + 32 + 28) == 1


class TestLfn:
    def test_seq(self):
        out = _mut(F12, [Op.LFN_SEQ], [0, LFN_SEQ_EDGES.index(0x00)])
        assert out[F12_ROOT] == 0x00

    def test_break_checksum(self):
        out = _mut(F12, [Op.LFN_BREAK, 0x80], [0])
        assert out[F12_ROOT + 13] == _chk(SHORT_NAME) ^ 0x80
        assert out[F12_ROOT + 13] != _chk(out[F12_FILE : F12_FILE + 11])

    def test_fat32_root_cluster_scanned(self):
        out = _mut(F32, [Op.LFN_SEQ], [0, LFN_SEQ_EDGES.index(0x7F)])
        assert out[F32_ROOT] == 0x7F

    @pytest.mark.parametrize("op", [Op.LFN_SEQ, Op.LFN_BREAK])
    def test_decline_without_lfn(self, op):
        no_lfn = bytearray(F12)
        no_lfn[F12_ROOT + 11] = 0x20
        assert _mut(bytes(no_lfn), [op, 1], [0, 0]) is None


class TestFsinfo:
    def test_free_count(self):
        out = _mut(F32, [Op.FSINFO, 0], [0])  # FREE, edge 0 = 0
        assert _u32(out, FSI + 488) == 0

    def test_next_free(self):
        out = _mut(F32, [Op.FSINFO, 1], [-1])
        assert _u32(out, FSI + 492) == 0xFFFFFFFF

    def test_break_signature(self):
        out = _mut(F32, [Op.FSINFO, 2, 1])
        assert _u32(out, FSI + 484) != 0x61417272
        assert _u32(out, FSI) == 0x41615252

    def test_declines_on_fat12(self):
        assert _mut(F12, [Op.FSINFO, 0], [0]) is None


class TestContract:
    def test_non_fat_declined(self):
        assert mutate_fat(bytes(4096), rng=ScriptedRng()) is None

    def test_max_len(self):
        out = _mut(F12, [Op.DIR_SIZE], [0, 0], max_len=100)
        assert len(out) == 100

    @pytest.mark.parametrize("kind", [Kind.FAT12, Kind.FAT32])
    @pytest.mark.parametrize("op", list(Op))
    def test_every_op_changes_or_declines(self, kind, op):
        data = build_fat(kind)
        pool = RandPool(seed=op)
        rng = _OpFirst(op, pool)
        out = mutate_fat(data, rng=rng)
        if op is Op.FSINFO and kind is Kind.FAT12:
            assert out is None
            return
        assert out is not None and out != data and len(out) == len(data)

    # Adversarial: truncations, header garbage and hostile BPB fields never raise.
    def test_hostile_inputs_bounded(self):
        pool = RandPool(seed=1234)
        seeds = [F12, F32, build_fat(Kind.FAT16)[:8192]]
        hostile = [s[:cut] for s in seeds for cut in range(0, len(s) + 1, 97)]
        for _ in range(600):
            buf = bytearray(pool.choice(seeds))
            for _ in range(pool.randint(1, 24)):
                buf[pool.randint(11, 90)] = pool.randint(0, 255)
            for _ in range(pool.randint(0, 16)):
                buf[pool.randint(0, len(buf) - 1)] = pool.randint(0, 255)
            hostile.append(bytes(buf))
        for data in hostile:
            for max_len in (64, 4096):
                out = mutate_fat(data, max_len, rng=pool)
                if out is None:
                    continue
                assert len(out) <= max_len
                assert len(data) > max_len or out != data


class _OpFirst:
    """RandPool proxy whose first randint is the op, the rest real draws."""

    def __init__(self, op: Op, pool: RandPool) -> None:
        self._op, self._pool, self._first = op, pool, True

    def randint(self, a: int, b: int) -> int:
        if self._first:
            self._first = False
            return int(self._op)
        return self._pool.randint(a, b)

    def choice(self, seq):
        return self._pool.choice(seq)
