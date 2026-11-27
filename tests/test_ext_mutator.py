"""Tests for the ext2/ext4 filesystem image mutators (`ext2_mutate`, `ext4_mutate`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.ext import (
    DESC_SIZE_EDGES,
    DX_FIELDS,
    EE_FIELDS,
    EE_LEN_EDGES,
    EH_FIELDS,
    EXT4_FEATURE_BITS,
    FLEX_EDGES,
    FT_EDGES,
    GD_FIELDS,
    INLINE_SIZE_EDGES,
    INODE_FIELDS,
    MODE_EDGES,
    SB_FIELDS,
    Op,
    Op4,
    mutate_ext2,
    mutate_ext4,
    sniff_ext2,
    sniff_ext4,
)
from fuzzer_tool.core.rand_pool import RandPool

from .support.ext_build import (
    BS,
    GDT_BLK,
    ITABLE_BLK,
    LPF_INO,
    NINODES,
    ROOT_INO,
    SB_OFF,
    SEED_ALL_ONES,
    build_ext2,
    build_ext4,
    crc32c_raw,
    csum_seed,
    csums_ok,
    dir_csum_ok,
    gd_crc16_ok,
    gd_csum_ok,
    inode_csum_ok,
    inode_off,
    root_dir_block,
    sb_csum_ok,
)
from .support.scripted_rng import ScriptedRng

E2 = build_ext2()
E4 = build_ext4()
E4_64 = build_ext4(64)
BLOCKS = 16
EXTENTS_FL = 0x80000
INLINE_DATA_FL = 0x10000000
INDEX_FL = 0x1000
INCOMPAT_INLINE_DATA = 0x8000
COMPAT_DIR_INDEX = 0x20
RO_GDT_CSUM = 0x10
IBLOCK = 0x28


def _u(img: bytes, off: int, fmt: str) -> int:
    return struct.unpack_from(fmt, img, off)[0]


def _field_idx(table, off: int) -> int:
    return next(i for i, f in enumerate(table) if f.off == off)


def _block_edges(self_blk: int) -> tuple[int, ...]:
    """Documented order: 0, 1, gdt, itable, self, last, count, max."""
    return (0, 1, GDT_BLK, ITABLE_BLK, self_blk, BLOCKS - 1, BLOCKS, 0xFFFFFFFF)


def _diff(a: bytes, b: bytes) -> list[int]:
    return [i for i in range(len(a)) if a[i] != b[i]]


def _resum_sb(img: bytearray) -> None:
    struct.pack_into(
        "<I", img, SB_OFF + 0x3FC, crc32c_raw(img[SB_OFF : SB_OFF + 0x3FC], SEED_ALL_ONES)
    )


class TestOracle:
    # Hard Rule 46 control: the oracle accepts the builder's images.
    def test_control_built_images_pass(self):
        assert csums_ok(E4)
        assert csums_ok(E4_64, desc=64)
        assert csums_ok(build_ext4())

    # Falsification of the oracle: one flipped bit per region is caught.
    @pytest.mark.parametrize(
        ("off", "check"),
        [
            (SB_OFF + 0x3FC, sb_csum_ok),
            (GDT_BLK * BS + 0x1E, lambda d: gd_csum_ok(d)),
            (inode_off(E4, ROOT_INO) + 0x7C, lambda d: inode_csum_ok(d, ROOT_INO)),
            (root_dir_block(E4) * BS + 30, lambda d: dir_csum_ok(d, root_dir_block(d), ROOT_INO)),
        ],
    )
    def test_planted_error_detected(self, off, check):
        bad = bytearray(E4)
        bad[off] ^= 1
        assert check(E4)
        assert not check(bytes(bad))


class TestSniff:
    def test_ext2(self):
        assert sniff_ext2(E2)
        assert not sniff_ext4(E2)

    def test_ext4(self):
        assert sniff_ext2(E4)
        assert sniff_ext4(E4)
        assert sniff_ext4(E4_64)

    def test_ext4_by_ro_compat_only(self):
        img = bytearray(E2)
        struct.pack_into("<I", img, SB_OFF + 0x64, 0x400)
        assert sniff_ext4(bytes(img))

    @pytest.mark.parametrize("data", [b"", b"\x53\xef" * 600, bytes(4096), E2[:2047]])
    def test_negative(self, data):
        assert not sniff_ext2(data)
        assert not sniff_ext4(data)


class TestExt2Ops:
    def test_sb_field_log_block_size(self):
        i = _field_idx(SB_FIELDS, 0x18)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.SB_FIELD, i], choice_idxs=[6]))
        assert _u(out, SB_OFF + 0x18, "<I") == SB_FIELDS[i].edges[6]
        assert _diff(E2, out) and all(SB_OFF + 0x18 <= p < SB_OFF + 0x1C for p in _diff(E2, out))

    def test_sb_field_inode_size_non_pow2(self):
        i = _field_idx(SB_FIELDS, 0x58)
        k = SB_FIELDS[i].edges.index(129)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.SB_FIELD, i], choice_idxs=[k]))
        assert _u(out, SB_OFF + 0x58, "<H") == 129

    # No-op guard: an edge equal to the current value still changes the field.
    def test_noop_edge_flips_low_bit(self):
        i = _field_idx(SB_FIELDS, 0x4C)
        k = SB_FIELDS[i].edges.index(1)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.SB_FIELD, i], choice_idxs=[k]))
        assert _u(out, SB_OFF + 0x4C, "<I") == 1 ^ 1

    def test_sb_feature_unknown_bit(self):
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.SB_FEATURE, 2, 31]))
        assert _u(out, SB_OFF + 0x64, "<I") == _u(E2, SB_OFF + 0x64, "<I") ^ (1 << 31)

    def test_gd_inode_table_overlaps_gdt(self):
        i = _field_idx(GD_FIELDS, 0x08)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.GD_FIELD, 0, i], choice_idxs=[4]))
        assert _u(out, GDT_BLK * BS + 8, "<I") == _block_edges(GDT_BLK)[4] == GDT_BLK

    def test_gd_free_count(self):
        i = _field_idx(GD_FIELDS, 0x0C)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.GD_FIELD, 0, i], choice_idxs=[4]))
        assert _u(out, GDT_BLK * BS + 0x0C, "<H") == GD_FIELDS[i].edges[4]

    def test_inode_mode_of_lost_found(self):
        k = MODE_EDGES.index(0xA1FF)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.INODE_FIELD, 1, 0], choice_idxs=[k]))
        assert _u(out, inode_off(E2, LPF_INO), "<H") == 0xA1FF

    def test_inode_links_zero(self):
        i = _field_idx(INODE_FIELDS, 0x1A)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.INODE_FIELD, 0, i], choice_idxs=[0]))
        assert _u(out, inode_off(E2, ROOT_INO) + 0x1A, "<H") == 0

    def test_inode_indirect_points_to_self(self):
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.INODE_BLOCK, 0, 12], choice_idxs=[4]))
        self_blk = inode_off(E2, ROOT_INO) // BS
        assert _u(out, inode_off(E2, ROOT_INO) + IBLOCK + 48, "<I") == self_blk

    def test_dirent_rec_len_overrun(self):
        root = root_dir_block(E2)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.DIRENT, 2, 1], choice_idxs=[4]))
        assert _u(out, root * BS + 24 + 4, "<H") == (BS - 24) + 4

    def test_dirent_name_len_past_record(self):
        root = root_dir_block(E2)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.DIRENT, 0, 2], choice_idxs=[2]))
        assert out[root * BS + 6] == 12 - 8 + 1

    def test_dirent_file_type(self):
        root = root_dir_block(E2)
        k = FT_EDGES.index(0xDE)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.DIRENT, 1, 3], choice_idxs=[k]))
        assert out[root * BS + 12 + 7] == 0xDE

    def test_dirent_inode_out_of_range(self):
        root = root_dir_block(E2)
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[Op.DIRENT, 2, 0], choice_idxs=[3]))
        assert _u(out, root * BS + 24, "<I") == NINODES + 1

    def test_break_declines_without_checksums(self):
        assert mutate_ext2(E2, rng=ScriptedRng(randints=[Op.BREAK_CSUM, 0])) is None

    def test_gdt_csum_crc16_repaired(self):
        img = bytearray(E2)
        struct.pack_into("<I", img, SB_OFF + 0x64, RO_GDT_CSUM)
        i = _field_idx(GD_FIELDS, 0x0E)
        out = mutate_ext2(
            bytes(img), rng=ScriptedRng(randints=[Op.GD_FIELD, 0, i], choice_idxs=[1])
        )
        assert _u(out, GDT_BLK * BS + 0x0E, "<H") == GD_FIELDS[i].edges[1]
        assert gd_crc16_ok(out)


class TestRepair:
    """metadata_csum: every edit lands behind valid checksums."""

    @pytest.mark.parametrize(
        "draws",
        [
            ([Op.SB_FIELD, 0], [1]),
            ([Op.SB_FEATURE, 0, 5], []),
            ([Op.GD_FIELD, 0, 1], [3]),
            ([Op.INODE_FIELD, 0, 1], [4]),
            ([Op.INODE_BLOCK, 1, 3], [6]),
            ([Op.DIRENT, 2, 1], [0]),
        ],
    )
    def test_ext2_ops_on_csum_image(self, draws):
        out = mutate_ext2(E4, rng=ScriptedRng(randints=draws[0], choice_idxs=draws[1]))
        assert out != E4
        assert csums_ok(out)

    def test_desc64_gd_repair(self):
        out = mutate_ext4(E4_64, rng=ScriptedRng(randints=[Op.GD_FIELD, 0, 0], choice_idxs=[7]))
        assert _u(out, GDT_BLK * BS, "<I") == 0xFFFFFFFF
        assert csums_ok(out, desc=64)

    # Edge: INCOMPAT_CSUM_SEED makes s_checksum_seed, not the uuid, the seed.
    def test_csum_seed_field_used(self):
        img = bytearray(E4)
        struct.pack_into("<I", img, SB_OFF + 0x270, csum_seed(E4))
        struct.pack_into("<I", img, SB_OFF + 0x60, _u(E4, SB_OFF + 0x60, "<I") | 0x2000)
        img[SB_OFF + 0x68 : SB_OFF + 0x78] = b"\xaa" * 16
        _resum_sb(img)
        assert csums_ok(bytes(img))
        out = mutate_ext2(
            bytes(img), rng=ScriptedRng(randints=[Op.INODE_FIELD, 0, 1], choice_idxs=[2])
        )
        assert csums_ok(out)

    @pytest.mark.parametrize(
        ("kind", "check"),
        [
            (0, sb_csum_ok),
            (1, gd_csum_ok),
            (2, lambda d: inode_csum_ok(d, ROOT_INO)),
        ],
    )
    def test_break_exactly_one(self, kind, check):
        out = mutate_ext2(E4, rng=ScriptedRng(randints=[Op.BREAK_CSUM, kind]))
        assert not check(out)
        others = [sb_csum_ok(out), gd_csum_ok(out), inode_csum_ok(out, ROOT_INO)]
        assert others.count(False) == 1


class TestExt4Ops:
    def test_feature_flip(self):
        k = EXT4_FEATURE_BITS.index((0x60, 0x80))
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.FEATURE4], choice_idxs=[k]))
        assert _u(out, SB_OFF + 0x60, "<I") == _u(E4, SB_OFF + 0x60, "<I") ^ 0x80
        assert sb_csum_ok(out)

    def test_metadata_csum_off_skips_repair(self):
        k = EXT4_FEATURE_BITS.index((0x64, 0x400))
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.FEATURE4], choice_idxs=[k]))
        assert _u(out, SB_OFF + 0x64, "<I") & 0x400 == 0
        assert _u(out, SB_OFF + 0x3FC, "<I") == _u(E4, SB_OFF + 0x3FC, "<I")

    def test_desc_size(self):
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.DESC_SIZE], choice_idxs=[4]))
        assert _u(out, SB_OFF + 0xFE, "<H") == DESC_SIZE_EDGES[4]
        assert sb_csum_ok(out)

    # Edge: a new descriptor size re-repairs group 0 over the new length.
    def test_desc_size_repairs_gd_with_new_length(self):
        k = DESC_SIZE_EDGES.index(128)
        out = mutate_ext4(E4_64, rng=ScriptedRng(randints=[Op4.DESC_SIZE], choice_idxs=[k]))
        assert gd_csum_ok(out, desc=128) and sb_csum_ok(out)

    def test_flex(self):
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.FLEX], choice_idxs=[5]))
        assert out[SB_OFF + 0x174] == FLEX_EDGES[5]

    def test_eh_entries_gt_max(self):
        i = _field_idx(EH_FIELDS, 2)
        k = EH_FIELDS[i].edges.index(5)
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.EXT_HDR, 0, i], choice_idxs=[k]))
        base = inode_off(E4, ROOT_INO) + IBLOCK
        assert _u(out, base + 2, "<H") == 5 > _u(out, base + 4, "<H")
        assert inode_csum_ok(out, ROOT_INO) and sb_csum_ok(out)

    def test_ee_len_uninit(self):
        i = _field_idx(EE_FIELDS, 4)
        k = EE_LEN_EDGES.index(32769)
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.EXT_ENTRY, 1, 0, i], choice_idxs=[k]))
        assert _u(out, inode_off(E4, LPF_INO) + IBLOCK + 12 + 4, "<H") == 32769
        assert inode_csum_ok(out, LPF_INO)

    def test_ee_start_out_of_range(self):
        i = _field_idx(EE_FIELDS, 8)
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.EXT_ENTRY, 0, 0, i], choice_idxs=[6]))
        self_blk = inode_off(E4, ROOT_INO) // BS
        assert _u(out, inode_off(E4, ROOT_INO) + IBLOCK + 12 + 8, "<I") == _block_edges(self_blk)[6]

    def test_order_duplicates_single_extent(self):
        base = inode_off(E4, ROOT_INO) + IBLOCK
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.EXT_ORDER, 0]))
        assert _u(out, base + 2, "<H") == 2
        assert out[base + 24 : base + 36] == E4[base + 12 : base + 24]
        assert csums_ok(out)

    def test_order_swaps_two_extents(self):
        img = bytearray(E4)
        base = inode_off(E4, ROOT_INO) + IBLOCK
        struct.pack_into("<H", img, base + 2, 2)
        struct.pack_into("<IHHI", img, base + 24, 7, 1, 0, 12)
        out = mutate_ext4(bytes(img), rng=ScriptedRng(randints=[Op4.EXT_ORDER, 0]))
        assert out[base + 12 : base + 24] == bytes(img[base + 24 : base + 36])
        assert out[base + 24 : base + 36] == bytes(img[base + 12 : base + 24])

    def test_index_points_to_self(self):
        base = inode_off(E4, ROOT_INO) + IBLOCK
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.EXT_INDEX, 0], choice_idxs=[4]))
        assert _u(out, base + 6, "<H") == 1
        assert _u(out, base + 12 + 4, "<I") == base // BS
        assert inode_csum_ok(out, ROOT_INO)

    def test_inline(self):
        off = inode_off(E4, LPF_INO)
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.INLINE, 1], choice_idxs=[2]))
        assert _u(out, off + 0x20, "<I") & INLINE_DATA_FL
        assert _u(out, off + 4, "<I") == INLINE_SIZE_EDGES[2]
        assert _u(out, SB_OFF + 0x60, "<I") & INCOMPAT_INLINE_DATA
        assert inode_csum_ok(out, LPF_INO) and sb_csum_ok(out)

    def test_htree_info_length(self):
        i = _field_idx(DX_FIELDS, 0x1D)
        k = DX_FIELDS[i].edges.index(9)
        out = mutate_ext4(E4, rng=ScriptedRng(randints=[Op4.HTREE, i], choice_idxs=[k]))
        root = root_dir_block(E4)
        assert out[root * BS + 0x1D] == 9
        assert _u(out, inode_off(E4, ROOT_INO) + 0x20, "<I") & INDEX_FL
        assert _u(out, SB_OFF + 0x5C, "<I") & COMPAT_DIR_INDEX
        assert csums_ok(out)

    def test_ext_ops_decline_without_extent_inodes(self):
        img = bytearray(E4)
        for ino in (ROOT_INO, LPF_INO):
            struct.pack_into("<I", img, inode_off(E4, ino) + 0x20, 0)
        for op in (Op4.EXT_HDR, Op4.EXT_ENTRY, Op4.EXT_ORDER, Op4.EXT_INDEX):
            rng = ScriptedRng(randints=[op, 0, 0, 0], choice_idxs=[0])
            assert mutate_ext4(bytes(img), rng=rng) is None


class TestDecline:
    def test_not_ext(self):
        assert mutate_ext2(bytes(4096), rng=ScriptedRng()) is None
        assert mutate_ext4(E2, rng=ScriptedRng()) is None

    def test_inode_ops_need_inode_table(self):
        for op in (Op.INODE_FIELD, Op.INODE_BLOCK, Op.DIRENT):
            assert (
                mutate_ext2(E2[:4096], rng=ScriptedRng(randints=[op, 0, 0], choice_idxs=[0]))
                is None
            )

    def test_dirent_needs_dir_block(self):
        cut = root_dir_block(E2) * BS
        assert (
            mutate_ext2(E2[:cut], rng=ScriptedRng(randints=[Op.DIRENT, 0, 0], choice_idxs=[0]))
            is None
        )

    def test_gd_needs_gdt_but_sb_ops_work(self):
        sb_only = E2[:2048]
        assert (
            mutate_ext2(sb_only, rng=ScriptedRng(randints=[Op.GD_FIELD, 0, 0], choice_idxs=[0]))
            is None
        )
        out = mutate_ext2(sb_only, rng=ScriptedRng(randints=[Op.SB_FEATURE, 0, 0]))
        assert out is not None and out != sb_only

    def test_max_len(self):
        out = mutate_ext2(E2, max_len=3000, rng=ScriptedRng(randints=[Op.SB_FEATURE, 0, 1]))
        assert len(out) == 3000


ALL_OPS = list(Op) + list(Op4)


class TestCoverage:
    @pytest.mark.parametrize("op", ALL_OPS)
    def test_every_op_changes_ext4(self, op):
        rng = ScriptedRng(randints=[op] + [0] * 6, choice_idxs=[0] * 4)
        out = mutate_ext4(E4, rng=rng)
        assert out is not None and out != E4 and len(out) == len(E4)

    @pytest.mark.parametrize("op", [o for o in Op if o is not Op.BREAK_CSUM])
    def test_every_ext2_op_changes_ext2(self, op):
        out = mutate_ext2(E2, rng=ScriptedRng(randints=[op] + [0] * 6, choice_idxs=[0] * 4))
        assert out is not None and out != E2


class TestAdversarial:
    # Every op on every truncation never raises and never grows the input.
    @pytest.mark.parametrize("img", [E2, E4, E4_64])
    def test_truncations(self, img):
        for cut in list(range(0, 2100, 97)) + list(range(2048, len(img) + 1, 512)):
            data = img[:cut]
            for op in ALL_OPS:
                out = mutate_ext4(
                    data, rng=ScriptedRng(randints=[op] + [0] * 6, choice_idxs=[0] * 4)
                )
                assert out is None or (out != data and len(out) == len(data))

    # Hostile superblocks/descriptors/inodes: random corruption, seeded RandPool.
    def test_hostile_images(self):
        rng = RandPool(seed=1234)
        bases = (E2, E4, E4_64)
        hot = (SB_OFF + 0x18, SB_OFF + 0x28, SB_OFF + 0x58, SB_OFF + 0xFE, GDT_BLK * BS + 8)
        for n in range(600):
            img = bytearray(bases[n % 3])
            for _ in range(rng.randint(1, 12)):
                pos = (
                    rng.choice(hot) + rng.randint(0, 3)
                    if rng.randint(0, 1)
                    else rng.randint(0, len(img) - 1)
                )
                img[pos] = rng.randint(0, 255)
            data = bytes(img[: rng.randint(2048, len(img))])
            max_len = rng.randint(1, 20000)
            for fn in (mutate_ext2, mutate_ext4):
                out = fn(data, max_len, rng=rng)
                assert out is None or (len(out) <= max_len and out != data)

    def test_garbage_with_magic(self):
        rng = RandPool(seed=99)
        for _ in range(300):
            img = bytearray(rng.randint(0, 255) for _ in range(rng.randint(2048, 9000)))
            struct.pack_into("<H", img, SB_OFF + 0x38, 0xEF53)
            out = (
                mutate_ext4(bytes(img), rng=rng)
                if sniff_ext4(bytes(img))
                else mutate_ext2(bytes(img), rng=rng)
            )
            assert out is None or (out != bytes(img) and len(out) == len(img))
