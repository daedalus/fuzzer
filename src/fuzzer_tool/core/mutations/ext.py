"""ext2 / ext4 filesystem image mutators (Linux ext4 driver, e2fsprogs, libext2fs).

  byte 0      1024           2048 (1 KiB blocks: block 2)
  | boot pad | superblock  | group descriptors | bitmaps | inode table | data ...
             s_magic 0xEF53 @ +0x38            desc[g]: bitmap / table locations

  inode (128 or s_inode_size bytes)       i_block[15] (60 bytes @ +0x28)
  +0x00 mode  +0x04 size  +0x1A links     ext2: 12 direct, ind, dind, tind
  +0x1C blocks +0x20 flags +0x6C size_hi  ext4 EXTENTS_FL: eh(12) + 4 x extent(12)
  +0x7C csum_lo  +0x82 csum_hi            eh: magic F30A, entries, max, depth
  dir block: {inode u32, rec_len u16, name_len u8, file_type u8, name}...
             [ext4 metadata_csum: 12-byte tail {0, 12, 0, 0xDE, crc32c}]

Why each edit matters (bug class it aims at):

* geometry (log_block_size shifts, per-group counts, inode_size < 128 or not a
  power of two, first_data_block): division by zero, shift overflow, wrong
  table strides -> out-of-bounds reads in group / inode lookups.
* feature bits (known and unknown): mount gating and dispatch to code paths
  (extents, 64bit descriptors, flex_bg, inline data) the image never set up.
* descriptor locations (bitmap / inode table at 0, the GDT itself, past the
  end): metadata overlap and out-of-range block reads.
* inodes (mode type, size vs blocks, links 0, block pointers to self or
  metadata, flags): type confusion, indirect-block loops, size overflow.
* dirents (rec_len 0 / unaligned / past block end, name_len > rec_len,
  inode 0 / out of range): the classic dir-walk overrun and infinite loop.
* extent trees (bad magic, entries > max, depth edges, overlapping or
  unsorted extents, uninit lengths, index nodes pointing to themselves).
* htree dx_root fields and inline data: hash / level / count validation.

metadata_csum: when ro_compat METADATA_CSUM is set (after the edit), the
superblock, touched group descriptor, inode and dir-block tail checksums are
recomputed (formulas as in ext4_superblock_csum / ext4_group_desc_csum /
ext4_inode_csum / ext4_dirblock_csum) so the edit reaches the parser behind
the check; GDT_CSUM (crc16) descriptors are repaired too. ``Op.BREAK_CSUM``
repairs then corrupts exactly one checksum to exercise the reject path.
Edits are in place and length-preserving.
"""

from __future__ import annotations

import struct
from enum import Enum, IntEnum
from typing import NamedTuple

from fuzzer_tool.core.berlekamp_massey import compute_checksum
from fuzzer_tool.core.rand_pool import RandPool

SB_OFF = 1024
SB_END = 2048
MAGIC_OFF = 0x38
EXT_MAGIC = 0xEF53
MIN_BLOCK = 1024
MAX_LOG_BLOCK = 6  # 64 KiB, the largest block size any driver accepts
GOOD_OLD_INODE = 128
DESC_LEN = 32
DESC64_MIN = 64
ROOT_INO = 2
MAX_GROUPS = 256
MAX_INODE_SCAN = 32
MAX_DIRENTS = 64
DIRENT_HDR = 8

# Superblock offsets.
COMPAT_OFF, INCOMPAT_OFF, RO_COMPAT_OFF = 0x5C, 0x60, 0x64
FEATURE_OFFS = (COMPAT_OFF, INCOMPAT_OFF, RO_COMPAT_OFF)
UUID_OFF = 0x68
DESC_SIZE_OFF = 0xFE
FLEX_OFF = 0x174
CSUM_SEED_OFF = 0x270
SB_CSUM_OFF = 0x3FC

COMPAT_DIR_INDEX = 0x20
INCOMPAT_EXTENTS = 0x40
INCOMPAT_64BIT = 0x80
INCOMPAT_FLEX_BG = 0x200
INCOMPAT_META_BG = 0x10
INCOMPAT_CSUM_SEED = 0x2000
INCOMPAT_INLINE_DATA = 0x8000
RO_HUGE_FILE = 0x8
RO_GDT_CSUM = 0x10
RO_DIR_NLINK = 0x20
RO_EXTRA_ISIZE = 0x40
RO_BIGALLOC = 0x200
RO_METADATA_CSUM = 0x400
EXT4_INCOMPAT = (
    INCOMPAT_EXTENTS | INCOMPAT_64BIT | INCOMPAT_FLEX_BG | INCOMPAT_CSUM_SEED | INCOMPAT_INLINE_DATA
)
EXT4_RO = (
    RO_HUGE_FILE | RO_GDT_CSUM | RO_DIR_NLINK | RO_EXTRA_ISIZE | RO_BIGALLOC | RO_METADATA_CSUM
)

# Group descriptor offsets.
GD_ITABLE_LO, GD_ITABLE_HI, GD_CSUM_OFF = 0x08, 0x28, 0x1E

# Inode offsets and flags.
I_FLAGS, I_BLOCK, I_GEN, I_CSUM_LO, I_EXTRA, I_CSUM_HI = 0x20, 0x28, 0x64, 0x7C, 0x80, 0x82
I_SIZE = 0x04
I_BLOCK_SLOTS = 15
CSUM_HI_EXTRA_END = 4  # i_extra_isize must cover i_checksum_hi
EXTENTS_FL = 0x80000
INDEX_FL = 0x1000
INLINE_DATA_FL = 0x10000000
EH_MAGIC = 0xF30A
EH_LEN = 12
EXT_LEN = 12
EXT_MAX_INODE = 4
EI_LEAF_LO, EI_LEAF_HI = 4, 8

# Dir-block checksum tail.
TAIL_LEN = 12
TAIL_FT = 0xDE

CRC32C_POLY = 0x82F63B78  # reflected Castagnoli
CRC16_POLY = 0xA001  # reflected 0x8005 (Linux crc16)
ALL_ONES32 = 0xFFFFFFFF
ALL_ONES16 = 0xFFFF
LOW16 = 0xFFFF
BREAK_MASK = 0x01


class Field(NamedTuple):
    """One fixed-width field; ``edges`` None means geometry-relative block edges."""

    off: int
    fmt: str
    edges: tuple[int, ...] | None


U32_EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
U16_EDGES = (0, 1, 0x7FFF, 0x8000, 0xFFFF)
FIRST_DATA_EDGES = (0, 1, 2, 0xFFFFFFFF)
LOG_BLOCK_EDGES = (0, 1, 2, 6, 7, 16, 31, 32, 0xFFFFFFFF)  # 1024 << v
PER_GROUP_EDGES = (0, 1, 7, 8, 8193, 65536, 0xFFFFFFFF)
REV_EDGES = (0, 1, 2, 0xFFFFFFFF)
FIRST_INO_EDGES = (0, 1, 2, 10, 0xFFFFFFFF)
INODE_SIZE_EDGES = (0, 64, 127, 129, 255, 384, 1024, 4096, 0xFFFF)
STATE_EDGES = (0, 2, 4, 0xFFFF)
ERRORS_EDGES = (0, 1, 3, 4, 0xFFFF)
GD_FLAG_EDGES = (0, 1, 2, 4, 7, 0xFFFF)  # INODE_UNINIT, BLOCK_UNINIT, ITABLE_ZEROED
# fifo, chr, dir, blk, reg, symlink, sock, invalid type, all bits.
MODE_EDGES = (0, 0x11A4, 0x2000, 0x41ED, 0x6000, 0x81A4, 0xA1FF, 0xC000, 0xF1FF, 0xFFFF)
LINK_EDGES = (0, 1, 0xFFFF)
FLAG_EDGES = (0, 0x10, EXTENTS_FL, INDEX_FL, INLINE_DATA_FL, 0x40000, 0xFFFFFFFF)
FT_EDGES = (0, 1, 2, 7, 8, TAIL_FT, 0xFF)

SB_FIELDS = (
    Field(0x00, "<I", U32_EDGES),  # s_inodes_count
    Field(0x04, "<I", U32_EDGES),  # s_blocks_count_lo
    Field(0x0C, "<I", U32_EDGES),  # s_free_blocks_count_lo
    Field(0x10, "<I", U32_EDGES),  # s_free_inodes_count
    Field(0x14, "<I", FIRST_DATA_EDGES),  # s_first_data_block
    Field(0x18, "<I", LOG_BLOCK_EDGES),  # s_log_block_size
    Field(0x20, "<I", PER_GROUP_EDGES),  # s_blocks_per_group
    Field(0x28, "<I", PER_GROUP_EDGES),  # s_inodes_per_group
    Field(0x3A, "<H", STATE_EDGES),  # s_state
    Field(0x3C, "<H", ERRORS_EDGES),  # s_errors
    Field(0x4C, "<I", REV_EDGES),  # s_rev_level
    Field(0x54, "<I", FIRST_INO_EDGES),  # s_first_ino
    Field(0x58, "<H", INODE_SIZE_EDGES),  # s_inode_size
)
GD_FIELDS = (
    Field(0x00, "<I", None),  # bg_block_bitmap_lo
    Field(0x04, "<I", None),  # bg_inode_bitmap_lo
    Field(0x08, "<I", None),  # bg_inode_table_lo
    Field(0x0C, "<H", U16_EDGES),  # bg_free_blocks_count_lo
    Field(0x0E, "<H", U16_EDGES),  # bg_free_inodes_count_lo
    Field(0x10, "<H", U16_EDGES),  # bg_used_dirs_count_lo
    Field(0x12, "<H", GD_FLAG_EDGES),  # bg_flags
    Field(0x1C, "<H", U16_EDGES),  # bg_itable_unused_lo
)
INODE_FIELDS = (
    Field(0x00, "<H", MODE_EDGES),  # i_mode
    Field(0x04, "<I", U32_EDGES),  # i_size_lo
    Field(0x6C, "<I", U32_EDGES),  # i_size_high
    Field(0x14, "<I", U32_EDGES),  # i_dtime
    Field(0x1A, "<H", LINK_EDGES),  # i_links_count
    Field(0x1C, "<I", U32_EDGES),  # i_blocks_lo
    Field(0x20, "<I", FLAG_EDGES),  # i_flags
)
DIRENT_FIELDS = ((0, "<I"), (4, "<H"), (6, "<B"), (7, "<B"))  # inode, rec_len, name_len, type

# ext4 tables.
EXT4_FEATURE_BITS = (
    (INCOMPAT_OFF, INCOMPAT_EXTENTS),
    (INCOMPAT_OFF, INCOMPAT_64BIT),
    (INCOMPAT_OFF, INCOMPAT_FLEX_BG),
    (INCOMPAT_OFF, INCOMPAT_INLINE_DATA),
    (INCOMPAT_OFF, INCOMPAT_META_BG),
    (INCOMPAT_OFF, INCOMPAT_CSUM_SEED),
    (RO_COMPAT_OFF, RO_METADATA_CSUM),
    (RO_COMPAT_OFF, RO_GDT_CSUM),
    (RO_COMPAT_OFF, RO_HUGE_FILE),
)
DESC_SIZE_EDGES = (0, 1, 16, 31, 33, 63, 64, 65, 128, 1024, 2048, 0xFFFF)
FLEX_EDGES = (0, 1, 4, 16, 31, 32, 63, 64, 255)
EH_FIELDS = (  # offsets inside i_block
    Field(0, "<H", (0, 0xF30B, 0x0AF3, 0xFFFF)),  # eh_magic
    Field(2, "<H", (0, 5, 0x7FFF, 0xFFFF)),  # eh_entries (> eh_max)
    Field(4, "<H", (0, 1, 5, 340, 0xFFFF)),  # eh_max
    Field(6, "<H", (1, 2, 5, 6, 0xFFFF)),  # eh_depth (5 is EXT4_MAX_EXTENT_DEPTH)
)
EE_BLOCK_EDGES = (0, 1, 0x7FFFFFFF, 0xFFFFFFFF)
EE_LEN_EDGES = (0, 32768, 32769, 0xFFFF)  # > 32768 = uninitialized
EE_FIELDS = (  # offsets inside one extent
    Field(0, "<I", EE_BLOCK_EDGES),  # ee_block
    Field(4, "<H", EE_LEN_EDGES),  # ee_len
    Field(6, "<H", (1, 0xFFFF)),  # ee_start_hi
    Field(8, "<I", None),  # ee_start_lo
)
INLINE_SIZE_EDGES = (0, 59, 60, 61, 160, 0xFFFF, 0xFFFFFFFF)
DX_FIELDS = (  # dx_root offsets inside the dir block
    Field(0x1C, "<B", (0, 1, 2, 3, 4, 5, 6, 0xFF)),  # hash_version
    Field(0x1D, "<B", (0, 7, 8, 9, 0xFF)),  # info_length
    Field(0x1E, "<B", (1, 2, 3, 0xFF)),  # indirect_levels
    Field(0x20, "<H", U16_EDGES),  # limit
    Field(0x22, "<H", U16_EDGES),  # count
    Field(0x24, "<I", None),  # block
)


class Op(IntEnum):
    """ext2 edit; values are the RNG draw."""

    SB_FIELD = 0
    SB_FEATURE = 1
    GD_FIELD = 2
    INODE_FIELD = 3
    INODE_BLOCK = 4
    DIRENT = 5
    BREAK_CSUM = 6


class Op4(IntEnum):
    """ext4-only edit; values continue Op's in ``mutate_ext4``'s draw."""

    FEATURE4 = 7
    DESC_SIZE = 8
    FLEX = 9
    EXT_HDR = 10
    EXT_ENTRY = 11
    EXT_ORDER = 12
    EXT_INDEX = 13
    INLINE = 14
    HTREE = 15


class CsumMode(Enum):
    """Which integrity scheme the (edited) superblock enables."""

    NONE = 0
    GDT_CRC16 = 1
    METADATA = 2


class _Geo(NamedTuple):
    bs: int
    isz: int
    ipg: int
    ninodes: int
    nblocks: int
    desc: int
    gdt_blk: int
    ngroups: int
    itable0: int


def _u16(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 2], "little")


def _u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little")


def _crc32c(data: bytes, crc: int) -> int:
    """Raw crc32c register update (Linux ``crc32c_le``: no inversion)."""
    return compute_checksum(data, CRC32C_POLY, 32, crc, 0, True, False)


def _crc16(data: bytes, crc: int) -> int:
    """Raw CRC-16/ARC register update (Linux ``crc16``)."""
    return compute_checksum(data, CRC16_POLY, 16, crc, 0, True, False)


def sniff_ext2(data: bytes) -> bool:
    """Full superblock at 1024 with s_magic 0xEF53 (any ext2/3/4 image)."""
    return len(data) >= SB_END and _u16(data, SB_OFF + MAGIC_OFF) == EXT_MAGIC


def sniff_ext4(data: bytes) -> bool:
    """ext magic plus an ext4-only incompat or ro_compat feature."""
    if not sniff_ext2(data):
        return False
    incompat = _u32(data, SB_OFF + INCOMPAT_OFF)
    ro = _u32(data, SB_OFF + RO_COMPAT_OFF)
    return bool(incompat & EXT4_INCOMPAT or ro & EXT4_RO)


def _desc_size(buf: bytes, bs: int) -> int:
    """32, or s_desc_size when 64bit and it is a sane size (>= 64, <= block)."""
    if not _u32(buf, SB_OFF + INCOMPAT_OFF) & INCOMPAT_64BIT:
        return DESC_LEN
    ds = _u16(buf, SB_OFF + DESC_SIZE_OFF)
    return ds if DESC64_MIN <= ds <= bs else DESC_LEN


def _geo(buf: bytes) -> _Geo | None:
    """Bounded geometry of group 0 onward; None when hostile or truncated."""
    if len(buf) < SB_END:
        return None
    log = _u32(buf, SB_OFF + 0x18)
    if log > MAX_LOG_BLOCK:
        return None
    bs = MIN_BLOCK << log

    isz = GOOD_OLD_INODE if _u32(buf, SB_OFF + 0x4C) == 0 else _u16(buf, SB_OFF + 0x58)
    ipg = _u32(buf, SB_OFF + 0x28)
    if not GOOD_OLD_INODE <= isz <= bs or ipg == 0:
        return None

    desc = _desc_size(buf, bs)
    gdt_blk = SB_OFF // bs + 1
    avail = (len(buf) - gdt_blk * bs) // desc
    if avail < 1:
        return None

    nblocks = _u32(buf, SB_OFF + 0x04)
    fdb = _u32(buf, SB_OFF + 0x14)
    bpg = _u32(buf, SB_OFF + 0x20)
    groups = -(-(nblocks - fdb) // bpg) if bpg and nblocks > fdb else 1
    ngroups = max(1, min(groups, avail, MAX_GROUPS))
    gd = gdt_blk * bs
    itable0 = _u32(buf, gd + GD_ITABLE_LO)
    if desc >= DESC64_MIN:
        itable0 |= _u32(buf, gd + GD_ITABLE_HI) << 32
    ninodes = _u32(buf, SB_OFF)
    return _Geo(bs, isz, ipg, ninodes, nblocks, desc, gdt_blk, ngroups, itable0)


def _csum_mode(buf: bytes) -> CsumMode:
    ro = _u32(buf, SB_OFF + RO_COMPAT_OFF)
    if ro & RO_METADATA_CSUM:
        return CsumMode.METADATA
    return CsumMode.GDT_CRC16 if ro & RO_GDT_CSUM else CsumMode.NONE


def _seed(buf: bytes) -> int:
    """s_checksum_seed with INCOMPAT_CSUM_SEED, else crc32c(~0, s_uuid)."""
    if _u32(buf, SB_OFF + INCOMPAT_OFF) & INCOMPAT_CSUM_SEED:
        return _u32(buf, SB_OFF + CSUM_SEED_OFF)
    return _crc32c(buf[SB_OFF + UUID_OFF : SB_OFF + UUID_OFF + 16], ALL_ONES32)


def _block_edges(geo: _Geo, self_blk: int) -> tuple[int, ...]:
    """0, 1, GDT, inode table, self, last, one-past-end, max (u32-masked)."""
    raw = (0, 1, geo.gdt_blk, geo.itable0, self_blk, geo.nblocks - 1, geo.nblocks, ALL_ONES32)
    return tuple(v & ALL_ONES32 for v in raw)


class _Mut:
    """One edit on a copy of the image, then checksum repair."""

    def __init__(self, data: bytes, rng: RandPool) -> None:
        self.src = data
        self.buf = bytearray(data)
        self.rng = rng
        self.geo = _geo(data)
        self.last: int | None = None  # offset of the last field written
        self.touch_gd: int | None = None
        self.touch_inos: list[int] = []
        self.touch_dir: int | None = None
        self.broken: int | None = None

    # ---- raw access ----

    def put(self, off: int, fmt: str, value: int) -> None:
        """Write *value* masked to the field width; skip out-of-range offsets."""
        size = struct.calcsize(fmt)
        if off < 0 or off + size > len(self.buf):
            return
        struct.pack_into(fmt, self.buf, off, value & ((1 << (8 * size)) - 1))
        self.last = off

    def put_raw(self, off: int, blob: bytes) -> None:
        if off < 0 or off + len(blob) > len(self.buf):
            return
        self.buf[off : off + len(blob)] = blob
        self.last = off

    # ---- locating structures ----

    def ino_off(self, geo: _Geo, ino: int) -> int | None:
        """Byte offset of inode *ino*, None when out of the image."""
        group, idx = divmod(ino - 1, geo.ipg)
        if ino < 1 or group >= geo.ngroups:
            return None
        gd = geo.gdt_blk * geo.bs + group * geo.desc
        table = _u32(self.buf, gd + GD_ITABLE_LO)
        if geo.desc >= DESC64_MIN:
            table |= _u32(self.buf, gd + GD_ITABLE_HI) << 32
        off = table * geo.bs + idx * geo.isz
        return off if off + geo.isz <= len(self.buf) else None

    def used_inodes(self, flag: int) -> list[tuple[int, int]]:
        """(ino, offset) of in-use group-0 inodes whose i_flags has *flag* (0 = any)."""
        geo = self.geo
        if geo is None:
            return []
        out = []
        for ino in range(1, min(geo.ipg, geo.ninodes, MAX_INODE_SCAN) + 1):
            off = self.ino_off(geo, ino)
            if off is None:
                continue
            if not (_u16(self.buf, off) or ino == ROOT_INO):
                continue
            if flag and not _u32(self.buf, off + I_FLAGS) & flag:
                continue
            out.append((ino, off))
        return out

    def pick_inode(self, flag: int = 0) -> tuple[int, int] | None:
        """Draw: randint over the matching inodes."""
        inodes = self.used_inodes(flag)
        if not inodes:
            return None
        ino, off = inodes[self.rng.randint(0, len(inodes) - 1)]
        self.touch_inos.append(ino)
        return ino, off

    def root_dir(self) -> tuple[int, int] | None:
        """(root inode offset, first dir block number) when both are in range."""
        geo = self.geo
        if geo is None:
            return None
        off = self.ino_off(geo, ROOT_INO)
        if off is None:
            return None
        ib = off + I_BLOCK
        blk = _u32(self.buf, ib)
        if _u32(self.buf, off + I_FLAGS) & EXTENTS_FL:
            if (
                _u16(self.buf, ib) != EH_MAGIC
                or _u16(self.buf, ib + 6)
                or not _u16(self.buf, ib + 2)
            ):
                return None
            blk = _u32(self.buf, ib + EH_LEN + 8) | _u16(self.buf, ib + EH_LEN + 6) << 32
        if blk == 0 or (blk + 1) * geo.bs > len(self.buf):
            return None
        return off, blk

    def dirents(self, base: int, bs: int) -> list[int]:
        """Entry offsets of a linear dir block, bounded by MAX_DIRENTS."""
        out = []
        pos = 0
        while pos + DIRENT_HDR <= bs and len(out) < MAX_DIRENTS:
            out.append(pos)
            rec = _u16(self.buf, base + pos + 4)
            if rec < DIRENT_HDR:
                break
            pos += rec
        return out

    # ---- ext2 ops ----

    def op_sb_field(self) -> bool:
        """Draws: field index, edge."""
        f = SB_FIELDS[self.rng.randint(0, len(SB_FIELDS) - 1)]
        self.put(SB_OFF + f.off, f.fmt, self.rng.choice(f.edges))
        return True

    def op_sb_feature(self) -> bool:
        """Draws: word (compat/incompat/ro_compat), bit 0..31 to flip."""
        off = SB_OFF + FEATURE_OFFS[self.rng.randint(0, len(FEATURE_OFFS) - 1)]
        bit = self.rng.randint(0, 31)
        self.put(off, "<I", _u32(self.buf, off) ^ (1 << bit))
        return True

    def op_gd_field(self) -> bool:
        """Draws: group, field, edge (block fields: self = GDT block)."""
        geo = self.geo
        if geo is None:
            return False
        group = self.rng.randint(0, geo.ngroups - 1)
        f = GD_FIELDS[self.rng.randint(0, len(GD_FIELDS) - 1)]
        edges = f.edges or _block_edges(geo, geo.gdt_blk)
        self.put(geo.gdt_blk * geo.bs + group * geo.desc + f.off, f.fmt, self.rng.choice(edges))
        self.touch_gd = group
        return True

    def op_inode_field(self) -> bool:
        """Draws: inode, field, edge."""
        picked = self.pick_inode()
        if picked is None:
            return False
        f = INODE_FIELDS[self.rng.randint(0, len(INODE_FIELDS) - 1)]
        self.put(picked[1] + f.off, f.fmt, self.rng.choice(f.edges))
        return True

    def op_inode_block(self) -> bool:
        """Draws: inode, i_block slot 0..14, block edge (self = its table block)."""
        picked = self.pick_inode()
        if picked is None or self.geo is None:
            return False
        off = picked[1]
        slot = self.rng.randint(0, I_BLOCK_SLOTS - 1)
        edges = _block_edges(self.geo, off // self.geo.bs)
        self.put(off + I_BLOCK + 4 * slot, "<I", self.rng.choice(edges))
        return True

    @staticmethod
    def dirent_edges(geo: _Geo, field: int, pos: int, rec: int) -> tuple[int, ...]:
        """Edges for inode / rec_len / name_len / file_type of the entry at *pos*."""
        if field == 0:
            return (0, 1, geo.ninodes, geo.ninodes + 1, ALL_ONES32)
        if field == 1:  # zero, short, unaligned, minimal, overrun, to end, past end
            return (0, 4, 9, 12, rec + 4, geo.bs - pos, geo.bs - pos + 4, LOW16)
        if field == 2:  # name longer than the record holds
            return (0, 1, rec - DIRENT_HDR + 1, 0xFF)
        return FT_EDGES

    def op_dirent(self) -> bool:
        """Draws: root-dir entry, field (inode/rec_len/name_len/type), edge."""
        root = self.root_dir()
        if root is None or self.geo is None:
            return False
        base = root[1] * self.geo.bs
        ents = self.dirents(base, self.geo.bs)
        pos = ents[self.rng.randint(0, len(ents) - 1)]
        field = self.rng.randint(0, len(DIRENT_FIELDS) - 1)
        rec = _u16(self.buf, base + pos + 4)
        rel, fmt = DIRENT_FIELDS[field]
        self.put(
            base + pos + rel, fmt, self.rng.choice(self.dirent_edges(self.geo, field, pos, rec))
        )
        self.touch_dir = root[1]
        return True

    def op_break_csum(self) -> bool:
        """Draw: which checksum (superblock / group 0 / root inode) to corrupt."""
        if _csum_mode(self.buf) is not CsumMode.METADATA:
            return False
        kind = self.rng.randint(0, 2)
        if kind == 0:
            self.broken = SB_OFF + SB_CSUM_OFF
            return True
        geo = self.geo
        if geo is None:
            return False
        if kind == 1:
            self.touch_gd = 0
            self.broken = geo.gdt_blk * geo.bs + GD_CSUM_OFF
            return True
        off = self.ino_off(geo, ROOT_INO)
        if off is None:
            return False
        self.touch_inos.append(ROOT_INO)
        self.broken = off + I_CSUM_LO
        return True

    # ---- ext4 ops ----

    def op_feature4(self) -> bool:
        """Draw: (word, bit) from EXT4_FEATURE_BITS to flip."""
        word, bit = self.rng.choice(EXT4_FEATURE_BITS)
        self.put(SB_OFF + word, "<I", _u32(self.buf, SB_OFF + word) ^ bit)
        self.touch_gd = 0  # 64BIT / csum flips change how group 0 is summed
        return True

    def op_desc_size(self) -> bool:
        """Draw: s_desc_size edge."""
        self.put(SB_OFF + DESC_SIZE_OFF, "<H", self.rng.choice(DESC_SIZE_EDGES))
        self.touch_gd = 0  # group 0 is re-summed over the new length
        return True

    def op_flex(self) -> bool:
        """Draw: s_log_groups_per_flex edge."""
        self.put(SB_OFF + FLEX_OFF, "<B", self.rng.choice(FLEX_EDGES))
        return True

    def op_ext_hdr(self) -> bool:
        """Draws: extent inode, header field, edge."""
        picked = self.pick_inode(EXTENTS_FL)
        if picked is None:
            return False
        f = EH_FIELDS[self.rng.randint(0, len(EH_FIELDS) - 1)]
        self.put(picked[1] + I_BLOCK + f.off, f.fmt, self.rng.choice(f.edges))
        return True

    def op_ext_entry(self) -> bool:
        """Draws: extent inode, entry (< min(eh_entries, 4)), field, edge."""
        picked = self.pick_inode(EXTENTS_FL)
        if picked is None or self.geo is None:
            return False
        ib = picked[1] + I_BLOCK
        n = min(max(_u16(self.buf, ib + 2), 1), EXT_MAX_INODE)
        entry = ib + EH_LEN + EXT_LEN * self.rng.randint(0, n - 1)
        f = EE_FIELDS[self.rng.randint(0, len(EE_FIELDS) - 1)]
        edges = f.edges or _block_edges(self.geo, picked[1] // self.geo.bs)
        self.put(entry + f.off, f.fmt, self.rng.choice(edges))
        return True

    def op_ext_order(self) -> bool:
        """Draw: extent inode. Swap extents 0/1 (unsorted) or duplicate 0 (overlap)."""
        picked = self.pick_inode(EXTENTS_FL)
        if picked is None:
            return False
        ib = picked[1] + I_BLOCK
        first, second = ib + EH_LEN, ib + EH_LEN + EXT_LEN
        a = bytes(self.buf[first:second])
        b = bytes(self.buf[second : second + EXT_LEN])
        if _u16(self.buf, ib + 2) >= 2:
            self.put_raw(second, a)
            self.put_raw(first, b)
            return True
        self.put_raw(second, a)
        self.put(ib + 2, "<H", 2)
        return True

    def op_ext_index(self) -> bool:
        """Draws: extent inode, leaf block edge. Depth 1, index 0 -> edge (self = own block)."""
        picked = self.pick_inode(EXTENTS_FL)
        if picked is None or self.geo is None:
            return False
        ib = picked[1] + I_BLOCK
        edges = _block_edges(self.geo, picked[1] // self.geo.bs)
        self.put(ib + 6, "<H", 1)
        self.put(ib + EH_LEN + EI_LEAF_HI, "<H", 0)
        self.put(ib + EH_LEN + EI_LEAF_LO, "<I", self.rng.choice(edges))
        return True

    def op_inline(self) -> bool:
        """Draws: inode, i_size edge. Sets INLINE_DATA_FL and the incompat bit."""
        picked = self.pick_inode()
        if picked is None:
            return False
        off = picked[1]
        size = self.rng.choice(INLINE_SIZE_EDGES)
        incompat = SB_OFF + INCOMPAT_OFF
        self.put(incompat, "<I", _u32(self.buf, incompat) | INCOMPAT_INLINE_DATA)
        self.put(off + I_FLAGS, "<I", _u32(self.buf, off + I_FLAGS) | INLINE_DATA_FL)
        self.put(off + I_SIZE, "<I", size)
        return True

    def op_htree(self) -> bool:
        """Draws: dx_root field, edge. Marks root INDEX_FL and sets compat DIR_INDEX."""
        root = self.root_dir()
        if root is None or self.geo is None:
            return False
        off, blk = root
        f = DX_FIELDS[self.rng.randint(0, len(DX_FIELDS) - 1)]
        value = self.rng.choice(f.edges or _block_edges(self.geo, blk))
        compat = SB_OFF + COMPAT_OFF
        self.put(compat, "<I", _u32(self.buf, compat) | COMPAT_DIR_INDEX)
        self.put(off + I_FLAGS, "<I", _u32(self.buf, off + I_FLAGS) | INDEX_FL)
        self.put(blk * self.geo.bs + f.off, f.fmt, value)
        self.touch_inos.append(ROOT_INO)
        self.touch_dir = blk
        return True

    # ---- checksum repair ----

    def fix_gd(self, geo: _Geo, mode: CsumMode, seed: int) -> None:
        group = self.touch_gd
        if group is None or group >= geo.ngroups:
            return
        off = geo.gdt_blk * geo.bs + group * geo.desc
        raw = bytearray(self.buf[off : off + geo.desc])
        raw[GD_CSUM_OFF : GD_CSUM_OFF + 2] = bytes(2)
        le_group = struct.pack("<I", group)
        if mode is CsumMode.METADATA:
            crc = _crc32c(bytes(raw), _crc32c(le_group, seed)) & LOW16
        else:  # GDT_CSUM covers bytes past the checksum only for 64-byte descriptors
            uuid = self.buf[SB_OFF + UUID_OFF : SB_OFF + UUID_OFF + 16]
            crc = _crc16(le_group, _crc16(bytes(uuid), ALL_ONES16))
            crc = _crc16(bytes(raw[:GD_CSUM_OFF]), crc)
            crc = _crc16(bytes(raw[GD_CSUM_OFF + 2 :]), crc) if geo.desc > DESC_LEN else crc
        struct.pack_into("<H", self.buf, off + GD_CSUM_OFF, crc)

    def fix_inode(self, geo: _Geo, ino: int, seed: int) -> None:
        off = self.ino_off(geo, ino)
        if off is None:
            return
        raw = bytearray(self.buf[off : off + geo.isz])
        has_hi = geo.isz > GOOD_OLD_INODE and _u16(raw, I_EXTRA) >= CSUM_HI_EXTRA_END
        raw[I_CSUM_LO : I_CSUM_LO + 2] = bytes(2)
        if has_hi:
            raw[I_CSUM_HI : I_CSUM_HI + 2] = bytes(2)
        crc = _crc32c(struct.pack("<I", ino), seed)
        crc = _crc32c(bytes(raw[I_GEN : I_GEN + 4]), crc)
        crc = _crc32c(bytes(raw), crc)
        struct.pack_into("<H", self.buf, off + I_CSUM_LO, crc & LOW16)
        if has_hi:
            struct.pack_into("<H", self.buf, off + I_CSUM_HI, crc >> 16)

    def fix_dir(self, geo: _Geo, seed: int) -> None:
        """Recompute the root dir block's linear tail checksum, if it has a tail."""
        ino_off = self.ino_off(geo, ROOT_INO)
        if self.touch_dir is None or ino_off is None:
            return
        base = self.touch_dir * geo.bs
        tail = base + geo.bs - TAIL_LEN
        if tail + TAIL_LEN > len(self.buf):
            return
        sig = struct.unpack_from("<IHBB", self.buf, tail)
        if sig != (0, TAIL_LEN, 0, TAIL_FT):
            return
        crc = _crc32c(struct.pack("<I", ROOT_INO), seed)
        crc = _crc32c(bytes(self.buf[ino_off + I_GEN : ino_off + I_GEN + 4]), crc)
        crc = _crc32c(bytes(self.buf[base:tail]), crc)
        struct.pack_into("<I", self.buf, tail + 8, crc)

    def repair(self) -> None:
        """Repair touched structures for the edited image's checksum mode."""
        mode = _csum_mode(self.buf)
        if mode is CsumMode.NONE:
            return
        geo = _geo(self.buf)
        seed = _seed(self.buf)
        if geo is not None:
            self.fix_gd(geo, mode, seed)
        if geo is not None and mode is CsumMode.METADATA:
            self.fix_dir(geo, seed)
            for ino in self.touch_inos:
                self.fix_inode(geo, ino, seed)
        if mode is CsumMode.METADATA:
            crc = _crc32c(bytes(self.buf[SB_OFF : SB_OFF + SB_CSUM_OFF]), ALL_ONES32)
            struct.pack_into("<I", self.buf, SB_OFF + SB_CSUM_OFF, crc)

    def run(self, op: int, max_len: int) -> bytes | None:
        if not getattr(self, _HANDLERS[op])():
            return None
        if self.last is None and self.broken is None:
            return None
        if self.last is not None and self.buf == self.src:
            self.buf[self.last] ^= 1  # edge equalled the current value
        self.repair()
        if self.broken is not None:
            self.buf[self.broken] ^= BREAK_MASK
        return bytes(self.buf[:max_len])


_HANDLERS: dict[int, str] = {
    Op.SB_FIELD: "op_sb_field",
    Op.SB_FEATURE: "op_sb_feature",
    Op.GD_FIELD: "op_gd_field",
    Op.INODE_FIELD: "op_inode_field",
    Op.INODE_BLOCK: "op_inode_block",
    Op.DIRENT: "op_dirent",
    Op.BREAK_CSUM: "op_break_csum",
    Op4.FEATURE4: "op_feature4",
    Op4.DESC_SIZE: "op_desc_size",
    Op4.FLEX: "op_flex",
    Op4.EXT_HDR: "op_ext_hdr",
    Op4.EXT_ENTRY: "op_ext_entry",
    Op4.EXT_ORDER: "op_ext_order",
    Op4.EXT_INDEX: "op_ext_index",
    Op4.INLINE: "op_inline",
    Op4.HTREE: "op_htree",
}


def mutate_ext2(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One ext2-level edit (any ext image); None if declined.

    Draw order: op = randint(0, top), then the op's draws (see each
    ``op_*`` docstring: structure index, field index, then the edge value).
    ``top`` excludes BREAK_CSUM (the last op) without metadata_csum, where
    it could only decline.
    """
    if not sniff_ext2(data):
        return None
    top = len(Op) - 1 if _csum_mode(data) is CsumMode.METADATA else Op.BREAK_CSUM - 1
    op = Op(rng.randint(0, top))
    return _Mut(data, rng).run(op, max_len)


def mutate_ext4(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One ext2- or ext4-level edit on an ext4 image; None if declined.

    Draw order: v = randint(0, max(Op4)); v < len(Op) selects an Op, else an
    Op4; then the op's own draws as in :func:`mutate_ext2`.
    """
    if not sniff_ext4(data):
        return None
    value = rng.randint(0, max(Op4))
    op: int = Op(value) if value < len(Op) else Op4(value)
    return _Mut(data, rng).run(op, max_len)
