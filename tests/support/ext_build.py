"""Minimal ext2 / ext4 images for the ext mutator tests (pure Python).

Layout (1 KiB blocks, one group, 16 inodes, ``blocks`` total)::

  blk 0     1    2    3    4    5 .. 5+T-1   R      L
      boot  SB   GDT  bbm  ibm  inode table  root/  lost+found/

T = 16 * inode_size / 1024 (2 for ext2, 4 for ext4). Root (#2) and
lost+found (#11) are directories. The ext4 flavour stores both as one-extent
trees and carries metadata_csum; every checksum is computed by the bitwise
crc32c oracle below, written independently of the mutator's CRC path. Both
images pass ``e2fsck -fn`` (checked by hand; tests never run e2fsprogs).
"""

from __future__ import annotations

import struct
from enum import Enum

BS = 1024
SB_OFF = 1024
GDT_BLK = 2
BBM_BLK = 3
IBM_BLK = 4
ITABLE_BLK = 5
NINODES = 16
FIRST_INO = 11
ROOT_INO = 2
LPF_INO = 11
BLOCKS = 16
BLOCKS_PER_GROUP = 8192
EXT_MAGIC = 0xEF53

INCOMPAT_FILETYPE = 0x2
INCOMPAT_EXTENTS = 0x40
INCOMPAT_64BIT = 0x80
RO_COMPAT_METADATA_CSUM = 0x400
EXTENTS_FL = 0x80000
EH_MAGIC = 0xF30A
S_IFDIR = 0x4000
FT_DIR = 2
TAIL_LEN = 12
DIRENT_TAIL_FT = 0xDE
CSUM_TYPE_CRC32C = 1
EXTRA_ISIZE = 32
CRC32C_POLY = 0x82F63B78
SEED_ALL_ONES = 0xFFFFFFFF
CRC16_POLY = 0xA001
INCOMPAT_CSUM_SEED = 0x2000


class Flavor(Enum):
    """Image flavour: plain ext2 or ext4 with extents + metadata_csum."""

    EXT2 = 0
    EXT4 = 1


def crc32c_raw(data: bytes, crc: int) -> int:
    """Bitwise crc32c with no pre/post inversion (Linux ``crc32c_le``)."""
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (CRC32C_POLY if crc & 1 else 0)
    return crc


def csum_seed(img: bytes) -> int:
    """Filesystem checksum seed: crc32c(~0, s_uuid)."""
    return crc32c_raw(img[SB_OFF + 0x68 : SB_OFF + 0x78], SEED_ALL_ONES)


def _dirent(ino: int, rec_len: int, name: bytes) -> bytes:
    pad = (-len(name)) % 4
    return struct.pack("<IHBB", ino, rec_len, len(name), FT_DIR) + name + bytes(pad)


def _dir_block(self_ino: int, parent: int, child: tuple[int, bytes] | None, tail: int) -> bytes:
    """'.', '..' and an optional child; the last entry spans to the tail."""
    end = BS - tail
    if child is None:
        body = _dirent(self_ino, 12, b".") + _dirent(parent, end - 12, b"..")
    else:
        body = _dirent(self_ino, 12, b".") + _dirent(parent, 12, b"..")
        body += _dirent(child[0], end - 24, child[1])
    return body + bytes(end - len(body)) + bytes(tail)


def _bitmap(used: int) -> bytes:
    """Bitmap block with the *used* low bits set."""
    bits = bytearray(BS)
    for i in range(used):
        bits[i // 8] |= 1 << (i % 8)
    return bytes(bits)


def _pad_bitmap(bm: bytes, valid_bits: int) -> bytes:
    """Set every bit past *valid_bits* (e2fsck checks the padding)."""
    out = bytearray(bm)
    for i in range(valid_bits, BS * 8):
        out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _put_inode(img: bytearray, isz: int, ino: int, blk: int, links: int, flavor: Flavor) -> None:
    off = ITABLE_BLK * BS + (ino - 1) * isz
    mode = S_IFDIR | (0o755 if ino == ROOT_INO else 0o700)
    struct.pack_into("<HHI", img, off, mode, 0, BS)
    struct.pack_into("<HHI", img, off + 0x18, 0, links, BS // 512)
    if flavor is Flavor.EXT4:
        struct.pack_into("<I", img, off + 0x20, EXTENTS_FL)
        struct.pack_into("<HHHHI", img, off + 0x28, EH_MAGIC, 1, 4, 0, 0)
        struct.pack_into("<IHHI", img, off + 0x34, 0, 1, 0, blk)
        struct.pack_into("<H", img, off + 0x80, EXTRA_ISIZE)
    else:
        struct.pack_into("<I", img, off + 0x28, blk)


def _superblock(img: bytearray, isz: int, used_blocks: int, flavor: Flavor, desc: int) -> None:
    sb = SB_OFF
    free_blocks = BLOCKS - 1 - used_blocks
    struct.pack_into("<IIIIII", img, sb, NINODES, BLOCKS, 0, free_blocks, NINODES - FIRST_INO, 1)
    struct.pack_into("<IIIII", img, sb + 0x18, 0, 0, BLOCKS_PER_GROUP, BLOCKS_PER_GROUP, NINODES)
    struct.pack_into("<HhHHHH", img, sb + 0x34, 0, -1, EXT_MAGIC, 1, 1, 0)
    struct.pack_into("<I", img, sb + 0x4C, 1)
    struct.pack_into("<IH", img, sb + 0x54, FIRST_INO, isz)
    img[sb + 0x68 : sb + 0x78] = bytes(range(1, 17))

    incompat = INCOMPAT_FILETYPE
    if flavor is Flavor.EXT2:
        struct.pack_into("<I", img, sb + 0x60, incompat)
        return
    incompat |= INCOMPAT_EXTENTS | (INCOMPAT_64BIT if desc == 64 else 0)
    struct.pack_into("<II", img, sb + 0x60, incompat, RO_COMPAT_METADATA_CSUM)
    struct.pack_into("<H", img, sb + 0xFE, desc if desc == 64 else 0)
    struct.pack_into("<HH", img, sb + 0x15C, EXTRA_ISIZE, EXTRA_ISIZE)
    img[sb + 0x175] = CSUM_TYPE_CRC32C


def _group_desc(img: bytearray, used_blocks: int, desc: int) -> None:
    gd = GDT_BLK * BS
    free = BLOCKS - 1 - used_blocks
    struct.pack_into("<IIIHHH", img, gd, BBM_BLK, IBM_BLK, ITABLE_BLK, free, NINODES - 11, 2)


def _inode_csum(img: bytearray, isz: int, ino: int, seed: int) -> None:
    off = ITABLE_BLK * BS + (ino - 1) * isz
    raw = bytearray(img[off : off + isz])
    raw[0x7C:0x7E] = bytes(2)
    raw[0x82:0x84] = bytes(2)
    crc = crc32c_raw(struct.pack("<I", ino), seed)
    crc = crc32c_raw(raw[0x64:0x68], crc)
    crc = crc32c_raw(raw, crc)
    struct.pack_into("<H", img, off + 0x7C, crc & 0xFFFF)
    struct.pack_into("<H", img, off + 0x82, crc >> 16)


def _dir_tail(img: bytearray, blk: int, ino: int, seed: int) -> None:
    off = blk * BS
    tail = off + BS - TAIL_LEN
    struct.pack_into("<IHBB", img, tail, 0, TAIL_LEN, 0, DIRENT_TAIL_FT)
    crc = crc32c_raw(struct.pack("<I", ino), seed)
    crc = crc32c_raw(bytes(4), crc)  # generation 0
    crc = crc32c_raw(img[off:tail], crc)
    struct.pack_into("<I", img, tail + 8, crc)


def _checksums(img: bytearray, isz: int, desc: int, root: int, lpf: int) -> None:
    """metadata_csum: bitmaps, dir tails, inodes, group descriptor, superblock."""
    seed = csum_seed(img)
    gd = GDT_BLK * BS
    bb = crc32c_raw(img[BBM_BLK * BS : BBM_BLK * BS + BLOCKS_PER_GROUP // 8], seed)
    ib = crc32c_raw(img[IBM_BLK * BS : IBM_BLK * BS + NINODES // 8], seed)
    struct.pack_into("<HH", img, gd + 0x18, bb & 0xFFFF, ib & 0xFFFF)
    if desc == 64:
        struct.pack_into("<HH", img, gd + 0x38, bb >> 16, ib >> 16)

    _dir_tail(img, root, ROOT_INO, seed)
    _dir_tail(img, lpf, LPF_INO, seed)
    _inode_csum(img, isz, ROOT_INO, seed)
    _inode_csum(img, isz, LPF_INO, seed)

    raw = bytearray(img[gd : gd + desc])
    raw[0x1E:0x20] = bytes(2)
    crc = crc32c_raw(raw, crc32c_raw(bytes(4), seed))
    struct.pack_into("<H", img, gd + 0x1E, crc & 0xFFFF)

    struct.pack_into("<I", img, SB_OFF + 0x3FC, crc32c_raw(img[SB_OFF : SB_OFF + 0x3FC], SEED_ALL_ONES))


def _build(isz: int, flavor: Flavor, desc: int = 32) -> bytes:
    img = bytearray(BLOCKS * BS)
    tblocks = NINODES * isz // BS
    root = ITABLE_BLK + tblocks
    lpf = root + 1
    used = lpf  # blocks 1..lpf

    _superblock(img, isz, used, flavor, desc)
    _group_desc(img, used, desc)
    img[BBM_BLK * BS : (BBM_BLK + 1) * BS] = _pad_bitmap(_bitmap(used), BLOCKS - 1)
    img[IBM_BLK * BS : (IBM_BLK + 1) * BS] = _pad_bitmap(_bitmap(FIRST_INO), NINODES)

    tail = TAIL_LEN if flavor is Flavor.EXT4 else 0
    img[root * BS : (root + 1) * BS] = _dir_block(ROOT_INO, ROOT_INO, (LPF_INO, b"lost+found"), tail)
    img[lpf * BS : (lpf + 1) * BS] = _dir_block(LPF_INO, ROOT_INO, None, tail)
    _put_inode(img, isz, ROOT_INO, root, 3, flavor)
    _put_inode(img, isz, LPF_INO, lpf, 2, flavor)

    if flavor is Flavor.EXT4:
        _checksums(img, isz, desc, root, lpf)
    return bytes(img)


def build_ext2() -> bytes:
    """16 KiB ext2: 128-byte inodes, filetype, block-mapped dirs, no checksums."""
    return _build(128, Flavor.EXT2)


def build_ext4(desc: int = 32) -> bytes:
    """16 KiB ext4: 256-byte inodes, extents, metadata_csum (64bit when desc=64)."""
    return _build(256, Flavor.EXT4, desc)


def root_dir_block(img: bytes) -> int:
    """Block number holding the root directory in a built image."""
    isz = struct.unpack_from("<H", img, SB_OFF + 0x58)[0]
    return ITABLE_BLK + NINODES * isz // BS


def inode_off(img: bytes, ino: int) -> int:
    """Byte offset of inode *ino* in a built image."""
    isz = struct.unpack_from("<H", img, SB_OFF + 0x58)[0]
    return ITABLE_BLK * BS + (ino - 1) * isz


# ---- independent checksum oracle (tests compare the mutator against it) ----


def _seed_of(img: bytes) -> int:
    """csum seed: s_checksum_seed when INCOMPAT_CSUM_SEED, else crc32c(~0, uuid)."""
    incompat = struct.unpack_from("<I", img, SB_OFF + 0x60)[0]
    if incompat & INCOMPAT_CSUM_SEED:
        return struct.unpack_from("<I", img, SB_OFF + 0x270)[0]
    return csum_seed(img)


def sb_csum_ok(img: bytes) -> bool:
    stored = struct.unpack_from("<I", img, SB_OFF + 0x3FC)[0]
    return stored == crc32c_raw(img[SB_OFF : SB_OFF + 0x3FC], SEED_ALL_ONES)


def gd_csum_ok(img: bytes, desc: int = 32) -> bool:
    gd = GDT_BLK * BS
    raw = bytearray(img[gd : gd + desc])
    stored = struct.unpack_from("<H", raw, 0x1E)[0]
    raw[0x1E:0x20] = bytes(2)
    return stored == crc32c_raw(raw, crc32c_raw(bytes(4), _seed_of(img))) & 0xFFFF


def crc16_raw(data: bytes, crc: int) -> int:
    """Bitwise CRC-16/ARC register update (Linux ``crc16``), no inversion."""
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (CRC16_POLY if crc & 1 else 0)
    return crc


def gd_crc16_ok(img: bytes) -> bool:
    """GDT_CSUM descriptor check: crc16(~0, uuid, group, desc[:0x1E])."""
    gd = GDT_BLK * BS
    crc = crc16_raw(img[SB_OFF + 0x68 : SB_OFF + 0x78], 0xFFFF)
    crc = crc16_raw(bytes(4), crc)
    crc = crc16_raw(img[gd : gd + 0x1E], crc)
    return struct.unpack_from("<H", img, gd + 0x1E)[0] == crc


def inode_csum_ok(img: bytes, ino: int) -> bool:
    off = inode_off(img, ino)
    isz = struct.unpack_from("<H", img, SB_OFF + 0x58)[0]
    raw = bytearray(img[off : off + isz])
    stored = struct.unpack_from("<H", raw, 0x7C)[0] | struct.unpack_from("<H", raw, 0x82)[0] << 16
    raw[0x7C:0x7E] = bytes(2)
    raw[0x82:0x84] = bytes(2)
    crc = crc32c_raw(struct.pack("<I", ino), _seed_of(img))
    crc = crc32c_raw(raw[0x64:0x68], crc)
    return stored == crc32c_raw(raw, crc)


def dir_csum_ok(img: bytes, blk: int, ino: int) -> bool:
    off = blk * BS
    tail = off + BS - TAIL_LEN
    gen = img[inode_off(img, ino) + 0x64 : inode_off(img, ino) + 0x68]
    crc = crc32c_raw(struct.pack("<I", ino), _seed_of(img))
    crc = crc32c_raw(gen, crc)
    return struct.unpack_from("<I", img, tail + 8)[0] == crc32c_raw(img[off:tail], crc)


def csums_ok(img: bytes, desc: int = 32) -> bool:
    """Every checksum the mutator may touch on a built ext4 image."""
    root = root_dir_block(img)
    return (
        sb_csum_ok(img)
        and gd_csum_ok(img, desc)
        and inode_csum_ok(img, ROOT_INO)
        and inode_csum_ok(img, LPF_INO)
        and dir_csum_ok(img, root, ROOT_INO)
    )
