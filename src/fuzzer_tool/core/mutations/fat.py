"""FAT12/FAT16/FAT32 filesystem-image mutator (Linux vfat, mtools, FatFs, Windows).

  sector 0 (boot)            FAT region              root      data region
  +-------------------+  +---------+---------+  +---------+  +--------+--------+--
  | jmp OEM BPB .. 55AA|  | FAT #0  | FAT #1  |  | 12/16:  |  | clus 2 | clus 3 | ..
  +-------------------+  +---------+---------+  | RootEnt |  +--------+--------+--
   ^ RsvdSecCnt sectors    NumFATs * FATSz        | * 32 B  |   SecPerClus * BytsPerSec
                                                  +---------+   (FAT32 root = RootClus)

  BPB  11 BytsPerSec u16 | 13 SecPerClus u8 | 14 RsvdSecCnt u16 | 16 NumFATs u8
       17 RootEntCnt u16 | 19 TotSec16 u16  | 21 Media u8       | 22 FATSz16 u16
       32 TotSec32 u32   | FAT32: 36 FATSz32 | 40 ExtFlags | 44 RootClus | 48 FSInfo
       50 BkBootSec; FilSysType label at 54 (12/16) or 82 (32)
  dirent  0 name[11] | 11 attr | 20 clus_hi | 26 clus_lo | 28 size     (32 bytes)
  LFN     0 seq | 11 attr 0x0F | 13 checksum of the 8.3 name that follows

Type rule (Linux vfat): FATSz16 == 0 means FAT32; otherwise fatgen103's
cluster count, < 4085 FAT12 else FAT16. Drivers disagree here (FatFs and
Windows count clusters for FAT32 too), hence TYPE_CONFUSE.

Why each edit matters:
* BPB edge values: divide-by-zero (BytsPerSec/SecPerClus 0), FAT/root/data
  offsets past EOF, RootClus/FSInfo/BkBootSec out of range, ExtFlags mirroring.
* TYPE_CONFUSE: cluster count on a 4085/65525 threshold flips the entry width,
  so the driver walks the same FAT as 12, 16 or 32-bit entries.
* FAT entries: chain loops (infinite walk), free/bad/EOC inside a chain,
  out-of-range or reserved next-cluster numbers (OOB cluster reads). All
  NumFATs copies stay in sync; FAT_DESYNC edits one copy (mirror mismatch).
* Directory entries: first cluster (0, past the end, own dir -> loop), size
  edges vs. chain length, attr combos (label+dir, 0x0F), '.'/'..' names,
  0xE5 / 0x05 / 0x00 first bytes. DIR_NAME repairs the LFN checksum so the
  long name still binds; LFN_BREAK breaks it (orphan-LFN path).
* FSInfo: free count / next-free hints are trusted for allocation; broken
  signatures exercise the fallback.

Edits are in place and length-preserving. Draw order: op, then the op's own
draws (documented on each handler).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

SECTOR = 512
BOOT_SIG = b"\x55\xaa"
BOOT_SIG_OFF = 510
JMP_SHORT, JMP_NOP, JMP_NEAR = 0xEB, 0x90, 0xE9
VALID_BPS = frozenset((512, 1024, 2048, 4096))
VALID_SPC = frozenset(1 << i for i in range(8))
LABEL_OFF_16, LABEL_OFF_32 = 54, 82
LABELS_16 = (b"FAT12   ", b"FAT16   ", b"FAT     ")
LABEL_32 = b"FAT32   "
LABELS = (*LABELS_16, LABEL_32, b"        ")

FAT12_MAX = 4085  # fatgen103: fewer clusters is FAT12
FAT16_MAX = 65525
CLUSTER_THRESHOLDS = (FAT12_MAX - 1, FAT12_MAX, FAT16_MAX - 1, FAT16_MAX)

ENTRY = 32
ATTR_OFF, CLUS_HI_OFF, CLUS_LO_OFF, SIZE_OFF = 11, 20, 26, 28
LFN_CHK_OFF = 13
ATTR_LFN, ATTR_DIR, ATTR_MASK = 0x0F, 0x10, 0x3F
DOT_NAMES = (b".          ", b"..         ")

FSI_FREE_OFF, FSI_NEXT_OFF = 488, 492
FSI_SIG_OFFS = (0, 484, 508)  # 0x41615252, 0x61417272, 0xAA550000
SIG_FLIP = 0x01

MAX_FAT_SCAN = 1 << 16  # FAT entries examined per mutation
MAX_ENTRIES = 512  # directory entries collected
MAX_DIR_CLUSTERS = 8  # FAT32 root chain length followed
MAX_SUBDIRS = 8
MAX_LFN_SLOTS = 20  # 255 chars / 13


class FatType(IntEnum):
    """FAT variant; value is the entry width in bits."""

    FAT12 = 12
    FAT16 = 16
    FAT32 = 32


class Field(NamedTuple):
    """BPB field and its edge values."""

    name: str
    off: int
    fmt: str
    edges: tuple[int, ...]


U32_EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
_COMMON = (
    Field("BytsPerSec", 11, "<H", (0, 1, 256, 511, 512, 513, 1024, 2048, 4096, 8192, 0x8000)),
    Field("SecPerClus", 13, "B", (0, 1, 2, 3, 64, 128, 255)),
    Field("RsvdSecCnt", 14, "<H", (0, 1, 2, 0x7FFF, 0xFFFF)),
    Field("NumFATs", 16, "B", (0, 1, 2, 3, 255)),
    Field("RootEntCnt", 17, "<H", (0, 1, 15, 16, 17, 512, 0xFFF0, 0xFFFF)),
    Field("TotSec16", 19, "<H", (0, 1, 0xFFFF)),
    Field("Media", 21, "B", (0x00, 0xEF, 0xF0, 0xF8, 0xFF)),
    Field("FATSz16", 22, "<H", (0, 1, 0xFFFF)),
    Field("TotSec32", 32, "<I", U32_EDGES),
)
_FAT32_ONLY = (
    Field("FATSz32", 36, "<I", U32_EDGES),
    Field("ExtFlags", 40, "<H", (0, 0x80, 0x81, 0x8F, 0xFFFF)),
    Field("FSVer", 42, "<H", (0, 1, 0xFFFF)),
    Field(
        "RootClus",
        44,
        "<I",
        (0, 1, 2, 3, 0x0FFFFFF7, 0x0FFFFFF8, 0x0FFFFFFF, 0x10000002, 0xFFFFFFFF),
    ),
    Field("FSInfo", 48, "<H", (0, 1, 2, 6, 0xFFFF)),
    Field("BkBootSec", 50, "<H", (0, 1, 6, 0xFFFF)),
)
BPB_FIELDS = {
    FatType.FAT12: _COMMON,
    FatType.FAT16: _COMMON,
    FatType.FAT32: _COMMON + _FAT32_ONLY,
}

# FAT entry values per type.
FAT_MASK = {FatType.FAT12: 0xFFF, FatType.FAT16: 0xFFFF, FatType.FAT32: 0xFFFFFFFF}
CLUS_MASK = {FatType.FAT12: 0xFFF, FatType.FAT16: 0xFFFF, FatType.FAT32: 0x0FFFFFFF}
BAD = {FatType.FAT12: 0xFF7, FatType.FAT16: 0xFFF7, FatType.FAT32: 0x0FFFFFF7}
EOC = {
    FatType.FAT12: (0xFF8, 0xFFE, 0xFFF),
    FatType.FAT16: (0xFFF8, 0xFFFE, 0xFFFF),
    FatType.FAT32: (0x0FFFFFF8, 0x0FFFFFFF, 0xFFFFFFF8, 0xFFFFFFFF),
}
RESERVED = {  # reserved next-cluster values; FAT32 adds high-nibble garbage
    FatType.FAT12: (0xFF0, 0xFF6),
    FatType.FAT16: (0xFFF0, 0xFFF6),
    FatType.FAT32: (0x0FFFFFF0, 0x0FFFFFF6, 0x10000002, 0xF0000003),
}

SIZE_EDGES = (0, 1, 511, 512, 513, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)
ATTR_EDGES = (0x00, 0x08, 0x0F, 0x10, 0x18, 0x1F, 0x3F, 0x40, 0x80, 0xFF)
NAME_EDITS = (  # 1 byte replaces name[0]; 11 bytes replace the name
    b"\xe5",  # deleted
    b"\x05",  # escaped 0xE5 lead byte
    b"\x00",  # end-of-directory marker mid-listing
    b" ",
    *DOT_NAMES,
    b".  .       ",
    b"\xff" * 11,
    b'A:/\\*?<>|" ',  # illegal 8.3 characters
)
LFN_SEQ_EDGES = (0x00, 0x01, 0x40, 0x41, 0x54, 0x5F, 0x7F, 0x80, 0xE5)


class Op(IntEnum):
    """FAT edit; values are the RNG draw."""

    BPB = 0
    LABEL = 1
    TYPE_CONFUSE = 2
    FAT_LOOP = 3
    FAT_FREE = 4
    FAT_BAD = 5
    FAT_EOC = 6
    FAT_RANGE = 7
    FAT_DESYNC = 8
    DIR_CLUSTER = 9
    DIR_SIZE = 10
    DIR_ATTR = 11
    DIR_NAME = 12
    LFN_SEQ = 13
    LFN_BREAK = 14
    FSINFO = 15


class FsiField(IntEnum):
    """FSInfo target; values are the RNG draw."""

    FREE = 0
    NEXT = 1
    SIG = 2


@dataclass(frozen=True, slots=True)
class Geometry:
    """Byte offsets and counts derived from the BPB."""

    kind: FatType
    bps: int
    spc: int
    nfats: int
    overhead: int  # sectors before cluster 2
    fat_off: int
    fat_bytes: int
    root_off: int
    root_bytes: int
    data_off: int
    clus_bytes: int
    clusters: int
    root_clus: int


class Entry(NamedTuple):
    """Directory entry offset and the first cluster of the dir holding it (0 = root)."""

    off: int
    parent: int


def _u16(data: bytes, off: int) -> int:
    return data[off] | data[off + 1] << 8


def _u32(data: bytes, off: int) -> int:
    return int.from_bytes(data[off : off + 4], "little")


def _has_label(data: bytes) -> bool:
    return data[LABEL_OFF_16 : LABEL_OFF_16 + 8] in LABELS_16 or (
        data[LABEL_OFF_32 : LABEL_OFF_32 + 8] == LABEL_32
    )


def sniff_fat(data: bytes) -> bool:
    """Boot signature, jump, sane BytsPerSec/SecPerClus, then a label or FAT geometry."""
    if len(data) < SECTOR or data[BOOT_SIG_OFF:SECTOR] != BOOT_SIG:
        return False
    if not ((data[0] == JMP_SHORT and data[2] == JMP_NOP) or data[0] == JMP_NEAR):
        return False
    if _u16(data, 11) not in VALID_BPS or data[13] not in VALID_SPC:
        return False
    if _has_label(data):
        return True

    # No label: demand reserved sectors, a FAT and a FAT size (rules out NTFS/exFAT).
    return _u16(data, 14) >= 1 and data[16] >= 1 and (_u16(data, 22) or _u32(data, 36)) > 0


def parse_fat(data: bytes) -> Geometry | None:
    """Geometry from the BPB, or None if not FAT. Never reads past the boot sector."""
    if not sniff_fat(data):
        return None

    bps, spc, rsvd, nfats = _u16(data, 11), data[13], _u16(data, 14), data[16]
    root_ent, fatsz16 = _u16(data, 17), _u16(data, 22)
    fatsz = fatsz16 or _u32(data, 36)
    totsec = _u16(data, 19) or _u32(data, 32)
    root_secs = (root_ent * ENTRY + bps - 1) // bps
    overhead = rsvd + nfats * fatsz + root_secs
    clusters = max(0, (totsec - overhead) // spc)

    # Linux: FATSz16 == 0 is FAT32 whatever the count; else fatgen103 counting.
    if fatsz16 == 0:
        kind = FatType.FAT32
    else:
        kind = FatType.FAT12 if clusters < FAT12_MAX else FatType.FAT16

    root_off = (rsvd + nfats * fatsz) * bps
    return Geometry(
        kind=kind,
        bps=bps,
        spc=spc,
        nfats=nfats,
        overhead=overhead,
        fat_off=rsvd * bps,
        fat_bytes=fatsz * bps,
        root_off=root_off,
        root_bytes=root_ent * ENTRY,
        data_off=root_off + root_secs * bps,
        clus_bytes=spc * bps,
        clusters=clusters,
        root_clus=_u32(data, 44) if kind is FatType.FAT32 else 0,
    )


# --- in-place writers ----------------------------------------------------------


def _put(buf: bytearray, off: int, fmt: str, val: int) -> None:
    """Write *val*; if the field already holds it, write ``val ^ 1`` (never a no-op)."""
    if struct.unpack_from(fmt, buf, off)[0] == val:
        val ^= 1
    struct.pack_into(fmt, buf, off, val)


def _entry_cap(geo: Geometry, avail: int) -> int:
    """Number of whole FAT entries in *avail* bytes."""
    if geo.kind is FatType.FAT12:
        return avail * 2 // 3
    return avail // (geo.kind // 8)


def _fat_last(buf: bytearray, geo: Geometry) -> int:
    """Highest cluster whose entry fits in FAT #0, the image and the scan cap."""
    avail = min(geo.fat_bytes, len(buf) - geo.fat_off)
    return min(geo.clusters + 1, _entry_cap(geo, max(0, avail)) - 1, MAX_FAT_SCAN)


def _fat_values(buf: bytearray, geo: Geometry, last: int) -> np.ndarray:
    """Entries 0..last of FAT #0, decoded to ints (vectorized)."""
    if geo.kind is FatType.FAT16:
        return np.frombuffer(buf, "<u2", last + 1, geo.fat_off)
    if geo.kind is FatType.FAT32:
        return np.frombuffer(buf, "<u4", last + 1, geo.fat_off) & CLUS_MASK[geo.kind]

    raw = np.frombuffer(buf, np.uint8, last * 3 // 2 + 2, geo.fat_off).astype(np.uint16)
    idx = np.arange(last + 1)
    o = idx * 3 // 2
    lo, hi = raw[o], raw[o + 1]
    return np.where(idx & 1, lo >> 4 | hi << 4, lo | (hi & 0xF) << 8)


def _fat_get(buf: bytearray, geo: Geometry, copy: int, c: int) -> int | None:
    """Entry *c* of FAT copy *copy*, or None when out of the image."""
    base = geo.fat_off + copy * geo.fat_bytes
    if geo.kind is FatType.FAT12:
        o = base + c * 3 // 2
        if o + 2 > len(buf):
            return None
        pair = _u16(buf, o)
        return pair >> 4 if c & 1 else pair & 0xFFF

    width = geo.kind // 8
    o = base + c * width
    if o + width > len(buf):
        return None
    return int.from_bytes(buf[o : o + width], "little")


def _fat_set(buf: bytearray, geo: Geometry, copy: int, c: int, val: int) -> None:
    """Store *val* in entry *c* of copy *copy*; silently skips entries past EOF."""
    base = geo.fat_off + copy * geo.fat_bytes
    if geo.kind is not FatType.FAT12:
        width = geo.kind // 8
        o = base + c * width
        if o + width <= len(buf):
            buf[o : o + width] = (val & FAT_MASK[geo.kind]).to_bytes(width, "little")
        return

    # FAT12: two entries share three bytes; keep the neighbour's nibble.
    o = base + c * 3 // 2
    if o + 2 > len(buf):
        return
    val &= 0xFFF
    if c & 1:
        buf[o] = (buf[o] & 0x0F) | (val << 4 & 0xF0)
        buf[o + 1] = val >> 4
    else:
        buf[o] = val & 0xFF
        buf[o + 1] = (buf[o + 1] & 0xF0) | val >> 8


# --- op handlers: each returns False to decline ---------------------------------


def _bpb(buf: bytearray, geo: Geometry, rng: RandPool, _op: Op) -> bool:
    """Draws: field, edge value."""
    field: Field = rng.choice(BPB_FIELDS[geo.kind])  # type: ignore[assignment]
    _put(buf, field.off, field.fmt, rng.choice(field.edges))  # type: ignore[arg-type]
    return True


def _label(buf: bytearray, geo: Geometry, rng: RandPool, _op: Op) -> bool:
    """Draws: label."""
    off = LABEL_OFF_32 if geo.kind is FatType.FAT32 else LABEL_OFF_16
    label: bytes = rng.choice(LABELS)  # type: ignore[assignment]
    if buf[off : off + 8] == label:
        label = label[:7] + bytes([label[7] ^ 1])
    buf[off : off + 8] = label
    return True


def _type_confuse(buf: bytearray, geo: Geometry, rng: RandPool, _op: Op) -> bool:
    """Draws: target cluster count. Resizes TotSec so the count sits on it."""
    target: int = rng.choice(CLUSTER_THRESHOLDS)  # type: ignore[assignment]
    tot = geo.overhead + target * geo.spc
    if tot < 0x10000 and _u16(buf, 19):
        _put(buf, 19, "<H", tot)
        return True
    struct.pack_into("<H", buf, 19, 0)
    _put(buf, 32, "<I", tot & 0xFFFFFFFF)
    return True


def _fat_candidates(buf: bytearray, geo: Geometry) -> list[int]:
    """Allocated clusters (non-zero entry), else every addressable cluster."""
    last = _fat_last(buf, geo)
    if last < 2:
        return []
    used = np.flatnonzero(_fat_values(buf, geo, last)[2:]) + 2
    return used.tolist() if used.size else list(range(2, last + 1))


def _fat_value(op: Op, geo: Geometry, c: int, rng: RandPool) -> int:
    """New entry value for *op* on cluster *c*."""
    if op is Op.FAT_LOOP:
        return rng.randint(2, c)  # self or an earlier cluster
    if op is Op.FAT_FREE:
        return 0
    if op is Op.FAT_BAD:
        return BAD[geo.kind]
    if op is Op.FAT_EOC:
        return rng.choice(EOC[geo.kind])  # type: ignore[return-value]
    if op is Op.FAT_RANGE:
        n = geo.clusters
        return rng.choice((1, n + 1, n + 2, *RESERVED[geo.kind]))  # type: ignore[return-value]
    return rng.randint(0, FAT_MASK[geo.kind])


def _fat_op(buf: bytearray, geo: Geometry, rng: RandPool, op: Op) -> bool:
    """Draws: [DESYNC: copy], cluster, [value]. Writes all copies except DESYNC."""
    copies = range(geo.nfats)
    if op is Op.FAT_DESYNC:
        if geo.nfats < 2:
            return False
        copy = rng.randint(0, geo.nfats - 1)
        copies = range(copy, copy + 1)

    cands = _fat_candidates(buf, geo)
    if not cands or not copies:
        return False
    c: int = rng.choice(cands)  # type: ignore[assignment]
    val = _fat_value(op, geo, c, rng) & FAT_MASK[geo.kind]
    old = _fat_get(buf, geo, copies[0], c)
    if old is None:  # DESYNC copy truncated off the image
        return False
    if old == val:
        val ^= 1

    for k in copies:
        _fat_set(buf, geo, k, c, val)
    return True


# --- directories ----------------------------------------------------------------


def _is_lfn(buf: bytearray, off: int) -> bool:
    return buf[off + ATTR_OFF] & ATTR_MASK == ATTR_LFN


def _first_clus(buf: bytearray, geo: Geometry, off: int) -> int:
    lo = _u16(buf, off + CLUS_LO_OFF)
    if geo.kind is FatType.FAT32:
        return _u16(buf, off + CLUS_HI_OFF) << 16 | lo
    return lo


def _clus_off(geo: Geometry, c: int) -> int | None:
    """Byte offset of cluster *c*, or None if not a data cluster."""
    if not 2 <= c <= geo.clusters + 1:
        return None
    return geo.data_off + (c - 2) * geo.clus_bytes


def _scan(buf: bytearray, start: int, size: int, parent: int, out: list[Entry]) -> bool:
    """Append entries in [start, start+size) until 0x00; True once listing ended."""
    end = min(start + size, len(buf)) - ENTRY + 1
    for off in range(start, end, ENTRY):
        if len(out) >= MAX_ENTRIES or buf[off] == 0:
            return True
        out.append(Entry(off, parent))
    return False


def _root_entries(buf: bytearray, geo: Geometry) -> list[Entry]:
    """Root dir: fixed region (12/16) or a bounded walk of the RootClus chain (32)."""
    out: list[Entry] = []
    if geo.kind is not FatType.FAT32:
        _scan(buf, geo.root_off, geo.root_bytes, 0, out)
        return out

    c, seen = geo.root_clus, set()
    while len(seen) < MAX_DIR_CLUSTERS and c not in seen:
        off = _clus_off(geo, c)
        if off is None or _scan(buf, off, geo.clus_bytes, geo.root_clus, out):
            break
        seen.add(c)
        c = (_fat_get(buf, geo, 0, c) or 0) & CLUS_MASK[geo.kind]
    return out


def _dir_entries(buf: bytearray, geo: Geometry) -> list[Entry]:
    """Root entries, then the first cluster of each root subdirectory."""
    out = _root_entries(buf, geo)
    subdirs = 0
    for e in list(out):
        if subdirs >= MAX_SUBDIRS:
            break
        attr = buf[e.off + ATTR_OFF]
        if attr & ATTR_MASK == ATTR_LFN or not attr & ATTR_DIR:
            continue
        if bytes(buf[e.off : e.off + 11]) in DOT_NAMES:
            continue
        c = _first_clus(buf, geo, e.off)
        off = _clus_off(geo, c)
        if off is None:
            continue
        subdirs += 1
        _scan(buf, off, geo.clus_bytes, c, out)
    return out


def lfn_checksum(name: bytes) -> int:
    """Microsoft ChkSum: rotate right, add each byte of the 8.3 name."""
    s = 0
    for c in name:
        s = ((s & 1) << 7 | s >> 1) + c & 0xFF
    return s


def _repair_lfn(buf: bytearray, entries: list[Entry], i: int) -> None:
    """Rewrite the checksum of the LFN slots directly before short entry *i*."""
    chk = lfn_checksum(bytes(buf[entries[i].off : entries[i].off + 11]))
    j = i - 1
    while j >= 0 and i - j <= MAX_LFN_SLOTS:
        off = entries[j].off
        if off != entries[j + 1].off - ENTRY or not _is_lfn(buf, off):
            return
        buf[off + LFN_CHK_OFF] = chk
        j -= 1


def _dir_value(buf: bytearray, geo: Geometry, e: Entry, op: Op, rng: RandPool) -> None:
    """Apply a DIR_* op to short entry *e*; draws: the op's edge value."""
    off = e.off
    if op is Op.DIR_SIZE:
        _put(buf, off + SIZE_OFF, "<I", rng.choice(SIZE_EDGES))  # type: ignore[arg-type]
        return
    if op is Op.DIR_ATTR:
        _put(buf, off + ATTR_OFF, "B", rng.choice(ATTR_EDGES))  # type: ignore[arg-type]
        return
    if op is Op.DIR_NAME:
        edit: bytes = rng.choice(NAME_EDITS)  # type: ignore[assignment]
        old = bytes(buf[off : off + 11])
        buf[off : off + len(edit)] = edit
        if buf[off : off + 11] == old:
            buf[off] ^= 1
        return

    # DIR_CLUSTER: FAT32 splits the cluster over hi/lo words.
    n, kind = geo.clusters, geo.kind
    edges = (0, 1, n + 1, n + 2, BAD[kind], CLUS_MASK[kind], e.parent)
    val: int = rng.choice(edges) & CLUS_MASK[kind]  # type: ignore[operator]
    if _first_clus(buf, geo, off) == val:
        val ^= 1
    struct.pack_into("<H", buf, off + CLUS_LO_OFF, val & 0xFFFF)
    if kind is FatType.FAT32:
        struct.pack_into("<H", buf, off + CLUS_HI_OFF, val >> 16)


def _dir_op(buf: bytearray, geo: Geometry, rng: RandPool, op: Op) -> bool:
    """Draws: short entry, edge value. DIR_NAME repairs the LFN checksum."""
    entries = _dir_entries(buf, geo)
    shorts = [i for i, e in enumerate(entries) if not _is_lfn(buf, e.off)]
    if not shorts:
        return False
    i: int = rng.choice(shorts)  # type: ignore[assignment]
    _dir_value(buf, geo, entries[i], op, rng)
    if op is Op.DIR_NAME:
        _repair_lfn(buf, entries, i)
    return True


def _lfn_op(buf: bytearray, geo: Geometry, rng: RandPool, op: Op) -> bool:
    """Draws: LFN slot, then seq edge (LFN_SEQ) or checksum XOR 1..255 (LFN_BREAK)."""
    lfns = [e.off for e in _dir_entries(buf, geo) if _is_lfn(buf, e.off)]
    if not lfns:
        return False
    off: int = rng.choice(lfns)  # type: ignore[assignment]
    if op is Op.LFN_SEQ:
        _put(buf, off, "B", rng.choice(LFN_SEQ_EDGES))  # type: ignore[arg-type]
        return True
    buf[off + LFN_CHK_OFF] ^= rng.randint(1, 0xFF)
    return True


def _fsinfo(buf: bytearray, geo: Geometry, rng: RandPool, _op: Op) -> bool:
    """FAT32 only. Draws: field, then edge (FREE/NEXT) or signature index (SIG)."""
    if geo.kind is not FatType.FAT32:
        return False
    base = _u16(buf, 48) * geo.bps
    if base == 0 or base + SECTOR > len(buf):
        return False

    field = FsiField(rng.randint(0, len(FsiField) - 1))
    if field is FsiField.SIG:
        off = base + FSI_SIG_OFFS[rng.randint(0, len(FSI_SIG_OFFS) - 1)]
        struct.pack_into("<I", buf, off, _u32(buf, off) ^ SIG_FLIP)
        return True

    n = geo.clusters
    edges = (0, 1, 2, n, n + 1, n + 2, 0xFFFFFFFE, 0xFFFFFFFF)
    off = base + (FSI_FREE_OFF if field is FsiField.FREE else FSI_NEXT_OFF)
    _put(buf, off, "<I", rng.choice(edges) & 0xFFFFFFFF)  # type: ignore[operator]
    return True


_Handler = Callable[[bytearray, Geometry, RandPool, Op], bool]
_FAT_OPS = (Op.FAT_LOOP, Op.FAT_FREE, Op.FAT_BAD, Op.FAT_EOC, Op.FAT_RANGE, Op.FAT_DESYNC)
_DIR_OPS = (Op.DIR_CLUSTER, Op.DIR_SIZE, Op.DIR_ATTR, Op.DIR_NAME)
_HANDLERS: dict[Op, _Handler] = {
    Op.BPB: _bpb,
    Op.LABEL: _label,
    Op.TYPE_CONFUSE: _type_confuse,
    **{op: _fat_op for op in _FAT_OPS},
    **{op: _dir_op for op in _DIR_OPS},
    Op.LFN_SEQ: _lfn_op,
    Op.LFN_BREAK: _lfn_op,
    Op.FSINFO: _fsinfo,
}


def mutate_fat(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One in-place structural edit of a FAT image; None if declined.

    Draw order: op (``randint(0, len(Op) - 1)``), then the handler's draws.
    """
    geo = parse_fat(data)
    if geo is None:
        return None

    op = Op(rng.randint(0, len(Op) - 1))
    buf = bytearray(data)
    if not _HANDLERS[op](buf, geo, rng, op):
        return None
    return bytes(buf[:max_len])
