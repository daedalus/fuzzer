"""Minimal FAT12/16/32 image builders for the FAT mutator tests (no mkfs).

Every image holds, in its root directory: an LFN entry + its 8.3 file
entry (one cluster of data) and a subdirectory with '.' / '..'.

  FAT12  4 KiB:  boot | FAT1 | FAT2 | root(16) | clus 2 file | 3 sub | 4,5 free
  FAT16  ~2 MiB: same, 4085 clusters (the smallest count fatgen103 calls FAT16)
  FAT32  4 KiB:  boot | FSInfo | FAT1 | FAT2 | clus 2 root | 3 file | 4 sub | 5 free
                 (FATSz16 == 0, so Linux vfat treats it as FAT32 at any size)
"""

from __future__ import annotations

import struct
from enum import Enum

BPS = 512
ENTRY = 32
FILE_DATA = b"hello\n"
LONG_NAME = "LongName.txt"
SHORT_NAME = b"LONGNA~1TXT"
SUB_NAME = b"SUB        "
DOT = b".          "
DOTDOT = b"..         "
ATTR_ARCHIVE = 0x20
ATTR_DIR = 0x10
ATTR_LFN = 0x0F
LFN_LAST = 0x40
MEDIA = 0xF8
FSI_LEAD, FSI_STRUC, FSI_TRAIL = 0x41615252, 0x61417272, 0xAA550000
FAT16_CLUSTERS = 4085


class Kind(Enum):
    """FAT variant to build."""

    FAT12 = 12
    FAT16 = 16
    FAT32 = 32


def lfn_checksum(name: bytes) -> int:
    """Microsoft ChkSum over the 11-byte 8.3 name."""
    s = 0
    for c in name:
        s = (((s & 1) << 7) | (s >> 1)) + c & 0xFF
    return s


def short_entry(name: bytes, attr: int, cluster: int, size: int) -> bytes:
    """32-byte 8.3 directory entry."""
    hi, lo = cluster >> 16, cluster & 0xFFFF
    return struct.pack("<11sBBBHHHHHHHI", name, attr, 0, 0, 0, 0, 0, hi, 0, 0, lo, size)


def lfn_entry(text: str, seq: int, chk: int) -> bytes:
    """One LFN slot holding up to 13 UTF-16 chars (NUL then 0xFFFF padded)."""
    units = list(text.encode("utf-16-le"))
    chars = [units[i] | units[i + 1] << 8 for i in range(0, len(units), 2)]
    if len(chars) < 13:
        chars.append(0)
    chars += [0xFFFF] * (13 - len(chars))
    head = struct.pack("<B5HBBB", seq, *chars[:5], ATTR_LFN, 0, chk)
    return head + struct.pack("<6HH2H", *chars[5:11], 0, *chars[11:])


def _root_entries(file_clus: int, sub_clus: int) -> bytes:
    lfn = lfn_entry(LONG_NAME, LFN_LAST | 1, lfn_checksum(SHORT_NAME))
    file = short_entry(SHORT_NAME, ATTR_ARCHIVE, file_clus, len(FILE_DATA))
    return lfn + file + short_entry(SUB_NAME, ATTR_DIR, sub_clus, 0)


def _sub_entries(sub_clus: int) -> bytes:
    return short_entry(DOT, ATTR_DIR, sub_clus, 0) + short_entry(DOTDOT, ATTR_DIR, 0, 0)


def _boot(kind: Kind, rsvd: int, fatsz: int, root_ent: int, totsec: int) -> bytearray:
    """Boot sector: jump, OEM, BPB, type-specific tail, 0x55AA."""
    b = bytearray(BPS)
    b[0:3] = b"\xeb\x3c\x90"
    b[3:11] = b"FATTEST "
    small = totsec if totsec < 0x10000 else 0
    big = 0 if small else totsec
    fatsz16 = 0 if kind is Kind.FAT32 else fatsz
    struct.pack_into(
        "<HBHBHHBHHHII", b, 11, BPS, 1, rsvd, 2, root_ent, small, MEDIA, fatsz16, 1, 1, 0, big
    )
    if kind is Kind.FAT32:
        struct.pack_into("<IHHIHH", b, 36, fatsz, 0, 0, 2, 1, 0)
        b[66] = 0x29
        b[82:90] = b"FAT32   "
    else:
        b[38] = 0x29
        b[54:62] = b"FAT12   " if kind is Kind.FAT12 else b"FAT16   "
    b[510:512] = b"\x55\xaa"
    return b


def _fat(kind: Kind, used: int, fatsz: int) -> bytes:
    """FAT copy: media/EOC reserved entries, then *used* single-cluster chains."""
    n = 2 + used
    if kind is Kind.FAT12:
        vals = [0xF00 | MEDIA] + [0xFFF] * (n - 1)
        raw = bytearray()
        for i in range(0, n + (n & 1), 2):
            a = vals[i]
            c = vals[i + 1] if i + 1 < n else 0
            raw += bytes([a & 0xFF, (a >> 8) | (c & 0xF) << 4, c >> 4])
    elif kind is Kind.FAT16:
        raw = bytearray(struct.pack(f"<{n}H", 0xFF00 | MEDIA, *[0xFFFF] * (n - 1)))
    else:
        raw = bytearray(struct.pack(f"<{n}I", 0x0FFFFF00 | MEDIA, *[0x0FFFFFFF] * (n - 1)))
    return bytes(raw) + bytes(fatsz * BPS - len(raw))


def _fsinfo(free: int, nxt: int) -> bytes:
    s = bytearray(BPS)
    struct.pack_into("<I", s, 0, FSI_LEAD)
    struct.pack_into("<III", s, 484, FSI_STRUC, free, nxt)
    struct.pack_into("<I", s, 508, FSI_TRAIL)
    return bytes(s)


def _put(img: bytearray, off: int, blob: bytes) -> None:
    img[off : off + len(blob)] = blob


def build_fat(kind: Kind = Kind.FAT12) -> bytes:
    """Smallest image of *kind* with an LFN file and a subdirectory."""
    if kind is Kind.FAT32:
        rsvd, fatsz, root_ent, clusters = 2, 1, 0, 4
        root_secs = 0
    else:
        clusters = FAT16_CLUSTERS if kind is Kind.FAT16 else 4
        rsvd, root_ent, root_secs = 1, 16, 1
        per = 2 if kind is Kind.FAT16 else 1.5
        fatsz = -(-int((clusters + 2) * per) // BPS)
    totsec = rsvd + 2 * fatsz + root_secs + clusters
    img = bytearray(totsec * BPS)
    _put(img, 0, _boot(kind, rsvd, fatsz, root_ent, totsec))

    # Cluster use: FAT32 puts the root at 2, so file/sub shift up one.
    used = 3 if kind is Kind.FAT32 else 2
    fat = _fat(kind, used, fatsz)
    _put(img, rsvd * BPS, fat)
    _put(img, (rsvd + fatsz) * BPS, fat)
    data_off = (rsvd + 2 * fatsz + root_secs) * BPS

    def clus(c: int) -> int:
        return data_off + (c - 2) * BPS

    if kind is Kind.FAT32:
        _put(img, BPS, _fsinfo(clusters - used, 2 + used))
        _put(img, clus(2), _root_entries(3, 4))
        file_clus, sub_clus = 3, 4
    else:
        _put(img, (rsvd + 2 * fatsz) * BPS, _root_entries(2, 3))
        file_clus, sub_clus = 2, 3
    _put(img, clus(file_clus), FILE_DATA)
    _put(img, clus(sub_clus), _sub_entries(sub_clus))
    return bytes(img)
