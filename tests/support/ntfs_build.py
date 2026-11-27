"""Minimal NTFS image builder for the ntfs mutator tests (no mkntfs needed).

Layout (512-byte sectors and clusters, 1024-byte MFT records):

  0     512            1536           2560
  | boot | FILE #0 $MFT | FILE #1 file |
          ^ MFT LCN 1
"""

from __future__ import annotations

import struct

SECTOR = 512
RECORD = 1024
USA_OFF = 0x30
FIRST_ATTR = 0x38
USN = 0x0001
END_MARKER = 0xFFFFFFFF
MFT_LCN = 1
INDX_ENTRIES = 0x28  # index entries offset, relative to the 0x18 index header


def _align8(n: int) -> int:
    return (n + 7) & ~7


def boot_sector(total_sectors: int, mft_lcn: int = MFT_LCN) -> bytes:
    """512-byte NTFS boot sector: 1 sector/cluster, 2 clusters/record."""
    bs = bytearray(SECTOR)
    bs[0:3] = b"\xeb\x52\x90"
    bs[3:11] = b"NTFS    "
    struct.pack_into("<HB", bs, 0x0B, SECTOR, 1)
    bs[0x15] = 0xF8
    struct.pack_into("<QQQ", bs, 0x28, total_sectors, mft_lcn, mft_lcn + 2)
    bs[0x40] = 2  # clusters per MFT record
    bs[0x44] = 2  # clusters per index record
    struct.pack_into("<Q", bs, 0x48, 0x1122334455667788)
    bs[510:512] = b"\x55\xaa"
    return bytes(bs)


def resident_attr(type_code: int, value: bytes, attr_id: int = 0) -> bytes:
    """Resident attribute: 24-byte header then the value."""
    reclen = _align8(0x18 + len(value))
    buf = bytearray(reclen)
    struct.pack_into("<IIBBHHH", buf, 0, type_code, reclen, 0, 0, 0x18, 0, attr_id)
    struct.pack_into("<IH", buf, 0x10, len(value), 0x18)
    buf[0x18 : 0x18 + len(value)] = value
    return bytes(buf)


def nonres_attr(type_code: int, runlist: bytes, clusters: int, attr_id: int = 0) -> bytes:
    """Non-resident attribute: 64-byte header then the mapping pairs."""
    reclen = _align8(0x40 + len(runlist))
    size = clusters * SECTOR
    buf = bytearray(reclen)
    struct.pack_into("<IIBBHHH", buf, 0, type_code, reclen, 1, 0, 0x40, 0, attr_id)
    struct.pack_into("<QQH", buf, 0x10, 0, clusters - 1, 0x40)
    struct.pack_into("<QQQ", buf, 0x28, size, size, size)
    buf[0x40 : 0x40 + len(runlist)] = runlist
    return bytes(buf)


def file_name_value(name: str, parent: int = 5) -> bytes:
    """$FILE_NAME value: parent ref, 4 times, sizes, flags, name."""
    enc = name.encode("utf-16-le")
    head = struct.pack("<Q4QQQII", parent, 0, 0, 0, 0, 0, 0, 0, 0)
    return head + bytes([len(name), 1]) + enc


def apply_fixups(rec: bytearray, usn: int = USN) -> None:
    """Move each sector tail into the USA and stamp the USN there."""
    usa_off, usa_count = struct.unpack_from("<HH", rec, 4)
    struct.pack_into("<H", rec, usa_off, usn)
    for k in range(usa_count - 1):
        tail = k * SECTOR + SECTOR - 2
        slot = usa_off + 2 + 2 * k
        rec[slot : slot + 2] = rec[tail : tail + 2]
        struct.pack_into("<H", rec, tail, usn)


def file_record(attrs: list[bytes], rec_no: int = 0, flags: int = 1) -> bytes:
    """1024-byte FILE record with the given attributes and valid fixups."""
    rec = bytearray(RECORD)
    body = b"".join(attrs) + struct.pack("<I", END_MARKER)
    used = _align8(FIRST_ATTR + len(body))
    rec[0:4] = b"FILE"
    struct.pack_into("<HHQHHHH", rec, 4, USA_OFF, RECORD // SECTOR + 1, 0, 1, 1, FIRST_ATTR, flags)
    struct.pack_into("<IIQH", rec, 0x18, used, RECORD, 0, len(attrs))
    struct.pack_into("<I", rec, 0x2C, rec_no)
    rec[FIRST_ATTR : FIRST_ATTR + len(body)] = body
    rec[SECTOR - 2 : SECTOR] = b"\xab\xcd"  # logical tail data the USA must preserve
    apply_fixups(rec)
    return bytes(rec)


def mft_record() -> bytes:
    """$MFT: $STANDARD_INFORMATION, $FILE_NAME, non-resident $DATA (1 run)."""
    attrs = [
        resident_attr(0x10, bytes(48), 0),
        resident_attr(0x30, file_name_value("$MFT"), 1),
        nonres_attr(0x80, bytes([0x11, 0x02, MFT_LCN, 0x00]), 2, 2),
    ]
    return file_record(attrs, 0)


def data_record() -> bytes:
    """A small file with resident $DATA."""
    attrs = [
        resident_attr(0x10, bytes(48), 0),
        resident_attr(0x30, file_name_value("a"), 1),
        resident_attr(0x80, b"hello", 2),
    ]
    return file_record(attrs, 1)


def indx_record() -> bytes:
    """1024-byte INDX block: one index entry plus the last-entry marker."""
    rec = bytearray(RECORD)
    rec[0:4] = b"INDX"
    struct.pack_into("<HHQQ", rec, 4, 0x28, RECORD // SECTOR + 1, 0, 0)
    first = 0x18 + INDX_ENTRIES
    struct.pack_into("<IIII", rec, 0x18, INDX_ENTRIES, INDX_ENTRIES + 0x68, RECORD - 0x18, 0)
    struct.pack_into("<QHHH", rec, first, 0, 0x58, 0x42, 0)
    struct.pack_into("<QHHH", rec, first + 0x58, 0, 0x10, 0, 2)
    apply_fixups(rec)
    return bytes(rec)


def build_image() -> bytes:
    """Boot sector + 2-record MFT (2560 bytes)."""
    recs = mft_record() + data_record()
    return boot_sector((SECTOR + len(recs)) // SECTOR) + recs
