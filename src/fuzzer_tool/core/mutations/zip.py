"""Structure-aware ZIP mutator.

ZIP structure:
  [local file headers + data...][central directory][EOCD]

  LFH:   PK\x03\x04 + 26-byte fixed header + name + extra
  CD:    PK\x01\x02 + 46-byte fixed header + name + extra + comment
  EOCD:  PK\x05\x06 + 18-byte fixed header + comment
  Data descriptor (flag bit 3): 12 or 16 raw bytes after file data

Detection: PK\x03\x04 at offset 0 AND a valid EOCD (PK\x05\x06 found by
backward scan within 65557 bytes of the end, comment length verified).
Any 0xFFFFFFFF zip64 sentinel makes parse return None (zip64 deliberately
unsupported in v1 — TODO follow-up).

Serialization recomputes local offsets, CD offsets, CD size and EOCD
counts from the actual entry layout; every other field is written
verbatim from stored values, so an untouched archive round-trips
byte-identically.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

from fuzzer_tool.core.crc32 import crc32

# Interesting compression methods to swap in
METHOD_VALUES = [0, 8, 9, 12, 14, 93, 0xFFFF]

# Interesting DOS time / date values
TIME_VALUES = [0x0000, 0x2020, 0x4040, 0x7FFF, 0xFFFF]
DATE_VALUES = [0x0021, 0x4A21, 0x7FFF, 0xFFFF]

# Interesting flags values (bit 3 = data descriptor, bit 11 = UTF-8)
FLAG_VALUES = [0x0000, 0x0008, 0x0800, 0x0808, 0xFFFF]

# Interesting EOCD field values (counts, sizes, offsets)
EOCD_VALUES = [0, 1, 0xFFFF, 0xFFFFFFFF]


@dataclass
class ZipEntry:
    """A single file entry (LFH + CD records)."""

    name: bytes
    extra: bytes
    comment: bytes
    method: int
    flags: int
    crc32: int
    csize: int
    usize: int
    modtime: int
    moddate: int
    data: bytes  # raw file data
    desc: bytes  # raw data descriptor (flag bit 3), else b""
    lfh_fixed: bytes  # 30-byte LFH fixed header (sig..extra_len)
    cd_fixed: bytes  # 46-byte CD fixed header (sig..local_offset)


@dataclass
class ZipDoc:
    """A parsed ZIP archive."""

    entries: list[ZipEntry]
    eocd_fixed: bytes  # 22-byte EOCD fixed header (sig..comment_len)
    eocd_comment: bytes


def _find_eocd(data: bytes) -> int:
    """Backward-scan for a valid EOCD record. Returns its offset or -1."""
    start = max(0, len(data) - 22 - 65535)
    for pos in range(len(data) - 22, start - 1, -1):
        if data[pos : pos + 4] == b"PK\x05\x06":
            comment_len = struct.unpack_from("<H", data, pos + 20)[0]
            if pos + 22 + comment_len == len(data):
                return pos
    return -1


def parse_zip(data: bytes) -> ZipDoc | None:
    """Parse a ZIP archive into a ZipDoc.

    Returns None unless PK\x03\x04 starts the buffer, a valid EOCD is
    found, the central directory parses cleanly, and no zip64 sentinel
    appears.
    """
    if len(data) < 30 or data[:4] != b"PK\x03\x04":
        return None

    eocd_pos = _find_eocd(data)
    if eocd_pos < 0:
        return None
    eocd_fixed = data[eocd_pos : eocd_pos + 22]
    eocd_comment = data[eocd_pos + 22 :]
    cd_offset = struct.unpack_from("<I", data, eocd_pos + 16)[0]
    total = struct.unpack_from("<H", data, eocd_pos + 10)[0]

    entries: list[ZipEntry] = []
    pos = cd_offset
    for _ in range(total):
        if pos + 46 > len(data) or data[pos : pos + 4] != b"PK\x01\x02":
            return None
        fixed = data[pos : pos + 46]
        name_len = struct.unpack_from("<H", fixed, 28)[0]
        extra_len = struct.unpack_from("<H", fixed, 30)[0]
        comment_len = struct.unpack_from("<H", fixed, 32)[0]
        if pos + 46 + name_len + extra_len + comment_len > len(data):
            return None
        name = data[pos + 46 : pos + 46 + name_len]
        extra = data[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        comment = data[
            pos + 46 + name_len + extra_len : pos + 46 + name_len + extra_len + comment_len
        ]
        method = struct.unpack_from("<H", fixed, 10)[0]
        flags = struct.unpack_from("<H", fixed, 8)[0]
        crc = struct.unpack_from("<I", fixed, 16)[0]
        csize = struct.unpack_from("<I", fixed, 20)[0]
        usize = struct.unpack_from("<I", fixed, 24)[0]
        modtime = struct.unpack_from("<H", fixed, 12)[0]
        moddate = struct.unpack_from("<H", fixed, 14)[0]
        lho = struct.unpack_from("<I", fixed, 42)[0]
        if 0xFFFFFFFF in (csize, usize, lho):
            return None  # zip64 sentinel — unsupported

        # Parse the local file header
        if lho + 30 > len(data) or data[lho : lho + 4] != b"PK\x03\x04":
            return None
        lfh_fixed = data[lho : lho + 30]
        lname_len = struct.unpack_from("<H", data, lho + 26)[0]
        lextra_len = struct.unpack_from("<H", data, lho + 28)[0]
        if lho + 30 + lname_len + lextra_len > len(data):
            return None
        data_start = lho + 30 + lname_len + lextra_len

        desc = b""
        if flags & 0x08:
            # Data descriptor follows the file data (12 or 16 bytes)
            if data_start + csize + 12 > len(data) or data_start + csize + 12 > eocd_pos:
                return None
            if data[data_start + csize : data_start + csize + 4] == b"PK\x07\x08":
                desc = data[data_start + csize : data_start + csize + 16]
            else:
                desc = data[data_start + csize : data_start + csize + 12]
        elif data_start + csize > eocd_pos:
            return None

        entry_data = data[data_start : data_start + csize]
        entries.append(
            ZipEntry(
                name=name,
                extra=extra,
                comment=comment,
                method=method,
                flags=flags,
                crc32=crc,
                csize=csize,
                usize=usize,
                modtime=modtime,
                moddate=moddate,
                data=entry_data,
                desc=desc,
                lfh_fixed=lfh_fixed,
                cd_fixed=fixed,
            )
        )
        pos += 46 + name_len + extra_len + comment_len

    if pos != eocd_pos:
        return None  # central directory must end exactly at the EOCD

    return ZipDoc(entries=entries, eocd_fixed=eocd_fixed, eocd_comment=eocd_comment)


def _patch_lfh(entry: ZipEntry) -> bytes:
    """Patch the stored LFH fixed header with current field values."""
    fixed = bytearray(entry.lfh_fixed)
    struct.pack_into("<H", fixed, 6, entry.flags & 0xFFFF)
    struct.pack_into("<H", fixed, 8, entry.method & 0xFFFF)
    struct.pack_into("<H", fixed, 10, entry.modtime & 0xFFFF)
    struct.pack_into("<H", fixed, 12, entry.moddate & 0xFFFF)
    struct.pack_into("<I", fixed, 14, entry.crc32 & 0xFFFFFFFF)
    struct.pack_into("<I", fixed, 18, entry.csize & 0xFFFFFFFF)
    struct.pack_into("<I", fixed, 22, entry.usize & 0xFFFFFFFF)
    struct.pack_into("<H", fixed, 26, len(entry.name) & 0xFFFF)
    struct.pack_into("<H", fixed, 28, len(entry.extra) & 0xFFFF)
    return bytes(fixed)


def _patch_cd(entry: ZipEntry, local_offset: int) -> bytes:
    """Patch the stored CD fixed header with current field values."""
    fixed = bytearray(entry.cd_fixed)
    struct.pack_into("<H", fixed, 8, entry.flags & 0xFFFF)
    struct.pack_into("<H", fixed, 10, entry.method & 0xFFFF)
    struct.pack_into("<H", fixed, 12, entry.modtime & 0xFFFF)
    struct.pack_into("<H", fixed, 14, entry.moddate & 0xFFFF)
    struct.pack_into("<I", fixed, 16, entry.crc32 & 0xFFFFFFFF)
    struct.pack_into("<I", fixed, 20, entry.csize & 0xFFFFFFFF)
    struct.pack_into("<I", fixed, 24, entry.usize & 0xFFFFFFFF)
    struct.pack_into("<H", fixed, 28, len(entry.name) & 0xFFFF)
    struct.pack_into("<H", fixed, 30, len(entry.extra) & 0xFFFF)
    struct.pack_into("<H", fixed, 32, len(entry.comment) & 0xFFFF)
    struct.pack_into("<I", fixed, 42, local_offset & 0xFFFFFFFF)
    return bytes(fixed)


def serialize_zip(doc: ZipDoc) -> bytes:
    """Serialize a ZipDoc back to bytes.

    Local offsets, CD offsets, CD size and EOCD counts are recomputed
    from the actual layout; all other fields are written verbatim.
    """
    out = bytearray()
    offsets: list[int] = []
    for entry in doc.entries:
        offsets.append(len(out))
        out.extend(_patch_lfh(entry))
        out.extend(entry.name)
        out.extend(entry.extra)
        out.extend(entry.data)
        out.extend(entry.desc)

    cd_start = len(out)
    for entry, local_offset in zip(doc.entries, offsets, strict=True):
        out.extend(_patch_cd(entry, local_offset))
        out.extend(entry.name)
        out.extend(entry.extra)
        out.extend(entry.comment)
    cd_size = len(out) - cd_start

    eocd = bytearray(doc.eocd_fixed)
    struct.pack_into("<H", eocd, 8, len(doc.entries) & 0xFFFF)
    struct.pack_into("<H", eocd, 10, len(doc.entries) & 0xFFFF)
    struct.pack_into("<I", eocd, 12, cd_size & 0xFFFFFFFF)
    struct.pack_into("<I", eocd, 16, cd_start & 0xFFFFFFFF)
    struct.pack_into("<H", eocd, 20, len(doc.eocd_comment) & 0xFFFF)
    out.extend(eocd)
    out.extend(doc.eocd_comment)

    return bytes(out)


def _locate_eocd(raw: bytes) -> int:
    """Find the EOCD offset in a serialized archive (via backward scan)."""
    return _find_eocd(raw)


class ZipMutator:
    """Structure-aware ZIP mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        doc = parse_zip(data)
        if doc is None:
            return self._generate_random_zip(max_len, rng=self._rng)

        op = self._rng.randint(0, 11)
        mutators = [
            self._swap_method,
            self._mutate_crc,
            self._rewrite_size,
            self._rewrite_name,
            self._mutate_flags,
            self._mutate_modtime,
            self._mutate_eocd_field,
            self._swap_entries,
            self._duplicate_entry,
            self._delete_entry,
            self._truncate_data,
            self._generate_random_zip,
        ]
        result = mutators[op](doc, max_len)
        if isinstance(result, ZipDoc):
            return serialize_zip(result)[:max_len]
        return result[:max_len]

    def _swap_method(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            current = entry.method
            options = [m for m in METHOD_VALUES if m != current]
            entry.method = self._rng.choice(options)
        return doc

    def _mutate_crc(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            entry.crc32 = self._rng.choice(
                [
                    0,
                    1,
                    0xFFFFFFFF,
                    crc32(entry.data) & 0xFFFFFFFF,
                    self._rng.randint(0, 0xFFFFFFFF),
                ]
            )
        return doc

    def _rewrite_size(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            entry.csize = self._rng.choice(
                [0, 1, len(entry.data), max_len, 0xFFFFFFFE, self._rng.randint(0, 0xFFFFFFFF)]
            )
            entry.usize = self._rng.choice(
                [0, 1, len(entry.data), max_len, 0xFFFFFFFE, self._rng.randint(0, 0xFFFFFFFF)]
            )
        return doc

    def _rewrite_name(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            mode = self._rng.randint(0, 2)
            if mode == 0:
                entry.name = entry.name[: self._rng.randint(0, len(entry.name))]
            elif mode == 1:
                entry.name = entry.name + self._rng.randbytes(self._rng.randint(0, 8))
            else:
                entry.name = self._rng.choice(
                    [b"a.txt", b"dir/", b"../evil", b"\x00\x01", b"PK\x03\x04", b"n" * 64]
                )
        return doc

    def _mutate_flags(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            entry.flags = self._rng.choice(FLAG_VALUES)
            # Keep flag bit 3 (data descriptor) consistent with desc presence
            if entry.flags & 0x08:
                if not entry.desc:
                    entry.desc = struct.pack(
                        "<III",
                        entry.crc32 & 0xFFFFFFFF,
                        entry.csize & 0xFFFFFFFF,
                        entry.usize & 0xFFFFFFFF,
                    )
            else:
                entry.desc = b""
        return doc

    def _mutate_modtime(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            entry.modtime = self._rng.choice(TIME_VALUES + [self._rng.randint(0, 0xFFFF)])
            entry.moddate = self._rng.choice(DATE_VALUES + [self._rng.randint(0, 0xFFFF)])
        return doc

    def _mutate_eocd_field(self, doc: ZipDoc, max_len: int) -> bytes:
        """Corrupt an EOCD field (deliberately inconsistent)."""
        raw = bytearray(serialize_zip(doc))
        eocd_pos = _locate_eocd(raw)
        if eocd_pos < 0:
            return bytes(raw)
        field_off = self._rng.choice([8, 10, 12, 16, 20])
        value = self._rng.choice(EOCD_VALUES + [self._rng.randint(0, 0xFFFF)])
        struct.pack_into("<H", raw, eocd_pos + field_off, value & 0xFFFF)
        return bytes(raw)

    def _swap_entries(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if len(doc.entries) >= 2:
            i, j = self._rng.sample(list(range(len(doc.entries))), 2)
            doc.entries[i], doc.entries[j] = doc.entries[j], doc.entries[i]
        return doc

    def _duplicate_entry(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            idx = self._rng.randint(0, len(doc.entries) - 1)
            orig = doc.entries[idx]
            dup = ZipEntry(
                name=orig.name[:],
                extra=orig.extra[:],
                comment=orig.comment[:],
                method=orig.method,
                flags=orig.flags,
                crc32=orig.crc32,
                csize=orig.csize,
                usize=orig.usize,
                modtime=orig.modtime,
                moddate=orig.moddate,
                data=orig.data[:],
                desc=orig.desc[:],
                lfh_fixed=orig.lfh_fixed,
                cd_fixed=orig.cd_fixed,
            )
            doc.entries.insert(idx + 1, dup)
        return doc

    def _delete_entry(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if len(doc.entries) > 1:
            doc.entries.pop(self._rng.randint(0, len(doc.entries) - 1))
        return doc

    def _truncate_data(self, doc: ZipDoc, max_len: int) -> ZipDoc:
        if doc.entries:
            entry = self._rng.choice(doc.entries)
            if len(entry.data) > 1:
                entry.data = entry.data[: self._rng.randint(0, len(entry.data))]
        return doc

    def _generate_random_zip(self, _doc=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a minimal random ZIP archive (stored, no encryption)."""
        self._rng = rng or self._rng

        name = self._rng.choice([b"a.txt", b"data.bin", b"dir/file"])
        data = self._rng.randbytes(self._rng.randint(0, 64))
        crc = crc32(data) & 0xFFFFFFFF
        modtime = self._rng.choice(TIME_VALUES)
        moddate = self._rng.choice(DATE_VALUES)

        lfh_fixed = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,
            0,
            0,
            modtime,
            moddate,
            crc,
            len(data),
            len(data),
            len(name),
            0,
        )
        cd_fixed = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            20,
            20,
            0,
            0,
            modtime,
            moddate,
            crc,
            len(data),
            len(data),
            len(name),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        entry = ZipEntry(
            name=name,
            extra=b"",
            comment=b"",
            method=0,
            flags=0,
            crc32=crc,
            csize=len(data),
            usize=len(data),
            modtime=modtime,
            moddate=moddate,
            data=data,
            desc=b"",
            lfh_fixed=lfh_fixed,
            cd_fixed=cd_fixed,
        )
        eocd_fixed = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, 0, 0, 0)
        doc = ZipDoc(entries=[entry], eocd_fixed=eocd_fixed, eocd_comment=b"")
        return serialize_zip(doc)[:max_len]
