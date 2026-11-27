"""NTFS boot sector / MFT record / INDX mutator with fixup repair.

::

    boot sector (offset 0)          FILE / INDX record (1 KiB typical)
    0x03 "NTFS    "                 0x00 magic | 0x04 USA off | 0x06 USA count
    0x0B bytes/sector u16           0x08 LSN | 0x10 seq | 0x12 links
    0x0D sectors/cluster u8         0x14 first attr | 0x16 flags | 0x18 used
    0x28 total sectors u64          0x1C allocated | 0x20 base ref | 0x28 next id
    0x30 MFT LCN | 0x38 MFTMirr     attributes ... 0xFFFFFFFF end marker
    0x40 clusters/record i8         every 512-byte sector ends in the USN; the
    0x44 clusters/index i8          real tail bytes live in the USA (fixups)
    0x1FE 55 AA

Edits target ntfs-3g / ntfs3 / Windows parsers: BPB geometry (signed
``2^-n`` record sizes, MFT LCN past the volume), record header lengths
(used > allocated, first attribute out of range), attribute headers
(type, length, resident/non-resident confusion, value and size overruns),
mapping pairs (oversized length/offset nibbles, negative deltas, missing
terminator) and INDX entry lengths. Record edits are applied to the
de-fixed record and the fixups are re-applied, so the edit passes the
torn-write check; FIXUP alone breaks one sector tail to hit the reject path.
USA offset/count are never edited: the re-fix keys on them.

Draw order: op (``choice`` over :func:`available_ops`), then ``randint``
selectors (record, attribute, field / run position / sector) and a
``choice`` value, as listed per op in ``_Editor``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeVar, cast

from fuzzer_tool.core.rand_pool import RandPool

SECTOR = 512
BOOT_LEN = 512
OEM_ID = b"NTFS    "
OEM_OFF = 3
BOOT_SIG = b"\x55\xaa"
BOOT_SIG_OFF = 510
FILE_MAGIC = b"FILE"
INDX_MAGIC = b"INDX"
USA_MIN_OFF = 0x28
USA_MIN_COUNT = 2
USA_MAX_COUNT = 9  # records up to 4 KiB
MAX_RECORDS = 256
MAX_ATTRS = 64
END_MARKER = 0xFFFFFFFF
FIRST_ATTR_OFF = 0x14
ATTR_MIN = 0x18
NONRES_MIN = 0x40
MAPPING_OFF = 0x20
INDX_HDR = 0x18
TAIL_FLIP = 0xFFFF

U16 = 0xFFFF
U32 = 0xFFFFFFFF
U64 = (1 << 64) - 1
_LEN_EDGES = (0, 1, 7, 0x17, 0x3FF, 0x400, 0x7FFF, U32)

# (offset, width, edges): boot-sector BPB fields.
BPB_FIELDS = (
    (0x0B, 2, (0, 1, 256, 511, 513, 1024, 4096, 0x8000, U16)),  # bytes/sector
    (0x0D, 1, (0, 1, 3, 0x80, 0xF4, 0xFF)),  # sectors/cluster (> 0x80: 2^-n)
    (0x28, 8, (0, 1, U32, 1 << 63, U64)),  # total sectors
    (0x30, 8, (0, 2, 3, U32, U64)),  # MFT LCN
    (0x38, 8, (0, 1, U32, U64)),  # MFTMirr LCN
    (0x40, 1, (0, 1, 0x80, 0xE1, 0xF6, 0xFF, 0x7F)),  # clusters/MFT record (i8)
    (0x44, 1, (0, 1, 0x80, 0xE1, 0xF6, 0xFF, 0x7F)),  # clusters/index record (i8)
    (0x48, 8, (0, U64)),  # serial
)
# FILE record header (USA offset/count excluded: the re-fix keys on them).
REC_FIELDS = (
    (0x08, 8, (0, U64)),  # $LogFile LSN
    (0x10, 2, (0, U16)),  # sequence
    (0x12, 2, (0, 0x7FFF, U16)),  # link count
    (0x14, 2, (0, 0x18, 0x3F8, 0x3FC, 0x400, U16)),  # first attribute
    (0x16, 2, (0, 2, 3, 0x0C, U16)),  # flags: in use / dir / ext / view
    (0x18, 4, (0, 0x38, 0x3FF, 0x401, U32)),  # bytes used
    (0x1C, 4, (0, 0x200, 0x7FFF, U32)),  # bytes allocated
    (0x20, 8, (1, 0x0001_0000_0000_0000, U64)),  # base record reference
    (0x28, 2, (0, U16)),  # next attribute id
)
# Attribute header: common, then resident or non-resident tails.
ATTR_COMMON = (
    (0x00, 4, (0, 0x10, 0x20, 0x30, 0x80, 0xA0, 0xB0, 0x100, 0x1000, U32)),  # type
    (0x04, 4, _LEN_EDGES),  # record length
    (0x08, 1, (0, 1, 0xFF)),  # non-resident flag
    (0x09, 1, (0, 1, 0x7F, 0xFF)),  # name length (UTF-16 units)
    (0x0A, 2, (0, 0x3FE, U16)),  # name offset
    (0x0C, 2, (0x0001, 0x4000, 0x8000, U16)),  # flags: compressed/encrypted/sparse
    (0x0E, 2, (0, U16)),  # attribute id
)
ATTR_RESIDENT = (
    (0x10, 4, (0, 0x3E8, 0x7FFFFFFF, U32)),  # value length
    (0x14, 2, (0, 0x17, 0x3FF, U16)),  # value offset
    (0x16, 1, (1, 0xFF)),  # indexed flag
)
ATTR_NONRES = (
    (0x10, 8, (0, 1, U32, U64)),  # starting VCN
    (0x18, 8, (0, U32, 1 << 63, U64)),  # last VCN
    (0x20, 2, (0, 0x3F, 0x3FF, U16)),  # mapping pairs offset
    (0x22, 2, (4, 0x10, U16)),  # compression unit
    (0x28, 8, (0, 1, U64)),  # allocated size
    (0x30, 8, (0, 1, 1 << 63, U64)),  # data size
    (0x38, 8, (0, 1, 1 << 63, U64)),  # initialized size
)
# (base, offset, width, edges): base 0 = record-relative, 1 = first index entry.
INDX_FIELDS = (
    (0, 0x10, 8, (1, U32, U64)),  # VCN of this block
    (0, 0x18, 4, (0, 8, 0x3FF, U32)),  # entries offset
    (0, 0x1C, 4, (0, 0x10, 0x7FFF, U32)),  # index size
    (0, 0x20, 4, (0, 0x10, U32)),  # allocated size
    (0, 0x24, 1, (1, 0xFF)),  # has-children flag
    (1, 0x00, 8, (0, U64)),  # file reference
    (1, 0x08, 2, (0, 8, 0x0F, 0x3FF, U16)),  # entry length
    (1, 0x0A, 2, (0x3FF, 0x7FFF, U16)),  # key length
    (1, 0x0C, 2, (1, 2, 3, U16)),  # flags: sub-node / last
)


_T = TypeVar("_T")
_Field = tuple[int, int, tuple[int, ...]]


def _choose(rng: RandPool, seq: tuple[_T, ...]) -> _T:
    """``rng.choice`` narrowed to the element type (same single draw)."""
    return cast(_T, rng.choice(seq))


class RunPos(IntEnum):
    """Mapping-pair byte to overwrite in the first run."""

    HEADER = 0
    LENGTH = 1
    OFFSET = 2
    TERM = 3


# Header: nibbles > 8 bytes, zero length size, 8/8; length/offset bytes:
# zero, sign-flipping, negative delta; terminator replaced by a new run.
RUN_EDGES = {
    RunPos.HEADER: (0xFF, 0x0F, 0x01, 0x10, 0x88, 0x9F),
    RunPos.LENGTH: (0, 0x7F, 0x80, 0xFF),
    RunPos.OFFSET: (0, 0x7F, 0x80, 0xFF),
    RunPos.TERM: (0x11, 0x01, 0x48, 0xFF),
}


class Op(IntEnum):
    """NTFS edit; ``choice`` over the live subset, in this order."""

    BOOT = 0
    REC_HEADER = 1
    ATTR = 2
    RUNLIST = 3
    FIXUP = 4
    INDX = 5


@dataclass(frozen=True)
class _Record:
    off: int
    size: int
    usa: int
    count: int
    magic: bytes


@dataclass(frozen=True)
class _Attr:
    rec: _Record
    pos: int  # record-relative
    length: int
    nonres: bool


def _record_at(data: bytes, off: int) -> _Record | None:
    """FILE/INDX record at *off* with a sane, in-bounds update sequence array."""
    magic = data[off : off + 4]
    if magic not in (FILE_MAGIC, INDX_MAGIC) or off + 8 > len(data):
        return None

    usa, count = struct.unpack_from("<HH", data, off + 4)
    if not USA_MIN_COUNT <= count <= USA_MAX_COUNT or usa < USA_MIN_OFF or usa & 1:
        return None
    size = (count - 1) * SECTOR
    if usa + 2 * count > size or off + size > len(data):
        return None
    return _Record(off, size, usa, count, bytes(magic))


def _records(data: bytes) -> list[_Record]:
    """Records at 512-byte-aligned offsets; a hit skips its own length."""
    recs: list[_Record] = []
    off, n = 0, len(data)
    while off < n and len(recs) < MAX_RECORDS:
        rec = _record_at(data, off)
        if rec is None:
            off += SECTOR
            continue
        recs.append(rec)
        off += rec.size
    return recs


def _has_boot(data: bytes) -> bool:
    return (
        len(data) >= BOOT_LEN
        and data[OEM_OFF : OEM_OFF + 8] == OEM_ID
        and data[BOOT_SIG_OFF:BOOT_LEN] == BOOT_SIG
    )


def _defix(data: bytes | bytearray, rec: _Record) -> bytearray:
    """Logical record: each sector tail restored from its USA slot."""
    out = bytearray(data[rec.off : rec.off + rec.size])
    for k in range(rec.count - 1):
        tail = k * SECTOR + SECTOR - 2
        slot = rec.usa + 2 + 2 * k
        out[tail : tail + 2] = data[rec.off + slot : rec.off + slot + 2]
    return out


def _refix(logical: bytearray, rec: _Record) -> bytearray:
    """Move each logical tail into the USA and stamp the USN (USA slot 0)."""
    out = bytearray(logical)
    usn = out[rec.usa : rec.usa + 2]
    for k in range(rec.count - 1):
        tail = k * SECTOR + SECTOR - 2
        slot = rec.usa + 2 + 2 * k
        out[slot : slot + 2] = logical[tail : tail + 2]
        out[tail : tail + 2] = usn
    return out


def _attrs(logical: bytes | bytearray, rec: _Record) -> list[_Attr]:
    """Attribute headers from the first-attribute offset to the end marker."""
    if rec.magic != FILE_MAGIC:
        return []

    out: list[_Attr] = []
    pos = struct.unpack_from("<H", logical, FIRST_ATTR_OFF)[0]
    while len(out) < MAX_ATTRS and pos + ATTR_MIN <= rec.size:
        type_code, length = struct.unpack_from("<II", logical, pos)
        if type_code == END_MARKER or length < ATTR_MIN or pos + length > rec.size:
            break
        out.append(_Attr(rec, pos, length, logical[pos + 8] != 0))
        pos += length
    return out


def _mapping(logical: bytes | bytearray, attr: _Attr) -> int | None:
    """Record-relative offset of the mapping pairs, if inside the attribute."""
    if not attr.nonres or attr.length < NONRES_MIN:
        return None
    rel = struct.unpack_from("<H", logical, attr.pos + MAPPING_OFF)[0]
    return attr.pos + rel if NONRES_MIN <= rel < attr.length else None


@dataclass
class _Scan:
    boot: bool
    files: list[_Record]
    indx: list[_Record]
    records: list[_Record]
    attr_recs: list[tuple[_Record, list[_Attr]]]
    runs: list[tuple[_Attr, int]]


def _scan(data: bytes) -> _Scan:
    records = _records(data)
    files = [r for r in records if r.magic == FILE_MAGIC]
    attr_recs, runs = [], []
    for rec in files:
        logical = _defix(data, rec)
        attrs = _attrs(logical, rec)
        if attrs:
            attr_recs.append((rec, attrs))
        for attr in attrs:
            mp = _mapping(logical, attr)
            if mp is not None:
                runs.append((attr, mp))
    indx = [r for r in records if r.magic == INDX_MAGIC]
    return _Scan(_has_boot(data), files, indx, records, attr_recs, runs)


def _live(scan: _Scan) -> tuple[Op, ...]:
    flags = (
        (Op.BOOT, scan.boot),
        (Op.REC_HEADER, scan.files),
        (Op.ATTR, scan.attr_recs),
        (Op.RUNLIST, scan.runs),
        (Op.FIXUP, scan.records),
        (Op.INDX, scan.indx),
    )
    return tuple(op for op, live in flags if live)


def available_ops(data: bytes) -> tuple[Op, ...]:
    """Ops whose target structure exists in *data*, in ``Op`` order."""
    return _live(_scan(data))


def sniff_ntfs(data: bytes) -> bool:
    """NTFS boot sector, or a FILE/INDX record with a sane fixup array."""
    return _has_boot(data) or bool(_records(data))


def _put(buf: bytearray, pos: int, width: int, value: int) -> None:
    """Little-endian store; flips the low bit if the value is already there."""
    value &= (1 << (8 * width)) - 1
    if int.from_bytes(buf[pos : pos + width], "little") == value:
        value ^= 1
    buf[pos : pos + width] = value.to_bytes(width, "little")


class _Editor:
    """One edit on *buf*; record edits run on the de-fixed copy."""

    def __init__(self, buf: bytearray, scan: _Scan, rng: RandPool) -> None:
        self.buf = buf
        self.scan = scan
        self.rng = rng

    def _idx(self, n: int) -> int:
        return self.rng.randint(0, n - 1)

    def _field(self, logical: bytearray, base: int, fields: tuple[_Field, ...]) -> None:
        off, width, edges = fields[self._idx(len(fields))]
        _put(logical, base + off, width, _choose(self.rng, edges))

    def _commit(self, rec: _Record, logical: bytearray) -> None:
        self.buf[rec.off : rec.off + rec.size] = _refix(logical, rec)

    def boot(self) -> None:
        """Draws: field, value."""
        self._field(self.buf, 0, BPB_FIELDS)

    def rec_header(self) -> None:
        """Draws: record, field, value."""
        rec = self.scan.files[self._idx(len(self.scan.files))]
        logical = _defix(self.buf, rec)
        self._field(logical, 0, REC_FIELDS)
        self._commit(rec, logical)

    def attr(self) -> None:
        """Draws: record, attribute, field, value."""
        rec, attrs = self.scan.attr_recs[self._idx(len(self.scan.attr_recs))]
        attr = attrs[self._idx(len(attrs))]
        tail = ATTR_NONRES if attr.nonres and attr.length >= NONRES_MIN else ATTR_RESIDENT
        logical = _defix(self.buf, rec)
        self._field(logical, attr.pos, ATTR_COMMON + tail)
        self._commit(rec, logical)

    def runlist(self) -> None:
        """Draws: non-resident attribute, run position, value."""
        attr, mp = self.scan.runs[self._idx(len(self.scan.runs))]
        where = RunPos(self._idx(len(RunPos)))
        logical = _defix(self.buf, attr.rec)

        head = logical[mp]
        len_size, off_size = head & 0x0F, head >> 4
        rel = {
            RunPos.HEADER: 0,
            RunPos.LENGTH: 1,
            RunPos.OFFSET: 1 + len_size,
            RunPos.TERM: 1 + len_size + off_size,
        }[where]
        pos = min(mp + rel, attr.pos + attr.length - 1)
        _put(logical, pos, 1, _choose(self.rng, RUN_EDGES[where]))
        self._commit(attr.rec, logical)

    def fixup(self) -> None:
        """Draws: record, sector. Tail no longer matches the USN: torn write."""
        rec = self.scan.records[self._idx(len(self.scan.records))]
        tail = rec.off + self._idx(rec.count - 1) * SECTOR + SECTOR - 2
        value = struct.unpack_from("<H", self.buf, tail)[0]
        struct.pack_into("<H", self.buf, tail, value ^ TAIL_FLIP)

    def indx(self) -> None:
        """Draws: record, field, value; entry fields key on the entries offset."""
        rec = self.scan.indx[self._idx(len(self.scan.indx))]
        logical = _defix(self.buf, rec)
        base, off, width, edges = INDX_FIELDS[self._idx(len(INDX_FIELDS))]
        if base:
            off += INDX_HDR + struct.unpack_from("<I", logical, INDX_HDR)[0]
        if off + width > rec.size:
            off, width, edges = INDX_FIELDS[1][1:]  # entry out of range: hit its offset
        _put(logical, off, width, _choose(self.rng, edges))
        self._commit(rec, logical)


_HANDLERS = {
    Op.BOOT: _Editor.boot,
    Op.REC_HEADER: _Editor.rec_header,
    Op.ATTR: _Editor.attr,
    Op.RUNLIST: _Editor.runlist,
    Op.FIXUP: _Editor.fixup,
    Op.INDX: _Editor.indx,
}


def mutate_ntfs(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One boot/record/attribute/runlist/fixup/INDX edit; None if not NTFS."""
    scan = _scan(data)
    ops = _live(scan)
    if not ops:
        return None

    buf = bytearray(data)
    _HANDLERS[_choose(rng, ops)](_Editor(buf, scan, rng))

    # A hostile USA overlapping the edited field can make the re-fix undo it.
    if buf == data and scan.records:
        buf[scan.records[0].off + SECTOR - 2] ^= 0xFF
    elif buf == data:
        buf[0] ^= 1
    return bytes(buf[:max_len])
