"""Field map: name the byte ranges of a known container format.

Crash explanation needs to say "``IHDR.width`` changed" rather than "byte 0x12
changed". This module turns raw bytes into an ordered list of named spans for
PNG, gzip, ZIP and RIFF.

The walkers here are NOT the mutators' parsers (``mutations/png.py`` and
friends). Those are built for mutation: they drop a chunk whose length runs
past the end, recompute CRCs on serialize, and so cannot report the malformed
values a crash is made of. A walker here must instead

* keep the offset of every byte it names,
* report the bytes that exist, never the lengths the file claims,
* never raise and always terminate: it runs on the crashing input.

Layout::

    bytes ──► sniff ──► walker ──► [FieldSpan, ...]   sorted, disjoint,
                                                       inside the input,
                                                       <= MAX_FIELDS

Valid files are covered byte for byte. Truncated or hostile ones get the
spans that fit; spans past ``MAX_FIELDS`` are dropped. Unknown formats map to
nothing -- no names are invented.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum
from typing import Literal, NamedTuple

MAX_FIELDS = 512  # per input; bounds a chunk flood
MAX_RIFF_DEPTH = 8  # LIST-in-LIST nesting followed
MAX_REPR_BYTES = 16  # raw bytes shown by span_repr
MAX_INT_BYTES = 8  # widest span read as an integer

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GZIP_MAGIC = b"\x1f\x8b"
ZIP_LOCAL_SIG = b"PK\x03\x04"
ZIP_CENTRAL_SIG = b"PK\x01\x02"
ZIP_EOCD_SIG = b"PK\x05\x06"
ZIP_DESC_SIG = b"PK\x07\x08"
RIFF_MAGIC = b"RIFF"

PNG_SIG_LEN = 8
PNG_LEN_LEN = 4
PNG_TYPE_LEN = 4
PNG_CRC_LEN = 4
PNG_CHUNK_IHDR = "IHDR"
PNG_CHUNK_IEND = "IEND"

GZIP_HEADER_LEN = 10
GZIP_TRAILER_LEN = 8
GZIP_FLAG_HCRC = 1 << 1
GZIP_FLAG_EXTRA = 1 << 2
GZIP_FLAG_NAME = 1 << 3
GZIP_FLAG_COMMENT = 1 << 4
ZIP_FLAG_DESCRIPTOR = 1 << 3

RIFF_HEADER_LEN = 12
RIFF_CHUNK_HDR_LEN = 8
RIFF_FORM_LEN = 4
RIFF_LIST_IDS = ("LIST", "RIFF")


class FieldKind(Enum):
    MAGIC = "magic"
    TAG = "tag"
    LENGTH = "length"
    COUNT = "count"
    OFFSET = "offset"
    CRC = "crc"
    FLAGS = "flags"
    VALUE = "value"
    DATA = "data"
    PADDING = "padding"
    UNKNOWN = "unknown"


class Endian(Enum):
    BIG = "big"
    LITTLE = "little"
    NONE = "none"  # not an integer: bytes, tags, single bytes


class FieldSpan(NamedTuple):
    offset: int
    width: int
    name: str
    kind: FieldKind
    endian: Endian = Endian.NONE


class FieldMap(NamedTuple):
    fmt: str  # "" when the format is not recognised
    spans: list[FieldSpan]


# (name, width, kind, endian) rows for the fixed part of a record.
_Layout = tuple[tuple[str, int, FieldKind, Endian], ...]

_BE = Endian.BIG
_LE = Endian.LITTLE
_NO = Endian.NONE

_PNG_IHDR: _Layout = (
    ("width", 4, FieldKind.VALUE, _BE),
    ("height", 4, FieldKind.VALUE, _BE),
    ("bit_depth", 1, FieldKind.VALUE, _NO),
    ("color_type", 1, FieldKind.VALUE, _NO),
    ("compression", 1, FieldKind.VALUE, _NO),
    ("filter", 1, FieldKind.VALUE, _NO),
    ("interlace", 1, FieldKind.VALUE, _NO),
)
_PNG_IHDR_LEN = sum(w for _n, w, _k, _e in _PNG_IHDR)

_GZIP_HEADER: _Layout = (
    ("magic", 2, FieldKind.MAGIC, _NO),
    ("method", 1, FieldKind.VALUE, _NO),
    ("flags", 1, FieldKind.FLAGS, _NO),
    ("mtime", 4, FieldKind.VALUE, _LE),
    ("xfl", 1, FieldKind.VALUE, _NO),
    ("os", 1, FieldKind.VALUE, _NO),
)

_ZIP_LOCAL: _Layout = (
    ("signature", 4, FieldKind.MAGIC, _NO),
    ("version", 2, FieldKind.VALUE, _LE),
    ("flags", 2, FieldKind.FLAGS, _LE),
    ("method", 2, FieldKind.VALUE, _LE),
    ("mtime", 2, FieldKind.VALUE, _LE),
    ("mdate", 2, FieldKind.VALUE, _LE),
    ("crc32", 4, FieldKind.CRC, _LE),
    ("comp_size", 4, FieldKind.LENGTH, _LE),
    ("uncomp_size", 4, FieldKind.LENGTH, _LE),
    ("name_len", 2, FieldKind.LENGTH, _LE),
    ("extra_len", 2, FieldKind.LENGTH, _LE),
)

_ZIP_CENTRAL: _Layout = (
    ("signature", 4, FieldKind.MAGIC, _NO),
    ("version_made", 2, FieldKind.VALUE, _LE),
    ("version_need", 2, FieldKind.VALUE, _LE),
    ("flags", 2, FieldKind.FLAGS, _LE),
    ("method", 2, FieldKind.VALUE, _LE),
    ("mtime", 2, FieldKind.VALUE, _LE),
    ("mdate", 2, FieldKind.VALUE, _LE),
    ("crc32", 4, FieldKind.CRC, _LE),
    ("comp_size", 4, FieldKind.LENGTH, _LE),
    ("uncomp_size", 4, FieldKind.LENGTH, _LE),
    ("name_len", 2, FieldKind.LENGTH, _LE),
    ("extra_len", 2, FieldKind.LENGTH, _LE),
    ("comment_len", 2, FieldKind.LENGTH, _LE),
    ("disk_start", 2, FieldKind.VALUE, _LE),
    ("int_attr", 2, FieldKind.FLAGS, _LE),
    ("ext_attr", 4, FieldKind.FLAGS, _LE),
    ("local_offset", 4, FieldKind.OFFSET, _LE),
)

_ZIP_EOCD: _Layout = (
    ("signature", 4, FieldKind.MAGIC, _NO),
    ("disk", 2, FieldKind.VALUE, _LE),
    ("cd_disk", 2, FieldKind.VALUE, _LE),
    ("entries_disk", 2, FieldKind.COUNT, _LE),
    ("entries_total", 2, FieldKind.COUNT, _LE),
    ("cd_size", 4, FieldKind.LENGTH, _LE),
    ("cd_offset", 4, FieldKind.OFFSET, _LE),
    ("comment_len", 2, FieldKind.LENGTH, _LE),
)

_ZIP_DESC: _Layout = (
    ("signature", 4, FieldKind.MAGIC, _NO),
    ("crc32", 4, FieldKind.CRC, _LE),
    ("comp_size", 4, FieldKind.LENGTH, _LE),
    ("uncomp_size", 4, FieldKind.LENGTH, _LE),
)

_ZIP_SIG_RE = re.compile(rb"PK(?:\x03\x04|\x01\x02|\x05\x06|\x07\x08)")


class _Spans:
    """Append-only span list, clamped to the input and capped.

    ``add`` returns the number of bytes it actually covered, so a walker
    advances by what exists: a length field that lies about the size of the
    rest of the file cannot push a walker past the end.
    """

    def __init__(self, size: int):
        self.size = size
        self.spans: list[FieldSpan] = []

    @property
    def full(self) -> bool:
        return len(self.spans) >= MAX_FIELDS

    def add(self, offset: int, width: int, name: str, kind: FieldKind, endian: Endian = _NO) -> int:
        if self.full:
            return 0

        width = min(width, self.size - offset)
        if width <= 0:
            return 0

        self.spans.append(FieldSpan(offset, width, name, kind, endian))
        return width


def _to_int(raw: bytes, endian: Endian) -> int:
    order: Literal["big", "little"] = "big" if endian is Endian.BIG else "little"
    return int.from_bytes(raw, order)


def _uint(data: bytes, offset: int, width: int, endian: Endian) -> int | None:
    """Integer at *offset*, or None when the field is not fully present."""
    if offset + width > len(data):
        return None
    return _to_int(data[offset : offset + width], endian)


def _tag_text(raw: bytes) -> str:
    """Printable name for a 4-byte tag; hex when it is not text."""
    text = raw.decode("latin-1")
    if text.isascii() and text.isprintable() and text.strip():
        return text.strip()
    return "x" + raw.hex()


def _fixed(
    s: _Spans, data: bytes, pos: int, prefix: str, layout: _Layout
) -> tuple[int, dict[str, int]]:
    """Add the fixed-width fields of a record.

    Returns the position after the record and the integer value of each
    field that was fully present, for the size fields that follow it.
    """
    values: dict[str, int] = {}
    for name, width, kind, endian in layout:
        value = _uint(data, pos, width, endian) if endian is not _NO else None
        if value is not None:
            values[name] = value

        pos += s.add(pos, width, f"{prefix}{name}", kind, endian)
    return pos, values


# ── PNG ────────────────────────────────────────────────────────────────


def _sniff_png(data: bytes) -> bool:
    return data[:PNG_SIG_LEN] == PNG_MAGIC


def _png_chunk(s: _Spans, data: bytes, pos: int, idx: int) -> int:
    """Map one chunk at *pos*; return the next position (len(data) if cut)."""
    n = len(data)
    tag = _tag_text(data[pos + PNG_LEN_LEN : pos + PNG_LEN_LEN + PNG_TYPE_LEN])
    base = f"{tag}[{idx}]"

    length = _uint(data, pos, PNG_LEN_LEN, _BE)
    got = s.add(pos, PNG_LEN_LEN, f"{base}.length", FieldKind.LENGTH, _BE)
    if length is None or got < PNG_LEN_LEN:
        return n

    got = s.add(pos + PNG_LEN_LEN, PNG_TYPE_LEN, f"{base}.type", FieldKind.TAG)
    if got < PNG_TYPE_LEN:
        return n

    body = pos + PNG_LEN_LEN + PNG_TYPE_LEN
    avail = min(length, n - body)
    if tag == PNG_CHUNK_IHDR and avail >= _PNG_IHDR_LEN:
        after, _vals = _fixed(s, data, body, f"{base}.", _PNG_IHDR)
        s.add(after, avail - _PNG_IHDR_LEN, f"{base}.extra", FieldKind.DATA)
    else:
        s.add(body, avail, f"{base}.data", FieldKind.DATA)

    if avail < length:
        return n

    crc_at = body + length
    if s.add(crc_at, PNG_CRC_LEN, f"{base}.crc", FieldKind.CRC, _BE) < PNG_CRC_LEN:
        return n
    return crc_at + PNG_CRC_LEN


def _walk_png(data: bytes) -> list[FieldSpan]:
    s = _Spans(len(data))
    s.add(0, PNG_SIG_LEN, "signature", FieldKind.MAGIC)

    pos, idx = PNG_SIG_LEN, 0
    while pos < len(data) and not s.full:
        is_end = _tag_text(data[pos + 4 : pos + 8]) == PNG_CHUNK_IEND
        pos = _png_chunk(s, data, pos, idx)
        idx += 1
        if is_end:
            break

    s.add(pos, len(data) - pos, "trailing", FieldKind.DATA)
    return s.spans


# ── gzip ───────────────────────────────────────────────────────────────


def _sniff_gzip(data: bytes) -> bool:
    # Looser than mutations.recompress.sniff_gzip (>= 18 bytes, CM == 8): a
    # mutated header with a corrupted method byte is exactly the input that
    # has to be mapped.
    return data[: len(GZIP_MAGIC)] == GZIP_MAGIC


def _gzip_zstring(s: _Spans, data: bytes, pos: int, name: str) -> int:
    """Add a NUL-terminated field (terminator included); return next pos."""
    end = data.find(b"\x00", pos)
    end = len(data) if end < 0 else end + 1
    return pos + s.add(pos, end - pos, name, FieldKind.DATA)


def _walk_gzip(data: bytes) -> list[FieldSpan]:
    s = _Spans(len(data))
    _fixed_end, _vals = _fixed(s, data, 0, "", _GZIP_HEADER)
    pos = GZIP_HEADER_LEN
    flags = data[3] if len(data) > 3 else 0

    if flags & GZIP_FLAG_EXTRA:
        xlen = _uint(data, pos, 2, _LE)
        pos += s.add(pos, 2, "xlen", FieldKind.LENGTH, _LE)
        if xlen is not None:
            pos += s.add(pos, xlen, "extra", FieldKind.DATA)
    if flags & GZIP_FLAG_NAME:
        pos = _gzip_zstring(s, data, pos, "fname")
    if flags & GZIP_FLAG_COMMENT:
        pos = _gzip_zstring(s, data, pos, "fcomment")
    if flags & GZIP_FLAG_HCRC:
        pos += s.add(pos, 2, "header_crc", FieldKind.CRC, _LE)

    tail = max(pos, len(data) - GZIP_TRAILER_LEN)
    s.add(pos, tail - pos, "deflate", FieldKind.DATA)
    s.add(tail, 4, "crc32", FieldKind.CRC, _LE)
    s.add(tail + 4, 4, "isize", FieldKind.LENGTH, _LE)
    return s.spans


# ── ZIP ────────────────────────────────────────────────────────────────


def _sniff_zip(data: bytes) -> bool:
    return data[:4] in (ZIP_LOCAL_SIG, ZIP_EOCD_SIG)


def _next_zip_sig(data: bytes, pos: int) -> int:
    m = _ZIP_SIG_RE.search(data, pos)
    return m.start() if m else len(data)


def _zip_names(s: _Spans, pos: int, prefix: str, sizes: tuple[tuple[str, int], ...]) -> int:
    for name, size in sizes:
        pos += s.add(pos, size, f"{prefix}{name}", FieldKind.DATA)
    return pos


def _zip_local(s: _Spans, data: bytes, pos: int, idx: int) -> int:
    p = f"lfh[{idx}]."
    pos, v = _fixed(s, data, pos, p, _ZIP_LOCAL)
    pos = _zip_names(s, pos, p, (("name", v.get("name_len", 0)), ("extra", v.get("extra_len", 0))))

    size = v.get("comp_size", 0)
    unsized = v.get("flags", 0) & ZIP_FLAG_DESCRIPTOR and size == 0
    if unsized:
        # Size lives in the data descriptor: the data runs to the next record.
        size = _next_zip_sig(data, pos) - pos
    return pos + s.add(pos, size, f"{p}data", FieldKind.DATA)


def _zip_central(s: _Spans, data: bytes, pos: int, idx: int) -> int:
    p = f"cdh[{idx}]."
    pos, v = _fixed(s, data, pos, p, _ZIP_CENTRAL)
    sizes = (
        ("name", v.get("name_len", 0)),
        ("extra", v.get("extra_len", 0)),
        ("comment", v.get("comment_len", 0)),
    )
    return _zip_names(s, pos, p, sizes)


def _zip_eocd(s: _Spans, data: bytes, pos: int, idx: int) -> int:
    pos, v = _fixed(s, data, pos, "eocd.", _ZIP_EOCD)
    return _zip_names(s, pos, "eocd.", (("comment", v.get("comment_len", 0)),))


def _zip_desc(s: _Spans, data: bytes, pos: int, idx: int) -> int:
    return _fixed(s, data, pos, f"dd[{idx}].", _ZIP_DESC)[0]


_ZIP_RECORDS: dict[bytes, Callable[[_Spans, bytes, int, int], int]] = {
    ZIP_LOCAL_SIG: _zip_local,
    ZIP_CENTRAL_SIG: _zip_central,
    ZIP_EOCD_SIG: _zip_eocd,
    ZIP_DESC_SIG: _zip_desc,
}


def _walk_zip(data: bytes) -> list[FieldSpan]:
    s = _Spans(len(data))
    counts: dict[bytes, int] = {}
    gaps = 0

    pos = 0
    while pos < len(data) and not s.full:
        sig = data[pos : pos + 4]
        record = _ZIP_RECORDS.get(sig)
        if record is None:
            # Bytes between records: skip to the next signature.
            end = _next_zip_sig(data, pos)
            pos += s.add(pos, end - pos, f"gap[{gaps}]", FieldKind.DATA)
            gaps += 1
            continue

        idx = counts.get(sig, 0)
        counts[sig] = idx + 1
        pos = record(s, data, pos, idx)
    return s.spans


# ── RIFF ───────────────────────────────────────────────────────────────


def _sniff_riff(data: bytes) -> bool:
    return len(data) >= RIFF_HEADER_LEN and data[:4] == RIFF_MAGIC


def _riff_list(s: _Spans, data: bytes, pos: int, end: int, path: str, depth: int) -> int:
    """Map the chunks in ``data[pos:end]``; return the position reached."""
    idx = 0
    while pos + RIFF_CHUNK_HDR_LEN <= end and not s.full:
        cid = _tag_text(data[pos : pos + 4])
        base = f"{path}{cid}[{idx}]"
        idx += 1

        size = _uint(data, pos + 4, 4, _LE) or 0
        s.add(pos, 4, f"{base}.id", FieldKind.TAG)
        s.add(pos + 4, 4, f"{base}.size", FieldKind.LENGTH, _LE)

        body = pos + RIFF_CHUNK_HDR_LEN
        avail = min(size, end - body)
        is_list = cid in RIFF_LIST_IDS and avail >= RIFF_FORM_LEN and depth < MAX_RIFF_DEPTH
        if is_list:
            s.add(body, RIFF_FORM_LEN, f"{base}.form", FieldKind.TAG)
            _riff_list(s, data, body + RIFF_FORM_LEN, body + avail, f"{base}/", depth + 1)
        else:
            s.add(body, avail, f"{base}.data", FieldKind.DATA)

        if avail < size:
            return end

        pad = size & 1
        if pad and body + size < end:
            s.add(body + size, 1, f"{base}.pad", FieldKind.PADDING)
        pos = body + size + pad

    if pos < end:
        s.add(pos, end - pos, f"{path}tail", FieldKind.DATA)
    return end


def _walk_riff(data: bytes) -> list[FieldSpan]:
    s = _Spans(len(data))
    s.add(0, 4, "riff.id", FieldKind.MAGIC)
    s.add(4, 4, "riff.size", FieldKind.LENGTH, _LE)
    s.add(8, RIFF_FORM_LEN, "riff.form", FieldKind.TAG)

    # riff.size is not trusted: chunks are walked to the end of the input,
    # which is what a lenient reader sees when the size field is mutated.
    _riff_list(s, data, RIFF_HEADER_LEN, len(data), "", 0)
    return s.spans


# ── registry ───────────────────────────────────────────────────────────

_FORMATS: tuple[tuple[str, Callable[[bytes], bool], Callable[[bytes], list[FieldSpan]]], ...] = (
    ("png", _sniff_png, _walk_png),
    ("gzip", _sniff_gzip, _walk_gzip),
    ("zip", _sniff_zip, _walk_zip),
    ("riff", _sniff_riff, _walk_riff),
)


def map_fields(data: bytes) -> FieldMap:
    """Name the fields of *data*; ``FieldMap("", [])`` if the format is unknown."""
    for fmt, sniff, walk in _FORMATS:
        if sniff(data):
            return FieldMap(fmt, walk(data))
    return FieldMap("", [])


def span_int(data: bytes, span: FieldSpan) -> int | None:
    """The integer held by *span*, or None if it is not an integer field."""
    if span.endian is _NO or span.width > MAX_INT_BYTES:
        return None
    return _to_int(data[span.offset : span.offset + span.width], span.endian)


def span_repr(data: bytes, span: FieldSpan) -> str:
    """Compact text for the bytes of *span*: fixed-width hex for integers."""
    value = span_int(data, span)
    if value is not None:
        return f"0x{value:0{span.width * 2}x}"

    raw = data[span.offset : span.offset + span.width]
    shown = raw[:MAX_REPR_BYTES].hex()
    return shown + "…" if len(raw) > MAX_REPR_BYTES else shown
