"""Structure-aware RAR mutations (RAR4 and RAR5).

Every RAR header carries a CRC over its own bytes, and libarchive drops a
header whose CRC mismatches before reading any field. Flat byte mutations
therefore almost never get past the first header. These mutators edit one
field and repair the CRC, so the edit reaches the parser.

Layouts::

    RAR5  "Rar!\\x1a\\x07\\x01\\x00" then per header:
          crc32 u32 | hsize vint | type vint | flags vint
          | [extra size vint] | [data size vint] | type fields | extra area
          followed by <data size> bytes of data area.
          crc32 covers hsize vint + header body.

    RAR4  "Rar!\\x1a\\x07\\x00" (itself the marker block) then per block:
          crc16 u16 | type u8 | flags u16 | size u16 | [add size u32] | ...
          crc16 = low 16 bits of crc32 over type..end of header.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

RAR5_SIG = b"Rar!\x1a\x07\x01\x00"
RAR4_SIG = b"Rar!\x1a\x07\x00"

_VINT_MAX_BYTES = 10
_VINT_BITS = 7
_VINT_MASK = 0x7F
_VINT_MORE = 0x80
_MAX_HEADERS = 4096
_MAX_FIELDS = 8
_CRC5_LEN = 4
_CRC4_LEN = 2

# RAR5 header flags / types.
_R5_HAS_EXTRA = 0x01
_R5_HAS_DATA = 0x02
_R5_MAIN, _R5_FILE, _R5_SERVICE, _R5_END = 1, 2, 3, 5
_R5_TYPES = (_R5_MAIN, _R5_FILE, _R5_END)
_R5_FILE_MTIME = 0x02
_R5_FILE_CRC = 0x04
_R5_ATTR_ARCHIVE = 0x20
_R5_HOST_UNIX = 1

# RAR4 block types / flags / layout.
_R4_MAIN, _R4_FILE, _R4_SUB, _R4_END = 0x73, 0x74, 0x7A, 0x7B
_R4_TYPES = (_R4_MAIN, _R4_FILE, _R4_SUB, _R4_END)
_R4_LONG_BLOCK = 0x8000
_R4_END_FLAGS = 0x4000
_R4_BASE = 7  # crc + type + flags + size
_R4_FLAGS_OFF = 3
_R4_SIZE_OFF = 5
_R4_ADD_OFF = 7
_R4_ADD_END = 11
_R4_MAIN_RESERVED = 6
_R4_UNP_VER = 29
_R4_METHOD_STORE = 0x30
_R4_FILE_FMT = "<IIBIIBBHI"  # pack, unp, host, crc, ftime, ver, method, name len, attr

# (offset from block start, struct format) of editable fixed-width fields.
_RAR4_COMMON = ((2, "<B"), (_R4_FLAGS_OFF, "<H"), (_R4_SIZE_OFF, "<H"), (_R4_ADD_OFF, "<I"))
_RAR4_FILE_FIELDS = (
    (11, "<I"),  # unpacked size
    (15, "<B"),  # host OS
    (16, "<I"),  # file CRC
    (24, "<B"),  # unpack version
    (25, "<B"),  # method
    (26, "<H"),  # name size
    (28, "<I"),  # attributes
)

# Boundary values: vint group edges, 32/63/64-bit limits.
_VINT_EDGES = (
    0,
    1,
    0x7F,
    0x80,
    0x3FFF,
    0x4000,
    0xFFFFFFFF,
    1 << 32,
    (1 << 63) - 1,
    1 << 63,
    (1 << 64) - 1,
)
# 0 = canonical; 10 = longest legal overlong; 11 = one byte past the limit.
_VINT_WIDTHS = (0, _VINT_MAX_BYTES, _VINT_MAX_BYTES + 1)

_GEN_MAX_PAYLOAD = 64
_GEN_NAME = b"a.txt"
_BLOCK_DUP, _BLOCK_DROP, _BLOCK_SWAP = 0, 1, 2


def encode_vint(value: int, width: int = 0) -> bytes:
    """RAR5 vint: 7-bit little-endian groups, 0x80 = more follows.

    *width* above the canonical length pads with continuation bytes, e.g.
    ``encode_vint(1, 3) == b"\\x81\\x80\\x00"`` (overlong, same value).
    """
    out = bytearray()
    while True:
        byte = value & _VINT_MASK
        value >>= _VINT_BITS
        if not value:
            out.append(byte)
            break
        out.append(byte | _VINT_MORE)

    while len(out) < width:
        out[-1] |= _VINT_MORE
        out.append(0)
    return bytes(out)


def decode_vint(data: bytes, pos: int, end: int) -> tuple[int, int] | None:
    """Return ``(value, next_pos)``, or None when truncated or over 10 bytes."""
    value = 0
    shift = 0
    limit = min(end, len(data), pos + _VINT_MAX_BYTES)
    i = pos
    while i < limit:
        byte = data[i]
        value |= (byte & _VINT_MASK) << shift
        i += 1
        if not byte & _VINT_MORE:
            return value, i
        shift += _VINT_BITS
    return None


@dataclass
class RarBlock:
    """One header plus its data area. Offsets are absolute.

    ``start`` is the CRC field; ``body`` the first CRC'd field after the
    size (RAR5: type vint; RAR4: type byte); ``end`` the header end;
    ``data_end`` the end of the data area (clamped to the buffer).
    """

    start: int
    body: int
    end: int
    data_end: int
    htype: int
    fields: list[tuple[int, int]] = field(default_factory=list)
    extra_slice: tuple[int, int] | None = None
    extra_size: int = 0
    data_slice: tuple[int, int] | None = None
    rest: int = 0


@dataclass
class RarArchive:
    version: int
    blocks: list[RarBlock]


def _vint_slice(data: bytes, pos: int, end: int) -> tuple[int, int, int] | None:
    """``(value, start, stop)`` of the vint at *pos*."""
    got = decode_vint(data, pos, end)
    if got is None:
        return None
    return got[0], pos, got[1]


def _rar5_common(data: bytes, blk: RarBlock) -> RarBlock | None:
    """Fill type/flags/extra/data vint slices of a RAR5 header."""
    htype = _vint_slice(data, blk.body, blk.end)
    if htype is None:
        return None
    flags = _vint_slice(data, htype[2], blk.end)
    if flags is None:
        return None

    blk.htype = htype[0]
    blk.fields = [(htype[1], htype[2]), (flags[1], flags[2])]
    pos = flags[2]
    if flags[0] & _R5_HAS_EXTRA:
        extra = _vint_slice(data, pos, blk.end)
        if extra is None:
            return None
        blk.extra_size, blk.extra_slice, pos = extra[0], extra[1:], extra[2]
        blk.fields.append(blk.extra_slice)

    if flags[0] & _R5_HAS_DATA:
        size = _vint_slice(data, pos, blk.end)
        if size is None:
            return None
        blk.data_slice, pos = size[1:], size[2]
        blk.fields.append(blk.data_slice)
        blk.data_end = min(len(data), blk.end + size[0])

    blk.rest = pos
    return blk


def _parse_rar5(data: bytes) -> list[RarBlock]:
    blocks: list[RarBlock] = []
    n = len(data)
    pos = len(RAR5_SIG)
    while pos + _CRC5_LEN < n and len(blocks) < _MAX_HEADERS:
        size = decode_vint(data, pos + _CRC5_LEN, n)
        if size is None or size[0] == 0:
            break
        end = size[1] + size[0]
        if end > n:
            break

        blk = _rar5_common(data, RarBlock(pos, size[1], end, end, 0))
        if blk is None:
            break
        blocks.append(blk)
        pos = blk.data_end
    return blocks


def _parse_rar4(data: bytes) -> list[RarBlock]:
    blocks: list[RarBlock] = []
    n = len(data)
    pos = len(RAR4_SIG)
    while pos + _R4_BASE <= n and len(blocks) < _MAX_HEADERS:
        htype, flags, size = struct.unpack_from("<BHH", data, pos + _CRC4_LEN)
        end = pos + size
        if size < _R4_BASE or end > n:
            break

        add = 0
        has_add = flags & _R4_LONG_BLOCK or htype == _R4_FILE
        if has_add and size >= _R4_ADD_END:
            add = struct.unpack_from("<I", data, pos + _R4_ADD_OFF)[0]
        data_end = min(n, end + add)
        blocks.append(RarBlock(pos, pos + _CRC4_LEN, end, data_end, htype))
        pos = data_end
    return blocks


def parse_rar(data: bytes) -> RarArchive | None:
    """Walk RAR4/RAR5 headers; None when the signature is absent."""
    if data.startswith(RAR5_SIG):
        return RarArchive(5, _parse_rar5(data))
    if data.startswith(RAR4_SIG):
        return RarArchive(4, _parse_rar4(data))
    return None


def _rar5_header(body: bytes) -> bytes:
    """crc32 | hsize vint | body, with hsize and CRC derived from *body*."""
    hsize = encode_vint(len(body))
    crc = zlib.crc32(hsize + body)
    return struct.pack("<I", crc) + hsize + body


def _rar4_header(htype: int, flags: int, body: bytes) -> bytes:
    rest = struct.pack("<BHH", htype, flags, _R4_BASE + len(body)) + body
    crc = zlib.crc32(rest) & 0xFFFF
    return struct.pack("<H", crc) + rest


def _fix_crc4(buf: bytearray, start: int) -> None:
    """Recompute a RAR4 CRC over the block's *declared* size."""
    size = struct.unpack_from("<H", buf, start + _R4_SIZE_OFF)[0]
    end = min(len(buf), start + max(size, _R4_BASE))
    crc = zlib.crc32(buf[start + _CRC4_LEN : end]) & 0xFFFF
    struct.pack_into("<H", buf, start, crc)


def _put4(data: bytes, blk: RarBlock, off: int, fmt: str, value: int) -> bytes:
    """Write a fixed-width RAR4 field, then repair the block CRC."""
    width = struct.calcsize(fmt)
    if blk.start + off + width > blk.end:
        return data

    buf = bytearray(data)
    struct.pack_into(fmt, buf, blk.start + off, value & ((1 << (8 * width)) - 1))
    _fix_crc4(buf, blk.start)
    return bytes(buf)


def _rewrite(data: bytes, blk: RarBlock, span: tuple[int, int], new: bytes) -> bytes:
    """Replace a RAR5 header slice; re-derive hsize, extra size and CRC."""
    s, e = span
    body = bytearray(data[blk.body : s] + new + data[e : blk.end])
    inside_extra = blk.extra_slice is not None and s >= blk.end - blk.extra_size
    if inside_extra:
        xs, xe = blk.extra_slice
        delta = len(new) - (e - s)
        body[xs - blk.body : xe - blk.body] = encode_vint(blk.extra_size + delta)
    return data[: blk.start] + _rar5_header(bytes(body)) + data[blk.end :]


def _file_slices(data: bytes, blk: RarBlock) -> list[tuple[int, int]]:
    """Vint slices of a RAR5 file/service header's type-specific fields."""
    out: list[tuple[int, int]] = []
    pos = blk.rest
    flags = 0
    for idx in range(6):  # flags, unp size, attr, comp info, host, name len
        got = _vint_slice(data, pos, blk.end)
        if got is None:
            break
        out.append(got[1:])
        pos = got[2]
        if idx == 0:
            flags = got[0]
        if idx != 2:
            continue
        # mtime / data CRC u32s sit between attributes and comp info.
        pos += 4 * bool(flags & _R5_FILE_MTIME) + 4 * bool(flags & _R5_FILE_CRC)
    return out


def _extra_slices(data: bytes, blk: RarBlock) -> list[tuple[int, int]]:
    """Size/type vint slices of RAR5 extra-area records."""
    out: list[tuple[int, int]] = []
    pos = blk.end - blk.extra_size
    while pos < blk.end and len(out) < _MAX_FIELDS:
        size = _vint_slice(data, pos, blk.end)
        if size is None:
            break
        rtype = _vint_slice(data, size[2], blk.end)
        out.append(size[1:])
        if rtype is None:
            break
        out.append(rtype[1:])
        pos = size[2] + max(size[0], 1)
    return out


def _has_extra(blk: RarBlock) -> bool:
    return blk.extra_slice is not None and 0 < blk.extra_size <= blk.end - blk.rest


def _build_rar5(payload: bytes) -> bytes:
    """Main + stored file (with data CRC) + end-of-archive."""
    main = _rar5_header(encode_vint(_R5_MAIN) + encode_vint(0) + encode_vint(0))
    size = encode_vint(len(payload))
    file_body = (
        encode_vint(_R5_FILE)
        + encode_vint(_R5_HAS_DATA)
        + size
        + encode_vint(_R5_FILE_CRC)
        + size
        + encode_vint(_R5_ATTR_ARCHIVE)
        + struct.pack("<I", zlib.crc32(payload))
        + encode_vint(0)  # compression info: store, version 0
        + encode_vint(_R5_HOST_UNIX)
        + encode_vint(len(_GEN_NAME))
        + _GEN_NAME
    )
    end = _rar5_header(encode_vint(_R5_END) + encode_vint(0) + encode_vint(0))
    return RAR5_SIG + main + _rar5_header(file_body) + payload + end


def _build_rar4(payload: bytes) -> bytes:
    """Main + stored file + end-of-archive."""
    main = _rar4_header(_R4_MAIN, 0, bytes(_R4_MAIN_RESERVED))
    n = len(payload)
    fields = struct.pack(
        _R4_FILE_FMT,
        n,
        n,
        0,
        zlib.crc32(payload),
        0,
        _R4_UNP_VER,
        _R4_METHOD_STORE,
        len(_GEN_NAME),
        _R5_ATTR_ARCHIVE,
    )
    file_hdr = _rar4_header(_R4_FILE, _R4_LONG_BLOCK, fields + _GEN_NAME)
    return RAR4_SIG + main + file_hdr + payload + _rar4_header(_R4_END, _R4_END_FLAGS, b"")


def _insert_body(version: int, htype: int) -> bytes:
    """A minimal, CRC-valid header of *htype*."""
    if version == 4:
        pad = struct.calcsize(_R4_FILE_FMT) if htype == _R4_FILE else 0
        return _rar4_header(htype, 0, bytes(pad))
    tail = encode_vint(0)
    if htype == _R5_FILE:
        tail = bytes(5) + encode_vint(1) + b"a"  # zero fields, name "a"
    return _rar5_header(encode_vint(htype) + encode_vint(0) + tail)


class RarMutator:
    """Structure-aware RAR mutator: header fields with CRC repair."""

    def __init__(self, seed=None):
        # One pool per mutator; a caller-owned pool passed as ``rng=`` wins
        # for that call (Hard Rule 16).
        self._rng = RandPool(seed=seed)

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one RAR-specific mutation."""
        self._rng = rng or self._rng
        arc = parse_rar(data)
        if arc is None or not arc.blocks:
            return self._generate_random_rar(max_len=max_len, rng=self._rng)

        menu = (
            self._hdr_field,
            self._data_size,
            self._file_field,
            self._extra_rec,
            self._corrupt_crc,
            self._block_op,
            self._insert_hdr,
            lambda _data, _arc: self._generate_random_rar(max_len=max_len, rng=self._rng),
        )
        op = self._rng.randint(0, len(menu) - 1)
        return menu[op](data, arc)[:max_len]

    def _edge_vint(self) -> bytes:
        rng = self._rng
        return encode_vint(rng.choice(_VINT_EDGES), rng.choice(_VINT_WIDTHS))

    def _hdr_field(self, data: bytes, arc: RarArchive) -> bytes:
        """Common field (type/flags/sizes) to a boundary value, CRC repaired."""
        blk = self._rng.choice(arc.blocks)
        if arc.version == 4:
            off, fmt = self._rng.choice(_RAR4_COMMON)
            return _put4(data, blk, off, fmt, self._rng.choice(_VINT_EDGES))

        span = self._rng.choice(blk.fields)
        return _rewrite(data, blk, span, self._edge_vint())

    def _data_size(self, data: bytes, arc: RarArchive) -> bytes:
        """Data/pack size vs. bytes actually left: 0, ±1, 2^32, 2^63."""
        if arc.version == 4:
            cands = [b for b in arc.blocks if b.end - b.start >= _R4_ADD_END]
        else:
            cands = [b for b in arc.blocks if b.data_slice is not None]
        if not cands:
            return data

        blk = self._rng.choice(cands)
        remaining = len(data) - blk.end
        edges = (0, max(remaining - 1, 0), remaining + 1, 1 << 32, 1 << 63)
        value = self._rng.choice(edges)
        if arc.version == 5:
            return _rewrite(data, blk, blk.data_slice, encode_vint(value))

        flags = struct.unpack_from("<H", data, blk.start + _R4_FLAGS_OFF)[0] | _R4_LONG_BLOCK
        out = _put4(data, blk, _R4_FLAGS_OFF, "<H", flags)
        return _put4(out, blk, _R4_ADD_OFF, "<I", value)

    def _file_field(self, data: bytes, arc: RarArchive) -> bytes:
        """File header field (sizes, attrs, method, name length)."""
        kinds = (_R4_FILE,) if arc.version == 4 else (_R5_FILE, _R5_SERVICE)
        files = [b for b in arc.blocks if b.htype in kinds]
        if not files:
            return data

        blk = self._rng.choice(files)
        if arc.version == 4:
            off, fmt = self._rng.choice(_RAR4_FILE_FIELDS)
            return _put4(data, blk, off, fmt, self._rng.choice(_VINT_EDGES))

        slices = _file_slices(data, blk)
        if not slices:
            return data
        return _rewrite(data, blk, self._rng.choice(slices), self._edge_vint())

    def _extra_rec(self, data: bytes, arc: RarArchive) -> bytes:
        """RAR5 extra-area record size/type; RAR4 has none -> file field."""
        if arc.version == 4:
            return self._file_field(data, arc)

        extras = [b for b in arc.blocks if _has_extra(b)]
        if not extras:
            return data
        blk = self._rng.choice(extras)
        slices = _extra_slices(data, blk)
        if not slices:
            return data
        return _rewrite(data, blk, self._rng.choice(slices), self._edge_vint())

    def _corrupt_crc(self, data: bytes, arc: RarArchive) -> bytes:
        """Deliberately break one header CRC (the reject path)."""
        blk = self._rng.choice(arc.blocks)
        buf = bytearray(data)
        buf[blk.start] ^= self._rng.randint(1, 0xFF)
        return bytes(buf)

    def _block_op(self, data: bytes, arc: RarArchive) -> bytes:
        """Duplicate, drop or swap whole header+data blocks."""
        blocks = arc.blocks
        chunks = [data[b.start : b.data_end] for b in blocks]
        prefix = data[: blocks[0].start]
        suffix = data[blocks[-1].data_end :]

        kind = self._rng.randint(0, _BLOCK_SWAP)
        i = self._rng.randint(0, len(chunks) - 1)
        if kind == _BLOCK_SWAP and len(chunks) > 1:
            j = (i + self._rng.randint(1, len(chunks) - 1)) % len(chunks)
            chunks[i], chunks[j] = chunks[j], chunks[i]
        elif kind == _BLOCK_DROP:
            del chunks[i]
        else:
            chunks.insert(i + 1, chunks[i])
        return prefix + b"".join(chunks) + suffix

    def _insert_hdr(self, data: bytes, arc: RarArchive) -> bytes:
        """Insert a minimal CRC-valid header at a block boundary."""
        types = _R4_TYPES if arc.version == 4 else _R5_TYPES
        hdr = _insert_body(arc.version, self._rng.choice(types))
        bounds = [b.start for b in arc.blocks] + [arc.blocks[-1].data_end]
        pos = bounds[self._rng.randint(0, len(bounds) - 1)]
        return data[:pos] + hdr + data[pos:]

    def _generate_random_rar(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Stored single-file archive, RAR5 or RAR4, all CRCs valid."""
        self._rng = rng or self._rng
        size = self._rng.randint(0, _GEN_MAX_PAYLOAD)
        payload = self._rng.randbytes(size)
        build = _build_rar5 if self._rng.randint(0, 1) else _build_rar4
        return build(payload)[:max_len]


__all__ = [
    "RAR4_SIG",
    "RAR5_SIG",
    "RarArchive",
    "RarBlock",
    "RarMutator",
    "decode_vint",
    "encode_vint",
    "parse_rar",
]
