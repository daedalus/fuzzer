"""Structure-aware LZ4 frame mutator.

Input layout follows targets/lz4_read.c: byte 0 is a mode selector (even ->
LZ4F frame decode, odd -> raw block decode), the frame starts at offset 1.

  [mode:1][magic 04 22 4D 18][FLG][BD][content size:8?][dict id:4?][HC]
  { [block size:4 LE, bit31 = stored][data][block xxh32:4?] }*
  [EndMark 00 00 00 00][content xxh32:4?]

HC = (xxh32(FLG..dict id) >> 8) & 0xFF gates the whole decoder, block and
content checksums gate everything after, so most branches here mutate one
field and then repair every checksum the frame declares.

Checksum fields left as ``None`` on the model are recomputed at serialize
time; a parsed value is kept verbatim until a mutation invalidates it, so an
untouched frame costs no hashing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from enum import IntEnum
from importlib.util import find_spec
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

# ── xxh32 ──────────────────────────────────────────────────────────────────

_U32 = 0xFFFFFFFF
_U64 = (1 << 64) - 1
_P1 = 0x9E3779B1
_P2 = 0x85EBCA77
_P3 = 0xC2B2AE3D
_P4 = 0x27D4EB2F
_P5 = 0x165667B1
_STRIPE = 16
_WORD = 4


def _xxh32_py(data: bytes, seed: int = 0) -> int:
    """Pure-Python XXH32 (fallback when the C ``xxhash`` module is absent)."""
    n = len(data)
    pos = 0
    if n >= _STRIPE:
        v1 = (seed + _P1 + _P2) & _U32
        v2 = (seed + _P2) & _U32
        v3 = seed & _U32
        v4 = (seed - _P1) & _U32
        nw = (n // _STRIPE) * 4
        words = struct.unpack_from(f"<{nw}I", data)
        # Lanes inlined: a helper call per word costs ~2x on 4 KiB input.
        for i in range(0, nw, 4):
            v1 = (v1 + words[i] * _P2) & _U32
            v1 = (((v1 << 13) | (v1 >> 19)) * _P1) & _U32
            v2 = (v2 + words[i + 1] * _P2) & _U32
            v2 = (((v2 << 13) | (v2 >> 19)) * _P1) & _U32
            v3 = (v3 + words[i + 2] * _P2) & _U32
            v3 = (((v3 << 13) | (v3 >> 19)) * _P1) & _U32
            v4 = (v4 + words[i + 3] * _P2) & _U32
            v4 = (((v4 << 13) | (v4 >> 19)) * _P1) & _U32
        h = (
            ((v1 << 1) | (v1 >> 31))
            + ((v2 << 7) | (v2 >> 25))
            + ((v3 << 12) | (v3 >> 20))
            + ((v4 << 18) | (v4 >> 14))
        )
        pos = nw * _WORD
    else:
        h = seed + _P5
    h = (h + n) & _U32

    # Tail: 4-byte words, then single bytes.
    while pos + _WORD <= n:
        h = (h + struct.unpack_from("<I", data, pos)[0] * _P3) & _U32
        h = (((h << 17) | (h >> 15)) * _P4) & _U32
        pos += _WORD
    while pos < n:
        h = (h + data[pos] * _P5) & _U32
        h = (((h << 11) | (h >> 21)) * _P1) & _U32
        pos += 1

    # Avalanche.
    h ^= h >> 15
    h = (h * _P2) & _U32
    h ^= h >> 13
    h = (h * _P3) & _U32
    return h ^ (h >> 16)


# C xxhash is ~100x faster; the pure version keeps the module dependency-free.
if find_spec("xxhash") is not None:
    import xxhash as _xxhash

    def xxh32(data: bytes, seed: int = 0) -> int:
        """XXH32 of *data* (C ``xxhash`` backend)."""
        return _xxhash.xxh32_intdigest(data, seed)
else:
    xxh32 = _xxh32_py


# ── format constants ───────────────────────────────────────────────────────

LZ4_MAGIC = 0x184D2204
SKIPPABLE_MAGIC_BASE = 0x184D2A50
SKIPPABLE_NIBBLE_MAX = 0xF

FLG_VERSION = 0x40  # version 01 in bits 7-6
FLG_VERSION_MASK = 0xC0
FLG_B_INDEP = 0x20
FLG_B_CHECKSUM = 0x10
FLG_C_SIZE = 0x08
FLG_C_CHECKSUM = 0x04
FLG_RESERVED = 0x02
FLG_DICT_ID = 0x01
FLG_BITS = (
    0x80,
    FLG_VERSION,
    FLG_B_INDEP,
    FLG_B_CHECKSUM,
    FLG_C_SIZE,
    FLG_C_CHECKSUM,
    FLG_RESERVED,
    FLG_DICT_ID,
)

BD_SHIFT = 4
BD_VALUE_MASK = 0x7
BD_MIN_VALID = 4
BD_MAX_VALID = 7
# BD 4..7 -> 64 KiB .. 4 MiB; invalid codes fall back to the largest.
_BLOCK_MAX = {4: 1 << 16, 5: 1 << 18, 6: 1 << 20, 7: 1 << 22}
_BLOCK_MAX_DEFAULT = 1 << 22

BLOCK_UNCOMPRESSED = 0x80000000
BLOCK_SIZE_MASK = 0x7FFFFFFF
END_MARK = 0

_MODE_OFFSET = 1  # frame starts after the mode selector byte
_MODE_PARITY = 1  # odd mode byte = raw block path
_MIN_LEN = _MODE_OFFSET + 7  # magic + FLG + BD + HC
_HC_SHIFT = 8
_BYTE = 0xFF

# Bounded-work caps: hostile frames cost a fixed amount.
_MAX_BLOCKS = 4096
_MAX_DECODE = 1 << 20
_MAX_HASH = 1 << 20
_GEN_MAX_BLOCKS = 3
_GEN_MAX_PAYLOAD = 64
_MAX_SKIP_PAYLOAD = 32

# LZ4 sequence token: high nibble literals, low nibble match length - 4.
_TOKEN_SHIFT = 4
_NIBBLE = 0xF
_LEN_EXT = 255
_MIN_MATCH = 4

_REPAIR_PROB = 0.75
_GEN_FLG_OPTS = FLG_B_INDEP | FLG_B_CHECKSUM | FLG_C_SIZE | FLG_C_CHECKSUM


class Lz4Op(IntEnum):
    """Top-level mutation menu (index drawn by ``mutate``)."""

    FLG = 0
    BD = 1
    CSIZE = 2
    BSIZE = 3
    CHECKSUM = 4
    BLOCKS = 5
    SKIPPABLE = 6


class ChecksumTarget(IntEnum):
    """Which checksum the CHECKSUM branch repairs or corrupts."""

    HEADER = 0
    BLOCK = 1
    CONTENT = 2


class BlockEdit(IntEnum):
    """Block-list edit applied by the BLOCKS branch."""

    INSERT = 0
    DUP = 1
    DROP = 2
    DROP_END = 3


class SkipPlace(IntEnum):
    """Where the SKIPPABLE branch puts its frame."""

    PREPEND = 0
    APPEND = 1


# ── model ──────────────────────────────────────────────────────────────────


@dataclass
class Lz4Block:
    """One data block: raw size word, payload, block checksum (None = compute)."""

    word: int
    data: bytes
    checksum: int | None = None


@dataclass
class Lz4Frame:
    """Parsed frame. ``None`` checksum/size fields are recomputed on serialize."""

    mode: int
    flg: int
    bd: int
    content_size: int | None = None
    dict_id: int | None = None
    hc: int | None = None
    blocks: list[Lz4Block] = field(default_factory=list)
    end_mark: bool = True
    content_checksum: int | None = None
    lead: bytes = b""  # bytes between mode byte and magic (skippable frames)
    trailer: bytes = b""


# ── LZ4 block decode (for content size / checksum repair) ─────────────────


def _read_len(src: bytes, i: int, n: int) -> tuple[int, int] | None:
    """Read an LZ4 extended-length run (255, 255, ..., <255)."""
    total = 0
    while i < n:
        b = src[i]
        i += 1
        total += b
        if b != _LEN_EXT:
            return total, i
    return None


def _copy_match(out: bytearray, off: int, ml: int) -> None:
    """Append *ml* bytes from *off* back; overlap handled in off-sized chunks."""
    while ml > 0:
        start = len(out) - off
        chunk = min(off, ml)
        out += out[start : start + chunk]
        ml -= chunk


def _decode_block(src: bytes, out: bytearray) -> bool:
    """Decode one compressed block onto *out* (prior output = history)."""
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> _TOKEN_SHIFT
        if lit == _NIBBLE:
            ext = _read_len(src, i, n)
            if ext is None:
                return False
            lit, i = lit + ext[0], ext[1]
        if i + lit > n or len(out) + lit > _MAX_DECODE:
            return False
        out += src[i : i + lit]
        i += lit
        if i == n:
            return True
        if i + 2 > n:
            return False
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = token & _NIBBLE
        if ml == _NIBBLE:
            ext = _read_len(src, i, n)
            if ext is None:
                return False
            ml, i = ml + ext[0], ext[1]
        ml += _MIN_MATCH
        if off == 0 or off > len(out) or len(out) + ml > _MAX_DECODE:
            return False
        _copy_match(out, off, ml)
    return True


def _decode_content(frame: Lz4Frame) -> bytes | None:
    """Decompressed content of all blocks, or None if undecodable/too big."""
    out = bytearray()
    for blk in frame.blocks:
        if blk.word & BLOCK_UNCOMPRESSED:
            if len(out) + len(blk.data) > _MAX_DECODE:
                return None
            out += blk.data
            continue
        if not _decode_block(blk.data, out):
            return None
    return bytes(out)


def _encode_literals(payload: bytes) -> bytes:
    """Literal-only compressed block (a single final sequence)."""
    n = len(payload)
    if n < _NIBBLE:
        return bytes([n << _TOKEN_SHIFT]) + payload
    rem = n - _NIBBLE
    ext = b"\xff" * (rem // _LEN_EXT) + bytes([rem % _LEN_EXT])
    return bytes([_NIBBLE << _TOKEN_SHIFT]) + ext + payload


# ── parse / serialize ──────────────────────────────────────────────────────


def _parse_header(data: bytes) -> tuple[Lz4Frame, int] | None:
    """Parse magic..HC; returns frame and offset of the first block."""
    if len(data) < _MIN_LEN or data[0] & _MODE_PARITY:
        return None
    if struct.unpack_from("<I", data, _MODE_OFFSET)[0] != LZ4_MAGIC:
        return None
    pos = _MODE_OFFSET + _WORD
    frame = Lz4Frame(mode=data[0], flg=data[pos], bd=data[pos + 1])
    pos += 2
    if frame.flg & FLG_C_SIZE:
        if pos + 8 > len(data):
            return None
        frame.content_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
    if frame.flg & FLG_DICT_ID:
        if pos + _WORD > len(data):
            return None
        frame.dict_id = struct.unpack_from("<I", data, pos)[0]
        pos += _WORD
    if pos >= len(data):
        return None
    return frame, pos + 1  # HC is always recomputed


def _parse_blocks(frame: Lz4Frame, data: bytes, pos: int) -> int:
    """Append blocks up to EndMark/truncation; returns the consumed offset."""
    n = len(data)
    cs = _WORD if frame.flg & FLG_B_CHECKSUM else 0
    frame.end_mark = False
    while pos + _WORD <= n and len(frame.blocks) < _MAX_BLOCKS:
        word = struct.unpack_from("<I", data, pos)[0]
        if word == END_MARK:
            frame.end_mark = True
            return pos + _WORD
        end = pos + _WORD + (word & BLOCK_SIZE_MASK)
        if end + cs > n:
            return pos
        checksum = struct.unpack_from("<I", data, end)[0] if cs else None
        frame.blocks.append(Lz4Block(word, data[pos + _WORD : end], checksum))
        pos = end + cs
    return pos


def parse_lz4(data: bytes) -> Lz4Frame | None:
    """Parse ``mode + LZ4 frame``; None unless mode is even and magic at 1.

    Anything past the last well-formed block (truncated block, garbage,
    trailing frames) is kept verbatim in ``trailer``.
    """
    head = _parse_header(data)
    if head is None:
        return None
    frame, pos = head
    pos = _parse_blocks(frame, data, pos)
    if frame.end_mark and frame.flg & FLG_C_CHECKSUM and pos + _WORD <= len(data):
        frame.content_checksum = struct.unpack_from("<I", data, pos)[0]
        pos += _WORD
    frame.trailer = data[pos:]
    return frame


def _descriptor(frame: Lz4Frame, content: _Content) -> bytes:
    """FLG..dict id bytes (the HC input)."""
    out = bytearray((frame.flg, frame.bd))
    if frame.flg & FLG_C_SIZE:
        size = frame.content_size
        out += struct.pack("<Q", (content.size() if size is None else size) & _U64)
    if frame.flg & FLG_DICT_ID:
        out += struct.pack("<I", (frame.dict_id or 0) & _U32)
    return bytes(out)


def header_checksum(desc: bytes) -> int:
    """HC byte for descriptor bytes FLG..dict id."""
    return (xxh32(desc) >> _HC_SHIFT) & _BYTE


class _Content:
    """Lazy decoded content: decoded at most once per serialize."""

    def __init__(self, frame: Lz4Frame):
        self._frame = frame
        self._done = False
        self._data: bytes | None = None

    def get(self) -> bytes | None:
        if not self._done:
            self._data = _decode_content(self._frame)
            self._done = True
        return self._data

    def size(self) -> int:
        data = self.get()
        if data is None:
            return sum(len(b.data) for b in self._frame.blocks)
        return len(data)

    def checksum(self) -> int:
        data = self.get()
        if data is None or len(data) > _MAX_HASH:
            return 0
        return xxh32(data)


def serialize_lz4(frame: Lz4Frame) -> bytes:
    """Emit ``mode + lead + frame + trailer``, computing ``None`` checksums."""
    content = _Content(frame)
    desc = _descriptor(frame, content)
    hc = header_checksum(desc) if frame.hc is None else frame.hc
    out = bytearray((frame.mode,))
    out += frame.lead
    out += struct.pack("<I", LZ4_MAGIC)
    out += desc
    out.append(hc & _BYTE)
    block_cs = frame.flg & FLG_B_CHECKSUM
    for blk in frame.blocks:
        out += struct.pack("<I", blk.word & _U32)
        out += blk.data
        if block_cs:
            cs = xxh32(blk.data) if blk.checksum is None else blk.checksum
            out += struct.pack("<I", cs & _U32)
    if frame.end_mark:
        out += struct.pack("<I", END_MARK)
    if frame.end_mark and frame.flg & FLG_C_CHECKSUM:
        cs = content.checksum() if frame.content_checksum is None else frame.content_checksum
        out += struct.pack("<I", cs & _U32)
    out += frame.trailer
    return bytes(out)


def _block_max(bd: int) -> int:
    return _BLOCK_MAX.get((bd >> BD_SHIFT) & BD_VALUE_MASK, _BLOCK_MAX_DEFAULT)


# ── mutator ────────────────────────────────────────────────────────────────


class Lz4Mutator:
    """Structure-aware LZ4 frame mutator (mode byte + frame, lz4_read.c)."""

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call (Hard Rule 16).
        self._rng = RandPool(seed=seed)

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one LZ4-frame mutation; generate a frame if *data* is not one."""
        self._rng = rng or self._rng
        frame = parse_lz4(data)
        if frame is None:
            return self._generate_random_lz4(max_len=max_len, rng=self._rng)

        ops = (
            self._mut_flg,
            self._mut_bd,
            self._mut_csize,
            self._mut_bsize,
            self._mut_checksum,
            self._mut_blocks,
            self._mut_skippable,
        )
        ops[self._rng.randint(0, len(ops) - 1)](frame)
        return serialize_lz4(frame)[:max_len]

    # ── header ────────────────────────────────────────────────────────────

    def _mut_flg(self, frame: Lz4Frame) -> None:
        """Flip one FLG bit (version/reserved included); fields follow the flags."""
        bit = self._rng.choice(FLG_BITS)
        frame.flg ^= bit
        # Newly-declared fields default to repaired values (None -> computed).
        if bit == FLG_C_SIZE:
            frame.content_size = None
        if bit == FLG_DICT_ID:
            frame.dict_id = None
        if bit == FLG_C_CHECKSUM:
            frame.content_checksum = None
        if bit == FLG_B_CHECKSUM:
            for blk in frame.blocks:
                blk.checksum = None

    def _mut_bd(self, frame: Lz4Frame) -> None:
        """Set BD block-max code to any of 0..7 (0..3 are invalid)."""
        value = self._rng.randint(0, BD_VALUE_MASK)
        frame.bd = (frame.bd & ~(BD_VALUE_MASK << BD_SHIFT)) | (value << BD_SHIFT)

    def _mut_csize(self, frame: Lz4Frame) -> None:
        """Declare content size and set it to a boundary around the truth."""
        true = _Content(frame).size()
        frame.flg |= FLG_C_SIZE
        values = (0, true, true - 1, true + 1, 1 << 32, _U64)
        frame.content_size = self._rng.choice(values) & _U64

    # ── blocks ────────────────────────────────────────────────────────────

    def _mut_bsize(self, frame: Lz4Frame) -> None:
        """Make a block size word lie (boundary sizes or stored-bit toggle)."""
        if not frame.blocks:
            self._insert_block(frame)
            return
        blk = frame.blocks[self._rng.randint(0, len(frame.blocks) - 1)]
        n = len(blk.data)
        bmax = _block_max(frame.bd)
        keep = blk.word & BLOCK_UNCOMPRESSED
        sizes = (0, n - 1, n + 1, bmax, bmax + 1, BLOCK_SIZE_MASK)
        words = [keep | (s & BLOCK_SIZE_MASK) for s in sizes]
        words.append(blk.word ^ BLOCK_UNCOMPRESSED)
        blk.word = self._rng.choice(words)

    def _mut_blocks(self, frame: Lz4Frame) -> None:
        """Insert / duplicate / drop a block, or drop the EndMark."""
        edit = self._rng.randint(0, len(BlockEdit) - 1)
        if edit == BlockEdit.DROP_END:
            frame.end_mark = False
            return
        if edit == BlockEdit.INSERT:
            self._insert_block(frame)
        elif not frame.blocks:
            return
        elif edit == BlockEdit.DUP:
            idx = self._rng.randint(0, len(frame.blocks) - 1)
            frame.blocks.insert(idx + 1, replace(frame.blocks[idx]))
        else:
            del frame.blocks[self._rng.randint(0, len(frame.blocks) - 1)]
        # Content changed: repair content size and checksum.
        frame.content_size = None
        frame.content_checksum = None

    def _insert_block(self, frame: Lz4Frame) -> None:
        """Insert a stored (uncompressed) random block."""
        payload = self._rng.randbytes(self._rng.randint(0, _GEN_MAX_PAYLOAD))
        pos = self._rng.randint(0, len(frame.blocks))
        frame.blocks.insert(pos, Lz4Block(BLOCK_UNCOMPRESSED | len(payload), payload))

    # ── checksums ─────────────────────────────────────────────────────────

    def _mut_checksum(self, frame: Lz4Frame) -> None:
        """Repair (recompute) or corrupt the header / a block / the content checksum."""
        target = self._rng.randint(0, len(ChecksumTarget) - 1)
        repair = self._rng.random() < _REPAIR_PROB
        if target == ChecksumTarget.HEADER:
            self._fix_hc(frame, repair)
            return
        if target == ChecksumTarget.BLOCK:
            self._fix_block_cs(frame, repair)
            return
        self._fix_content_cs(frame, repair)

    def _flip_bit(self, value: int, width: int) -> int:
        return value ^ (1 << self._rng.randint(0, width - 1))

    def _fix_hc(self, frame: Lz4Frame, repair: bool) -> None:
        frame.hc = None
        if repair:
            return
        good = header_checksum(_descriptor(frame, _Content(frame)))
        frame.hc = self._flip_bit(good, 8)

    def _fix_block_cs(self, frame: Lz4Frame, repair: bool) -> None:
        if not frame.blocks:
            return
        if not frame.flg & FLG_B_CHECKSUM:
            frame.flg |= FLG_B_CHECKSUM
            for blk in frame.blocks:
                blk.checksum = None
        blk = frame.blocks[self._rng.randint(0, len(frame.blocks) - 1)]
        blk.checksum = None if repair else self._flip_bit(xxh32(blk.data), 32)

    def _fix_content_cs(self, frame: Lz4Frame, repair: bool) -> None:
        frame.flg |= FLG_C_CHECKSUM
        frame.end_mark = True
        frame.content_checksum = None
        if repair:
            return
        frame.content_checksum = self._flip_bit(_Content(frame).checksum(), 32)

    # ── skippable frames ──────────────────────────────────────────────────

    def _mut_skippable(self, frame: Lz4Frame) -> None:
        """Prepend/append a skippable frame, possibly with a lying size."""
        magic = SKIPPABLE_MAGIC_BASE + self._rng.randint(0, SKIPPABLE_NIBBLE_MAX)
        payload = self._rng.randbytes(self._rng.randint(0, _MAX_SKIP_PAYLOAD))
        n = len(payload)
        size = self._rng.choice((n, n + 1, n - 1, _U32)) & _U32
        skip = struct.pack("<II", magic, size) + payload
        if self._rng.randint(0, len(SkipPlace) - 1) == SkipPlace.PREPEND:
            frame.lead = skip + frame.lead
            return
        frame.trailer = skip + frame.trailer

    # ── generator ─────────────────────────────────────────────────────────

    def _generate_random_lz4(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate ``b"\\x00"`` + a valid frame (stored and literal-only blocks)."""
        self._rng = rng or self._rng
        r = self._rng
        flg = FLG_VERSION | (r.randint(0, _GEN_FLG_OPTS) & _GEN_FLG_OPTS)
        bd = r.randint(BD_MIN_VALID, BD_MAX_VALID) << BD_SHIFT
        frame = Lz4Frame(mode=0, flg=flg, bd=bd)
        for _ in range(r.randint(1, _GEN_MAX_BLOCKS)):
            payload = r.randbytes(r.randint(0, _GEN_MAX_PAYLOAD))
            if r.randint(0, 1):
                frame.blocks.append(Lz4Block(BLOCK_UNCOMPRESSED | len(payload), payload))
                continue
            enc = _encode_literals(payload)
            frame.blocks.append(Lz4Block(len(enc), enc))
        return serialize_lz4(frame)[:max_len]


__all__ = [
    "Lz4Block",
    "Lz4Frame",
    "Lz4Mutator",
    "parse_lz4",
    "serialize_lz4",
    "xxh32",
]
