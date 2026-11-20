"""Core Audio Format mutator (FFmpeg ``cafdec``).

  0      4    6     8
  "caff" ver  flags  then chunks:  type[4] | size i64 BE | payload[size]
  desc (first, 32 B): rate f64 | codec[4] | flags bpp fpp channels bits (u32 BE)
  pakt: num_packets i64 | nb_frames i64 | priming u32 | remainder u32 | varints

size -1 means "to EOF" (data). FFmpeg 9.0.2 adds priming/remainder to
nb_frames unchecked (signed overflow at INT64_MAX); bpp/fpp of 0 make the
pakt table drive packet sizes and durations through 7-bit varints.
Field ops overwrite in place; chunk ops (dup/drop/swap) keep desc first.
"""

from __future__ import annotations

import struct
from enum import IntEnum

from fuzzer_tool.core.mutations.frame_stream import FrameOp, reframe
from fuzzer_tool.core.rand_pool import RandPool

MAGIC = b"caff"
FILE_HDR_LEN = 8
CHUNK_HDR_LEN = 12  # type[4] + size i64
DESC_LEN = 32
PAKT_HDR_LEN = 24

VERSION_OFF = 4
RATE_OFF = 0  # within desc payload
CODEC_OFF = 8
DESC_U32_OFF = 12  # flags, bpp, fpp, channels, bits
DESC_U32_COUNT = 5
PAKT_U32_OFF = 16  # priming, remainder

INT64_MAX = (1 << 63) - 1
U64_MASK = (1 << 64) - 1
# AVIndexEntry is 24 bytes; cafdec rejects num_packets > INT32_MAX / 24.
_PAKT_CAP = 0x7FFFFFFF // 24

VERSION_EDGES = (0, 1, 2, 0xFFFF)
CHUNK_TYPES = (b"desc", b"data", b"chan", b"pakt", b"kuki", b"info", b"free")
CODECS = (b"lpcm", b"ima4", b"aac ", b"alac", b"opus", b"flac", b".mp3", b"ulaw", b"alaw")
RATE_EDGES = (0.0, -1.0, float("nan"), float("inf"), 1.0, 2147483648.0, 5e-324)
U32_EDGES = (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
# Signed i64 as u64 bit patterns: -1 is "to EOF"; INT_MAX - 64 is the kuki cap.
SIZE_EDGES = (0, 1, U64_MASK, INT64_MAX, 1 << 63, 0x7FFFFFFF - 64)
I64_EDGES = (0, 1, INT64_MAX, INT64_MAX - 1, 1 << 63, U64_MASK, _PAKT_CAP, _PAKT_CAP + 1)
# ff_mp4_read_descr_len: up to 4 bytes, 7 bits each, MSB = continue.
VARINT_EDGES = (b"\x00", b"\x7f", b"\xff\xff\xff\x7f", b"\x80\x80\x80\x00", b"\xff\xff\xff\xff")


class Op(IntEnum):
    """CAF edit; values are the RNG draw."""

    VERSION = 0
    CHUNK_SIZE = 1
    CHUNK_TYPE = 2
    RATE = 3
    CODEC = 4
    DESC_U32 = 5
    PAKT_COUNT = 6
    PAKT_EDGE = 7
    PAKT_VARINT = 8
    DUP = 9
    DROP = 10
    SWAP = 11


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}
_DESC_OPS = (Op.RATE, Op.CODEC, Op.DESC_U32)
_PAKT_OPS = (Op.PAKT_COUNT, Op.PAKT_EDGE, Op.PAKT_VARINT)


def sniff_caf(data: bytes) -> bool:
    """Magic plus the file header and one chunk header."""
    return len(data) >= FILE_HDR_LEN + CHUNK_HDR_LEN and data[:4] == MAGIC


def parse_caf(data: bytes) -> tuple[list[bytes], bytes] | None:
    """(chunks, tail) after the file header, or None if not CAF.

    Each chunk keeps its 12-byte header. A negative size runs to EOF; a
    size past EOF ends the walk and it plus everything after is the tail.
    """
    if not sniff_caf(data):
        return None

    n = len(data)
    pos = FILE_HDR_LEN
    chunks: list[bytes] = []
    while pos + CHUNK_HDR_LEN <= n:
        size = struct.unpack_from(">q", data, pos + 4)[0]
        end = n if size < 0 else pos + CHUNK_HDR_LEN + size
        if end > n:
            break
        chunks.append(data[pos:end])
        pos = end
    return chunks, data[pos:]


def _find(chunks: list[bytes], tag: bytes, need: int) -> int:
    """Index of the first *tag* chunk with *need* payload bytes, else -1."""
    for i, c in enumerate(chunks):
        if c[:4] == tag and len(c) >= CHUNK_HDR_LEN + need:
            return i
    return -1


def _chunk_hdr(op: Op, chunk: bytearray, rng: RandPool) -> None:
    """Overwrite one chunk's size or type."""
    if op is Op.CHUNK_SIZE:
        struct.pack_into(">Q", chunk, 4, rng.choice(SIZE_EDGES))
        return
    chunk[:4] = rng.choice(CHUNK_TYPES)


def _desc(op: Op, chunk: bytearray, rng: RandPool) -> None:
    """Overwrite one desc field (payload starts after the chunk header)."""
    base = CHUNK_HDR_LEN
    if op is Op.RATE:
        struct.pack_into(">d", chunk, base + RATE_OFF, rng.choice(RATE_EDGES))
    elif op is Op.CODEC:
        off = base + CODEC_OFF
        chunk[off : off + 4] = rng.choice(CODECS)
    else:
        off = base + DESC_U32_OFF + 4 * rng.randint(0, DESC_U32_COUNT - 1)
        struct.pack_into(">I", chunk, off, rng.choice(U32_EDGES))


def _pakt(op: Op, chunk: bytearray, rng: RandPool) -> bool:
    """Overwrite one pakt field or table bytes; False if no table to edit."""
    base = CHUNK_HDR_LEN
    if op is Op.PAKT_COUNT:
        off = base + 8 * rng.randint(0, 1)
        struct.pack_into(">Q", chunk, off, rng.choice(I64_EDGES))
        return True
    if op is Op.PAKT_EDGE:
        off = base + PAKT_U32_OFF + 4 * rng.randint(0, 1)
        struct.pack_into(">I", chunk, off, rng.choice(U32_EDGES))
        return True

    # Varint table: length-preserving overwrite at a random byte.
    table = base + PAKT_HDR_LEN
    if len(chunk) <= table:
        return False
    off = table + rng.randint(0, len(chunk) - table - 1)
    edge = rng.choice(VARINT_EDGES)[: len(chunk) - off]
    chunk[off : off + len(edge)] = edge
    return True


def _field_op(op: Op, chunks: list[bytes], rng: RandPool) -> list[bytes] | None:
    """Apply a header/desc/pakt field op to one chunk; None if declined."""
    if op in _DESC_OPS:
        i = _find(chunks, b"desc", DESC_LEN)
    elif op in _PAKT_OPS:
        i = _find(chunks, b"pakt", PAKT_HDR_LEN)
    else:
        i = rng.randint(0, len(chunks) - 1) if chunks else -1
    if i < 0:
        return None

    chunk = bytearray(chunks[i])
    if op in _DESC_OPS:
        _desc(op, chunk, rng)
    elif op in _PAKT_OPS and not _pakt(op, chunk, rng):
        return None
    elif op not in _PAKT_OPS:
        _chunk_hdr(op, chunk, rng)

    out = list(chunks)
    out[i] = bytes(chunk)
    return out


def mutate_caf(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Mutate one header/chunk field or chunk order; None if declined.

    Draw order: op, then the op's own draws (field selector / chunk index,
    then the value from its edge table).
    """
    parsed = parse_caf(data)
    if parsed is None:
        return None
    chunks, tail = parsed
    head = bytearray(data[:FILE_HDR_LEN])

    op = Op(rng.randint(0, len(Op) - 1))
    if op is Op.VERSION:
        struct.pack_into(">H", head, VERSION_OFF, rng.choice(VERSION_EDGES))
        new_chunks: list[bytes] | None = chunks
    elif op in _FRAME_OPS:
        # desc must stay first for cafdec to read anything past the header.
        rest = reframe(chunks[1:], _FRAME_OPS[op], rng)
        new_chunks = None if rest is None else chunks[:1] + rest
    else:
        new_chunks = _field_op(op, chunks, rng)

    if new_chunks is None:
        return None
    out = bytes(head) + b"".join(new_chunks) + tail
    return out[:max_len]
