"""IVF container mutator (FFmpeg ``ivfdec``: VP8/VP9/AV1 raw streams).

  0      4    6    8       12   14   16     20      24        28     32
  "DKIF" ver  hlen fourcc  w    h    rate   scale   nframes   unused
  then per frame:  size u32 | pts u64 | payload[size]     (all LE)

ivfdec reads the fourcc through the BMP tag table, so a swapped fourcc
feeds one codec's frames to another decoder. rate/scale become the time
base (0 is rejected), and frame size/pts drive packet reads and timing.
Field ops overwrite in place; frame ops (dup/drop/swap) keep frames whole.
"""

from __future__ import annotations

import struct
from enum import IntEnum

from fuzzer_tool.core.mutations.frame_stream import FrameOp, reframe
from fuzzer_tool.core.rand_pool import RandPool

MAGIC = b"DKIF"
HEADER_LEN = 32
FRAME_HDR_LEN = 12  # size u32 + pts u64

FOURCC_OFF = 8
DIMS_OFF = 12  # width u16, height u16
TIMEBASE_OFF = 16  # rate u32, scale u32
NFRAMES_OFF = 24

# Codec tags in FFmpeg's ff_codec_bmp_tags that ivfdec resolves.
FOURCCS = (b"VP80", b"VP90", b"AV01", b"H264", b"HEVC", b"MJPG")
DIM_EDGES = (0, 1, 2, 16383, 16384, 0xFFFF)
U32_EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
PTS_EDGES = (0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1)


class Op(IntEnum):
    """IVF edit; values are the RNG draw."""

    DIMS = 0
    FOURCC = 1
    TIMEBASE = 2
    NFRAMES = 3
    FRAME_SIZE = 4
    PTS = 5
    DUP = 6
    DROP = 7
    SWAP = 8


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}


def sniff_ivf(data: bytes) -> bool:
    """Magic plus a full 32-byte header."""
    return len(data) >= HEADER_LEN and data[:4] == MAGIC


def parse_ivf(data: bytes) -> tuple[list[bytes], bytes] | None:
    """(frames, tail) after the header, or None if not IVF.

    Each frame keeps its 12-byte header. A frame running past EOF ends the
    walk; it and everything after it is the tail.
    """
    if not sniff_ivf(data):
        return None

    n = len(data)
    pos = HEADER_LEN
    frames: list[bytes] = []
    while pos + FRAME_HDR_LEN <= n:
        end = pos + FRAME_HDR_LEN + int.from_bytes(data[pos : pos + 4], "little")
        if end > n:
            break
        frames.append(data[pos:end])
        pos = end
    return frames, data[pos:]


def _header_op(op: Op, head: bytearray, rng: RandPool) -> None:
    """Overwrite one header field in place."""
    if op is Op.DIMS:
        off = DIMS_OFF + 2 * rng.randint(0, 1)
        struct.pack_into("<H", head, off, rng.choice(DIM_EDGES))
    elif op is Op.FOURCC:
        head[FOURCC_OFF : FOURCC_OFF + 4] = rng.choice(FOURCCS)
    elif op is Op.TIMEBASE:
        off = TIMEBASE_OFF + 4 * rng.randint(0, 1)
        struct.pack_into("<I", head, off, rng.choice(U32_EDGES))
    else:
        struct.pack_into("<I", head, NFRAMES_OFF, rng.choice(U32_EDGES))


def _frame_field(op: Op, frames: list[bytes], rng: RandPool) -> list[bytes] | None:
    """Overwrite one frame's size or pts; payload bytes are untouched."""
    if not frames:
        return None
    i = rng.randint(0, len(frames) - 1)
    frame = bytearray(frames[i])

    if op is Op.FRAME_SIZE:
        struct.pack_into("<I", frame, 0, rng.choice(U32_EDGES))
    else:
        struct.pack_into("<Q", frame, 4, rng.choice(PTS_EDGES))

    out = list(frames)
    out[i] = bytes(frame)
    return out


def mutate_ivf(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Mutate one header field, frame field or frame order; None if declined.

    Draw order: op, then the op's own draws (field selector / frame index,
    then the value from its edge table).
    """
    parsed = parse_ivf(data)
    if parsed is None:
        return None
    frames, tail = parsed
    head = bytearray(data[:HEADER_LEN])

    op = Op(rng.randint(0, len(Op) - 1))
    if op <= Op.NFRAMES:
        _header_op(op, head, rng)
        new_frames: list[bytes] | None = frames
    elif op in _FRAME_OPS:
        new_frames = reframe(frames, _FRAME_OPS[op], rng)
    else:
        new_frames = _frame_field(op, frames, rng)

    if new_frames is None:
        return None
    out = bytes(head) + b"".join(new_frames) + tail
    return out[:max_len]
