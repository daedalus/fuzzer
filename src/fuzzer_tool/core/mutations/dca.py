"""DTS core frame mutator (FFmpeg ``dts`` demux, ``dca`` decode).

Core frame header, bit offsets from the sync word (ETSI TS 102 114 §5.3.1):

   0 SYNC 7FFE8001 | 32 FTYPE | 33 SHORT 5 | 38 CPF | 39 NBLKS 7 | 46 FSIZE 14
  60 AMODE 6 | 66 SFREQ 4 | 70 RATE 5 | 75 FIXEDBIT DYNF TIMEF AUXF HDCD
  80 EXT_AUDIO_ID 3 | 83 EXT_AUDIO | 84 ASPF | 85 LFF 2 | 87 HFLAG
  88 [HCRC 16 if CPF] | FILTS | VERNUM 4 | CHIST 2 | PCMR 3 ...

Frame bytes = FSIZE + 1 (FSIZE >= 95). NBLKS + 1 PCM blocks set the
sub-frame count; AMODE/LFF/SFREQ pick channel layout and rate tables;
EXT_AUDIO selects XCh/X96/XXCh parsing. The header CRC is checked only
under ``-err_detect crccheck``, so it is left alone. 14-bit and
little-endian sync variants are not parsed.
"""

from __future__ import annotations

from enum import IntEnum

from fuzzer_tool.core.mutations.frame_stream import FrameOp, get_bits, reframe, set_bits
from fuzzer_tool.core.rand_pool import RandPool

SYNC = b"\x7f\xfe\x80\x01"
SCAN_LIMIT = 4096
HEADER_LEN = 15  # through PCMR with the optional header CRC
MIN_FSIZE = 95

# Bit offsets from the frame start.
FTYPE_BIT = 32
CPF_BIT = 38
NBLKS_BIT = 39
FSIZE_BIT = 46
AMODE_BIT = 60
SFREQ_BIT = 66
EXT_BIT = 80  # EXT_AUDIO_ID 3 + EXT_AUDIO 1
LFF_BIT = 85
PCMR_BIT = 95  # without the header CRC
HCRC_BITS = 16

NBLKS_EDGES = (0, 4, 5, 6, 7, 8, 15, 31, 127)
FSIZE_EDGES = (0, 94, 95, 96, 0x3FFF)
AMODE_EDGES = (0, 1, 2, 9, 15, 16, 63)


class Op(IntEnum):
    """Frame edit; values are the RNG draw."""

    FTYPE = 0
    NBLKS = 1
    FSIZE = 2
    AMODE = 3
    SFREQ = 4
    LFF = 5
    EXT = 6
    PCMR = 7
    DUP = 8
    DROP = 9
    SWAP = 10


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}

# (bit, width, edge table or None for a uniform draw over the width).
_FIELDS = {
    Op.NBLKS: (NBLKS_BIT, 7, NBLKS_EDGES),
    Op.FSIZE: (FSIZE_BIT, 14, FSIZE_EDGES),
    Op.AMODE: (AMODE_BIT, 6, AMODE_EDGES),
    Op.SFREQ: (SFREQ_BIT, 4, None),
    Op.LFF: (LFF_BIT, 2, None),
    Op.EXT: (EXT_BIT, 4, None),
}


def frame_size(data: bytes | bytearray, pos: int) -> int | None:
    """Declared byte size of the core frame at *pos*, or None."""
    if pos + HEADER_LEN > len(data) or data[pos : pos + 4] != SYNC:
        return None
    fsize = get_bits(data, pos * 8 + FSIZE_BIT, 14)
    return fsize + 1 if fsize >= MIN_FSIZE else None


def sniff_dca(data: bytes) -> bool:
    """Cheap :func:`parse_dca` gate: checks at most two frames, not the chain."""
    start = data.find(SYNC, 0, SCAN_LIMIT)
    if start < 0:
        return False
    size = frame_size(data, start)
    if size is None or start + size > len(data):
        return False
    if start == 0:
        return True
    nxt = frame_size(data, start + size)
    return nxt is not None and start + size + nxt <= len(data)


def parse_dca(data: bytes) -> tuple[int, list[bytes], bytes] | None:
    """(start, frames, tail), or None; two chained frames after a prefix."""
    start = data.find(SYNC, 0, SCAN_LIMIT)
    if start < 0:
        return None

    n = len(data)
    pos = start
    frames: list[bytes] = []
    while (size := frame_size(data, pos)) is not None and pos + size <= n:
        frames.append(data[pos : pos + size])
        pos += size

    if len(frames) < (1 if start == 0 else 2):
        return None
    return start, frames, data[pos:]


def _edit(op: Op, frame: bytearray, rng: RandPool) -> None:
    """Apply one header-field edit in place."""
    if op is Op.FTYPE:
        set_bits(frame, FTYPE_BIT, 1, get_bits(frame, FTYPE_BIT, 1) ^ 1)
        return
    if op is Op.PCMR:
        bit = PCMR_BIT + HCRC_BITS * get_bits(frame, CPF_BIT, 1)
        set_bits(frame, bit, 3, rng.randint(0, 7))
        return

    bit, width, edges = _FIELDS[op]
    value = rng.choice(edges) if edges else rng.randint(0, (1 << width) - 1)
    set_bits(frame, bit, width, value)


def mutate_dca(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one core header field or the frame order; None if declined.

    Draw order: op, frame index, then the op's value draw.
    """
    parsed = parse_dca(data)
    if parsed is None:
        return None
    start, frames, tail = parsed

    op = Op(rng.randint(0, len(Op) - 1))
    if op in _FRAME_OPS:
        new_frames = reframe(frames, _FRAME_OPS[op], rng)
        if new_frames is None:
            return None
    else:
        i = rng.randint(0, len(frames) - 1)
        frame = bytearray(frames[i])
        _edit(op, frame, rng)
        new_frames = [*frames[:i], bytes(frame), *frames[i + 1 :]]

    out = data[:start] + b"".join(new_frames) + tail
    return out[:max_len]
