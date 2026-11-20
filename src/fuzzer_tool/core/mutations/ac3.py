"""AC-3 / E-AC-3 frame mutator (FFmpeg ``ac3``/``eac3`` demux + decode).

Frame header, bit offsets from the sync word (ATSC A/52):

  AC-3  (bsid <= 10)  sync16 crc1 16 | fscod 2 | frmsizecod 6 | bsid 5 | bsmod 3 | acmod 3 | ...
  E-AC-3 (bsid 11-16) sync16 strmtyp 2 | substreamid 3 | frmsiz 11 | fscod 2 | numblkscod 2
                      | acmod 3 | lfeon 1 | bsid 5
                                                        bsid sits at bit 40 in both
Frame size: AC-3 from the (fscod, frmsizecod) table, E-AC-3 (frmsiz + 1) words.

FFmpeg (``-err_detect crccheck``/careful) and the probe require the CRC-16
(0x8005, MSB-first) over the frame after the sync word to be zero. The
last two bytes (crc2) are rewritten after every in-frame edit, so a
payload bit flip reaches the audio block parser instead of the CRC check.
"""

from __future__ import annotations

from enum import IntEnum

from fuzzer_tool.core.berlekamp_massey import compute_checksum
from fuzzer_tool.core.mutations.frame_stream import FrameOp, get_bits, reframe, set_bits
from fuzzer_tool.core.rand_pool import RandPool

SYNC = b"\x0b\x77"
SCAN_LIMIT = 4096  # first sync searched for within this prefix
HEADER_LEN = 8  # bytes needed to read every header field used here
CRC_LEN = 2
MIN_FRAME_LEN = HEADER_LEN + CRC_LEN + 1  # room for one payload bit
CRC16_POLY = 0x8005  # FFmpeg AV_CRC_16_ANSI

AC3_MAX_BSID = 10
EAC3_BSID = 16
RESERVED_FSCOD = 3
MAX_FRMSIZECOD = 37

# Bit offsets from the frame start.
FSCOD_BIT = 32
FRMSIZECOD_BIT = 34
EAC3_FRMSIZ_BIT = 21
BSID_BIT = 40
AC3_ACMOD_BIT = 48
EAC3_ACMOD_BIT = 36
EAC3_LFEON_BIT = 39

# A/52 Table 5.18 nominal bit rates (kbps), one per frmsizecod pair.
_BITRATES = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640)
_SAMPLE_RATES = (48000, 44100, 32000)
_WORDS_PER_KBPS = 96000  # 1536 samples * 1000 / 16 bits per word
AC3_44K_FSCOD = 1

FRMSIZECOD_EDGES = (0, 1, 36, 37, 38, 63)
EAC3_FRMSIZ_EDGES = (0, 1, 2046, 2047)
BSID_EDGES = (0, 6, 8, 9, 10, 11, 16, 17, 31)


class Op(IntEnum):
    """Frame edit; values are the RNG draw."""

    FSCOD = 0
    SIZE_CODE = 1
    BSID = 2
    ACMOD = 3
    LFEON = 4
    PAYLOAD = 5
    DUP = 6
    DROP = 7
    SWAP = 8


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}


def _ac3_bytes(fscod: int, code: int) -> int:
    """Frame bytes from A/52 Table 5.18; 44.1 kHz odd codes add one word."""
    words = _BITRATES[code >> 1] * _WORDS_PER_KBPS // _SAMPLE_RATES[fscod]
    if fscod == AC3_44K_FSCOD:
        words += code & 1
    return 2 * words


def _is_eac3(frame: bytes | bytearray, pos: int = 0) -> bool:
    return get_bits(frame, pos * 8 + BSID_BIT, 5) > AC3_MAX_BSID


def frame_size(data: bytes | bytearray, pos: int) -> int | None:
    """Declared byte size of the frame at *pos*, or None if not a frame."""
    if pos + HEADER_LEN > len(data) or data[pos : pos + 2] != SYNC:
        return None
    base = pos * 8
    bsid = get_bits(data, base + BSID_BIT, 5)

    if bsid > EAC3_BSID:
        return None
    if bsid > AC3_MAX_BSID:
        size = 2 * (get_bits(data, base + EAC3_FRMSIZ_BIT, 11) + 1)
        return size if size >= MIN_FRAME_LEN else None

    fscod = get_bits(data, base + FSCOD_BIT, 2)
    code = get_bits(data, base + FRMSIZECOD_BIT, 6)
    if fscod == RESERVED_FSCOD or code > MAX_FRMSIZECOD:
        return None
    return _ac3_bytes(fscod, code)


def _chain(data: bytes, start: int) -> tuple[list[bytes], int]:
    """Frames back to back from *start*; returns (frames, end offset)."""
    n = len(data)
    pos = start
    frames: list[bytes] = []
    while True:
        size = frame_size(data, pos)
        if size is None or pos + size > n:
            return frames, pos
        frames.append(data[pos : pos + size])
        pos += size


def sniff_ac3(data: bytes) -> bool:
    """Cheap :func:`parse_ac3` gate: checks at most two frames, not the chain."""
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


def parse_ac3(data: bytes) -> tuple[int, list[bytes], bytes] | None:
    """(start, frames, tail), or None.

    A stream starting at offset 0 needs one whole frame; after a prefix
    (ID3, junk) it needs two chained frames, so a stray ``0B 77`` in other
    data does not pass.
    """
    start = data.find(SYNC, 0, SCAN_LIMIT)
    if start < 0:
        return None

    frames, end = _chain(data, start)
    if len(frames) < (1 if start == 0 else 2):
        return None
    return start, frames, data[end:]


def fix_crc(frame: bytearray) -> None:
    """Rewrite crc2 so the CRC over ``frame[2:]`` is zero."""
    crc = compute_checksum(bytes(frame[2:-CRC_LEN]), CRC16_POLY, 16)
    frame[-CRC_LEN:] = crc.to_bytes(CRC_LEN, "big")


def _lfeon_bit(frame: bytearray) -> int:
    """AC-3 lfeon follows optional mix-level fields chosen by acmod."""
    if _is_eac3(frame):
        return EAC3_LFEON_BIT
    acmod = get_bits(frame, AC3_ACMOD_BIT, 3)
    bit = AC3_ACMOD_BIT + 3
    bit += 2 if acmod & 1 and acmod != 1 else 0  # cmixlev
    bit += 2 if acmod & 4 else 0  # surmixlev
    bit += 2 if acmod == 2 else 0  # dsurmod
    return bit


def _edit(op: Op, frame: bytearray, rng: RandPool) -> None:
    """Apply one in-frame edit (not the CRC)."""
    eac3 = _is_eac3(frame)

    if op is Op.FSCOD:
        set_bits(frame, FSCOD_BIT, 2, rng.randint(0, RESERVED_FSCOD))
    elif op is Op.SIZE_CODE and eac3:
        set_bits(frame, EAC3_FRMSIZ_BIT, 11, rng.choice(EAC3_FRMSIZ_EDGES))
    elif op is Op.SIZE_CODE:
        set_bits(frame, FRMSIZECOD_BIT, 6, rng.choice(FRMSIZECOD_EDGES))
    elif op is Op.BSID:
        set_bits(frame, BSID_BIT, 5, rng.choice(BSID_EDGES))
    elif op is Op.ACMOD:
        set_bits(frame, EAC3_ACMOD_BIT if eac3 else AC3_ACMOD_BIT, 3, rng.randint(0, 7))
    elif op is Op.LFEON:
        bit = _lfeon_bit(frame)
        set_bits(frame, bit, 1, get_bits(frame, bit, 1) ^ 1)
    else:
        bit = rng.randint(HEADER_LEN * 8, (len(frame) - CRC_LEN) * 8 - 1)
        set_bits(frame, bit, 1, get_bits(frame, bit, 1) ^ 1)


def mutate_ac3(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one frame (CRC repaired) or the frame order; None if declined.

    Draw order: op, frame index, then the op's value draw.
    """
    parsed = parse_ac3(data)
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
        fix_crc(frame)
        new_frames = [*frames[:i], bytes(frame), *frames[i + 1 :]]

    out = data[:start] + b"".join(new_frames) + tail
    return out[:max_len]
