"""FLAC frame mutator (FFmpeg ``flacdec``/``flac_parser``).

``flac_chunk_mutate`` stops at the metadata blocks; frames get flat byte
noise, which the header CRC-8 rejects almost always. Frame layout
(RFC 9639 §9.1):

  FF F8|F9   bs code 4 | rate code 4   channels 4 | size code 3 | 0
  coded number (UTF-8-like, 1-7 bytes)   [bs 8/16 bit] [rate 8/16 bit]
  CRC-8 (0x07) ── subframes ... ── CRC-16 (0x8005) over the whole frame

Frames have no length: a boundary is a strict header (reserved bits,
codes, CRC-8). After an edit the header is rebuilt (extra
bs/rate bytes follow their codes) and both CRCs are recomputed, so the
edit reaches subframe decoding even under ``-err_detect crccheck``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from fuzzer_tool.core.berlekamp_massey import compute_checksum
from fuzzer_tool.core.mutations.flac import parse_flac
from fuzzer_tool.core.mutations.frame_stream import FrameOp, reframe
from fuzzer_tool.core.rand_pool import RandPool

SYNC = b"\xff\xf8"
_SYNC_RE = re.compile(rb"\xff[\xf8\xf9]")
STRATEGY_BIT = 0x01  # byte 1: 0 fixed block size, 1 variable
FIXED_LEN = 4
CRC8_POLY, CRC16_POLY = 0x07, 0x8005
CRC16_LEN = 2

RESERVED_BS_CODE = 0
BS_8BIT, BS_16BIT = 6, 7
INVALID_RATE_CODE = 15
RATE_8BIT, RATE_16BIT = 12, (13, 14)
MAX_CHANNEL_CODE = 10
RESERVED_SIZE_CODE = 3

CONT_PAYLOAD_BITS = 6  # 10xxxxxx continuation bytes
MAX_NUMBER_BYTES = 7
NUMBER_EDGES = (0, 1, 0x7F, 0x80, 0x7FFFFFFF, 2**36 - 1)
EXTRA_FILL = 0xFF  # new bs/rate field: largest value


class Op(IntEnum):
    """Frame edit; values are the RNG draw."""

    BLOCK_SIZE = 0
    SAMPLE_RATE = 1
    CHANNELS = 2
    SAMPLE_SIZE = 3
    STRATEGY = 4
    NUMBER = 5
    PAYLOAD = 6
    DUP = 7
    DROP = 8
    SWAP = 9


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}


def encode_number(value: int) -> bytes:
    """UTF-8-like coded number, up to 36 bits. ``0x80 -> C2 80``."""
    if value < 0x80:
        return bytes([value])
    n = 2
    while value >> (5 * n + 1):
        n += 1
    shift = CONT_PAYLOAD_BITS * (n - 1)
    out = [((0xFF << (8 - n)) & 0xFF) | (value >> shift)]
    out += [0x80 | ((value >> (CONT_PAYLOAD_BITS * k)) & 0x3F) for k in range(n - 2, -1, -1)]
    return bytes(out)


def decode_number(data: bytes, pos: int) -> tuple[int, int] | None:
    """(value, byte count) of the coded number at *pos*, or None."""
    if pos >= len(data):
        return None
    lead = data[pos]
    if lead < 0x80:
        return lead, 1
    n = 8 - (lead ^ 0xFF).bit_length()  # leading one bits
    if n < 2 or n > MAX_NUMBER_BYTES or pos + n > len(data):
        return None

    value = lead & (0x7F >> n)
    for b in data[pos + 1 : pos + n]:
        if b & 0xC0 != 0x80:
            return None
        value = (value << CONT_PAYLOAD_BITS) | (b & 0x3F)
    return value, n


def _extra_len(bs_code: int, rate_code: int) -> tuple[int, int]:
    """Bytes of the optional block-size and sample-rate fields."""
    bs = {BS_8BIT: 1, BS_16BIT: 2}.get(bs_code, 0)
    rate = 1 if rate_code == RATE_8BIT else 2 if rate_code in RATE_16BIT else 0
    return bs, rate


@dataclass
class FrameHeader:
    """A frame header split into its rebuildable parts."""

    fixed: bytearray  # sync .. reserved bit
    number: int
    extra_bs: bytes
    extra_rate: bytes


def _valid_fixed(fixed: bytes) -> bool:
    bs_code, rate_code = fixed[2] >> 4, fixed[2] & 0xF
    ch_code, size_code = fixed[3] >> 4, (fixed[3] >> 1) & 7
    return (
        fixed[0] == SYNC[0]
        and fixed[1] & 0xFE == SYNC[1]
        and bs_code != RESERVED_BS_CODE
        and rate_code != INVALID_RATE_CODE
        and ch_code <= MAX_CHANNEL_CODE
        and size_code != RESERVED_SIZE_CODE
        and not fixed[3] & 1
    )


def parse_header(data: bytes, pos: int) -> tuple[FrameHeader, int] | None:
    """(header, length incl. CRC-8) of a strict, CRC-valid header at *pos*."""
    fixed = data[pos : pos + FIXED_LEN]
    if len(fixed) < FIXED_LEN or not _valid_fixed(fixed):
        return None
    num = decode_number(data, pos + FIXED_LEN)
    if num is None:
        return None

    bs_len, rate_len = _extra_len(fixed[2] >> 4, fixed[2] & 0xF)
    p = pos + FIXED_LEN + num[1]
    end = p + bs_len + rate_len
    if end >= len(data) or compute_checksum(data[pos : end + 1], CRC8_POLY, 8):
        return None
    header = FrameHeader(bytearray(fixed), num[0], data[p : p + bs_len], data[p + bs_len : end])
    return header, end + 1 - pos


def _audio_start(data: bytes) -> int | None:
    if data[:4] != b"fLaC":
        return 0
    parsed = parse_flac(data)
    return None if parsed is None else len(data) - len(parsed[1])


def sniff_flac_frame(data: bytes) -> bool:
    """A strict frame header where the audio starts (offset 0 or after metadata)."""
    start = _audio_start(data)
    return start is not None and parse_header(data, start) is not None


def parse_frames(data: bytes) -> tuple[int, list[bytes]] | None:
    """(audio start, frames) or None; the last frame runs to EOF.

    Boundaries are strict CRC-8-valid headers only. A span CRC-16 check
    would also reject header lookalikes inside audio, but costs 17x
    (12 KB file: 172 -> 10 us) to catch a ~1-in-16M-per-byte event.
    """
    start = _audio_start(data)
    if start is None or parse_header(data, start) is None:
        return None

    frames: list[bytes] = []
    begin = start
    for m in _SYNC_RE.finditer(data, start + 1):
        if parse_header(data, m.start()) is not None:
            frames.append(data[begin : m.start()])
            begin = m.start()
    frames.append(data[begin:])
    return start, frames


def _rebuild(header: FrameHeader, body: bytes) -> bytes:
    """Header (extra fields sized to the codes) + CRC-8 + body + CRC-16."""
    bs_len, rate_len = _extra_len(header.fixed[2] >> 4, header.fixed[2] & 0xF)
    extra_bs = header.extra_bs if len(header.extra_bs) == bs_len else bytes([EXTRA_FILL]) * bs_len
    extra_rate = header.extra_rate
    if len(extra_rate) != rate_len:
        extra_rate = bytes([EXTRA_FILL]) * rate_len

    head = bytes(header.fixed) + encode_number(header.number) + extra_bs + extra_rate
    head += bytes([compute_checksum(head, CRC8_POLY, 8)])
    frame = head + body
    return frame + compute_checksum(frame, CRC16_POLY, 16).to_bytes(CRC16_LEN, "big")


def _set_nibble(buf: bytearray, idx: int, high: bool, value: int) -> None:
    buf[idx] = (buf[idx] & 0x0F) | (value << 4) if high else (buf[idx] & 0xF0) | value


def _edit(op: Op, header: FrameHeader, body: bytearray, rng: RandPool) -> bool:
    """Apply one in-frame edit; False when it cannot apply."""
    fixed = header.fixed
    if op is Op.BLOCK_SIZE:
        _set_nibble(fixed, 2, True, rng.randint(0, 15))
    elif op is Op.SAMPLE_RATE:
        _set_nibble(fixed, 2, False, rng.randint(0, 15))
    elif op is Op.CHANNELS:
        _set_nibble(fixed, 3, True, rng.randint(0, 15))
    elif op is Op.SAMPLE_SIZE:
        fixed[3] = (fixed[3] & 0xF1) | (rng.randint(0, 7) << 1)
    elif op is Op.STRATEGY:
        fixed[1] ^= STRATEGY_BIT
    elif op is Op.NUMBER:
        header.number = rng.choice(NUMBER_EDGES)
    elif not body:
        return False
    else:
        bit = rng.randint(0, len(body) * 8 - 1)
        body[bit >> 3] ^= 0x80 >> (bit & 7)
    return True


def mutate_flac_frame(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one frame (both CRCs repaired) or the frame order; None if declined.

    Draw order: op, frame index, then the op's value draw.
    """
    parsed = parse_frames(data)
    if parsed is None:
        return None
    start, frames = parsed

    op = Op(rng.randint(0, len(Op) - 1))
    if op in _FRAME_OPS:
        new_frames = reframe(frames, _FRAME_OPS[op], rng)
    else:
        i = rng.randint(0, len(frames) - 1)
        new_frames = _edit_frame(op, frames, i, rng)

    if new_frames is None:
        return None
    out = data[:start] + b"".join(new_frames)
    return out[:max_len]


def _edit_frame(op: Op, frames: list[bytes], i: int, rng: RandPool) -> list[bytes] | None:
    frame = frames[i]
    parsed = parse_header(frame, 0)
    if parsed is None or len(frame) < parsed[1] + CRC16_LEN:
        return None
    header, hdr_len = parsed

    body = bytearray(frame[hdr_len:-CRC16_LEN])
    if not _edit(op, header, body, rng):
        return None
    return [*frames[:i], _rebuild(header, bytes(body)), *frames[i + 1 :]]
