"""G.711 A-law/µ-law sample mutator (FFmpeg ``pcm_alaw``/``pcm_mulaw``).

Byte-level ops on companded audio hit a log-scale code: flipping one bit
jumps a sample by up to 32k. This mutator works in the linear domain:

  WAV (fmt tag 6/7) / AU (.snd enc 27/1)
        │ find_payload            header bytes left untouched
        ▼
  codes ──decode──► int16 samples ──op on a span──► clip ──encode──► codes

Ops: gain (incl. inversion), DC offset, saturating square wave, silence,
reverse, noise. Length-preserving, so every header size stays valid.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from functools import cache

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

RIFF_HEADER_LEN = 12  # "RIFF" + size + "WAVE"
CHUNK_HEADER_LEN = 8  # fourcc + size
WAV_FMT_ALAW, WAV_FMT_ULAW = 6, 7  # RFC 2361 wFormatTag
AU_MAGIC = b".snd"
AU_HEADER_LEN = 24
AU_ENC_ULAW, AU_ENC_ALAW = 1, 27
AU_SIZE_UNKNOWN = 0xFFFFFFFF
MAX_CHUNKS = 4096  # walk bound for hostile RIFF chunk lists

INT16_MIN, INT16_MAX = -32768, 32767
GAINS = (-1, 0.25, 0.5, 2, 4, 16)
DC_MAX = 16384
SQUARE_MAX_PERIOD = 64  # samples per half-cycle
NOISE_SCALE = 256  # int8 noise byte -> int16 step


class Law(IntEnum):
    """G.711 companding law."""

    ULAW = 1
    ALAW = 2


class Op(IntEnum):
    """Linear-domain sample op; values are the RNG draw."""

    GAIN = 0
    OFFSET = 1
    SATURATE = 2
    SILENCE = 3
    REVERSE = 4
    NOISE = 5


_WAV_LAW = {WAV_FMT_ALAW: Law.ALAW, WAV_FMT_ULAW: Law.ULAW}
_AU_LAW = {AU_ENC_ALAW: Law.ALAW, AU_ENC_ULAW: Law.ULAW}


def _alaw_level(a: int) -> int:
    """ITU-T G.711 A-law code -> linear (FFmpeg ``alaw2linear``)."""
    a ^= 0x55
    t = a & 0xF
    seg = (a & 0x70) >> 4
    t = ((t + t + 1 + 32) << (seg + 2)) if seg else ((t + t + 1) << 3)
    return t if a & 0x80 else -t


def _ulaw_level(u: int) -> int:
    """ITU-T G.711 µ-law code -> linear (FFmpeg ``ulaw2linear``)."""
    u = ~u & 0xFF
    t = (((u & 0xF) << 3) + 0x84) << ((u & 0x70) >> 4)
    return 0x84 - t if u & 0x80 else t - 0x84


@cache
def _decode_table(law: Law) -> np.ndarray:
    level = _alaw_level if law is Law.ALAW else _ulaw_level
    return np.array([level(c) for c in range(256)], dtype=np.int32)


@cache
def _encode_table(law: Law) -> np.ndarray:
    """65536-entry nearest-level quantizer, indexed by sample & 0xFFFF.

    Ties go up: duplicate levels (µ-law ±0) to the higher code 0xFF,
    a sample midway between two levels (A-law 0) to the larger level.
    """
    dec = _decode_table(law)
    order = np.argsort(dec, kind="stable")  # equal levels: higher code last
    levels = dec[order]
    mids = (levels[:-1] + levels[1:]) / 2

    lin = np.arange(INT16_MIN, INT16_MAX + 1)
    table = np.empty(1 << 16, dtype=np.uint8)
    table[lin.astype(np.uint16)] = order[np.searchsorted(mids, lin, side="right")]
    return table


def decode(codes: bytes, law: Law) -> np.ndarray:
    """Companded bytes -> int32 linear samples."""
    return _decode_table(law)[np.frombuffer(codes, dtype=np.uint8)]


def encode(samples, law: Law) -> bytes:
    """Linear samples (clipped to int16) -> companded bytes."""
    x = np.clip(np.asarray(samples, dtype=np.int64), INT16_MIN, INT16_MAX)
    return _encode_table(law)[x.astype(np.uint16)].tobytes()


def _wav_payload(data: bytes) -> tuple[int, int, Law] | None:
    """Walk top-level chunks; the first ``data`` after a G.711 ``fmt ``."""
    n = len(data)
    pos = RIFF_HEADER_LEN
    law = None

    for _ in range(MAX_CHUNKS):
        if pos + CHUNK_HEADER_LEN > n:
            return None
        fourcc = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + CHUNK_HEADER_LEN

        if fourcc == b"fmt " and body + 2 <= n:
            law = _WAV_LAW.get(int.from_bytes(data[body : body + 2], "little"))
        if fourcc == b"data":
            return (body, min(body + size, n), law) if law else None

        pos = body + size + (size & 1)
    return None


def _au_payload(data: bytes) -> tuple[int, int, Law] | None:
    offset, size, enc = struct.unpack_from(">III", data, 4)
    law = _AU_LAW.get(enc)
    if law is None or offset < AU_HEADER_LEN:
        return None

    end = len(data) if size == AU_SIZE_UNKNOWN else min(offset + size, len(data))
    return (offset, end, law)


def find_payload(data: bytes) -> tuple[int, int, Law] | None:
    """(start, end, law) of the G.711 sample bytes, or None.

    Example: a µ-law WAV whose ``data`` chunk body sits at 44..1044 gives
    ``(44, 1044, Law.ULAW)``.
    """
    if len(data) >= RIFF_HEADER_LEN and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        span = _wav_payload(data)
    elif len(data) >= AU_HEADER_LEN and data[:4] == AU_MAGIC:
        span = _au_payload(data)
    else:
        return None

    if span is None or span[0] >= span[1]:
        return None
    return span


def _apply(op: Op, x: np.ndarray, rng: RandPool) -> np.ndarray:
    """Return the op's result on span *x* (int64 samples)."""
    if op is Op.GAIN:
        return (x * rng.choice(GAINS)).astype(np.int64)
    if op is Op.OFFSET:
        return x + rng.randint(-DC_MAX, DC_MAX)
    if op is Op.SATURATE:
        period = rng.randint(1, SQUARE_MAX_PERIOD)
        high = (np.arange(x.size) // period) % 2 == 0
        return np.where(high, INT16_MAX, INT16_MIN)
    if op is Op.SILENCE:
        return np.zeros_like(x)
    if op is Op.REVERSE:
        return x[::-1]

    noise = np.frombuffer(rng.randbytes(x.size), dtype=np.int8).astype(np.int64)
    return x + noise * NOISE_SCALE


def mutate_g711(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Mutate one span of decoded G.711 samples and re-encode; None if no payload."""
    span = find_payload(data)
    if span is None:
        return None
    start, end, law = span

    # Draw order: op, span start, span end, then op-specific draws.
    op = Op(rng.randint(0, len(Op) - 1))
    n = end - start
    lo = rng.randint(0, n - 1)
    hi = rng.randint(lo + 1, n)

    samples = decode(data[start + lo : start + hi], law).astype(np.int64)
    codes = encode(_apply(op, samples, rng), law)
    out = data[: start + lo] + codes + data[start + hi :]
    return out[:max_len]
