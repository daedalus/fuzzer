"""Hashing without third-party code.

Two families, split by whether the digest outlives the process:

* ``xxh64`` / ``key16`` -- XXH64, bit-exact with the reference. Corpus
  filenames and ``seed_meta`` keys are persisted, so the scheme must not
  change under an existing corpus. Pure Python: ~150 us per 4 KiB.
* ``hash64`` / ``hash64_len`` / ``hash64_seeded`` -- CRC32 | Adler32 << 32. Keys for
  in-memory and pickled tables. Stable across processes (unlike the
  builtin ``hash``, which PYTHONHASHSEED salts). ~2.6 us per 4 KiB
  (XXH3 was 0.22, BLAKE2b-64 6.7). Not collision-resistant: CRC32 is
  linear, so ~32 effective bits for random data. CRC32 does guarantee
  that any 1-bit flip or burst <= 32 bits changes the key, which is
  what the per-op "buffer changed" check needs. A merged key only
  pools two seeds' statistics; never use it for identity on disk.

    data --+--> key16()  -> "ef46db3751d8e999"   persisted names
           '--> hash64() -> 0x024d0127352441c2   table keys
"""

import struct
from zlib import adler32, crc32

_U64 = (1 << 64) - 1
_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P3 = 0x165667B19E3779F9
_P4 = 0x85EBCA77C2B2AE63
_P5 = 0x27D4EB2F165667C5

_STRIPE = 32
_LANE_FMT = "<4Q"
_WORD64 = 8
_WORD32 = 4
_ROT_LANES = (1, 7, 12, 18)
_U32 = 0xFFFFFFFF
_HI_SHIFT = 32
_LEN_SHIFT = 64
_KEY_HEX_CHARS = 16


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _U64


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * _P2) & _U64, 31) * _P1) & _U64


def _merge(h: int, acc: int) -> int:
    return (((h ^ _round(0, acc)) * _P1) + _P4) & _U64


def _bulk(data: bytes, seed: int) -> tuple[int, int]:
    """Consume whole 32-byte stripes; return (state, bytes consumed).

    Four independent lanes fold each stripe, then merge into one word.
    """
    v1 = (seed + _P1 + _P2) & _U64
    v2 = (seed + _P2) & _U64
    v3 = seed & _U64
    v4 = (seed - _P1) & _U64
    used = len(data) - len(data) % _STRIPE

    # Lanes inlined: a helper call per lane costs ~2x.
    for a, b, c, d in struct.iter_unpack(_LANE_FMT, data[:used]):
        v1 = (_rotl((v1 + a * _P2) & _U64, 31) * _P1) & _U64
        v2 = (_rotl((v2 + b * _P2) & _U64, 31) * _P1) & _U64
        v3 = (_rotl((v3 + c * _P2) & _U64, 31) * _P1) & _U64
        v4 = (_rotl((v4 + d * _P2) & _U64, 31) * _P1) & _U64

    h = 0
    for lane, rot in zip((v1, v2, v3, v4), _ROT_LANES, strict=True):
        h += _rotl(lane, rot)
    h &= _U64
    for lane in (v1, v2, v3, v4):
        h = _merge(h, lane)
    return h, used


def _tail(h: int, data: bytes, pos: int) -> int:
    """Fold the <32 trailing bytes: 8-byte words, one 4-byte word, bytes."""
    n = len(data)
    while pos + _WORD64 <= n:
        word = int.from_bytes(data[pos : pos + _WORD64], "little")
        h ^= _round(0, word)
        h = (_rotl(h, 27) * _P1 + _P4) & _U64
        pos += _WORD64

    if pos + _WORD32 <= n:
        word = int.from_bytes(data[pos : pos + _WORD32], "little")
        h ^= (word * _P1) & _U64
        h = (_rotl(h, 23) * _P2 + _P3) & _U64
        pos += _WORD32

    while pos < n:
        h ^= (data[pos] * _P5) & _U64
        h = (_rotl(h, 11) * _P1) & _U64
        pos += 1
    return h


def xxh64(data: bytes, seed: int = 0) -> int:
    """XXH64 of *data* as an int; bit-exact with the reference."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"bytes-like required, got {type(data).__name__}")

    data = bytes(data)
    n = len(data)
    if n >= _STRIPE:
        h, pos = _bulk(data, seed)
    else:
        h, pos = (seed + _P5) & _U64, 0

    h = _tail((h + n) & _U64, data, pos)

    # Avalanche.
    h ^= h >> 33
    h = (h * _P2) & _U64
    h ^= h >> 29
    h = (h * _P3) & _U64
    return h ^ (h >> 32)


def key16(data: bytes) -> str:
    """16-hex XXH64: the persisted corpus filename / ``seed_meta`` key."""
    return f"{xxh64(data):0{_KEY_HEX_CHARS}x}"


def hash64(data: bytes) -> int:
    """64-bit content key for tables; stable across processes."""
    return crc32(data) | (adler32(data) << _HI_SHIFT)


def hash64_len(data: bytes) -> int:
    """``hash64`` with the length above bit 64: key for length-sized state.

    A digest collision across lengths cannot share a table whose bins
    were sized for another length.
    """
    return hash64(data) | (len(data) << _LEN_SHIFT)


def hash64_seeded(data: bytes, seed: int) -> int:
    """``hash64`` with the seed as the initial state of both sums.

    The seed is wrapped to 64 bits: low half starts CRC32, high half Adler32.
    """
    seed &= _U64
    lo = crc32(data, seed & _U32)
    hi = adler32(data, seed >> _HI_SHIFT)
    return lo | (hi << _HI_SHIFT)
