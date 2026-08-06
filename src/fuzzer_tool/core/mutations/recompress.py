"""Round-trip mutation of compressed streams: inflate, mutate, re-deflate.

The existing ``zlib``/``gzip`` mutators corrupt the *compressed* bytes in
place. That reliably breaks the DEFLATE stream, so the target bails out in
its decompression step and the parser behind the compression layer is never
reached. This module does the opposite: it decompresses, applies a normal
byte-level mutation to the *plaintext*, then recompresses and fixes up the
trailer (Adler-32 for zlib, CRC-32 + ISIZE for gzip) so the result inflates
cleanly and the payload parser actually runs.

Throughput notes (this operator is the most expensive one in the tree, so
every step is bounded):

- ``sniff_*`` is a magic-byte check; nothing is decompressed until the
  operator has actually been selected.
- Decompression is capped at ``_MAX_INFLATE`` via ``decompressobj`` with an
  explicit ``max_length``, so a compression bomb costs a bounded amount of
  work instead of exhausting RAM.
- Inputs above ``_MAX_COMPRESSED_IN`` are skipped outright.
- Recompression uses level 1. Fuzzing cares that the stream inflates, not
  that it is small, and level 1 is several times faster than level 6.
- Successful round-trips are memoized by input hash in a small bounded
  cache, so re-selecting the operator on the same seed skips the inflate.
"""

from __future__ import annotations

import binascii
import random
import struct
import zlib

# Bounds chosen so a single call stays in the sub-millisecond range for
# typical seeds and cannot blow up on adversarial input.
_MAX_COMPRESSED_IN = 1 << 20  # 1 MiB of compressed input
_MAX_INFLATE = 1 << 22  # 4 MiB decompressed ceiling
# Deflate cost is linear in plaintext size and dominates this operator, so the
# amount actually mutated + recompressed is capped well below _MAX_INFLATE.
# 256 KiB keeps the worst case near 1ms instead of ~20ms at the 4 MiB ceiling.
_MAX_PLAIN_WORK = 1 << 18
_COMPRESS_LEVEL = 1  # speed over ratio
_CACHE_MAXSIZE = 256

# (wbits, hash(compressed bytes)) -> plaintext, so repeat selections skip the
# inflate. wbits is part of the key: the same bytes must not be served from a
# zlib-decoded entry when asked for as gzip (and vice versa).
_inflate_cache: dict[tuple[int, int], bytes] = {}


def _cache_get(data: bytes, wbits: int) -> bytes | None:
    return _inflate_cache.get((wbits, hash(data)))


def _cache_put(data: bytes, wbits: int, plain: bytes) -> None:
    if len(_inflate_cache) >= _CACHE_MAXSIZE:
        _inflate_cache.clear()
    _inflate_cache[(wbits, hash(data))] = plain


def _get_rng(rng=None):
    return rng or random


# ── Sniffers (cheap; safe to call on every selection) ──────────────────


def sniff_zlib(data: bytes) -> bool:
    """True if *data* plausibly starts with a zlib header.

    CMF/FLG must be a multiple of 31 and the compression method must be
    DEFLATE. This rejects almost all non-zlib input without allocating.
    """
    if len(data) < 6 or (data[0] & 0x0F) != 8:
        return False
    return ((data[0] << 8) | data[1]) % 31 == 0


def sniff_gzip(data: bytes) -> bool:
    return len(data) >= 18 and data[:3] == b"\x1f\x8b\x08"


# ── Bounded inflate ────────────────────────────────────────────────────


def _inflate(data: bytes, wbits: int) -> bytes | None:
    """Decompress at most ``_MAX_INFLATE`` bytes, or return None.

    Uses ``decompressobj`` rather than ``zlib.decompress`` so the ceiling is
    enforced during decompression instead of after it. A truncated stream is
    still usable — fuzzing corpora are full of partially-valid files — so
    whatever came out before the error is kept.
    """
    if not data or len(data) > _MAX_COMPRESSED_IN:
        return None
    cached = _cache_get(data, wbits)
    if cached is not None:
        return cached
    try:
        obj = zlib.decompressobj(wbits)
        plain = obj.decompress(data, _MAX_INFLATE)
    except zlib.error:
        return None
    except (MemoryError, OverflowError):
        return None
    if not plain:
        return None
    _cache_put(data, wbits, plain)
    return plain


def inflate_zlib(data: bytes) -> bytes | None:
    return _inflate(data, 15)


def inflate_gzip(data: bytes) -> bytes | None:
    return _inflate(data, 15 | 16)


# ── Plaintext mutation ─────────────────────────────────────────────────


def _mutate_plain(plain: bytes, max_len: int, rng=None) -> bytes:
    """Apply one cheap byte-level mutation to the decompressed payload.

    Deliberately limited to O(1)-ish edits rather than calling back into the
    full operator engine: the value here is reaching the inner parser at all,
    and the outer scheduler already re-selects this operator often enough to
    explore. Anything heavier would show up directly in EPS.
    """
    r = _get_rng(rng)
    if not plain:
        return plain
    buf = bytearray(plain)
    n = len(buf)
    op = r.randint(0, 5)

    if op == 0:  # bit flip
        i = r.randint(0, n - 1)
        buf[i] ^= 1 << r.randint(0, 7)
    elif op == 1:  # interesting byte
        i = r.randint(0, n - 1)
        buf[i] = r.choice((0x00, 0x01, 0x7F, 0x80, 0xFF))
    elif op == 2:  # small arithmetic
        i = r.randint(0, n - 1)
        buf[i] = (buf[i] + r.randint(-16, 16)) & 0xFF
    elif op == 3 and n > 2:  # delete a span
        start = r.randint(0, n - 2)
        length = min(r.randint(1, 16), n - start)
        del buf[start : start + length]
    elif op == 4 and n < max_len:  # duplicate a span
        start = r.randint(0, n - 1)
        length = min(r.randint(1, 16), n - start, max_len - n)
        if length > 0:
            buf[start:start] = buf[start : start + length]
    else:  # overwrite a short run
        i = r.randint(0, n - 1)
        length = min(r.randint(1, 8), n - i)
        for k in range(length):
            buf[i + k] = r.randint(0, 255)

    return bytes(buf[:max_len])


# ── Recompression ──────────────────────────────────────────────────────


def deflate_zlib(plain: bytes) -> bytes:
    """Recompress to a valid zlib stream (header + DEFLATE + Adler-32)."""
    return zlib.compress(plain, _COMPRESS_LEVEL)


def deflate_gzip(plain: bytes, mtime: int = 0, os_byte: int = 3) -> bytes:
    """Recompress to a valid gzip member with a correct CRC-32 and ISIZE.

    Built by hand rather than via ``gzip.compress`` so the header fields stay
    under our control (a later mutation may want to corrupt exactly one of
    them) and so no BytesIO wrapper is allocated per call.
    """
    co = zlib.compressobj(_COMPRESS_LEVEL, zlib.DEFLATED, -15)
    body = co.compress(plain) + co.flush()
    header = struct.pack("<BBBBIBB", 0x1F, 0x8B, 8, 0, mtime & 0xFFFFFFFF, 0, os_byte & 0xFF)
    trailer = struct.pack("<II", binascii.crc32(plain) & 0xFFFFFFFF, len(plain) & 0xFFFFFFFF)
    return header + body + trailer


# ── Public operators ───────────────────────────────────────────────────


def _fit(plain: bytes, deflate, max_len: int) -> bytes:
    """Deflate *plain*, shrinking the plaintext until the result fits.

    Truncating the *compressed* output would corrupt the stream and undo the
    whole point of this operator, so the plaintext is trimmed instead. The
    first retry scales by the observed compression ratio, which lands within
    budget in one step for essentially all real input; the loop is bounded at
    three attempts so a pathological ratio cannot spin.
    """
    out = deflate(plain)
    for _ in range(3):
        if len(out) <= max_len or not plain:
            break
        ratio = len(plain) / len(out)
        plain = plain[: max(1, int(max_len * ratio * 0.9))]
        out = deflate(plain)
    return out


def recompress_zlib(data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
    """Inflate, mutate the plaintext, re-deflate as zlib.

    Returns None when *data* is not an inflatable zlib stream, so the caller
    can fall through to another operator instead of emitting garbage.
    """
    plain = inflate_zlib(data)
    if plain is None:
        return None
    plain = plain[:_MAX_PLAIN_WORK]
    mutated = _mutate_plain(plain, _MAX_PLAIN_WORK, rng=rng)
    return _fit(mutated, deflate_zlib, max_len)


def recompress_gzip(data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
    """Inflate, mutate the plaintext, re-deflate as gzip."""
    plain = inflate_gzip(data)
    if plain is None:
        return None
    plain = plain[:_MAX_PLAIN_WORK]
    mutated = _mutate_plain(plain, _MAX_PLAIN_WORK, rng=rng)
    return _fit(mutated, deflate_gzip, max_len)
