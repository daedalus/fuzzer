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
import struct
import zlib
from dataclasses import replace

from fuzzer_tool.core.int_checksum import ADLER32, IntModel

# Bounds chosen so a single call stays in the sub-millisecond range for
# typical seeds and cannot blow up on adversarial input.
_MAX_COMPRESSED_IN = 1 << 20  # 1 MiB of compressed input
_MAX_INFLATE = 1 << 22  # 4 MiB decompressed ceiling
# Deflate cost is linear in plaintext size and dominates this operator, so the
# amount actually mutated + recompressed is capped well below _MAX_INFLATE.
# 256 KiB keeps the worst case near 1ms instead of ~20ms at the 4 MiB ceiling.
_MAX_PLAIN_WORK = 1 << 18
_COMPRESS_LEVEL = 1  # speed over ratio

_CACHE_MAX_BYTES = 8 << 20
"""Total plaintext held by the inflate cache.

The cache was originally bounded by *entry count* (256), which is not a
memory bound at all: each entry can hold up to _MAX_INFLATE (4 MiB), so the
ceiling was 256 x 4 MiB = 1 GB, and a corpus of highly compressible inputs
reached it — 251 entries holding 1053 MB, RSS 1040 MB against a 47 MB
baseline. Bounding the bytes is the actual invariant wanted."""

_CACHE_MAX_ENTRY = _MAX_PLAIN_WORK
"""Largest plaintext worth caching. Only the first _MAX_PLAIN_WORK bytes are
ever mutated, so caching more buys nothing, and one huge entry would evict
every useful small one."""

# (wbits, hash(compressed bytes)) -> plaintext, so repeat selections skip the
# inflate. wbits is part of the key: the same bytes must not be served from a
# zlib-decoded entry when asked for as gzip (and vice versa).
_inflate_cache: dict[tuple[int, int], bytes] = {}
_inflate_cache_bytes = 0


def _cache_get(data: bytes, wbits: int) -> bytes | None:
    return _inflate_cache.get((wbits, hash(data)))


def _cache_put(data: bytes, wbits: int, plain: bytes) -> None:
    """Insert under a byte budget, evicting oldest-first.

    Entries above _CACHE_MAX_ENTRY are not cached: they exceed what the
    operators read, and admitting one would flush the rest of the cache.
    """
    global _inflate_cache_bytes
    if len(plain) > _CACHE_MAX_ENTRY:
        return
    key = (wbits, hash(data))
    existing = _inflate_cache.pop(key, None)
    if existing is not None:
        _inflate_cache_bytes -= len(existing)
    # dicts preserve insertion order, so the first key is the oldest.
    while _inflate_cache and _inflate_cache_bytes + len(plain) > _CACHE_MAX_BYTES:
        oldest = next(iter(_inflate_cache))
        _inflate_cache_bytes -= len(_inflate_cache.pop(oldest))
    _inflate_cache[key] = plain
    _inflate_cache_bytes += len(plain)


def _cache_reset() -> None:
    """Drop everything. Exposed for tests and long-run housekeeping."""
    global _inflate_cache_bytes
    _inflate_cache.clear()
    _inflate_cache_bytes = 0


def cache_stats() -> dict:
    return {"entries": len(_inflate_cache), "bytes": _inflate_cache_bytes}


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


def _mutate_plain(plain: bytes, max_len: int, rng) -> bytes:
    """Apply one cheap byte-level mutation to the decompressed payload.

    Deliberately limited to O(1)-ish edits rather than calling back into the
    full operator engine: the value here is reaching the inner parser at all,
    and the outer scheduler already re-selects this operator often enough to
    explore. Anything heavier would show up directly in EPS.
    """
    r = rng
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


def recompress_zlib(data: bytes, max_len: int = 4096, *, rng) -> bytes | None:
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


def recompress_gzip(data: bytes, max_len: int = 4096, *, rng) -> bytes | None:
    """Inflate, mutate the plaintext, re-deflate as gzip."""
    plain = inflate_gzip(data)
    if plain is None:
        return None
    plain = plain[:_MAX_PLAIN_WORK]
    mutated = _mutate_plain(plain, _MAX_PLAIN_WORK, rng=rng)
    return _fit(mutated, deflate_gzip, max_len)


# ── LZ4 frame and PNG IDAT round-trips ─────────────────────────────────

_LZ4_COMPRESSED = 0
# A literal-only block adds one length byte per 255 plus the token; shrink
# the chunk so the encoded block still fits the frame's block maximum.
_LZ4_LEN_EXT = 255
_LZ4_TOKEN_SLACK = 2


def _lz4_rebuild(frame, plain: bytes, store: int) -> bytes:
    """Serialize *frame* carrying *plain*, every checksum and size recomputed.

    Blocks are literal-only compressed (``store`` 0) or stored raw (1); both
    decode everywhere, and literal runs exercise the sequence decoder.
    """
    from fuzzer_tool.core.mutations.lz4 import (  # noqa: PLC0415
        BLOCK_UNCOMPRESSED,
        Lz4Block,
        _block_max,
        _encode_literals,
        serialize_lz4,
    )

    step = _block_max(frame.bd)
    if store == _LZ4_COMPRESSED:
        step -= step // _LZ4_LEN_EXT + _LZ4_TOKEN_SLACK

    blocks = []
    for i in range(0, len(plain), step):
        chunk = plain[i : i + step]
        if store == _LZ4_COMPRESSED:
            enc = _encode_literals(chunk)
            blocks.append(Lz4Block(len(enc), enc))
            continue
        blocks.append(Lz4Block(len(chunk) | BLOCK_UNCOMPRESSED, chunk))

    return serialize_lz4(
        replace(
            frame, blocks=blocks, content_size=None, content_checksum=None, hc=None, end_mark=True
        )
    )


def recompress_lz4(data: bytes, max_len: int = 4096, *, rng) -> bytes | None:
    """Decode an LZ4 frame, mutate the plaintext, re-encode with valid checksums.

    ``lz4_chunk_mutate`` edits frame fields; the frame decoder then rejects
    most outputs before ``lz4_read.c`` sees the content. Returns None when
    *data* is not a decodable ``mode + frame`` or holds no content.
    """
    from fuzzer_tool.core.mutations.lz4 import _decode_content, parse_lz4  # noqa: PLC0415

    frame = parse_lz4(data)
    if frame is None:
        return None

    plain = _decode_content(frame)
    if not plain:
        return None

    mutated = _mutate_plain(plain[:_MAX_PLAIN_WORK], _MAX_PLAIN_WORK, rng=rng)
    store = rng.randint(0, 1)
    out = _fit(mutated, lambda p: _lz4_rebuild(frame, p, store), max_len)
    return out if len(out) <= max_len else None


def recompress_idat(data: bytes, max_len: int = 4096, *, rng) -> bytes | None:
    """Inflate a PNG's IDAT stream, mutate the scanlines, re-deflate.

    ``png_chunk_mutate`` flips compressed IDAT bytes, which breaks inflate
    or the Adler-32, so the filter/unfilter code behind it is rarely reached.
    Here all IDATs merge into one at the first IDAT's position, with a fresh
    zlib stream (valid Adler-32) and chunk CRC. Returns None when *data* is
    not a PNG with an inflatable IDAT stream.
    """
    from fuzzer_tool.core.mutations.png import (  # noqa: PLC0415
        PngChunk,
        parse_png_chunks,
        serialize_png_chunks,
    )

    chunks = parse_png_chunks(data)
    if not chunks:
        return None

    idat = [i for i, c in enumerate(chunks) if c.chunk_type == b"IDAT"]
    if not idat:
        return None

    plain = inflate_zlib(b"".join(chunks[i].data for i in idat))
    if plain is None:
        return None

    # The first IDAT's index equals the count of non-IDAT chunks before it.
    rest = [c for c in chunks if c.chunk_type != b"IDAT"]
    head, tail = rest[: idat[0]], rest[idat[0] :]

    def build(p: bytes) -> bytes:
        return serialize_png_chunks([*head, PngChunk(b"IDAT", deflate_zlib(p)), *tail])

    mutated = _mutate_plain(plain[:_MAX_PLAIN_WORK], _MAX_PLAIN_WORK, rng=rng)
    out = _fit(mutated, build, max_len)
    return out if len(out) <= max_len else None


# ── Adler-32 trailer patcher ───────────────────────────────────────────
#
# A byte-level mutation that leaves DEFLATE decodable still breaks the
# zlib trailer, and the target rejects the stream on the Adler-32 before
# its parser runs. Unlike recompress_*, this keeps the compressed bytes
# and rewrites only the checksum(s) that went stale:
#
#   zlib:  [CMF FLG] [DEFLATE ............] [ADLER32]   <- adler of plaintext
#   PNG:   ... IDAT(len,"IDAT",data,CRC32) ...          <- trailer ends the
#              joined IDAT data; chunk CRC is CRC-32 by spec
#
# The model only gates *when* to patch: the format fixes the algorithm,
# so only the recovered Adler-32 model qualifies (see ``_op_crc_learn``).

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_HEAD_LEN = 8  # length(4) + type(4)
_PNG_CRC_LEN = 4
_ZLIB_HDR_LEN = 2
_ZLIB_FDICT = 0x20
_ZLIB_DICT_ID_LEN = 4
_ADLER_LEN = 4
_RAW_DEFLATE = -15


def _zlib_trailer(stream: bytes) -> tuple[int, int] | None:
    """Return ``(trailer offset, Adler-32 of the inflated data)`` or None.

    Inflates raw DEFLATE, which never checks the trailer, so a stale one
    does not matter. None when the stream is truncated, corrupt, over the
    inflate bound, or has no room for a 4-byte trailer.
    """
    if len(stream) > _MAX_COMPRESSED_IN or not sniff_zlib(stream):
        return None

    start = _ZLIB_HDR_LEN
    if stream[1] & _ZLIB_FDICT:
        start += _ZLIB_DICT_ID_LEN

    obj = zlib.decompressobj(_RAW_DEFLATE)
    try:
        plain = obj.decompress(stream[start:], _MAX_INFLATE)
    except zlib.error:
        return None
    if not obj.eof:
        return None

    end = len(stream) - len(obj.unused_data)
    if end + _ADLER_LEN > len(stream):
        return None
    return end, zlib.adler32(plain)


def _patch_zlib(stream: bytes) -> bytes | None:
    """Rewrite a zlib stream's Adler-32 trailer; None when not a stream."""
    found = _zlib_trailer(stream)
    if found is None:
        return None
    end, adler = found
    return stream[:end] + struct.pack(">I", adler) + stream[end + _ADLER_LEN :]


def _idat_spans(data: bytes) -> list[tuple[int, int]]:
    """``(payload offset, length)`` of every complete IDAT chunk."""
    spans = []
    pos = len(_PNG_SIG)
    while pos + _PNG_HEAD_LEN + _PNG_CRC_LEN <= len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        kind = data[pos + 4 : pos + _PNG_HEAD_LEN]
        body = pos + _PNG_HEAD_LEN
        if body + length + _PNG_CRC_LEN > len(data):
            break
        if kind == b"IDAT":
            spans.append((body, length))
        if kind == b"IEND":
            break
        pos = body + length + _PNG_CRC_LEN
    return spans


def _patch_png(data: bytes) -> bytes:
    """Repair the zlib Adler-32 across a PNG's IDATs and their chunk CRCs.

    When there is no complete IDAT, or the joined data is not a patchable
    stream, the PNG is returned as is: it is still a PNG, and the generic
    trailing patch would overwrite the IEND CRC.
    """
    spans = _idat_spans(data)
    if not spans:
        return data

    stream = b"".join(data[off : off + n] for off, n in spans)
    patched = _patch_zlib(stream)
    if patched is None or patched == stream:
        return data

    out = bytearray(data)
    done = 0
    for off, n in spans:
        new = patched[done : done + n]
        if new != stream[done : done + n]:
            out[off : off + n] = new
            crc = binascii.crc32(out[off - 4 : off + n]) & 0xFFFFFFFF
            struct.pack_into(">I", out, off + n, crc)
        done += n
    return bytes(out)


def patch_adler(data: bytes, model: IntModel) -> bytes | None:
    """Repair the Adler-32 of a bare zlib stream or a PNG's IDAT stream.

    Returns the patched bytes (same length), or None when *model* is not
    the zlib Adler-32 or *data* is neither a PNG nor a zlib stream, so the
    caller can fall through to the generic trailing-field patch.
    """
    if model != ADLER32:
        return None
    if data[: len(_PNG_SIG)] == _PNG_SIG:
        return _patch_png(data)
    return _patch_zlib(data)
