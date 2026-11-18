"""Heap-chunk SIZE-field mutators targeting the glibc smallbin range.

Phrack #73 "Poor Little Chunks" (Ax): across three commits glibc 2.41/2.43
removed the fastbin size/``PROTECT_PTR`` checks and the unsorted-bin's five
integrity checks.  A chunk whose SIZE field lands in the ``0x20``-``0x3F0``
smallbin range is no longer validated on the malloc fast path, so a corrupted
SIZE that was inert before -- it would have tripped an abort -- can now sleep
and become an overlapping-chunk primitive.

These operators push plausible size/length fields in the *input* into that
range, so a target that copies a field into a heap chunk header (prev_size /
size), or sizes an allocation from it, is driven through the regressed
allocator path.  They know nothing of the target's heap layout; they work on
file bytes, the same domain :func:`fuzzer_tool.core.mutations.structured.size_field_overflow`
works in.  Two complementary strategies cover "before vs after free":

* :func:`smallbin_size` -- *absolute*: overwrite a field with a crafted
  smallbin-range chunk size (size | flag bits).  Direct SIZE corruption.
* :func:`chunk_overlap` -- *relative*: slide an existing field by a few chunk
  granularities, clamped into the range, preserving its flag bits.  Models the
  off-by-a-chunk SIZE that makes a neighbour boundary overlap once the chunk is
  later freed / reallocated.

Both are length-preserving and take their draw source as a required ``rng``
(Hard Rule 16), matching the structured-regularity family they register beside.

glibc 64-bit chunk SIZE layout (``malloc/malloc.c``)::

    63                                3 2 1 0
    +--------------------------------+-+-+-+
    |          chunk size (16-aligned)   |N|M|P|   P = PREV_INUSE
    +--------------------------------+-+-+-+       M = IS_MMAPPED
                                                   N = NON_MAIN_ARENA

``chunksize(p) = p->size & ~0x7``; a valid size is also 16-aligned, so bit 3 is
zero and the three low bits are pure flags.
"""

from __future__ import annotations

import struct

#: MINSIZE -- smallest allocatable chunk on 64-bit glibc.
MIN_CHUNK = 0x20
#: Largest smallbin chunk: one granule below MIN_LARGE_SIZE (0x400).
MAX_SMALLBIN_CHUNK = 0x3F0
#: Chunk size granularity (2 * SIZE_SZ on 64-bit).
CHUNK_ALIGN = 0x10
#: Mask of the three flag bits folded into the SIZE field (PREV_INUSE etc.).
SIZE_BITS = 0x7

#: Every 16-aligned chunk size a smallbin holds: 0x20, 0x30, ... 0x3F0.
SMALLBIN_SIZES = tuple(range(MIN_CHUNK, MAX_SMALLBIN_CHUNK + CHUNK_ALIGN, CHUNK_ALIGN))

#: Flag-bit combinations worth stamping onto a crafted size (PREV_INUSE is the
#: common live-neighbour case; the arena/mmapped bits probe the branches that
#: reject or reroute a chunk).
FLAG_CHOICES = (0x0, 0x1, 0x2, 0x4, 0x5)

#: Candidate field widths: a size_t is 8 on 64-bit / 4 on 32-bit, and 2 covers
#: the u16 length fields many container formats feed to an allocator.
WIDTHS = (2, 4, 8)
_FMT = {2: "H", 4: "I", 8: "Q"}

#: Below one 4-byte field there is nothing to interpret as a size.
_MIN_FIELD = 4


def _is_text(window: bytes) -> bool:
    """A window of only printable ASCII is text, not a packed size field."""
    return all(32 <= b < 127 for b in window)


def _size_field_offsets(data: bytes, width: int) -> list[int]:
    """Width-aligned offsets whose current bytes don't look like text.

    The same heuristic ``size_field_overflow`` uses: a packed integer field is
    non-printable far more often than payload text is, so skipping all-ASCII
    windows biases toward real length/size fields without parsing the format.
    """
    n = len(data)
    max_off = n - width
    return [off for off in range(0, max_off + 1, width) if not _is_text(data[off : off + width])]


def _pick_offset(data: bytes, width: int, rng) -> int:
    """A candidate size-field offset, else a random in-range fallback."""
    candidates = _size_field_offsets(data, width)
    if candidates:
        return candidates[rng.randint(0, len(candidates) - 1)]
    return rng.randint(0, len(data) - width)


def _pick_width(data: bytes, rng) -> int | None:
    """Draw a field width that fits, or ``None`` when the buffer is too small."""
    width = rng.choice(WIDTHS)
    if len(data) < width + 1:
        return None
    return width


def smallbin_size(data: bytes, rng) -> bytes:
    """Overwrite a plausible size field with a crafted smallbin chunk size.

    Picks a 2/4/8-byte field that looks like a size, then stamps it with a
    16-aligned size drawn from ``SMALLBIN_SIZES`` OR'd with a flag combo -- the
    exact shape the 2.41/2.43 fast path now accepts without a check.

    Length-preserving.  Returns *data* unchanged when it is too small to hold a
    field.

    Args:
        data: Input bytes.
        rng: Draw source (``RandPool`` API; tests inject ``ScriptedRng``).
    """
    if len(data) < _MIN_FIELD:
        return data

    width = _pick_width(data, rng)
    if width is None:
        return data

    off = _pick_offset(data, width, rng)
    fmt = ("<" if rng.randint(0, 1) == 0 else ">") + _FMT[width]
    size = SMALLBIN_SIZES[rng.randint(0, len(SMALLBIN_SIZES) - 1)]
    flags = FLAG_CHOICES[rng.randint(0, len(FLAG_CHOICES) - 1)]

    out = bytearray(data)
    struct.pack_into(fmt, out, off, size | flags)
    return bytes(out)


def _clamp_smallbin(size: int) -> int:
    """Clamp a 16-aligned size into ``[MIN_CHUNK, MAX_SMALLBIN_CHUNK]``."""
    if size < MIN_CHUNK:
        return MIN_CHUNK
    if size > MAX_SMALLBIN_CHUNK:
        return MAX_SMALLBIN_CHUNK
    return size


def chunk_overlap(data: bytes, rng) -> bytes:
    """Slide an existing size field by a few chunk granules, staying in range.

    Reads the current field, decodes ``(chunk_size, flags)``, nudges the size
    by ``±k`` granules (k in 1..3), clamps back into the smallbin range and
    re-stamps it with the original flag bits.  A small, flag-preserving delta is
    the "sleeping corruption": the neighbour chunk boundary the allocator
    recomputes is now off by a chunk, which -- unvalidated on 2.41/2.43 --
    overlaps rather than aborting.

    Length-preserving.  Returns *data* unchanged when it is too small.

    Args:
        data: Input bytes.
        rng: Draw source (``RandPool`` API; tests inject ``ScriptedRng``).
    """
    if len(data) < _MIN_FIELD:
        return data

    width = _pick_width(data, rng)
    if width is None:
        return data

    off = _pick_offset(data, width, rng)
    fmt = ("<" if rng.randint(0, 1) == 0 else ">") + _FMT[width]

    current = struct.unpack_from(fmt, data, off)[0]
    flags = current & SIZE_BITS
    size = (current & ~SIZE_BITS) & ~(CHUNK_ALIGN - 1)  # strip flags, align down

    k = rng.randint(1, 3)
    sign = 1 if rng.randint(0, 1) == 0 else -1
    size = _clamp_smallbin(size + sign * k * CHUNK_ALIGN)

    out = bytearray(data)
    struct.pack_into(fmt, out, off, (size | flags) & ((1 << (width * 8)) - 1))
    return bytes(out)
