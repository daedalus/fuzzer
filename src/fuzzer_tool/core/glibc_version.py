"""glibc little-chunk bin-entry protections, by version.

Source: "Poor Little Chunks" (Ax, Phrack #73, 2026). Four commits across
two releases dropped the validation small chunks (0x20-0x3F0 bytes, i.e.
"little chunks") used to pass through on their way into a bin:

- 2.41 (e2436d6f5a): the unsorted bin's five integrity checks (size in
  bounds, neighbor's size in bounds, neighbor agrees on size, fd/bk
  consistent, marked free) are skipped for the 55 sizes 0x90-0x3F0
  (144-1008 bytes). free() now drops them straight into smallbin.
- 2.43 (bf1015fb2d / bb5a4f5295): fastbin is gone. The seven sizes
  0x20-0x80 (32-128 bytes) lose PROTECT_PTR on `fd` (safe-linking) and the
  `fastbin_index(chunksize(victim)) != idx` check at retrieval; they too
  now go straight from free() into smallbin.
- 0b9210bd76: tcache per-size capacity doubled, 7 -> 16 (fewer chunks
  reach the arena path at all; not a protection, a mitigation of degree).

Neither tcache (`tcache_put`/`tcache_get`) nor smallbin's exact-fit
retrieval (`bck->fd == victim`, one direction only) ever validates a
chunk's SIZE field or calls `set_head`. So on an exposed glibc, a SIZE
field corrupted after free() survives retrieval and a second free()
completely unnoticed ("sleeping corruption"): what used to abort
deterministically at the fastbin/unsorted-bin gate now doesn't abort at
all, and if a crash results it lands somewhere else entirely, later, with
a stack trace unrelated to the actual corruption site.

This module only encodes which protections a given (major, minor) and
chunk size had; it does not detect corruption, and it says nothing about
chunks of 0x400 bytes or more (large chunks; full safe-unlinking and
bin-fit checks there are untouched by this change).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


def _query_gnu_libc_version() -> str | None:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    except OSError:
        return None
    fn = getattr(libc, "gnu_get_libc_version", None)
    if fn is None:  # musl, macOS libc: not glibc, nothing in this module applies
        return None
    fn.restype = ctypes.c_char_p
    raw = fn()
    return raw.decode("ascii", errors="replace") if raw else None


_cache_set = False
_cache: tuple[int, int] | None = None


def detect(*, _refresh: bool = False) -> tuple[int, int] | None:
    """(major, minor) of the glibc this process is linked against.

    None on musl / macOS / anything without ``gnu_get_libc_version``, or
    on a version string this doesn't recognize. Cached after the first
    call -- the host's glibc does not change mid-run; pass
    ``_refresh=True`` (tests only) to bypass the cache.
    """
    global _cache_set, _cache
    if _refresh or not _cache_set:
        raw = _query_gnu_libc_version()
        m = _VERSION_RE.match(raw) if raw else None
        _cache = (int(m.group(1)), int(m.group(2))) if m else None
        _cache_set = True
    return _cache


# Version at which each fence fell.
UNSORTED_CHECKS_DROPPED = (2, 41)  # e2436d6f5a
FASTBIN_KILLED = (2, 43)  # bf1015fb2d / bb5a4f5295

# Size-class boundaries, in *chunksize* (glibc's internal, 16-byte-aligned
# size including its header), not malloc(n)'s requested n.
_FALLEN_SEVEN = frozenset(range(0x20, 0x81, 0x10))  # 0x20, 0x30, ..., 0x80
_MIDDLE_RANGE = (0x90, 0x3F0)


def _size_class(size: int) -> str | None:
    if size in _FALLEN_SEVEN:
        return "fallen_seven"
    if _MIDDLE_RANGE[0] <= size <= _MIDDLE_RANGE[1]:
        return "middle_fiftyfive"
    return None


@dataclass(frozen=True)
class LittleChunkExposure:
    """What a chunk of a given size lost, on a given glibc."""

    size_class: str  # "fallen_seven" (0x20-0x80) or "middle_fiftyfive" (0x90-0x3F0)
    lost: tuple[str, ...]  # protections no longer enforced for this size, on this glibc
    note: str  # one-line, operator-facing


def little_chunk_exposure(size: int, version: tuple[int, int] | None) -> LittleChunkExposure | None:
    """What a *size*-byte little chunk lost on glibc *version*.

    None when *size* falls outside 0x20-0x3F0, *version* is None
    (non-glibc or undetected), or *version* predates the commit that
    affects this size class -- the chunk still has its pre-2.41/2.43
    fences.
    """
    if version is None:
        return None
    size_class = _size_class(size)
    if size_class is None:
        return None

    if size_class == "fallen_seven":
        if version < FASTBIN_KILLED:
            return None
        return LittleChunkExposure(
            size_class,
            (
                "PROTECT_PTR (fd mangling)",
                "fastbin_index size-index check at retrieval",
                "_int_free early return (the merge path is now reachable)",
            ),
            f"0x{size:x}-byte chunk on glibc {version[0]}.{version[1]}: fastbin is gone "
            "(bf1015fb2d/bb5a4f5295) -- fd/bk are raw pointers, nothing checks SIZE at "
            "retrieval, and all three overlapping-chunk approaches from 'Glibc "
            "adventures: The forgotten chunks' now apply to this size, including "
            "sleeping corruption surviving through smallbin.",
        )

    # middle_fiftyfive
    if version < UNSORTED_CHECKS_DROPPED:
        return None
    return LittleChunkExposure(
        size_class,
        ("unsorted bin's five integrity checks (b90ddd08f6)",),
        f"0x{size:x}-byte chunk on glibc {version[0]}.{version[1]}: the unsorted bin's "
        "five checks are skipped for this size (e2436d6f5a) -- a corrupted SIZE now "
        "lands in smallbin unvalidated, so sleeping corruption survives the arena path.",
    )


def version_note(version: tuple[int, int] | None) -> str:
    """One-line, size-unaware summary of this glibc's little-chunk exposure.

    For a crash where no chunk size is known (a raw glibc heap-corruption
    abort, not a sanitizer report with a region size): tells the operator
    which little chunk sizes, on this glibc, might not abort at all for a
    similar corruption -- rather than scoring severity on data this path
    doesn't have.
    """
    if version is None:
        return ""
    bits = []
    if version >= FASTBIN_KILLED:
        bits.append("0x20-0x80 chunks: fastbin gone, no PROTECT_PTR/size check (>=2.43)")
    if version >= UNSORTED_CHECKS_DROPPED:
        bits.append("0x90-0x3F0 chunks: unsorted-bin's 5 checks gone (>=2.41)")
    if not bits:
        return ""
    return f"glibc {version[0]}.{version[1]}: " + "; ".join(bits)
