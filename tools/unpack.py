"""Members of an archive or compressed file, bounded, for tools/dir_seeds.py --decompress.

Detected by magic, not extension (stdlib only):

    zip                    PK\\x03\\x04 / PK\\x05\\x06   each file member
    tar (plain/gz/bz2/xz)  ustar at 257, or a        each regular-file member
                           compressed tar stream
    gzip / bzip2 / xz      1f 8b / BZh / fd 37 7a..   the decompressed stream

One level only: a .gz inside a zip is a member, not unpacked again (tar
inside gzip/bzip2/xz is the format, not nesting). Tar means POSIX/GNU tar
with the "ustar" magic; a pre-POSIX v7 tar is read as a plain stream. Directories, links and
devices are skipped; nothing is extracted to disk.

Bounds (Hard Rule 54): each member's bytes are read up to *max_bytes*. Its
full size comes from the archive header, or, for a bare stream, by counting
decompressed chunks without keeping them, stopping once past *size_limit*.
At most MAX_UNPACKED bytes are decompressed per stream or declared per
archive, so a decompression bomb ends early, reported as too big.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

# Ceiling on decompressed bytes per stream / declared bytes per archive.
MAX_UNPACKED = 1 << 30
CHUNK = 1 << 20
HEAD_LEN = 512  # enough for the tar "ustar" magic at offset 257

ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
TAR_MAGIC = b"ustar"
TAR_MAGIC_OFF = 257
STREAM_OPENERS = (
    (b"\x1f\x8b", gzip.open),
    (b"BZh", bz2.open),
    (b"\xfd7zXZ\x00", lzma.open),
)

# What corrupt or truncated input raises across the stdlib readers.
UNPACK_ERRORS = (
    tarfile.TarError,
    zipfile.BadZipFile,
    zlib.error,
    lzma.LZMAError,
    EOFError,
    OSError,
    ValueError,
)


class Member(NamedTuple):
    """One unpacked file: its first max_bytes, and its full size."""

    name: str
    data: bytes
    size: int


def is_packed(path: Path) -> bool:
    """True when *path* starts like a zip, tar or gzip/bzip2/xz file."""
    with path.open("rb") as fh:
        head = fh.read(HEAD_LEN)
    if head.startswith(ZIP_MAGICS):
        return True
    return _has_tar_magic(head) or _stream_opener(head) is not None


def members(path: Path, max_bytes: int, size_limit: int) -> Iterator[Member]:
    """Members of *path* (see module doc). Raises one of UNPACK_ERRORS on corrupt input."""
    with path.open("rb") as fh:
        head = fh.read(HEAD_LEN)

    if head.startswith(ZIP_MAGICS):
        yield from _zip_members(path, max_bytes)
        return
    if _is_tar(path, head):
        with tarfile.open(path, "r:*") as tar:
            yield from _tar_members(tar, max_bytes)
        return
    yield _stream_member(path, head, max_bytes, size_limit)


def _has_tar_magic(block: bytes) -> bool:
    return block[TAR_MAGIC_OFF : TAR_MAGIC_OFF + len(TAR_MAGIC)] == TAR_MAGIC


def _is_tar(path: Path, head: bytes) -> bool:
    """Tar only with the ustar magic, plain or after decompressing the first block.

    tarfile alone accepts a stream of zeros as an empty tar (zero blocks end
    an archive), so a gzip of zeros would yield nothing instead of its bytes.
    """
    if _has_tar_magic(head):
        return True
    opener = _stream_opener(head)
    if opener is None:
        return False
    with opener(path, "rb") as fh:
        return _has_tar_magic(fh.read(HEAD_LEN))


def _stream_opener(head: bytes):
    """gzip.open / bz2.open / lzma.open matching *head*'s magic, or None."""
    return next((op for magic, op in STREAM_OPENERS if head.startswith(magic)), None)


def _zip_members(path: Path, max_bytes: int) -> Iterator[Member]:
    """File members in archive order; declared sizes summed against MAX_UNPACKED."""
    declared = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            declared += info.file_size
            if declared > MAX_UNPACKED:
                yield Member(info.filename, b"", MAX_UNPACKED + 1)
                return
            with zf.open(info) as fh:
                yield Member(info.filename, fh.read(max_bytes), info.file_size)


def _tar_members(tar: tarfile.TarFile, max_bytes: int) -> Iterator[Member]:
    """Regular-file members in order; links, dirs and devices skipped."""
    declared = 0
    for info in tar:
        if not info.isreg():
            continue
        declared += info.size
        if declared > MAX_UNPACKED:
            yield Member(info.name, b"", MAX_UNPACKED + 1)
            return
        fh = tar.extractfile(info)
        if fh is None:
            continue
        with fh:
            yield Member(info.name, fh.read(max_bytes), info.size)


def _stream_member(path: Path, head: bytes, max_bytes: int, size_limit: int) -> Member:
    """The whole decompressed stream as one member, sized by counting chunks."""
    opener = _stream_opener(head)
    if opener is None:
        raise ValueError(f"not an archive or compressed stream: {path}")

    # Count past the caller's limit only as far as needed to know it is exceeded.
    stop_at = min(size_limit, MAX_UNPACKED)
    with opener(path, "rb") as fh:
        data = fh.read(max_bytes)
        size = len(data)
        while size <= stop_at and (chunk := fh.read(CHUNK)):
            size += len(chunk)
    return Member(path.name, data, size)
