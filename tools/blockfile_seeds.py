#!/usr/bin/env python3
"""Read block files front to back; send each embedded media file as a TCP seed.

For anything holding many messages back to back: block devices (/dev/sda1,
root needed), disk images, memory dumps, Chromium blockfile caches
(``data_1..3``, Windows / old layouts). Each
message is found by its magic, its end by walking the format's own
structure, then it is sent and the read continues after it:

    data_2: [junk][PNG ......IEND][junk][JPEG ......FFD9][GIF ...;]...
                   ^start   end^ send  ^start      end^ send

    PNG   chunks to IEND           GIF   blocks to trailer 0x3B
    JPEG  segments + entropy data  WEBP  RIFF size
          to EOI (EXIF thumbnails  MP4   top-level boxes (ftyp first)
          skipped by length)       WEBM  EBML header + Segment

A body found inside another (a PNG in an MP4 mdat) stays part of it. A
message cut by the end of the file is sent as far as it goes. Invalid
structure after a magic: not a message, the scan moves one byte on.

Flags are those of tools/chrome_cache_seeds.py (--types, --host, --port,
--delay, --max-bytes, --retries, --retry-backoff, --sent-bloom, --dry-run,
--root). Paths may be files or directories; none: every Chrome/Chromium
profile cache.

Usage:
    sudo python tools/blockfile_seeds.py /dev/sda1 --types jpg,png --port 4444
    python tools/blockfile_seeds.py ~/cache_copy/data_2 ~/cache_copy/data_3 --port 4444
    python tools/blockfile_seeds.py disk.img --types jpg,png --dry-run
"""

from __future__ import annotations

import argparse
import mmap
import os
import re
import stat
import struct
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import chrome_cache_seeds as ccs
from chrome_cache_seeds import Media

# _end_of() result: the message runs past the end of the data.
TRUNCATED = -1

# Magic search span per step: first seeds come out without reading a whole disk.
SCAN_WINDOW = 64 << 20
# Bytes between "scanned X of Y GiB" lines.
GIB = 1 << 30
PROGRESS_BYTES = GIB

# Inputs that mmap: regular files and block devices (sized by seeking).
READABLE_KINDS = frozenset({stat.S_IFREG, stat.S_IFBLK})
NOT_A_DISK_KINDS = {
    stat.S_IFCHR: "character device",
    stat.S_IFIFO: "FIFO",
    stat.S_IFSOCK: "socket",
}
# /dev/nvme0 is the controller (char device); namespaces are /dev/nvme0n1.
NVME_CONTROLLER = re.compile(r"/dev/nvme\d+")

# Magic probes: (magic, bytes before the magic where the file starts, media).
PROBES = (
    *((magic, 0, media) for media, magic in ccs.SIGNATURES),
    (ccs.RIFF, 0, Media.WEBP),
    (ccs.FTYP, ccs.BOX_SIZE_LEN, Media.MP4),
)

PNG_SIG_LEN = 8
PNG_MAX_CHUNK = (1 << 31) - 1  # PNG spec: chunk length fits in 31 bits
PNG_CHUNK_OVERHEAD = 12  # length + type + CRC

JPEG_EOI = 0xD9
JPEG_SOS = 0xDA
JPEG_RST = range(0xD0, 0xD8)
JPEG_TEM = 0x01
JPEG_SEGMENTS = range(0xC0, 0xFF)  # length-prefixed markers (minus RST/SOI/EOI)

GIF_HEADERS = (b"GIF87a", b"GIF89a")
GIF_LSD_END = 13  # header 6 + logical screen descriptor 7
GIF_PACKED_OFF = 10
GIF_IMAGE_DESC_LEN = 10
GIF_EXTENSION = 0x21
GIF_IMAGE = 0x2C
GIF_TRAILER = 0x3B
GIF_COLOR_TABLE_FLAG = 0x80

RIFF_HEADER_LEN = 8
WEBP_MIN_SIZE = 4

MP4_BOX_HEADER = 8
MP4_LARGE_HEADER = 16
MP4_TOP_BOXES = frozenset(
    {b"ftyp", b"styp", b"moov", b"mdat", b"free", b"skip", b"wide", b"uuid", b"moof"}
    | {b"mfra", b"sidx", b"ssix", b"meta", b"pdin", b"emsg", b"prft", b"udta"}
)

EBML_ID = bytes.fromhex("1a45dfa3")
WEBM_SEGMENT = bytes.fromhex("18538067")
WEBM_MAX_ID_LEN = 4
WEBM_SEGMENT_CHILDREN = frozenset(
    bytes.fromhex(h)
    for h in (
        "114d9b74",  # SeekHead
        "1549a966",  # Info
        "1654ae6b",  # Tracks
        "1f43b675",  # Cluster
        "1c53bb6b",  # Cues
        "1043a770",  # Chapters
        "1254c367",  # Tags
        "1941a469",  # Attachments
        "ec",  # Void
        "bf",  # CRC-32
    )
)


class _Truncated(Exception):
    """The structure runs past the end of the data."""


class _Invalid(Exception):
    """The bytes after a magic are not that format."""


def _need(data, p: int, n: int) -> None:
    if p + n > len(data):
        raise _Truncated


def _u8(data, p: int) -> int:
    _need(data, p, 1)
    return data[p]


# ---------------------------------------------------------------- end walkers
# Each takes (data, start) and returns the end offset (exclusive), raising
# _Truncated or _Invalid.


def _png_end(data, start: int) -> int:
    """After the IEND chunk; the first chunk must be IHDR."""
    p = start + PNG_SIG_LEN
    first = True
    while True:
        _need(data, p, 8)
        length, kind = struct.unpack_from(">I4s", data, p)
        if length > PNG_MAX_CHUNK or not kind.isalpha() or (first and kind != b"IHDR"):
            raise _Invalid
        first = False

        p += PNG_CHUNK_OVERHEAD + length
        if kind == b"IEND":
            _need(data, start, p - start)
            return p


def _entropy_end(data, p: int) -> int:
    """Offset of the marker ending JPEG entropy data (FF00 stuffing and RSTn skipped)."""
    while True:
        i = data.find(b"\xff", p)
        if i < 0:
            raise _Truncated
        nxt = _u8(data, i + 1)
        if nxt == 0 or nxt in JPEG_RST:
            p = i + 2
            continue
        return i


def _jpg_end(data, start: int) -> int:
    """After EOI. Segments are skipped by length, so EXIF thumbnails stay inside."""
    p = start + 2
    while True:
        _need(data, p, 2)
        if data[p] != 0xFF:
            raise _Invalid
        marker = data[p + 1]
        if marker == 0xFF:  # fill byte
            p += 1
            continue
        if marker == JPEG_EOI:
            return p + 2
        if marker in JPEG_RST or marker == JPEG_TEM:
            p += 2
            continue
        if marker not in JPEG_SEGMENTS or marker in JPEG_RST or marker == 0xD8:
            raise _Invalid

        _need(data, p + 2, 2)
        (length,) = struct.unpack_from(">H", data, p + 2)
        if length < 2:
            raise _Invalid
        p += 2 + length
        if marker == JPEG_SOS:
            p = _entropy_end(data, p)


def _color_table(packed: int) -> int:
    """Bytes of a GIF color table flagged in *packed* (0 when absent)."""
    if not packed & GIF_COLOR_TABLE_FLAG:
        return 0
    return 3 << ((packed & 7) + 1)


def _sub_blocks(data, p: int) -> int:
    """After a GIF sub-block chain (length-prefixed, 0-terminated)."""
    while True:
        n = _u8(data, p)
        p += 1 + n
        if n == 0:
            return p


def _gif_end(data, start: int) -> int:
    """After the 0x3B trailer."""
    _need(data, start, GIF_LSD_END)
    if bytes(data[start : start + 6]) not in GIF_HEADERS:
        raise _Invalid
    p = start + GIF_LSD_END + _color_table(data[start + GIF_PACKED_OFF])

    while True:
        block = _u8(data, p)
        if block == GIF_TRAILER:
            return p + 1
        if block == GIF_EXTENSION:
            p = _sub_blocks(data, p + 2)
            continue
        if block != GIF_IMAGE:
            raise _Invalid
        _need(data, p, GIF_IMAGE_DESC_LEN)
        p += GIF_IMAGE_DESC_LEN + _color_table(data[p + 9]) + 1  # + LZW min code size
        p = _sub_blocks(data, p)


def _webp_end(data, start: int) -> int:
    """RIFF size says it: header + size, padded to even."""
    _need(data, start, 12)
    (size,) = struct.unpack_from("<I", data, start + 4)
    if bytes(data[start + 8 : start + 12]) != ccs.WEBP_TAG or size < WEBP_MIN_SIZE:
        raise _Invalid
    end = start + RIFF_HEADER_LEN + size + (size & 1)
    _need(data, start, end - start)
    return end


def _mp4_end(data, start: int) -> int:
    """After the last known top-level box; the first must be ftyp."""
    p = start
    n = len(data)
    while p + MP4_BOX_HEADER <= n:
        size, kind = struct.unpack_from(">I4s", data, p)
        if p == start and kind != ccs.FTYP:
            raise _Invalid
        if kind not in MP4_TOP_BOXES:
            return p

        header = MP4_BOX_HEADER
        if size == 1:  # 64-bit size follows
            _need(data, p, MP4_LARGE_HEADER)
            (size,) = struct.unpack_from(">Q", data, p + 8)
            header = MP4_LARGE_HEADER
        elif size == 0:  # "to end of file": on a disk, the rest of the disk
            raise _Truncated
        if size < header:
            return p

        p += size
        if p > n:
            raise _Truncated
    return p


def _ebml_size(data, p: int) -> tuple[int | None, int]:
    """(size, offset after it) of an EBML vint; size None when 'unknown' (all ones)."""
    first = _u8(data, p)
    if first == 0:
        raise _Invalid
    length = 9 - first.bit_length()
    _need(data, p, length)
    value = first & (0xFF >> length)
    for i in range(1, length):
        value = (value << 8) | data[p + i]
    unknown = value == (1 << (7 * length)) - 1
    return (None if unknown else value), p + length


def _ebml_id(data, p: int) -> tuple[bytes, int]:
    """(element id with its marker bits, offset after it)."""
    first = _u8(data, p)
    length = 9 - first.bit_length()
    if first == 0 or length > WEBM_MAX_ID_LEN:
        raise _Invalid
    _need(data, p, length)
    return bytes(data[p : p + length]), p + length


def _webm_end(data, start: int) -> int:
    """After the Segment; an unknown-size Segment ends after its last known child."""
    size, p = _ebml_size(data, start + len(EBML_ID))
    if size is None:
        raise _Invalid
    p += size
    ident, p = _ebml_id(data, p)
    if ident != WEBM_SEGMENT:
        raise _Invalid

    size, p = _ebml_size(data, p)
    if size is not None:
        _need(data, p, size)
        return p + size
    return _webm_children_end(data, p)


def _webm_children_end(data, p: int) -> int:
    """Walk Segment children of known size; stop at the first foreign element."""
    n = len(data)
    while p < n:
        try:
            ident, q = _ebml_id(data, p)
        except _Invalid:
            return p
        if ident not in WEBM_SEGMENT_CHILDREN:
            return p
        size, q = _ebml_size(data, q)
        if size is None:  # live stream cluster: end unknown
            raise _Truncated
        p = q + size
        if p > n:
            raise _Truncated
    return p


ENDS: dict[Media, Callable[[object, int], int]] = {
    Media.PNG: _png_end,
    Media.JPG: _jpg_end,
    Media.GIF: _gif_end,
    Media.WEBP: _webp_end,
    Media.MP4: _mp4_end,
    Media.WEBM: _webm_end,
}


# -------------------------------------------------------------------- scanner
class _Finder:
    """Earliest magic in a bounded window; caches each probe's result.

    Windows bound every find(): a rare magic is never searched to the end of a
    disk before the first seed goes out. Per probe, a hit at or after the
    window start is reused; a miss remembers how far it searched.
    """

    def __init__(self, data) -> None:
        self._data = data
        self._n = len(data)
        self._hit: list[int | None] = [None] * len(PROBES)  # file start of the probe's next hit
        self._upto = [0] * len(PROBES)  # starts below this were searched (when no hit)

    def next(self, pos: int) -> tuple[int, Media] | None:
        """Earliest (start, media) with start >= pos, or None; searched window by window."""
        while pos < self._n:
            hit, hi = self.in_window(pos)
            if hit is not None:
                return hit
            pos = hi
        return None

    def in_window(self, pos: int) -> tuple[tuple[int, Media] | None, int]:
        """(earliest (start, media) in [pos, end), or None; end of the window)."""
        hi = min(pos + SCAN_WINDOW, self._n)
        hits, upto = self._hit, self._upto
        best, best_media = hi, None

        # Hot loop (once per message): cached hit and cached miss are checked
        # inline; only a real search pays for a call.
        for i, (magic, back, media) in enumerate(PROBES):
            hit = hits[i]
            if hit is None or hit < pos:
                frm = pos if hit is not None else max(pos, upto[i])
                if frm >= hi:
                    continue
                hit = self._search(i, magic, back, frm, hi)
                if hit is None:
                    continue
            if hit < best:
                best, best_media = hit, media

        return ((best, best_media) if best_media is not None else None), hi

    def _search(self, i: int, magic: bytes, back: int, frm: int, hi: int) -> int | None:
        """First file start of probe *i* in [frm, hi); caches hit or miss."""
        # A magic straddling hi still counts: its file start is below hi.
        j = self._data.find(magic, frm + back, min(hi + back + len(magic) - 1, self._n))
        if j < 0:
            self._hit[i], self._upto[i] = None, hi
            return None
        self._hit[i] = j - back
        return j - back


def _end_of(media: Media, data, start: int) -> int | None:
    """End of the message at *start*, TRUNCATED when it runs past the data, None if not one."""
    try:
        end = ENDS[media](data, start)
    except _Truncated:
        return TRUNCATED
    except _Invalid:
        return None
    return end if end > start else None


def _no_progress(_pos: int) -> None:
    """Default on_progress: report nothing."""


def scan(
    data,
    wanted: set[Media],
    max_bytes: int,
    on_progress: Callable[[int], None] = _no_progress,
) -> Iterator[tuple[Media, bytes]]:
    """(media, bytes) of each message in *data*, in order, capped at *max_bytes*.

    *data* is bytes-like with ``find`` (bytes, mmap). Unwanted types are walked
    over, not searched inside. *on_progress* gets the scan offset at least once
    per SCAN_WINDOW, stretches without any magic included.
    """
    finder = _Finder(data)
    n = len(data)
    pos = 0
    report_at = SCAN_WINDOW
    while pos < n:
        if pos >= report_at:
            on_progress(pos)
            report_at = pos + SCAN_WINDOW

        hit, hi = finder.in_window(pos)
        if hit is None:
            pos = hi
            continue

        start, media = hit
        end = _end_of(media, data, start)
        if end is None:
            pos = start + 1
            continue

        # Truncated: sent as far as it goes, but the scan resumes at start + 1,
        # or a bogus length near the start of a raw disk would skip all the rest.
        resume = end
        if end is TRUNCATED:
            end, resume = n, start + 1
        if media in wanted:
            yield media, bytes(data[start : min(end, start + max_bytes)])
        pos = resume
    on_progress(n)


class _Progress:
    """on_progress for one file: a stderr line each PROGRESS_BYTES scanned."""

    def __init__(self, path: Path, size: int) -> None:
        self._path = path
        self._size = size
        self._next = PROGRESS_BYTES

    def __call__(self, pos: int) -> None:
        if pos < self._next:
            return
        print(
            f"[*] {self._path}: scanned {pos / GIB:.1f} of {self._size / GIB:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )
        self._next = (pos // PROGRESS_BYTES + 1) * PROGRESS_BYTES


# ---------------------------------------------------------------------- files
def _files(paths: Iterable[Path]) -> Iterator[Path]:
    """Each path; a directory gives its regular files, sorted."""
    for path in paths:
        if path.is_dir():
            yield from (p for p in sorted(path.iterdir()) if p.is_file())
        else:
            yield path


def _scan_file(path: Path, wanted: set[Media], max_bytes: int) -> Iterator[tuple[Media, bytes]]:
    """scan() over an mmap of *path*: memory stays bounded for multi-GB dumps / disks.

    Size by seeking to the end, not st_size: a block device (/dev/sda1) reports
    st_size 0, and mmap length 0 ("whole file") fails on it.
    """
    # stat before open: a FIFO open blocks forever; char devices don't seek.
    fmt = stat.S_IFMT(path.stat().st_mode)
    if fmt not in READABLE_KINDS:
        raise OSError(not_a_disk(str(path), fmt))

    with path.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return
        with mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ) as mm:
            yield from scan(mm, wanted, max_bytes, _Progress(path, size))


def not_a_disk(path: str, fmt: int) -> str:
    """Why *path* (file type *fmt*) can't be scanned, and what to pass instead."""
    kind = NOT_A_DISK_KINDS.get(fmt, "not a regular file or block device")
    if NVME_CONTROLLER.fullmatch(path):
        return f"{kind} (NVMe controller); the disk is {path}n1, its partitions {path}n1p1, ..."
    return f"{kind}; pass a block device (/dev/sda1, /dev/nvme0n1p1), an image or a dump"


def carve_files(
    paths: Iterable[Path], wanted: set[Media], max_bytes: int, stats: ccs.ScanStats
) -> Iterator[tuple[Media, bytes]]:
    """Messages of every file in order; unreadable files counted, not fatal."""
    for path in _files(paths):
        try:
            yield from _scan_file(path, wanted, max_bytes)
        except OSError as e:
            # Not fatal, but said: /dev/sda1 without root would otherwise read as "sent 0".
            print(f"[!] cannot read {path}: {e}", file=sys.stderr)
            stats.unreadable += 1


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="block devices (/dev/sda1), image/dump files or dirs (default: browser caches)",
    )
    ccs.add_send_args(ap)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"[-] no such file: {', '.join(missing)}", file=sys.stderr)
        return 1
    paths = args.paths or ccs.discover(args)
    if not paths:
        return 1

    stats = ccs.ScanStats()
    seeds = ccs.unique(carve_files(paths, args.types, args.max_bytes, stats), stats)
    return ccs.send_seeds(seeds, args, stats)


if __name__ == "__main__":
    sys.exit(main())
