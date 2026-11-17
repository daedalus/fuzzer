#!/usr/bin/env python3
"""Walk directories; send every file as one TCP seed.

For corpora, sample collections, extracted archives: each regular file is one
message (send, close), read up to --max-bytes. Feeds a run started with
``fuzzer-tool fuzz ... --tcp-listen-port PORT``.

    corpus/            sent in this order:
    |-- y              1. y          (a directory's own files, sorted,
    |-- z              2. z           then its subdirectories, sorted)
    |-- a/1            3. a/1
    |   `-- q/0        4. a/q/0
    `-- b/2            5. b/2

Type comes from the magic at offset 0 (jpg, png, gif, webp, mp4, webm), else
``other``; ``--types png,other`` filters on it (default: every file).
Directory symlinks are not followed (no loops); FIFOs and devices are skipped
and counted unreadable; the --sent-bloom file is never sent.

Flags are those of tools/chrome_cache_seeds.py (--types, --host, --port,
--delay, --max-bytes, --retries, --retry-backoff, --sent-bloom, --dry-run;
--root is accepted and unused: paths are required).

Usage:
    python tools/dir_seeds.py ~/fuzzing/corpuses/png_samples --port 4444
    python tools/dir_seeds.py ~/samples --types jpg,png --sent-bloom ~/fuzzing/dir_sent.pkl.gz
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import chrome_cache_seeds as ccs
from chrome_cache_seeds import Media


def walk(paths: Iterable[Path]) -> Iterator[Path]:
    """Files under *paths*: per directory its files then subdirs, both sorted.

    A file argument is yielded as is. Directory symlinks are listed by os.walk
    but not entered, so a link back to an ancestor cannot loop.
    """
    for path in paths:
        if not path.is_dir():
            yield path
            continue
        for top, dirs, files in os.walk(path, followlinks=False):
            dirs.sort()  # in place: os.walk descends in this order
            for name in sorted(files):
                yield Path(top) / name


def classify(data: bytes) -> Media:
    """Media whose magic sits at offset 0; OTHER when none does."""
    hit = ccs.sniff(data[: ccs.SNIFF_WINDOW])
    if hit is None or hit[1] != 0:
        return Media.OTHER
    return hit[0]


def _read(path: Path, max_bytes: int) -> bytes:
    """First *max_bytes* of a regular file; OSError for anything else."""
    # stat before open: opening a FIFO for reading blocks until a writer appears.
    if not stat.S_ISREG(path.stat().st_mode):
        raise OSError(f"not a regular file: {path}")
    with path.open("rb") as fh:
        return fh.read(max_bytes)


def iter_dir_seeds(
    paths: Iterable[Path],
    wanted: set[Media],
    max_bytes: int,
    stats: ccs.ScanStats,
    skip: frozenset[Path] = frozenset(),
) -> Iterator[tuple[Media, bytes]]:
    """(media, bytes) of every non-empty file of the *wanted* types, in walk order."""
    for path in walk(paths):
        if path.resolve() in skip:
            continue
        try:
            data = _read(path, max_bytes)
        except OSError:
            # Dangling link, permission, FIFO/device: counted, not fatal.
            stats.unreadable += 1
            continue
        if not data:
            continue
        media = classify(data)
        if media in wanted:
            yield media, data


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", type=Path, nargs="+", help="directories (walked) or files")
    ccs.add_send_args(ap)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"[-] no such file: {', '.join(missing)}", file=sys.stderr)
        return 1
    for path in args.paths:
        print(f"[*] walking: {path}")

    stats = ccs.ScanStats()
    skip = frozenset({args.sent_bloom.resolve()}) if args.sent_bloom else frozenset()
    found = iter_dir_seeds(args.paths, args.types, args.max_bytes, stats, skip)
    return ccs.send_seeds(ccs.unique(found, stats), args, stats)


if __name__ == "__main__":
    sys.exit(main())
