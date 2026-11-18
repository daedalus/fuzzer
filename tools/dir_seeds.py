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
import enum
import fnmatch
import os
import stat
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import chrome_cache_seeds as ccs
import unpack
from chrome_cache_seeds import Media

from fuzzer_tool.core.rand_pool import RandPool

# Size given to members past unpack.MAX_UNPACKED: larger than any --max-size.
TOO_BIG = sys.maxsize + 1
# Drawn seeds stay printable and pasteable.
MAX_SEED = (1 << 32) - 1


class Unpack(enum.Enum):
    """--decompress: send an archive's members instead of the archive."""

    OFF = "off"
    ON = "on"


class WalkOrder(enum.Enum):
    """Order of files and subdirectories within each directory (--order)."""

    SORTED = "sorted"
    RANDOM = "random"


def arranger(order: WalkOrder, seed: int) -> Callable[[list], None]:
    """In-place list orderer for walk(): sort, or sort then shuffle with *seed*.

    Sorting first makes the shuffle depend on the seed only, not on the
    order the filesystem lists entries in: same seed, same walk, any machine.
    """
    if order is WalkOrder.SORTED:
        return list.sort
    rng = RandPool(seed)

    def shuffled(items: list) -> None:
        items.sort()
        rng.shuffle(items)

    return shuffled


def _dir_excluded(name: str, full: str, patterns: list[str]) -> bool:
    """True when *name* or *full* path matches any glob in *patterns*."""
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(full, p) for p in patterns)


def walk(
    paths: Iterable[Path],
    arrange: Callable[[list], None] = list.sort,
    exclude_dirs: list[str] | None = None,
) -> Iterator[Path]:
    """Files under *paths*: per directory its files then subdirs, each list ordered by *arrange*.

    A file argument is yielded as is. Directory symlinks are listed by os.walk
    but not entered, so a link back to an ancestor cannot loop.
    Directories matching *exclude_dirs* glob patterns (on basename or full path)
    are pruned and not descended into.
    """
    patterns = list(exclude_dirs or [])
    for path in paths:
        if not path.is_dir():
            yield path
            continue
        for top, dirs, files in os.walk(path, followlinks=False):
            if patterns:
                dirs[:] = [d for d in dirs if not _dir_excluded(d, os.path.join(top, d), patterns)]
            arrange(dirs)  # in place: os.walk descends in this order
            arrange(files)
            for name in files:
                yield Path(top) / name


def classify(data: bytes) -> Media:
    """Media whose magic sits at offset 0; OTHER when none does."""
    hit = ccs.sniff(data[: ccs.SNIFF_WINDOW])
    if hit is None or hit[1] != 0:
        return Media.OTHER
    return hit[0]


def _read(path: Path, max_bytes: int) -> tuple[bytes, int]:
    """(first *max_bytes*, file size) of a regular file; OSError for anything else."""
    # stat before open: opening a FIFO for reading blocks until a writer appears.
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise OSError(f"not a regular file: {path}")
    with path.open("rb") as fh:
        return fh.read(max_bytes), st.st_size


def iter_dir_seeds(
    paths: Iterable[Path],
    wanted: set[Media],
    max_bytes: int,
    stats: ccs.ScanStats,
    skip: frozenset[Path] = frozenset(),
    max_size: int = ccs.NO_LIMIT,
    arrange: Callable[[list], None] = list.sort,
    unpack_mode: Unpack = Unpack.OFF,
    exclude_dirs: list[str] | None = None,
) -> Iterator[tuple[Media, bytes]]:
    """(media, bytes) of every non-empty file of the *wanted* types, in walk order.

    A wanted file over *max_size* bytes (file size, before *max_bytes*) is
    skipped and counted in stats.too_big. With Unpack.ON an archive or
    compressed file yields its members instead (tools/unpack.py).
    Directories matching *exclude_dirs* glob patterns are pruned before walking.
    """
    # Ifs up: the file -> payloads step is chosen once, not per file.
    payloads = _unpacked if unpack_mode is Unpack.ON else _as_is
    for path in walk(paths, arrange, exclude_dirs):
        if path.resolve() in skip:
            continue
        try:
            head, file_size = _read(path, max_bytes)
            for data, size in payloads(path, head, file_size, max_bytes, max_size):
                if (offer := _offer(data, size, wanted, max_size, stats)) is not None:
                    yield offer
        except OSError:
            # Dangling link, permission, FIFO/device: counted, not fatal.
            stats.unreadable += 1
        except unpack.UNPACK_ERRORS as e:
            # Corrupt archive: members before the damage were already sent.
            print(f"[!] cannot unpack {path}: {e}", file=sys.stderr)
            stats.unreadable += 1


def _offer(
    data: bytes, size: int, wanted: set[Media], max_size: int, stats: ccs.ScanStats
) -> tuple[Media, bytes] | None:
    """(media, data) if wanted and within *max_size*; None otherwise (too big counted)."""
    media = classify(data)
    if media not in wanted:
        return None
    if size > max_size:
        stats.too_big += 1
        return None
    if not data:
        return None
    return media, data


def _as_is(
    path: Path, data: bytes, size: int, _max_bytes: int, _max_size: int
) -> Iterator[tuple[bytes, int]]:
    """Unpack.OFF: the file itself."""
    yield data, size


def _unpacked(
    path: Path, data: bytes, size: int, max_bytes: int, max_size: int
) -> Iterator[tuple[bytes, int]]:
    """Unpack.ON: an archive's / compressed file's members; any other file as is."""
    if not unpack.is_packed(path):
        yield data, size
        return
    for member in unpack.members(path, max_bytes, max_size):
        # Past the hard cap (a bomb, or a lying header): always too big.
        over_cap = member.size > unpack.MAX_UNPACKED
        yield member.data, (TOO_BIG if over_cap else member.size)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", type=Path, nargs="+", help="directories (walked) or files")
    ap.add_argument(
        "--order",
        choices=[o.value for o in WalkOrder],
        default=WalkOrder.SORTED.value,
        help="sorted (default) or random: shuffle the paths, and each directory's files and subdirs",
    )
    ap.add_argument("--seed", type=int, help="seed for --order random (default: drawn and printed)")
    ap.add_argument(
        "--decompress",
        action="store_true",
        help="send the members of zip / tar(.gz/.bz2/.xz) / gzip / bzip2 / xz files, one level deep",
    )
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="GLOB",
        help="exclude directories matching glob pattern (can repeat, e.g. --exclude-dir '*/.git' --exclude-dir '*/__pycache__')",
    )
    ccs.add_send_args(ap)
    args = ap.parse_args(argv)
    args.order = WalkOrder(args.order)
    if args.seed is not None and args.order is not WalkOrder.RANDOM:
        ap.error("--seed needs --order random")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"[-] no such file: {', '.join(missing)}", file=sys.stderr)
        return 1
    # Ifs up: the ordering is chosen once and bound into walk().
    seed = args.seed if args.seed is not None else RandPool().randint(0, MAX_SEED)
    arrange = arranger(args.order, seed)
    paths = list(args.paths)
    if args.order is WalkOrder.RANDOM:
        arrange(paths)
        print(f"[*] walk order: random (seed {seed})")
    for path in paths:
        print(f"[*] walking: {path}")

    stats = ccs.ScanStats()
    skip = frozenset({args.sent_bloom.resolve()}) if args.sent_bloom else frozenset()
    unpack_mode = Unpack.ON if args.decompress else Unpack.OFF
    if args.exclude_dir:
        print(f"[*] excluding dirs matching: {', '.join(args.exclude_dir)}")
    found = iter_dir_seeds(
        paths,
        args.types,
        args.max_bytes,
        stats,
        skip,
        args.max_size,
        arrange,
        unpack_mode,
        args.exclude_dir,
    )
    return ccs.send_seeds(ccs.unique(found, stats), args, stats)


if __name__ == "__main__":
    sys.exit(main())
