"""Read-only ``corpus/seeds.tar.gz``: seeds archived beside ``seeds/``.

Member layout follows ``seeds.zip`` (``adapters/seed_zip.py``)::

    seeds.tar.gz
    ├── [./][seeds/]ab/id_ab12..           live seed
    ├── [./][seeds/]irreplaceable/..       protected (also crashing/, timeouts/)
    └── [./][seeds/]pruned/.. .pruned/..   skipped: pruned stays pruned

Seeds are keyed by content, so leaf names are free. Streamed (``r|``): one
member in memory at a time, nothing extracted to disk. Only regular files
are read; traversal, absolute names, links and devices are ignored. A torn
or invalid archive is logged and yields what precedes the tear.
"""

from __future__ import annotations

import gzip
import logging
import tarfile
import zlib
from collections.abc import Collection, Iterator
from pathlib import Path

from fuzzer_tool.adapters.seed_zip import SeedTree
from fuzzer_tool.core.hashing import key16

log = logging.getLogger(__name__)

TAR_NAME = "seeds.tar.gz"

_ROOT_DIR = "seeds"
_SKIP_TREES = frozenset(("pruned", ".pruned"))
_PROTECTED = frozenset(t.value for t in SeedTree if t is not SeedTree.MAIN)

# Torn gzip stream, bad header, short read.
_READ_ERRORS = (tarfile.TarError, EOFError, OSError, zlib.error)


def _tree(name: str) -> str | None:
    """Subtree of a member (``""`` = main), or None to skip it.

    e.g. ``./seeds/irreplaceable/x`` -> ``irreplaceable``; ``../x`` -> None.
    """
    if name.startswith("/"):
        return None

    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None

    # A tar of the seeds/ directory carries it as a prefix: same level.
    while len(parts) > 1 and parts[0] == _ROOT_DIR:
        parts = parts[1:]

    head = parts[0] if len(parts) > 1 else ""
    if head in _SKIP_TREES:
        return None
    return head if head in _PROTECTED else ""


def _members(corpus_dir: Path) -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo, str]]:
    """``(archive, member, tree)`` for every readable seed member, in order."""
    path = Path(corpus_dir) / TAR_NAME
    if not path.is_file():
        return

    # gzip.open, not tarfile's "r|gz": tarfile ends a torn stream silently,
    # GzipFile raises EOFError on a missing end-of-stream marker.
    try:
        with gzip.open(path, "rb") as gz, tarfile.open(fileobj=gz, mode="r|") as tf:
            for info in tf:
                if not info.isreg():
                    continue
                tree = _tree(info.name)
                if tree is None:
                    log.debug("%s: ignoring member %r", TAR_NAME, info.name)
                    continue
                yield tf, info, tree
    except _READ_ERRORS as e:
        log.warning("%s: unreadable past this point (%s); kept seeds read so far", path, e)


def _from_leaf(name: str) -> str | None:
    """Content key from a member leaf ``id_<16 hex>``; None when unnamed."""
    leaf = name.rsplit("/", 1)[-1]
    if not leaf.startswith("id_") or len(leaf) != 19:
        return None
    try:
        int(leaf[3:], 16)
    except ValueError:
        return None
    return leaf[3:]


def named(corpus_dir: str | Path) -> Iterator[tuple[str, int, bool, str]]:
    """Yield (seed id, size, protected, hh) for every live member.

    hh is the two-character shard (e.g. ``ab``) from the member path ``seeds/ab/id_...``.
    """
    for _tf, info, tree in _members(Path(corpus_dir)):
        sid = _from_leaf(info.name)
        if sid is not None:
            hh = info.name.split("/")[1] if "/" in info.name else ""
            yield sid, info.size, bool(tree), hh
        # skip non-seed members (tree is not None)


def pick(corpus_dir: str | Path, wanted: Collection[str]) -> Iterator[tuple[str, bytes, str]]:
    """Yield (seed id, data, hh) for the wanted live members, in one streaming pass."""
    for _tf, info, _tree in _members(Path(corpus_dir)):
        sid = _from_leaf(info.name)
        if sid is not None and sid not in wanted:
            continue

        f = _tf.extractfile(info)
        if f is None:
            continue
        try:
            data = f.read()
        except _READ_ERRORS as e:
            log.warning("%s: member %r torn (%s); stopping", TAR_NAME, info.name, e)
            return
        sid = sid or key16(data)
        if sid in wanted:
            hh = info.name.split("/")[1] if "/" in info.name else info.name
            yield sid, data, hh


def live(corpus_dir: str | Path) -> Iterator[tuple[bytes, bool]]:
    """``(data, protected)`` for every live seed member."""
    for tf, info, tree in _members(Path(corpus_dir)):
        f = tf.extractfile(info)
        if f is None:
            continue

        # A tear inside a member's data surfaces here, not in _members.
        try:
            data = f.read()
        except _READ_ERRORS as e:
            log.warning("%s: member %r torn (%s); stopping", TAR_NAME, info.name, e)
            return
        yield data, bool(tree)


def count(corpus_dir: str | Path) -> int:
    """Live unprotected seed members, without reading their data into Python."""
    return sum(1 for _, _, tree in _members(Path(corpus_dir)) if not tree)
