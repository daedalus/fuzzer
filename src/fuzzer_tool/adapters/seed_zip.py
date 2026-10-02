"""``--zip-seed-corpus``: seeds written to ``corpus/seeds.zip`` instead of files.

Reads union ``seeds/**`` (files, unchanged) with ``seeds.zip``; writes go to
the zip only. Deltas stay in ``deltas/`` -- they are not seeds.

Member layout mirrors the file tree::

    seeds.zip
    ├── ab/id_ab12...                 live corpus seed      (seeds/ab/...)
    ├── irreplaceable/cd/id_cd34...   protected seed        (seeds/irreplaceable/...)
    ├── crashing/.. timeouts/..       protected seed
    └── .pruned/ab/id_ab12...         empty tombstone: hides the main member

A zip member cannot be moved, so retiring (``seeds/pruned/`` for files) is an
appended tombstone. Members are processed in archive order, so a seed
re-admitted after its tombstone is live again. The data member is never
dropped: rehydration still finds pruned seeds, as it does under pruned/.

Block mode: writes are buffered and appended ``BLOCK_SEEDS`` / ``BLOCK_BYTES``
at a time, plus on every state save. The append handle stays open for the
run; each block ends by rewriting the central directory in place, so the file
on disk is a valid archive after every flush. Reopening per block instead
re-parses the whole directory: measured 55 ms per block at 10k entries, half
of the total write cost (tools/bench_seed_zip.py). The block size grows with
the archive (BLOCK_GROWTH) because the in-place directory rewrite is itself
O(entries).

A kill mid-append leaves local entries intact and the central directory
torn. ``zipfile`` in "a" mode would then silently start a NEW archive after
the garbage, hiding every old seed -- so the archive is salvaged by walking
local headers before anything is read or appended.
"""

from __future__ import annotations

import atexit
import logging
import os
import struct
import warnings
import zipfile
import zlib
from collections.abc import Iterator
from enum import Enum, StrEnum
from pathlib import Path

log = logging.getLogger(__name__)

ZIP_NAME = "seeds.zip"
COMPRESS_LEVEL = 9
BLOCK_SEEDS = 64
BLOCK_BYTES = 8 << 20  # bounds pending memory and data lost to a kill
# Block grows to entries/BLOCK_GROWTH seeds: each flush rewrites the whole
# central directory, so a fixed block makes total directory writes O(N^2);
# a proportional one makes them ~(BLOCK_GROWTH + 1) entries per seed.
BLOCK_GROWTH = 16

_TOMB = ".pruned"
_HASH_LEN = 16
_HEX = frozenset("0123456789abcdef")

# Local file header (APPNOTE 4.3.7).
_LFH_SIG = b"PK\x03\x04"
_LFH = struct.Struct("<4s5H3L2H")
_DATA_DESCRIPTOR_FLAG = 0x08


class ZipMode(Enum):
    OFF = "off"
    ON = "on"


class SeedTree(StrEnum):
    """Zip subtree, named like the seeds/ subdirectory it replaces."""

    MAIN = ""
    IRREPLACEABLE = "irreplaceable"
    CRASHING = "crashing"
    TIMEOUTS = "timeouts"


_PROTECTED = frozenset(t.value for t in SeedTree if t is not SeedTree.MAIN)


def _member(tree: str, h: str) -> str:
    prefix = f"{tree}/" if tree else ""
    return f"{prefix}{h[:2]}/id_{h}"


def _parse(name: str) -> tuple[str, str] | None:
    """``(tree, hash)`` for a well-formed member name, else None.

    Rejects traversal, absolute paths, foreign subtrees and non-hash leaves:
    a name is only ever looked up, never joined onto a filesystem path, but
    an unexpected name must not be treated as corpus data either.
    """
    parts = name.split("/")
    if len(parts) == 2:
        tree, (shard, leaf) = "", parts
    elif len(parts) == 3 and (parts[0] in _PROTECTED or parts[0] == _TOMB):
        tree, shard, leaf = parts
    else:
        return None

    h = leaf[3:]
    if not leaf.startswith("id_") or len(h) != _HASH_LEN or not set(h) <= _HEX:
        return None
    if shard != h[:2]:
        return None
    return tree, h


def _salvage(path: Path) -> None:
    """Rebuild *path* from its local headers; keep the original as .corrupt.

    Stops at the first torn or inconsistent entry: everything before it was
    written completely, nothing after it can be trusted.
    """
    raw = path.read_bytes()
    entries: list[tuple[str, bytes]] = []
    off = 0
    while off + _LFH.size <= len(raw):
        sig, _v, flag, method, _t, _d, crc, csize, usize, nlen, xlen = _LFH.unpack_from(raw, off)
        if sig != _LFH_SIG or flag & _DATA_DESCRIPTOR_FLAG:
            break

        start = off + _LFH.size + nlen + xlen
        end = start + csize
        if end > len(raw):
            break

        data = _inflate(method, raw[start:end])
        if data is None or len(data) != usize or zlib.crc32(data) != crc:
            break

        entries.append(
            (raw[off + _LFH.size : off + _LFH.size + nlen].decode("utf-8", "replace"), data)
        )
        off = end

    n = 0
    while (corrupt := path.with_name(f"{path.name}.corrupt.{n}")).exists():
        n += 1
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL) as zf:
        _write_all(zf, entries)
    os.replace(path, corrupt)
    os.replace(tmp, path)
    log.warning("seeds.zip: salvaged %d members, original kept as %s", len(entries), corrupt.name)


def _inflate(method: int, payload: bytes) -> bytes | None:
    if method == zipfile.ZIP_STORED:
        return payload
    if method != zipfile.ZIP_DEFLATED:
        return None
    try:
        return zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
    except zlib.error:
        return None


def _checkpoint(zf: zipfile.ZipFile) -> bool:
    """Write the central directory now, keeping *zf* open for the next block.

    The next writestr() seeks back to ``start_dir`` and overwrites this
    directory, exactly as close()+reopen would, without re-parsing it.
    Private zipfile API: False (caller closes instead) if it is gone.
    """
    end_record = getattr(zf, "_write_end_record", None)
    if end_record is None or not hasattr(zf, "start_dir"):
        return False
    with zf._lock:
        zf.fp.seek(zf.start_dir)
        end_record()
    return True


def _write_all(zf: zipfile.ZipFile, entries) -> None:
    # Re-admission after a tombstone reuses a name; order carries meaning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Duplicate name", UserWarning)
        for name, data in entries:
            zf.writestr(name, data)


class SeedZip:
    """Append-only seed archive for one corpus directory."""

    def __init__(self, corpus_dir: Path, block_seeds: int, block_bytes: int, growth: int):
        self._path = Path(corpus_dir) / ZIP_NAME
        self._block_seeds = block_seeds
        self._block_bytes = block_bytes
        self._growth = growth
        self._pending: list[tuple[str, bytes]] = []
        self._pending_data: dict[str, bytes] = {}
        self._pending_bytes = 0
        self._indexed = False
        self._zf: zipfile.ZipFile | None = None
        self._where: dict[str, str] = {}  # hash -> latest data member
        self._main_seen: set[str] = set()
        self._main_live: set[str] = set()
        self._protected: set[tuple[str, str]] = set()

    # ── state ────────────────────────────────────────────────────────

    def _index(self) -> None:
        """Load member names (not data) once; salvage a torn archive first."""
        if self._indexed:
            return
        self._indexed = True
        if not self._path.is_file():
            return

        try:
            infos = self._infolist()
        except zipfile.BadZipFile:
            _salvage(self._path)
            infos = self._infolist()

        for info in infos:
            self._note(info.filename)

    def _infolist(self) -> list[zipfile.ZipInfo]:
        with zipfile.ZipFile(self._path) as zf:
            return zf.infolist()

    def _note(self, name: str) -> None:
        """Apply one member, in archive order, to the in-memory view."""
        parsed = _parse(name)
        if parsed is None:
            log.debug("seeds.zip: ignoring member %r", name)
            return

        tree, h = parsed
        if tree == _TOMB:
            self._main_live.discard(h)
            return

        self._where[h] = name
        if tree:
            self._protected.add((tree, h))
            return
        self._main_seen.add(h)
        self._main_live.add(h)

    def _queue(self, name: str, data: bytes) -> None:
        self._index()
        self._pending.append((name, data))
        self._pending_bytes += len(data)
        self._note(name)
        if data:
            self._pending_data[name] = data
        if len(self._pending) >= self._block_len() or self._pending_bytes >= self._block_bytes:
            self.flush()

    def _block_len(self) -> int:
        if not self._growth:
            return self._block_seeds
        return max(self._block_seeds, len(self._where) // self._growth)

    # ── writes ───────────────────────────────────────────────────────

    def put(self, tree: SeedTree, h: str, data: bytes) -> None:
        self._queue(_member(tree.value, h), data)

    def retire(self, h: str) -> bool:
        """Tombstone the live main member for *h*. False if there is none."""
        self._index()
        if h not in self._main_live:
            return False
        self._queue(_member(_TOMB, h), b"")
        return True

    def flush(self) -> None:
        """Append the pending block; the archive is valid on disk afterwards."""
        if not self._pending:
            return
        zf = self._writer()
        _write_all(zf, self._pending)
        self._pending.clear()
        self._pending_data.clear()
        self._pending_bytes = 0
        if not _checkpoint(zf):
            zf.close()
            self._zf = None

    def close(self) -> None:
        self.flush()
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def _writer(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._index()  # salvages a torn archive before "a" mode sees it
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._zf = zipfile.ZipFile(
                self._path, "a", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL
            )
        return self._zf

    # ── reads ────────────────────────────────────────────────────────

    def has(self, tree: SeedTree, h: str) -> bool:
        self._index()
        if tree is SeedTree.MAIN:
            return h in self._main_live
        return (tree.value, h) in self._protected

    def get(self, h: str) -> bytes | None:
        """Bytes of the latest data member named *h* (live or pruned)."""
        self._index()
        name = self._where.get(h)
        if name is None:
            return None
        if name in self._pending_data:
            return self._pending_data[name]
        with zipfile.ZipFile(self._path) as zf:
            return zf.read(name)

    def live(self) -> Iterator[tuple[bytes, bool]]:
        """``(data, protected)`` for every live seed; flushes first."""
        self.flush()
        self._index()
        protected = {h for _, h in self._protected}
        wanted = self._main_live | protected
        if not wanted:
            return
        with zipfile.ZipFile(self._path) as zf:
            for h in sorted(wanted):
                yield zf.read(self._where[h]), h in protected

    def main_hashes(self) -> set[str]:
        self._index()
        return set(self._main_live)

    def pruned_count(self) -> int:
        self._index()
        return len(self._main_seen - self._main_live)

    def pruned(self) -> Iterator[bytes]:
        self.flush()
        gone = sorted(self._main_seen - self._main_live)
        if not gone:
            return
        with zipfile.ZipFile(self._path) as zf:
            for h in gone:
                yield zf.read(self._where[h])


# ── registry: one store per corpus dir ───────────────────────────────

_STORES: dict[Path, SeedZip] = {}


def configure(
    corpus_dir: str | Path,
    mode: ZipMode,
    block_seeds: int = BLOCK_SEEDS,
    block_bytes: int = BLOCK_BYTES,
    growth: int = BLOCK_GROWTH,
) -> SeedZip | None:
    """Switch *corpus_dir* to *mode*; a replaced store is closed first."""
    key = Path(corpus_dir)
    old = _STORES.pop(key, None)
    if old is not None:
        old.close()
    if mode is ZipMode.OFF:
        return None
    store = SeedZip(key, block_seeds, block_bytes, growth)
    _STORES[key] = store
    return store


def lookup(corpus_dir: str | Path) -> SeedZip | None:
    """Store for *corpus_dir* in zip mode, else None. Free when unused."""
    if not _STORES:
        return None
    return _STORES.get(Path(corpus_dir))


def flush_all() -> None:
    for store in _STORES.values():
        store.flush()


def _close_all() -> None:
    for store in _STORES.values():
        store.close()


atexit.register(_close_all)
