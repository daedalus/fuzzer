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
of the total write cost (tools/lib/bench_seed_zip.py). The block size grows with
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
from dataclasses import dataclass
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
# seeds.zip sits beside seeds/, so members are rooted at the seeds/ level
# (``ab/id_..``). A zip made by archiving the seeds/ directory itself
# (``zip -r seeds.zip seeds``) carries that directory as a prefix; it names
# the same tree, so it is read as the same level, never as foreign data.
_ROOT_DIR = "seeds"
_COLD_DIR = "pruned"
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


def _strip_root(name: str) -> str:
    """Drop leading ``seeds/`` components: ``seeds/seeds/ab/id_x`` is ``ab/id_x``.

    Repeated, because an archive re-zipped from a tree that already held a
    ``seeds/`` level nests it again (``seeds/seeds/..``).
    """
    while True:
        head, sep, rest = name.partition("/")
        if not (sep and head == _ROOT_DIR and rest):
            return name
        name = rest


def _canonical(tree: str, h: str) -> str:
    return _member(tree, h)


def _parse(name: str) -> tuple[str, str] | None:
    """``(tree, hash)`` for a well-formed member name, else None.

    A leading ``seeds/`` is the same level as the archive root (see
    ``_ROOT_DIR``). Rejects traversal, absolute paths, foreign subtrees and
    non-hash leaves: a name is only ever looked up, never joined onto a
    filesystem path, but an unexpected name must not be treated as corpus
    data either.
    """
    parts = _strip_root(name).split("/")
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


def _is_foreign(name: str) -> bool:
    """A plain file member with an arbitrary name (a third-party seed zip).

    Names are only ever used as archive keys, never joined onto a path, so
    only directories, absolute names, traversal and our tombstone namespace
    are refused.
    """
    if not name or name.endswith("/") or name.startswith(("/", f"{_TOMB}/")):
        return False
    # seeds/pruned/ is the file layout's cold tier: pruned seeds stay pruned.
    stripped = _strip_root(name)
    if stripped != name and stripped.startswith(f"{_COLD_DIR}/"):
        return False
    return ".." not in name.split("/")


def _rooted(name: str) -> str | None:
    """*name* at the archive root (``seeds/ab/id_x`` -> ``ab/id_x``); None drops it.

    Directory entries (``seeds/``) carry no data. Cold (``seeds/pruned/..``)
    and refused names stay verbatim: stripping must not turn them foreign.
    """
    if name.endswith("/"):
        return None
    parsed = _parse(name)
    if parsed is not None:
        return _canonical(*parsed)
    stripped = _strip_root(name)
    return stripped if _is_foreign(name) and _is_foreign(stripped) else name


def _uprooted(names: list[str]) -> bool:
    """True if any member is not at the archive root."""
    return any(_rooted(n) != n for n in names)


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

    def __init__(
        self,
        corpus_dir: Path,
        block_seeds: int,
        block_bytes: int,
        growth: int,
        readonly: bool = False,
    ):
        self._readonly = readonly
        self._foreign: list[str] = []  # members not named ab/id_<hash>
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
            if _is_foreign(name):
                self._foreign.append(name)
            else:
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
        foreign = self._read_foreign()
        if foreign and not self._readonly:
            self._adopt(foreign)
            foreign = {}
        protected = {h for _, h in self._protected}
        wanted = self._main_live | protected
        if wanted:
            with zipfile.ZipFile(self._path) as zf:
                for h in sorted(wanted):
                    yield zf.read(self._where[h]), h in protected
        # Read-only: foreign members are served by content, never rewritten.
        for data in foreign.values():
            yield data, False

    def _read_foreign(self) -> dict[str, bytes]:
        """``{content hash: data}`` of foreign members not already canonical."""
        if not self._foreign:
            return {}
        from fuzzer_tool.adapters.filesystem import hash_data  # noqa: PLC0415

        out: dict[str, bytes] = {}
        with zipfile.ZipFile(self._path) as zf:
            for name in self._foreign:
                data = zf.read(name)
                h = hash_data(data)
                if h not in self._main_seen:  # pruned or already adopted: skip
                    out.setdefault(h, data)
        return out

    def _adopt(self, foreign: dict[str, bytes]) -> None:
        """Re-add foreign seeds as canonical members so prune can tombstone them."""
        for h, data in foreign.items():
            self.put(SeedTree.MAIN, h, data)
        self._foreign = []
        self.flush()
        log.info("seeds.zip: adopted %d foreign members as seeds", len(foreign))

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


def open_readonly(corpus_dir: str | Path) -> SeedZip:
    """Unregistered store for reading *corpus_dir*/seeds.zip; callers never write."""
    return SeedZip(Path(corpus_dir), BLOCK_SEEDS, BLOCK_BYTES, BLOCK_GROWTH, readonly=True)


_PEEK: dict[Path, tuple[tuple[int, int], SeedZip]] = {}


def peek(corpus_dir: str | Path) -> SeedZip | None:
    """Cached read-only view of an existing seeds.zip (zip mode off).

    Zip mode off never writes the archive, so its (mtime, size) is a sound
    cache key: the index is built once, not once per saved seed.
    """
    path = Path(corpus_dir) / ZIP_NAME
    try:
        st = path.stat()
    except OSError:
        _PEEK.pop(path, None)
        return None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _PEEK.get(path)
    if hit is not None and hit[0] == sig:
        return hit[1]
    store = open_readonly(corpus_dir)
    _PEEK[path] = (sig, store)
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


# ── compaction ───────────────────────────────────────────────────────

_COLD = Path("seeds") / "pruned"


class StoreOpenError(RuntimeError):
    """compact() needs the archive closed: a live store appends to it."""


class CompactError(RuntimeError):
    """The rewritten archive disagrees with the original; nothing replaced."""


@dataclass(frozen=True)
class CompactStats:
    moved: int = 0  # pruned seeds now in seeds/pruned/ only
    dropped: int = 0  # members removed from the archive
    bytes_before: int = 0
    bytes_after: int = 0


def _spill(corpus: Path, h: str, data: bytes) -> None:
    """Write pruned seed *h* to seeds/pruned/ (the file-mode layout), durably."""
    dest = corpus / _COLD / h[:2] / f"id_{h}"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)


def _plan(names: list[str], cold: set[str]) -> list[int]:
    """Indices of members to keep.

    Dropped: every tombstone (cold ones leave with their data, the rest were
    superseded by re-admission), data of cold seeds, directory entries, and
    all but the last copy of a name at the archive root (``seeds/ab/id_x``
    and ``ab/id_x`` are one). Foreign members are kept.
    """
    last = {_rooted(n): i for i, n in enumerate(names)}
    keep = []
    for i, name in enumerate(names):
        rooted = _rooted(name)
        if rooted is None or last[rooted] != i:
            continue

        parsed = _parse(name)
        if parsed is not None and (parsed[0] == _TOMB or (not parsed[0] and parsed[1] in cold)):
            continue
        keep.append(i)
    return keep


def _rewrite(path: Path, tmp: Path, infos: list[zipfile.ZipInfo], keep) -> None:
    """Copy kept members into *tmp* at the archive root, one at a time (bounded memory)."""
    out = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL)
    with zipfile.ZipFile(path) as zin, out as zout, warnings.catch_warnings():
        # uproot() keeps re-admitted duplicates: order carries meaning.
        warnings.filterwarnings("ignore", "Duplicate name", UserWarning)
        for i in keep:
            name = _rooted(infos[i].filename)
            if name is not None:
                zout.writestr(name, zin.read(infos[i]))


def _same_view(a: SeedZip, b: SeedZip) -> bool:
    return (
        a._main_live == b._main_live
        and a._protected == b._protected
        and {_rooted(n) for n in a._foreign} == set(b._foreign)
    )


def _swap(corpus: Path, store: SeedZip, infos: list[zipfile.ZipInfo], keep) -> None:
    """Replace seeds.zip with its *keep* members, rooted, after a replay check."""
    path = corpus / ZIP_NAME
    tmp = path.with_name(path.name + ".tmp")
    try:
        _rewrite(path, tmp, infos, keep)
        check = open_readonly(corpus)
        check._path = tmp
        check._index()
        if not _same_view(store, check):
            raise CompactError(f"{path}: rewritten archive differs")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def compact(corpus_dir: str | Path) -> CompactStats:
    """Move pruned seeds out of seeds.zip into seeds/pruned/; rewrite the archive.

    Offline. Nothing is deleted: a pruned seed is on disk under seeds/pruned/
    (where rehydration and the cuckoo filter already look) before the archive
    is swapped in. The swap is one os.replace after a replay check, so a
    failure leaves the original; a rerun is safe::

        seeds.zip  [a b c .pruned/b b' ...]  ->  seeds.zip [a c]
                                                 seeds/pruned/<hh>/id_b
    """
    corpus = Path(corpus_dir)
    if lookup(corpus) is not None:
        raise StoreOpenError(f"{corpus}: store is open")

    path = corpus / ZIP_NAME
    if not path.is_file():
        return CompactStats()

    store = open_readonly(corpus)
    store._index()  # salvages a torn archive first
    infos = store._infolist()
    cold = store._main_seen - store._main_live
    names = [i.filename for i in infos]
    keep = _plan(names, cold)
    dropped = len(infos) - len(keep)
    if not dropped and not _uprooted(names):
        return CompactStats()

    before = path.stat().st_size
    with zipfile.ZipFile(path) as zf:
        for h in sorted(cold):
            _spill(corpus, h, zf.read(store._where[h]))

    _swap(corpus, store, infos, keep)
    return CompactStats(len(cold), dropped, before, path.stat().st_size)


def uproot(corpus_dir: str | Path) -> bool:
    """Move every seeds.zip member to the archive root; True if rewritten.

    seeds.zip sits beside seeds/, so a zip of the seeds/ directory names the
    same tree one level down. Unlike compact(), order and tombstones are
    kept, so nothing changes liveness::

        seeds.zip [seeds/ seeds/ab/id_x seeds/.pruned/ab/id_x]
               -> seeds.zip [ab/id_x .pruned/ab/id_x]
    """
    corpus = Path(corpus_dir)
    if lookup(corpus) is not None:
        raise StoreOpenError(f"{corpus}: store is open")
    if not (corpus / ZIP_NAME).is_file():
        return False

    store = open_readonly(corpus)
    store._index()  # salvages a torn archive first
    infos = store._infolist()
    if not _uprooted([i.filename for i in infos]):
        return False

    _swap(corpus, store, infos, range(len(infos)))
    return True


def pruned_ratio(corpus_dir: str | Path) -> float:
    """Pruned seeds per live seed in seeds.zip (0.0 if there is no archive).

    inf when seeds are pruned and none are live.

    The archive's space amplification: how much of it is cold data a load
    still has to index and skip.
    """
    corpus = Path(corpus_dir)
    if not (corpus / ZIP_NAME).is_file():
        return 0.0

    store = open_readonly(corpus)
    store._index()
    pruned = len(store._main_seen - store._main_live)
    live = len(store._main_live)
    return pruned / live if live else float("inf") if pruned else 0.0


def compact_over(corpus_dir: str | Path, ratio: float) -> CompactStats:
    """compact() when pruned_ratio exceeds *ratio*; a ratio <= 0 (or NaN) is off."""
    if not ratio > 0 or pruned_ratio(corpus_dir) <= ratio:
        return CompactStats()

    return compact(corpus_dir)


atexit.register(_close_all)
