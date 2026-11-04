"""Minimal corpus that still reaches every edge recorded in an edge dump.

A corpus minimized twice shrinks to the edges its own survivors reach: the
second run's universe is the first run's output, so edges only the pruned
seeds hit are silently gone. The dump is the record of everything reached, so
it, not the current corpus, defines "all coverage"::

    e.pkl --stream batches--> dump union   (every seed ever dumped)
                              reach union  (seeds still on disk or archived)
                              greedy pool  (reaches every reachable edge)
           --> min_cover(pool, reach) --> verify --> copy | move to pruned/

The cover is heuristic (greedy + reductions, core/set_cover.py), but the
coverage claim is checked: an incomplete cover is refused, never written.

Seeds are read from loose files, ``seeds.zip`` and ``seeds.tar.gz``; archives
are read-only, so a corpus that lives in one needs ``-o``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters import seed_tar, seed_zip
from fuzzer_tool.adapters.edge_pickle import iter_edges_pickle
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.set_cover import min_cover
from fuzzer_tool.services.minimize import _commit_results, _discover_corpus_files

# Seed filenames are content hashes: seeds/<hh>/id_<hash> (corpus_manager).
_SEED_PREFIX = "id_"


class UnreachableEdges(RuntimeError):
    """The dump holds edges no seed left in the corpus can reach.

    Carries a partial ``CoverReport`` so the caller can print coverage.
    """

    def __init__(self, msg: str, partial: CoverReport | None = None):
        super().__init__(msg)
        self.partial = partial


class IncompleteCover(RuntimeError):
    """The chosen seeds miss edges the pool could reach; refuse to prune."""


class EmptyDump(RuntimeError):
    """The dump yields no seed edges, so it defines no coverage; refuse to prune."""


class ArchiveReadOnly(RuntimeError):
    """The corpus lives in an archive, which cannot be pruned in place."""


@dataclass(frozen=True)
class CoverReport:
    """One full-cover run: what the dump had, what the corpus could reach."""

    dump_seeds: int
    on_disk: int
    unmeasured: int
    gone_seeds: int
    union_edges: int
    reachable_edges: int
    unreachable_edges: int
    kept: int
    removed: int


@dataclass(frozen=True)
class _Corpus:
    """Where the seeds live: loose files, seeds.zip, seeds.tar.gz."""

    loose: dict[str, Path]
    sizes: dict[str, int]
    zipped: bool
    tarred: bool
    seed_id_to_hh: dict[str, str]

    @property
    def archived(self) -> bool:
        return self.zipped or self.tarred


def load_corpus(corpus_dir: str | Path) -> _Corpus:
    """Index every live seed by content hash, with its byte size."""
    corpus_path = Path(corpus_dir)

    loose: dict[str, Path] = {}
    for path in _discover_corpus_files(corpus_path):
        if not _is_seed_file(path, corpus_path):
            continue
        loose.setdefault(_seed_id(path), path)
    sizes = {sid: _size(path) for sid, path in loose.items()}

    zipped = tarred = False
    seed_id_to_hh: dict[str, str] = {}
    store = seed_zip.peek(corpus_path)
    if store is not None:
        zipped = True
        for sid, size, _protected, hh in store.named():
            sizes.setdefault(sid, size)
            seed_id_to_hh.setdefault(sid, hh)
    for sid, size, _protected, hh in seed_tar.named(corpus_path):
        tarred = True
        sizes.setdefault(sid, size)
        seed_id_to_hh.setdefault(sid, hh)

    return _Corpus(loose, sizes, zipped, tarred, seed_id_to_hh)


def _is_seed_file(path: Path, _corpus_path: Path) -> bool:
    return path.name.startswith(_SEED_PREFIX) and len(path.name) == 19 and _is_hex(path.name[3:])


def _is_hex(text: str) -> bool:
    hex_set = set("0123456789abcdef")
    return len(text) == 16 and set(text) <= hex_set


def _seed_id(path: Path) -> str:
    if path.name.startswith(_SEED_PREFIX):
        return path.name[len(_SEED_PREFIX) :]
    return hash_data(path.read_bytes())


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _grow(mask: np.ndarray, top: int) -> np.ndarray:
    """Extend a boolean edge mask to cover edge id *top*."""
    if top < mask.size:
        return mask
    return np.concatenate([mask, np.zeros(top + 1 - mask.size, dtype=bool)])


def _pick_batch(
    batch: dict[str, np.ndarray], ids: dict[str, int], reach: np.ndarray
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """Greedy picks in one batch, gains measured against the global *reach*.

    Stops when no live seed adds an edge, which is what makes *reach* the
    union of every counted seed, not just the picks.
    """
    live = [(sid, edges) for sid, edges in batch.items() if sid in ids and edges.size]
    picks = []
    while live:
        gains = []
        for _, edges in live:
            reach = _grow(reach, int(edges.max()))
            gains.append(np.count_nonzero(~reach[edges]))

        j = int(np.argmax(gains))
        if gains[j] == 0:
            break

        sid, edges = live.pop(j)
        reach[edges] = True
        picks.append((sid, edges))

    return picks, reach


def _scan(
    dump_path: str | Path, ids: dict[str, int]
) -> tuple[dict, np.ndarray, np.ndarray, set, set, int]:
    """Stream the dump once: per-batch picks, the dump union, the reach union."""
    pool: dict[str, np.ndarray] = {}
    dump = np.zeros(1, dtype=bool)
    reach = np.zeros(1, dtype=bool)
    unmeasured = set(ids)
    gone: set[str] = set()
    dump_seeds = 0

    for batch in iter_edges_pickle(dump_path):
        dump_seeds += len(batch)
        gone.update(sid for sid in batch if sid not in ids)
        unmeasured.difference_update(batch)

        for edges in batch.values():
            if edges.size:
                dump = _grow(dump, int(edges.max()))
                dump[edges] = True

        picks, reach = _pick_batch(batch, ids, reach)
        pool.update(picks)

    return pool, dump, reach, unmeasured, gone, dump_seeds


def _verify(kept: list[str], pool: dict[str, np.ndarray], universe: np.ndarray) -> None:
    """Every universe edge must be held by a kept seed."""
    mask = np.zeros(int(universe.max()) + 1 if universe.size else 1, dtype=bool)
    for sid in kept:
        mask[pool[sid]] = True

    miss = int(np.count_nonzero(~mask[universe]))
    if miss:
        raise IncompleteCover(f"cover misses {miss} of {universe.size} reachable edges")


def _write_cover(
    corpus: _Corpus, corpus_path: Path, kept: set[str], output_dir: str | Path
) -> None:
    """Copy the kept seeds out of every backend into *output_dir*.

    Layout preserved: ``min_corpus/seeds/<hh>/id_<hash>``. Seeds are keyed by
    content, so an id stored under more than one path (a shard and a protected
    tree, or an archive beside loose files) is written once.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = set(corpus.loose)

    for sid in kept:
        path = corpus.loose.get(sid)
        if path is not None:
            hh = corpus.seed_id_to_hh.get(sid, sid[:2])
            dest_dir = out / "seeds" / hh
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / path.name)
            meta = path.with_suffix(".txt")
            if meta.exists():
                shutil.copy2(meta, dest_dir / meta.name)

    if corpus.tarred:
        for sid, data, hh in seed_tar.pick(corpus_path, kept):
            if sid in written:
                continue
            written.add(sid)
            dest_dir = out / "seeds" / hh
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / f"id_{sid}").write_bytes(data)

    if corpus.zipped:
        store = seed_zip.peek(corpus_path)
        if store is not None:
            for sid, data, hh in store.pick(kept):
                if sid in written:
                    continue
                written.add(sid)
                dest_dir = out / "seeds" / hh
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / f"id_{sid}").write_bytes(data)


def full_cover(
    dump_path: str | Path,
    corpus_dir: str | Path,
    output_dir: str | Path | None = None,
    allow_unreachable: bool = False,
) -> CoverReport:
    """Keep the fewest corpus seeds that still reach every dumped edge.

    Raises:
        UnreachableEdges: the dump has edges no seed in the corpus reaches,
            and *allow_unreachable* is off.
        IncompleteCover: the cover misses reachable edges (never written).
        ArchiveReadOnly: pruning in place was asked of an archived corpus.
        EmptyDump: the dump holds no seed edges.
    """
    corpus_path = Path(corpus_dir)
    corpus = load_corpus(corpus_path)

    pool, dump, reach, unmeasured, gone, dump_seeds = _scan(dump_path, corpus.sizes)

    # No dump, no coverage claim: keeping everything would look like success.
    if not dump_seeds:
        raise EmptyDump(f"{dump_path} holds no seed edges; nothing to cover")

    top = max(dump.size, reach.size)
    dump, reach = _grow(dump, top - 1), _grow(reach, top - 1)
    unreachable = int(np.count_nonzero(dump & ~reach))
    if unreachable and not allow_unreachable:
        partial = CoverReport(
            dump_seeds=dump_seeds,
            on_disk=len(corpus.sizes),
            unmeasured=len(unmeasured),
            gone_seeds=len(gone),
            union_edges=int(dump.sum()),
            reachable_edges=int(reach.sum()),
            unreachable_edges=unreachable,
            kept=0,
            removed=0,
        )
        raise UnreachableEdges(
            f"{unreachable} edges unreachable by the {len(corpus.sizes)} seeds in the corpus "
            f"({len(gone)} dumped seeds are gone); --allow-unreachable covers the rest",
            partial=partial,
        )

    universe = np.flatnonzero(reach)
    sizes = {sid: corpus.sizes[sid] for sid in pool}
    kept = set(min_cover(pool, sizes))
    _verify(list(kept), pool, universe)

    # Seeds never dumped are unmeasured: keep them, never prune on a guess.
    kept |= unmeasured

    report = CoverReport(
        dump_seeds=dump_seeds,
        on_disk=len(corpus.sizes),
        unmeasured=len(unmeasured),
        gone_seeds=len(gone),
        union_edges=int(dump.sum()),
        reachable_edges=int(universe.size),
        unreachable_edges=unreachable,
        kept=len(kept),
        removed=len(corpus.sizes) - len(kept),
    )

    if output_dir is not None:
        _write_cover(corpus, corpus_path, kept, output_dir)
        return report

    if corpus.archived:
        raise ArchiveReadOnly(
            "seeds live in an archive (read-only); pass -o DIR to write the cover"
        )

    files = list(corpus.loose.values())
    survivors = [str(corpus.loose[sid]) for sid in kept if sid in corpus.loose]
    n_kept, n_removed = _commit_results(files, survivors, None, corpus_path)
    return replace(report, kept=n_kept, removed=n_removed)
