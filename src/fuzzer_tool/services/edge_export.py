"""Edge export: replay every corpus seed and pickle its edge ids. No pruning.

    corpus/seeds/** --replay--> ShmCoverage --> {hash_data(seed): uint32[edge ids]} --> out.pkl
                                                                                   +-> out.sqlite (edge-keyed)

Replay backends (spawn, loader, direct_lite, direct) are minimize's.
"""

import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.edge_sqlite import read_edges_sqlite, write_edges_sqlite
from fuzzer_tool.adapters.filesystem import count_archive_seeds, hash_data, iter_archive_seeds
from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.minimize import (
    _TIMEOUT_RC,
    _TRIM_EVERY,
    DEFAULT_INPROCESS_FUNC,
    ExecMode,
    _discover_corpus_files,
    _maxrss_kb,
    _open_runner,
    _print_progress,
    _runner_replay,
    _spawn_replay,
    _split_oversized,
    _trim_heap,
)

# Seeds between progress lines, as in minimize.
_PROGRESS_EVERY = 10

# Output format by file extension (case-insensitive); see _FORMATS.
_PKL = ".pkl"
_SQLITE = ".sqlite"


class _CorpusSeeds:
    """Live seeds: loose files, then seeds.zip / seeds.tar.gz members; size-capped.

    Archive sizes are only known once read, so ``oversized`` is final after
    iteration.
    """

    def __init__(self, corpus_dir: Path, max_seed_bytes: int) -> None:
        self._corpus = corpus_dir
        self._max = max_seed_bytes
        self._files, over = _split_oversized(_discover_corpus_files(corpus_dir), max_seed_bytes)
        self.oversized = len(over)
        self.total = len(self._files) + count_archive_seeds(corpus_dir)

    def __iter__(self) -> Iterator[bytes]:
        for fpath in self._files:
            if not fpath.exists():
                print(f"[!] Skipping missing seed: {fpath}")
                continue
            yield fpath.read_bytes()

        for data in iter_archive_seeds(self._corpus):
            if self._max and len(data) > self._max:
                self.oversized += 1
                continue
            yield data


class _Sink:
    """One -o file: its format's reader/writer and the seeds it already holds."""

    def __init__(self, path: Path, read, write) -> None:
        self.path = path
        self._read = read
        self._write = write
        self.prior: dict[str, np.ndarray] = {}

    def load(self) -> bool:
        """Read the existing file into ``prior``; False (reported) if unreadable."""
        if not self.path.exists():
            return True

        try:
            self.prior = self._read(self.path)
        except (OSError, pickle.UnpicklingError, sqlite3.DatabaseError, ValueError, EOFError) as e:
            print(f"[-] Cannot read existing output {self.path}: {e}", file=sys.stderr)
            return False
        return True

    def save(self, new: dict[str, np.ndarray]) -> int:
        """Write prior + *new* (new wins); returns seeds written."""
        merged = {**self.prior, **new}
        self._write(self.path, merged)
        print(f"[+] Exported {len(merged)} seeds (+{len(new)}) -> {self.path}")
        return len(merged)


def export_edges(
    target: str,
    corpus_dir: str,
    out_paths: list[str],
    timeout: float = 1.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    exec_mode: ExecMode = ExecMode.SPAWN,
    inprocess_func: str = DEFAULT_INPROCESS_FUNC,
    max_seed_bytes: int = 0,
) -> int:
    """Export ``{seed_id: sorted uint32 edge ids}`` for every live seed.

    Each of *out_paths* picks its format by extension: ``.pkl`` pickles the
    dict, ``.sqlite`` writes the edge-keyed inverse (``adapters/edge_sqlite.py``).
    Any other extension fails before replay.

    Incremental: seed_ids already in every existing output are not replayed;
    new results merge into each file. An unreadable output aborts untouched.

    seed_id is ``hash_data(seed)``, the corpus filename key. The corpus is
    read only. New seeds with no edges at all are a failed measurement:
    nothing is written, so they are retried next run instead of cached.

    Returns:
        Seeds in the outputs; 0 when nothing was written.
    """
    # Writers resolved before replay: a bad extension costs no replay.
    sinks = _resolve_sinks(out_paths)
    if sinks is None or not all(s.load() for s in sinks):
        return 0

    # Known only if every output holds it, e.g. pkl {a,b} + sqlite {a} -> skip {a}.
    known = set.intersection(*(set(s.prior) for s in sinks))

    seeds = _CorpusSeeds(Path(corpus_dir), max_seed_bytes)
    if not seeds.total:
        print(f"[-] Corpus is empty: {corpus_dir}", file=sys.stderr)
        return 0

    edges = _replay_all(
        seeds, target, timeout, file_mode, target_args, exec_mode, inprocess_func, known
    )
    if seeds.oversized:
        print(f"[*] Skipped: {seeds.oversized} seeds over {max_seed_bytes} bytes")
    if edges is None:
        return 0

    if not edges:
        print(f"[*] Up to date: {len(known)} seeds already exported")
        return len(known)

    # Blackout guard, same reading as cmin: no edges means no measurement.
    if not any(a.size for a in edges.values()):
        print("[-] No edges recorded for any new seed -- nothing written.", file=sys.stderr)
        return 0

    return max(s.save(edges) for s in sinks)


def _resolve_sinks(out_paths: list[str]) -> list[_Sink] | None:
    """A ``_Sink`` per output, or None (reported) on an unknown extension.

    e.g. ["e.pkl", "e.sqlite"] -> [pickle sink, sqlite sink]
    """
    if not out_paths:
        print(f"[-] No output given: -o FILE{_PKL} and/or -o FILE{_SQLITE}", file=sys.stderr)
        return None

    sinks = []
    for p in map(Path, out_paths):
        io = _FORMATS.get(p.suffix.lower())
        if io is None:
            print(f"[-] Unknown output extension: {p} (use {_PKL} or {_SQLITE})", file=sys.stderr)
            return None
        sinks.append(_Sink(p, *io))
    return sinks


def _replay_all(seeds, target, timeout, file_mode, target_args, exec_mode, func, known):
    """``{seed_id: edge ids}`` over *seeds* not in *known*, or None when the runner cannot start.

    A seed seen twice (loose and archived) replays once.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="edump_"))

    # Caller-context edge ids hash a return address; ASLR makes them noise.
    disable_aslr()

    shm = ShmCoverage()
    env = os.environ.copy()
    env["__AFL_SHM_ID"] = shm.env_id
    env["AFL_MAP_SIZE"] = str(shm.num_entries)

    edges: dict[str, np.ndarray] = {}
    timeouts = 0
    skipped = 0
    runner = None
    try:
        # Backend chosen once; the loop does not branch on mode.
        if exec_mode is ExecMode.SPAWN:
            replay = _spawn_replay(target, timeout, file_mode, tmp_dir, target_args, env)
        else:
            runner = _open_runner(target, exec_mode, func, timeout, shm)
            if runner is None:
                return None
            replay = _runner_replay(runner)

        n = seeds.total
        start_time = time.time()
        unique: set[int] = set()
        peak_rss_kb = _maxrss_kb()
        i = -1
        for i, data in enumerate(seeds):
            key = hash_data(data)
            if key in known:
                skipped += 1
                continue
            if key in edges:
                continue

            shm.reset_edge_map()
            timeouts += replay(data)[0] == _TIMEOUT_RC
            ids = shm.edge_id_array()
            edges[key] = ids
            unique.update(ids.tolist())
            peak_rss_kb = max(peak_rss_kb, _maxrss_kb())

            if (i + 1) % _TRIM_EVERY == 0:
                _trim_heap()

            if (i + 1) % _PROGRESS_EVERY == 0:
                _print_progress(i + 1, n, start_time, len(unique), peak_rss_kb)

        # Final line: archive members over the cap never reach i + 1 == n.
        _print_progress(i + 1, n, start_time, len(unique), peak_rss_kb)
    finally:
        if runner is not None:
            runner.stop()
        shm.cleanup()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    if skipped:
        print(f"[*] Already exported: {skipped} seeds skipped")
    if timeouts:
        print(f"[*] Timed out: {timeouts}/{len(edges)} seeds (partial edges kept)")
    return edges


def _write_atomic(out: Path, edges: dict[str, np.ndarray]) -> None:
    """Pickle *edges* to *out* via tmp + rename: no half-written file on crash."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)


def _read_pkl(path: Path) -> dict[str, np.ndarray]:
    """A pickle this module wrote; ValueError on any other shape."""
    with open(path, "rb") as f:
        edges = pickle.load(f)
    if not isinstance(edges, dict) or not all(
        isinstance(k, str) and isinstance(v, np.ndarray) for k, v in edges.items()
    ):
        raise ValueError("not a {seed_id: ndarray} edge export")
    return edges


# Extension -> (reader, writer).
_FORMATS = {_PKL: (_read_pkl, _write_atomic), _SQLITE: (read_edges_sqlite, write_edges_sqlite)}
