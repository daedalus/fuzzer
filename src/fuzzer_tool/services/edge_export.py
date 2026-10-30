"""Edge export: replay every corpus seed and write its edge ids. No pruning.

    corpus --replay--> (seed_id, ids) --> _Collector --batch--> out.pkl    (appended batches)
                                             ^   |         +--> out.sqlite (one txn per batch)
                                             |   v
                                          AdaptiveBatch (size climbs on seeds/s)

Results are written in batches as they arrive: memory holds one batch, and
a killed run keeps every written batch. Replay backends are minimize's.
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

from fuzzer_tool.adapters.edge_pickle import EdgePickleWriter
from fuzzer_tool.adapters.edge_sqlite import EdgeSqliteWriter
from fuzzer_tool.adapters.filesystem import count_archive_seeds, hash_data, iter_archive_seeds
from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.adaptive_batch import AdaptiveBatch
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

# Write batch, in seeds: starts here, climbs on seeds/s within the bounds.
# Measured (20k seeds, direct_lite, ~10k seeds/s): fixed 8 = 6.5 s, fixed
# 4096 = 1.9 s; start 64 / tol 0 collapsed on window noise (2.0-2.3 s),
# start 256 / tol 0.2 held near the top (1.94 s). See DEEP_DIVE §Edge dump.
_BATCH_START = 256
_BATCH_MIN = 8
_BATCH_MAX = 4096
# Rate must drop by more than this to reverse the climb (window noise).
_BATCH_TOL = 0.2
# Memory cap: a batch is also written once it buffers this many edge ids.
_MAX_ROWS = 1 << 20

_EMPTY = np.empty(0, dtype=np.uint32)


class _CorpusSeeds:
    """Live seeds: loose files, then seeds.zip / seeds.tar.gz members; size-capped.

    Archives are streamed in place, one member at a time, in one pass.
    Archive sizes are only known once read, so ``oversized`` and ``seen``
    are final after iteration. ``total`` is None when a tar makes it unknown.
    """

    def __init__(self, corpus_dir: Path, max_seed_bytes: int) -> None:
        self._corpus = corpus_dir
        self._max = max_seed_bytes
        self._files, over = _split_oversized(_discover_corpus_files(corpus_dir), max_seed_bytes)
        self.oversized = len(over)
        self.seen = 0
        archived = count_archive_seeds(corpus_dir)
        self.total = None if archived is None else len(self._files) + archived

    def __iter__(self) -> Iterator[bytes]:
        for fpath in self._files:
            if not fpath.exists():
                print(f"[!] Skipping missing seed: {fpath}")
                continue
            self.seen += 1
            yield fpath.read_bytes()

        for data in iter_archive_seeds(self._corpus):
            if self._max and len(data) > self._max:
                self.oversized += 1
                continue
            self.seen += 1
            yield data


class _Sink:
    """One -o file: its format's batch writer and the seeds it already holds."""

    def __init__(self, path: Path, writer_cls) -> None:
        self.path = path
        self._writer = writer_cls(path)
        self.prior: set[str] = set()

    def load(self) -> bool:
        """Read the seed ids already in the file; False (reported) if unreadable."""
        try:
            self.prior = self._writer.open()
        except (OSError, pickle.UnpicklingError, sqlite3.DatabaseError, ValueError, EOFError) as e:
            print(f"[-] Cannot read existing output {self.path}: {e}", file=sys.stderr)
            return False
        return True

    def write(self, items: list[tuple[str, np.ndarray]]) -> None:
        self._writer.write(items)

    def close(self) -> int:
        """Close; report and return the seeds now in the file."""
        self._writer.close()
        total = len(self._writer.known)
        added = total - len(self.prior)
        if added:
            print(f"[+] Exported {total} seeds (+{added}) -> {self.path}")
        return total


class _Collector:
    """Buffers replay results and writes them to every sink in adaptive batches.

    Edgeless seeds are held (ids only) until the first seed with edges: if
    none ever comes, the target was not measured and nothing is written.
    """

    def __init__(self, sinks: list[_Sink]) -> None:
        self._sinks = sinks
        self._batch = AdaptiveBatch(_BATCH_START, _BATCH_MIN, _BATCH_MAX, tol=_BATCH_TOL)
        self._buf: list[tuple[str, np.ndarray]] = []
        self._rows = 0
        self._held: list[str] = []
        self.live = False
        self.count = 0

    def add(self, key: str, ids: np.ndarray) -> None:
        self.count += 1
        if not self.live:
            if not ids.size:
                self._held.append(key)
                return
            self.live = True
            self._buf.extend((k, _EMPTY) for k in self._held)
            self._held.clear()

        self._buf.append((key, ids))
        self._rows += ids.size
        if len(self._buf) >= self._batch.size or self._rows >= _MAX_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        for sink in self._sinks:
            sink.write(self._buf)
        self._batch.observe(len(self._buf))
        self._buf = []
        self._rows = 0

    def close(self) -> int:
        """Write the tail; close sinks; seeds in the fullest output."""
        self.flush()
        return max(sink.close() for sink in self._sinks)


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
    max_seeds: int = 0,
) -> int:
    """Export ``{seed_id: sorted uint32 edge ids}`` for every live seed.

    *max_seeds* > 0 stops after that many new seeds, in corpus order (known
    seeds do not count, so repeated runs walk the corpus in chunks). It is
    also the progress total, which gives a streamed seeds.tar.gz an ETA.

    Each of *out_paths* picks its format by extension: ``.pkl`` appends one
    pickled dict per batch (``adapters/edge_pickle.py``), ``.sqlite`` writes
    the edge-keyed inverse (``adapters/edge_sqlite.py``). Any other
    extension fails before replay. Batches are written as results arrive;
    their size adapts to seeds/s (``core/adaptive_batch.py``).

    Incremental: seed_ids already in every existing output are not replayed;
    new results are appended to each file. An unreadable output aborts untouched.

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
    known = set.intersection(*(s.prior for s in sinks))

    seeds = _CorpusSeeds(Path(corpus_dir), max_seed_bytes)
    if seeds.total == 0 and not seeds.oversized:
        print(f"[-] Corpus is empty: {corpus_dir}", file=sys.stderr)
        return 0

    collector = _Collector(sinks)
    try:
        ok = _replay_all(
            seeds,
            target,
            timeout,
            file_mode,
            target_args,
            exec_mode,
            inprocess_func,
            known,
            collector,
            max(max_seeds, 0),
        )
    finally:
        # Written batches and the buffered tail survive a crash or Ctrl-C.
        exported = collector.close()

    if seeds.oversized:
        print(f"[*] Skipped: {seeds.oversized} seeds over {max_seed_bytes} bytes")
    if not ok:
        return 0

    # A streamed tar has no up-front count: emptiness shows only now.
    if not seeds.seen:
        print(f"[-] No seeds to replay in {corpus_dir}", file=sys.stderr)
        return 0

    if not collector.count:
        print(f"[*] Up to date: {len(known)} seeds already exported")
        return len(known)

    # Blackout guard, same reading as cmin: no edges means no measurement.
    if not collector.live:
        print("[-] No edges recorded for any new seed -- nothing written.", file=sys.stderr)
        return 0

    return exported


def _resolve_sinks(out_paths: list[str]) -> list[_Sink] | None:
    """A ``_Sink`` per output, or None (reported) on an unknown extension.

    e.g. ["e.pkl", "e.sqlite"] -> [pickle sink, sqlite sink]
    """
    if not out_paths:
        print(f"[-] No output given: -o FILE{_PKL} and/or -o FILE{_SQLITE}", file=sys.stderr)
        return None

    sinks = []
    for p in map(Path, out_paths):
        writer_cls = _FORMATS.get(p.suffix.lower())
        if writer_cls is None:
            print(f"[-] Unknown output extension: {p} (use {_PKL} or {_SQLITE})", file=sys.stderr)
            return None
        sinks.append(_Sink(p, writer_cls))
    return sinks


def _replay_all(seeds, target, timeout, file_mode, target_args, exec_mode, func, known, out, limit):
    """Replay *seeds* not in *known* into collector *out*; False if the runner cannot start.

    A seed seen twice (loose and archived) replays once. *limit* > 0 stops
    after that many replays.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="edump_"))

    # Caller-context edge ids hash a return address; ASLR makes them noise.
    disable_aslr()

    shm = ShmCoverage()
    env = os.environ.copy()
    env["__AFL_SHM_ID"] = shm.env_id
    env["AFL_MAP_SIZE"] = str(shm.num_entries)

    done: set[str] = set()
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
                return False
            replay = _runner_replay(runner)

        n, done_of = _progress_basis(limit, seeds.total, done)
        start_time = time.time()
        unique: set[int] = set()
        peak_rss_kb = _maxrss_kb()
        i = -1
        for i, data in enumerate(seeds):
            key = hash_data(data)
            if key in known:
                skipped += 1
                continue
            if key in done:
                continue
            done.add(key)

            shm.reset_edge_map()
            timeouts += replay(data)[0] == _TIMEOUT_RC
            ids = shm.edge_id_array()
            out.add(key, ids)
            unique.update(ids.tolist())
            peak_rss_kb = max(peak_rss_kb, _maxrss_kb())

            if (i + 1) % _TRIM_EVERY == 0:
                _trim_heap()

            if (i + 1) % _PROGRESS_EVERY == 0:
                _print_progress(done_of(i), n, start_time, len(unique), peak_rss_kb)

            # Leaving a streamed archive here closes it (GeneratorExit), no read error.
            if limit and len(done) >= limit:
                break

        # Final line: archive members over the size cap never reach i + 1 == n.
        _print_progress(done_of(i), n, start_time, len(unique), peak_rss_kb)
    finally:
        if runner is not None:
            runner.stop()
        shm.cleanup()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    if skipped:
        print(f"[*] Already exported: {skipped} seeds skipped")
    if timeouts:
        print(f"[*] Timed out: {timeouts}/{len(done)} seeds (partial edges kept)")
    return True


def _progress_basis(limit: int, total: int | None, done: set[str]):
    """``(total, done_of(i))`` for the progress line, chosen once per run.

    With a cap: replays out of the cap (a streamed tar then has a total).
    Else: seeds read out of the corpus; total 0 = unknown, no ETA.
    """
    if limit:
        return limit, lambda i: len(done)
    return total or 0, lambda i: i + 1


# Extension -> batch writer.
_FORMATS = {_PKL: EdgePickleWriter, _SQLITE: EdgeSqliteWriter}
