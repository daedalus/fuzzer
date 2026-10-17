"""Clang source-coverage driver: branch counts and the coverage frontier.

A ``<name>_cov`` build (``tools/build_targets.sh --frontier``) writes an LLVM
profile per run; ``llvm-cov export`` maps it to branch conditions::

    condition (file, line, col) -> (true count, false count)
    frontier = conditions taken one way only; Side = the missing side

``Coverage`` runs inputs in batch (``tools/frontier.py``). ``FrontierOracle``
measures seeds one at a time as the fuzzer admits them, in a worker thread,
and answers which open frontier conditions a seed executes::

    submit(key, data) -> queue (bounded, drops when full)
    worker            -> run _cov, export, fold counts into the corpus totals
    open_reached(key) -> conditions the seed executes that are still one-sided

Cost is one ``llvm-cov export`` per seed: it reloads the coverage mapping
each call (fuzzgoat ~0.5 s, FFmpeg ~4 s).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import weakref
from collections import OrderedDict, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# llvm-cov export branch record: [line, col, end_line, end_col, true, false, ...]
_LINE, _COL, _TRUE, _FALSE = 0, 1, 4, 5
RUN_TIMEOUT = 10
MERGE_BATCH = 256  # raw profiles per llvm-profdata merge
STUB_NAME = "frontier_stub.c"

MAX_PENDING = 256  # queued seeds before submit() drops
MAX_SEEDS = 4096  # seeds whose reach sets are kept (LRU)

Branch = tuple[str, int, int]  # (file basename, line, col)
Counts = dict[Branch, tuple[int, int]]


class Side(Enum):
    """The side of a frontier condition never taken."""

    TRUE = "true"
    FALSE = "false"


class OracleMode(Enum):
    """Where FrontierOracle measures seeds."""

    THREAD = "thread"  # background worker (fuzzing)
    INLINE = "inline"  # in drain() on the caller's thread (tests)


def frontier_of(branches: Counts) -> dict[Branch, Side]:
    """Conditions with exactly one side counted, mapped to the missing side."""
    return {
        b: Side.FALSE if t else Side.TRUE for b, (t, f) in branches.items() if bool(t) != bool(f)
    }


@dataclass
class Run:
    """One input's branch counts."""

    path: Path
    branches: Counts


class Coverage:
    """Runs inputs through a ``_cov`` binary and reads branch counts back."""

    def __init__(self, binary: Path, workdir: Path, workers: int = 4) -> None:
        self._bin = Path(binary)
        self._work = Path(workdir)
        self._workers = workers
        self.sources: dict[str, str] = {}  # basename -> full path

    def run(self, files: Sequence[Path]) -> list[Run]:
        """Branch counts per input, in input order."""
        if not files:
            return []
        self._work.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(self._workers) as ex:
            return list(ex.map(self._one, range(len(files)), files))

    def profile(self, files: Sequence[Path]) -> Counts:
        """Corpus-wide counts: every input run, profiles merged, one export.

        ``llvm-cov export`` reloads the binary's coverage mapping per call
        (FFmpeg: ~4 s), so this is the cheap path; ``run`` exports per input.
        Raw profiles are folded into the merged one MERGE_BATCH at a time,
        which bounds disk use (FFmpeg: 3.5 MB per raw).
        """
        if not files:
            return {}
        self._work.mkdir(parents=True, exist_ok=True)
        acc = self._work / "all.profdata"
        for start in range(0, len(files), MERGE_BATCH):
            batch = files[start : start + MERGE_BATCH]
            with ThreadPoolExecutor(self._workers) as ex:
                raws = [r for r in ex.map(self._exec, range(start, start + len(batch)), batch) if r]
            self._fold(acc, raws)
        return self._export(acc) if acc.exists() else {}

    @staticmethod
    def merged(runs: Sequence[Run]) -> Counts:
        """Summed counts over runs (same as merging the profiles)."""
        out: Counts = {}
        for r in runs:
            for b, (t, f) in r.branches.items():
                t0, f0 = out.get(b, (0, 0))
                out[b] = (t0 + t, f0 + f)
        return out

    def measure(self, tag: str, data: bytes) -> Counts:
        """Branch counts of one input given as bytes."""
        self._work.mkdir(parents=True, exist_ok=True)
        path = self._work / f"{tag}.in"
        path.write_bytes(data)
        try:
            return self._one(tag, path).branches
        finally:
            path.unlink()

    def _exec(self, i: int | str, path: Path) -> Path | None:
        """Run one input; its raw profile, or None when none was written."""
        raw = self._work / f"{i}.profraw"
        env = {**os.environ, "LLVM_PROFILE_FILE": str(raw)}
        # Crashes and timeouts are expected; the stub dumps the profile on a fatal signal.
        try:
            subprocess.run(
                [str(self._bin), str(path)], env=env, capture_output=True, timeout=RUN_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            print(f"timeout: {path}", file=sys.stderr)
        return raw if raw.exists() else None

    @staticmethod
    def _fold(acc: Path, raws: Sequence[Path]) -> None:
        """Merge raws (and acc, if present) into acc; delete the raws."""
        if not raws:
            return
        inputs = [str(acc)] if acc.exists() else []
        tmp = acc.with_suffix(".tmp")
        cmd = ["llvm-profdata", "merge", "-sparse", *inputs, *map(str, raws), "-o", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(acc)
        for r in raws:
            r.unlink()

    def _one(self, i: int | str, path: Path) -> Run:
        raw = self._exec(i, path)
        if raw is None:
            return Run(path, {})

        prof = self._work / f"{i}.profdata"
        self._fold(prof, [raw])
        branches = self._export(prof)
        prof.unlink()
        return Run(path, branches)

    def _export(self, prof: Path) -> Counts:
        cmd = ["llvm-cov", "export", str(self._bin), f"-instr-profile={prof}", "-skip-expansions"]
        data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        out: Counts = {}
        for fdata in data["data"][0]["files"]:
            name = os.path.basename(fdata["filename"])
            if name == STUB_NAME:
                continue
            self.sources[name] = fdata["filename"]
            for b in fdata["branches"]:
                key = (name, b[_LINE], b[_COL])
                t, f = out.get(key, (0, 0))
                out[key] = (t + b[_TRUE], f + b[_FALSE])
        return out


class FrontierOracle:
    """Per-seed reach of open frontier conditions, measured as seeds arrive.

    Corpus totals sum every measured seed (evicted ones too), so the
    frontier only moves forward. ``reachers(c)`` counts live seeds that
    execute c: rare reach is the scheduling signal.
    """

    def __init__(
        self,
        binary: Path,
        workdir: Path,
        mode: OracleMode = OracleMode.THREAD,
        max_pending: int = MAX_PENDING,
        max_seeds: int = MAX_SEEDS,
    ) -> None:
        self._work = Path(workdir)
        self._cov = Coverage(binary, self._work, workers=1)
        self._max_pending = max_pending
        self._max_seeds = max_seeds

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: deque[tuple[object, bytes]] = deque()
        self._queued: set = set()
        self._busy = False
        self._closing = False

        self._totals: Counts = {}
        self._frontier: dict[Branch, Side] = {}
        self._reach: OrderedDict[object, frozenset[Branch]] = OrderedDict()
        self._reachers: dict[Branch, int] = {}
        self._seq = 0
        self._dropped = 0
        self._closed = 0
        self.version = 0  # bumps when the frontier changes
        self.epoch = 0  # bumps on every measured seed

        # The workdir goes at close() or interpreter exit, whichever comes first.
        self._cleanup = weakref.finalize(self, shutil.rmtree, self._work, True)

        self._thread = None
        if mode is OracleMode.THREAD:
            self._thread = threading.Thread(target=self._loop, name="frontier-oracle", daemon=True)
            self._thread.start()

    # -- producer ---------------------------------------------------------------

    def submit(self, key, data: bytes) -> None:
        """Queue a seed for measurement; dropped (counted) when the queue is full.

        A key already measured or queued is skipped: the totals count each seed once.
        """
        with self._lock:
            if key in self._reach or key in self._queued:
                return
            if len(self._queue) >= self._max_pending:
                self._dropped += 1
                return
            self._queue.append((key, bytes(data)))
            self._queued.add(key)
            self._wake.notify()

    def drain(self) -> None:
        """Measure everything queued: inline, or wait for the worker."""
        if self._thread is None:
            while self._step():
                pass
            return
        with self._lock:
            while self._queue or self._busy:
                self._wake.wait()

    def close(self) -> None:
        """Stop the worker and remove the work directory."""
        with self._lock:
            self._closing = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join()
        self._cleanup()

    # -- queries ----------------------------------------------------------------

    def frontier(self) -> dict[Branch, Side]:
        with self._lock:
            return dict(self._frontier)

    def open_reached(self, key) -> frozenset[Branch] | None:
        """Open conditions the seed executes; None if never measured (or evicted)."""
        with self._lock:
            reach = self._reach.get(key)
            if reach is None:
                return None
            return frozenset(c for c in reach if c in self._frontier)

    def rare_reach(self, keys) -> list[float]:
        """Per key: sum of 1/reachers over the open conditions it executes; 0 if unmeasured."""
        with self._lock:
            front, reachers = self._frontier, self._reachers
            out = []
            for k in keys:
                reach = self._reach.get(k)
                out.append(sum(1.0 / reachers[c] for c in reach if c in front) if reach else 0.0)
            return out

    def reachers(self, cond: Branch) -> int:
        with self._lock:
            return self._reachers.get(cond, 0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "seeds": len(self._reach),
                "pending": len(self._queue),
                "dropped": self._dropped,
                "frontier": len(self._frontier),
                "closed": self._closed,
            }

    # -- worker -----------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._closing:
                    self._wake.wait()
                if self._closing:
                    return
            self._step()

    def _step(self) -> bool:
        """Measure one queued seed; False when the queue is empty."""
        with self._lock:
            if not self._queue:
                return False
            key, data = self._queue.popleft()
            self._queued.discard(key)
            self._busy = True
            self._seq += 1
            tag = f"s{self._seq}"

        # busy clears only after the fold, so drain() never sees a measured
        # seed missing from the totals.
        counts: Counts = {}
        try:
            counts = self._cov.measure(tag, data)
        finally:
            with self._lock:
                self._fold(key, counts)
                self._busy = False
                self._wake.notify_all()
        return True

    def _fold(self, key, counts: Counts) -> None:
        """Add one seed's counts (caller holds the lock)."""
        executed = frozenset(b for b, (t, f) in counts.items() if t or f)
        for b, (t, f) in counts.items():
            t0, f0 = self._totals.get(b, (0, 0))
            self._totals[b] = (t0 + t, f0 + f)

        before = self._frontier
        self._frontier = frontier_of(self._totals)
        self._closed += sum(1 for b in before if b not in self._frontier)
        if self._frontier.keys() != before.keys():
            self.version += 1

        old = self._reach.pop(key, None)
        if old is not None:
            self._drop_reach(old)
        self._reach[key] = executed
        for b in executed:
            self._reachers[b] = self._reachers.get(b, 0) + 1
        while len(self._reach) > self._max_seeds:
            _, evicted = self._reach.popitem(last=False)
            self._drop_reach(evicted)
        self.epoch += 1

    def _drop_reach(self, executed: frozenset[Branch]) -> None:
        for b in executed:
            n = self._reachers[b] - 1
            if n:
                self._reachers[b] = n
            else:
                del self._reachers[b]
