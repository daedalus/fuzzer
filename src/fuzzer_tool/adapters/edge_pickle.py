"""Pickle sink for per-seed edge ids: one pickled ``{seed_id: uint32 ids}`` dict.

The file is a single pickle at all times; batches extend its dict in place::

    PROTO 3  EMPTY_DICT  [MARK k v k v ... SETITEMS]  [MARK ... SETITEMS]  STOP
    \\_______ head _____/  \\______ batch 0 ________/  \\__ batch 1 ___/    ^ moved
                                                                          on each write

Each write overwrites the trailing STOP with the batch's set-items opcodes
and a new STOP, so one ``pickle.load`` returns every seed after any batch,
and memory holds one batch. A kill mid-write leaves no STOP: the next
``open`` cuts back to the last complete batch.

Batches are pickled at protocol 3: raw bytes, no frames, and memo opcodes
with explicit indices (BINPUT n / BINGET n). Every batch re-puts what it
gets, so reusing indices across batches is safe; protocol 4+ numbers memo
entries implicitly (MEMOIZE) and would not splice. 1-D arrays pickle as
``numpy.frombuffer(bytes, dtype)``: one raw copy, so loaded arrays are
read-only (``.copy()`` to mutate). Measured per 256-seed batch, 300
edges/seed: frombuffer 228 us, numpy's own protocol-3 reduce 547 us,
protocol 5 (the old layout) 374 us; memo-less (``fast``) pickling was 3x.

Exports written before this layout (one protocol-5 dict, or back-to-back
batch pickles) are rewritten once into it on ``open``.
"""

from __future__ import annotations

import copyreg
import io
import logging
import os
import pickle
import pickletools
from pathlib import Path
from typing import BinaryIO

import numpy as np

log = logging.getLogger(__name__)

_PROTO = 3
_HEAD = b"\x80\x03}"  # PROTO 3, EMPTY_DICT
# A batch dump's own head: PROTO 3, EMPTY_DICT, BINPUT 0 (memoizes the dict).
_BATCH_HEAD = _HEAD + b"q\x00"
_STOP = b"."
# Opcodes that close a batch: SETITEMS (many items), SETITEM (one).
_BATCH_END = frozenset(("SETITEMS", "SETITEM"))

# A cut pickle can surface as any of these, depending on where the cut fell.
_TORN = (EOFError, pickle.UnpicklingError, ValueError, AttributeError, IndexError, TypeError)


class _Stub:
    """Stand-in for numpy objects when only the keys are wanted."""

    def __init__(self, *args: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        pass


class _KeysOnly(pickle.Unpickler):
    """Unpickles the dict with every numpy object replaced by ``_Stub``."""

    def find_class(self, module: str, name: str):
        if module.partition(".")[0] == "numpy":
            return _Stub
        return super().find_class(module, name)


def _reduce_ids(a: np.ndarray):
    """1-D array -> ``numpy.frombuffer(raw bytes, dtype)``; anything else as numpy does."""
    if a.ndim != 1:
        return a.__reduce_ex__(_PROTO)
    return np.frombuffer, (a.tobytes(), a.dtype.str)


_DISPATCH = copyreg.dispatch_table.copy()
_DISPATCH[np.ndarray] = _reduce_ids


def _splice(batch: dict[str, np.ndarray]) -> bytes:
    """Set-items opcodes of *batch*, ready to follow the head or a prior batch.

    e.g. {k: v} pickles to HEAD BINPUT 0 MARK k v SETITEMS STOP -> MARK k v SETITEMS.
    """
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, _PROTO)
    pickler.dispatch_table = _DISPATCH
    pickler.dump(batch)
    data = buf.getvalue()
    if not data.startswith(_BATCH_HEAD):
        raise RuntimeError(f"unexpected pickle head {data[: len(_BATCH_HEAD)]!r}")
    return data[len(_BATCH_HEAD) : -len(_STOP)]


def _valid(batch: object) -> bool:
    return isinstance(batch, dict) and all(
        isinstance(k, str) and isinstance(v, np.ndarray) for k, v in batch.items()
    )


def read_edges_pickle(path: str | Path) -> dict[str, np.ndarray]:
    """The export in *path*; old back-to-back batch files are merged (later wins)."""
    merged: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
            except _TORN:
                return merged
            if not _valid(batch):
                return merged
            merged.update(batch)


class EdgePickleWriter:
    """Batch writer: each ``write`` extends the file's one pickled dict."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.known: set[str] = set()
        self._f: BinaryIO | None = None

    def open(self) -> set[str]:
        """Seed ids already in the file, read without rebuilding arrays.

        A torn tail or trailing junk is cut off (logged); an old-layout
        export is rewritten into this one.

        Raises:
            ValueError: the file is not an edge export.
        """
        if not self.path.exists() or not self.path.stat().st_size:
            return set()

        with open(self.path, "rb") as f:
            head = f.read(len(_HEAD))
        if head != _HEAD:
            self._migrate()
            return set(self.known)

        end = self._repair()
        with open(self.path, "rb") as f:
            batch = _KeysOnly(f).load()
            size = f.seek(0, 2)
        if not isinstance(batch, dict) or not all(isinstance(k, str) for k in batch):
            raise ValueError("not a {seed_id: ndarray} edge export")

        if end < size:
            log.warning("%s: %d bytes after the pickle cut off", self.path, size - end)
            with open(self.path, "r+b") as f:
                f.truncate(end)
        self.known = set(batch)
        return set(self.known)

    def write(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Add *items* to the pickled dict; seeds already written are dropped."""
        batch = {k: v for k, v in items if k not in self.known}
        if not batch:
            return

        f = self._handle()
        f.seek(-len(_STOP), os.SEEK_END)
        f.write(_splice(batch) + _STOP)
        f.flush()
        self.known.update(batch)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def _handle(self) -> BinaryIO:
        """Read-write handle; a new file starts as an empty pickled dict."""
        if self._f is not None:
            return self._f

        if not self.path.exists() or not self.path.stat().st_size:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(_HEAD + _STOP)
        self._f = open(self.path, "r+b")  # noqa: SIM115  (held across batches)
        return self._f

    def _repair(self) -> int:
        """End offset of the complete pickle; a torn one is cut back and re-closed."""
        with open(self.path, "rb") as f:
            try:
                _KeysOnly(f).load()
                return f.tell()
            except _TORN:
                pass

        good = _last_batch_end(self.path)
        log.warning("%s: torn last batch cut off at byte %d", self.path, good)
        with open(self.path, "r+b") as f:
            f.truncate(good)
            f.seek(good)
            f.write(_STOP)
        return good + len(_STOP)

    def _migrate(self) -> None:
        """Rewrite an old-layout export (one or more dict pickles) as one pickle."""
        merged = read_edges_pickle(self.path)
        if not merged:
            raise ValueError("not a {seed_id: ndarray} edge export")

        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(_HEAD + _splice(merged) + _STOP)
        os.replace(tmp, self.path)
        log.info("%s: rewritten as one pickle (%d seeds)", self.path, len(merged))
        self.known = set(merged)


def _last_batch_end(path: Path) -> int:
    """Offset just past the last complete batch (the head if there is none).

    Walks opcodes without building objects, stopping where the cut falls.
    """
    good = len(_HEAD)
    with open(path, "rb") as f:
        try:
            for op, _, pos in pickletools.genops(f):
                if op.name in _BATCH_END:
                    good = pos + 1
        except _TORN:
            pass
    return good
