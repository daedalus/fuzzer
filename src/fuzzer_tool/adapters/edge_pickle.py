"""Pickle sink for per-seed edge ids: one ``{seed_id: uint32 ids}`` dict per batch.

    e.pkl = pickle(batch 0) pickle(batch 1) ... pickle(batch n)

Batches are appended as they are written, so memory holds one batch and a
killed run keeps every complete batch. A single ``pickle.load`` returns the
first batch only; :func:`read_edges_pickle` merges all of them. An export
written before batching is one dict, i.e. one batch, and reads unchanged.

A torn last batch (kill mid-append) is cut off on the next ``open``.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

log = logging.getLogger(__name__)

# A cut batch can surface as any of these, depending on where the cut fell.
_TORN = (EOFError, pickle.UnpicklingError, ValueError, AttributeError, IndexError, TypeError)


def _valid(batch: object) -> bool:
    return isinstance(batch, dict) and all(
        isinstance(k, str) and isinstance(v, np.ndarray) for k, v in batch.items()
    )


def _batches(f: BinaryIO) -> Iterator[tuple[dict[str, np.ndarray], int]]:
    """``(batch, end offset)`` per complete batch; stops at EOF or the first bad one."""
    while True:
        try:
            batch = pickle.load(f)
        except _TORN:
            return
        if not _valid(batch):
            return
        yield batch, f.tell()


def read_edges_pickle(path: str | Path) -> dict[str, np.ndarray]:
    """Every batch in *path*, merged (later batches win)."""
    merged: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        for batch, _ in _batches(f):
            merged.update(batch)
    return merged


class EdgePickleWriter:
    """Batch writer: one appended pickle per ``write``; file created on the first one."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.known: set[str] = set()
        self._f: BinaryIO | None = None

    def open(self) -> set[str]:
        """Seed ids already in the file; a torn tail is cut off (logged).

        Raises:
            ValueError: the file holds no complete batch (not an edge export).
        """
        if not self.path.exists():
            return set()

        good = 0  # end offset of the last complete batch
        with open(self.path, "rb") as f:
            for batch, end in _batches(f):
                self.known.update(batch)
                good = end
            size = f.seek(0, 2)

        if good == 0 and size:
            raise ValueError("not a {seed_id: ndarray} edge export")
        if good < size:
            log.warning("%s: torn last batch (%d bytes) cut off", self.path, size - good)
            with open(self.path, "r+b") as f:
                f.truncate(good)
        return set(self.known)

    def write(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Append *items* as one batch; seeds already written are dropped."""
        batch = {k: v for k, v in items if k not in self.known}
        if not batch:
            return

        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "ab")  # noqa: SIM115  (held across batches)
        pickle.dump(batch, self._f, protocol=pickle.HIGHEST_PROTOCOL)
        self._f.flush()
        self.known.update(batch)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
