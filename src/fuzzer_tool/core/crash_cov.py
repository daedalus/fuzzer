"""Crash-only virgin map (Fuzzilli ``cov_evaluate_crash``).

Fuzzilli keeps a second virgin bitmap fed only by crashing runs: a crash is
coverage-new when it hits an edge no earlier crash hit. Signatures dedup
by stack; this dedups by path, so a signature duplicate that reaches new
crash edges hints that two bugs share one signature.

    crash ── edge ids ──> seen? ── no ──> mark, cov_new += 1
                                          └─ signature dup? ──> sig_dup_cov_new += 1
"""

from __future__ import annotations

import enum
import logging

import numpy as np

log = logging.getLogger(__name__)

# Shim edge ids are capped below 2^24 (afl_shim.c __AFL_GUARD_MAX_BITS);
# synthetic ids (bit 31) and anything larger take the sparse set.
DENSE_MAX = 1 << 24

# Bumped when the to_dict() layout changes; older payloads load as fresh.
_STATE_VERSION = 1
_N_COUNTS = 3


class SigNovelty(enum.Enum):
    """Whether signature dedup kept the crash."""

    NEW = "new"
    DUP = "dup"


class CrashCoverage:
    """Edges seen by any crashing run, plus crash novelty counters."""

    def __init__(self) -> None:
        self._crashes = 0
        self._cov_new = 0
        self._sig_dup_cov_new = 0
        self._clear_edges()

    def _clear_edges(self) -> None:
        # Lazily committed: untouched pages of np.zeros cost no memory.
        self._dense: np.ndarray | None = None
        self._sparse: set[int] = set()
        self._edges = 0

    def record(self, ids: np.ndarray, sig: SigNovelty) -> int:
        """Mark one crash's edge ids; return how many no earlier crash hit."""
        self._crashes += 1
        ids = np.asarray(ids, dtype=np.uint64)
        dense = ids < DENSE_MAX
        n_new = self._mark_dense(ids[dense].astype(np.intp)) + self._mark_sparse(ids[~dense])
        if n_new == 0:
            return 0

        self._edges += n_new
        self._cov_new += 1
        if sig is SigNovelty.DUP:
            self._sig_dup_cov_new += 1
        return n_new

    def _mark_dense(self, ids: np.ndarray) -> int:
        if ids.size == 0:
            return 0
        if self._dense is None:
            self._dense = np.zeros(DENSE_MAX, dtype=np.uint8)

        # Dedup first: a repeated new id must count once.
        fresh = np.unique(ids[self._dense[ids] == 0])
        self._dense[fresh] = 1
        return int(fresh.size)

    def _mark_sparse(self, ids: np.ndarray) -> int:
        before = len(self._sparse)
        self._sparse.update(ids.tolist())
        return len(self._sparse) - before

    def summary(self) -> dict[str, int]:
        """Crash and edge counters for the run summary."""
        return {
            "crashes": self._crashes,
            "cov_new": self._cov_new,
            "sig_dup_cov_new": self._sig_dup_cov_new,
            "edges": self._edges,
        }

    def to_dict(self) -> dict:
        """Seen edges and counters for ``--resume``."""
        dense = [] if self._dense is None else np.flatnonzero(self._dense).tolist()
        return {
            "version": _STATE_VERSION,
            "edges": dense + sorted(self._sparse),
            "counts": [self._crashes, self._cov_new, self._sig_dup_cov_new],
        }

    def from_dict(self, data) -> None:
        """Restore :meth:`to_dict` output; a malformed payload is ignored whole."""
        if not data:
            return
        try:
            if data.get("version") != _STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            edges = np.array([int(e) for e in data["edges"]], dtype=np.int64)
            counts = [int(c) for c in data["counts"]]
            if len(counts) != _N_COUNTS or min(counts) < 0 or (edges < 0).any():
                raise ValueError("bad counts or edges")
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as e:
            log.warning("crash coverage unreadable, starting fresh: %s", e)
            return

        self._clear_edges()
        self.record(edges.astype(np.uint64), SigNovelty.NEW)
        self._crashes, self._cov_new, self._sig_dup_cov_new = counts
