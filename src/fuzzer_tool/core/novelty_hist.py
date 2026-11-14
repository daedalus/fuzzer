"""New-edges-per-admitted-seed histogram (Fuzzilli corpus diagnostic).

Fuzzilli reports how many new edges each corpus sample contributed; on
JavaScriptCore 48% added one edge. A corpus dominated by the ``1`` bucket
finds features one at a time and rarely combines them.

    admit(n new edges) -> bucket:  0 | 1 | 2-5 | 6-10 | 11+

``0`` counts admissions without new edges (initial seeds, hit-count gains).
"""

from __future__ import annotations

import logging
from array import array
from bisect import bisect_left

log = logging.getLogger(__name__)

# Inclusive upper bound of each bucket but the last; 11+ is open-ended.
_UPPER = (0, 1, 5, 10)
BUCKET_LABELS = ("0", "1", "2-5", "6-10", "11+")

# Bumped when the to_dict() layout changes; older payloads load as fresh.
_STATE_VERSION = 1


class NoveltyHist:
    """Counts admitted seeds by how many new edges each one found."""

    def __init__(self) -> None:
        self._counts = array("Q", bytes(8 * len(BUCKET_LABELS)))

    def add(self, n_new: int) -> None:
        """Count one admission that found *n_new* edges (negatives clamp to 0)."""
        self._counts[bisect_left(_UPPER, max(0, n_new))] += 1

    def counts(self) -> tuple[int, ...]:
        """Per-bucket admissions, ordered as :data:`BUCKET_LABELS`."""
        return tuple(self._counts)

    def total(self) -> int:
        """Admissions counted so far."""
        return sum(self._counts)

    def shares(self) -> tuple[float, ...]:
        """Per-bucket fraction of admissions; all zero when empty."""
        total = self.total()
        if total == 0:
            return tuple(0.0 for _ in self._counts)
        return tuple(c / total for c in self._counts)

    def to_dict(self) -> dict:
        """Bucket counts for ``--resume``."""
        return {"version": _STATE_VERSION, "counts": list(self._counts)}

    def from_dict(self, data) -> None:
        """Restore :meth:`to_dict` output; a malformed payload is ignored whole."""
        if not data:
            return
        try:
            if data.get("version") != _STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            counts = array("Q", (int(c) for c in data["counts"]))
            if len(counts) != len(BUCKET_LABELS):
                raise ValueError(f"{len(counts)} buckets")
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as e:
            log.warning("novelty histogram unreadable, starting fresh: %s", e)
            return

        self._counts = counts
