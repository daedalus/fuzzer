"""Observed branch counts for ``--distance-metric likelihood``.

The shim bumps a counter per distance-table slot on every probe hit
(``DistHitsShm``). Every ``REFRESH_EXECS`` executions this sums site hits
into block hits, re-derives ``p(v|u)`` and rewrites the table in place::

    shim probe hit ──► counts[slot]++            (__AFL_DIST_HITS_ID)
                          │  every REFRESH_EXECS
                          ▼
    site_hits()  ──► key → block (site_blocks) ──► block_hits
                          ▼
    TargetDistance.reweight(block_hits) ──► DistanceTableShm.rewrite()

Seeds already tagged keep their old distance; new executions see the
re-weighted table.
"""

from __future__ import annotations

from fuzzer_tool.core.cadence import bucket

# Load-time Dijkstra per refresh; a few seconds of fuzzing between them.
REFRESH_EXECS = 5000
_SITE = "fuzzer.branch_counts"


class BranchCounts:
    """Feeds shim site hits back into a LIKELIHOOD ``TargetDistance``."""

    def __init__(self, distance, table, hits):
        self._distance = distance
        self._table = table
        self._hits = hits
        # Probe key -> block start, fixed for the run: scan the text once.
        self._block_of = distance.site_blocks()
        self._last: dict[int, int] = {}
        self._bucket = 0

    def tick(self, exec_count: int) -> bool:
        """Refresh once per ``REFRESH_EXECS`` bucket; True if re-weighted."""
        b = bucket(exec_count, REFRESH_EXECS, _SITE)
        if b == self._bucket:
            return False
        self._bucket = b
        return self.refresh()

    def refresh(self) -> bool:
        """Re-weight from the current hits; False when nothing changed."""
        block_hits: dict[int, int] = {}
        block_of = self._block_of
        for key, n in self._hits.site_hits().items():
            block = block_of.get(key)
            if block is not None:
                block_hits[block] = block_hits.get(block, 0) + n

        if not block_hits or block_hits == self._last:
            return False
        self._last = block_hits

        self._distance.reweight(block_hits)
        self._table.rewrite(self._distance.pc_distance_table())
        return True

    def cleanup(self) -> None:
        self._hits.cleanup()
