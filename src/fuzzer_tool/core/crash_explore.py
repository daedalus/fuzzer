r"""Crash exploration (AFL ``-C``): grow a corpus of related crash variants.

Seeds are crashing inputs. A mutant survives only if it still crashes and
reaches an edge no earlier crash reached; everything else is dropped::

    crash seed --mutate--> no crash          -> drop
                       \-> crash, old path   -> drop
                        \-> crash, new edge  -> queue + save

Comparing the variants' fault addresses shows how far the input steers the
fault (lcamtuf's unrtf example: the faulting address tracked an RTF integer).

Coverage here is crash-path coverage only, separate from the campaign map:
a non-crashing mutant must not use up the novelty of an edge that a later
crashing mutant reaches.
"""

from collections.abc import Iterable

import numpy as np


class CrashExplorer:
    """Union of edges reached by crashing executions."""

    def __init__(self) -> None:
        # Bounded by the coverage map size: edge ids are map indices.
        self._edges: set[int] = set()
        self.variants = 0

    def observe(self, edges: Iterable[int] | bytes) -> bool:
        """Record a crash's edges; True when any edge is new.

        *edges* is an edge-id iterable or a byte bitmap (ptrace backend),
        where a nonzero byte at index i means edge i was hit. An empty trace
        is never new: a crash with no readable coverage would otherwise
        admit every such mutant.
        """
        if isinstance(edges, bytes | bytearray):
            edges = np.flatnonzero(np.frombuffer(edges, np.uint8)).tolist()
        new = set(edges) - self._edges
        if not new:
            return False
        self._edges |= new
        self.variants += 1
        return True
