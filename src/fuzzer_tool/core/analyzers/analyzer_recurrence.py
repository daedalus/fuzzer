"""Limit-cycle monitor over the campaign's own exec stream (``--recurrence``).

Each execution pushes one symbol, hash(seed, path hash), and whether it found
new coverage. Every ``window`` pushes the last ``window`` symbols get an RQA
reading (``core/recurrence.py``)::

    push x window ──► ring ──► rqa(embed, l_min) ──► verdict
                                                     │
          no novelty in window and DET >= det_trapped ─► TRAPPED
          otherwise                                   ─► FREE

TRAPPED means the campaign replays the same (seed, path) sequence without
finding anything. ``SeedPicker._saturation_gate`` reads it as "plateau is a
loop, not saturation" and re-enables the discovery analyses.

State is one window of symbols, so it is not persisted: a resumed campaign
re-reads a window before its first verdict.
"""

from __future__ import annotations

import enum

import numpy as np

from fuzzer_tool.core.recurrence import Rqa, rqa

# Execs per reading. 256 keeps the O(window^2) matrix at 64 KiB, ~0.4 ms.
_WINDOW = 256

# m-gram length: one repeated symbol is not a repeated sequence.
_EMBED = 3

# Diagonal length counted as deterministic. With embed 3, a line needs 10
# consecutive matching symbols; a 3-symbol iid stream does so at p ~ 3^-10.
_L_MIN = 8

# DET at or above this reads as a loop. Exact cycles score ~0.99.
_DET_TRAPPED = 0.9

# Smallest window that holds one line of length l_min.
_MIN_WINDOW = 8


class Novelty(enum.Enum):
    """Whether an execution found new coverage."""

    NONE = "none"
    NEW = "new"


class Recurrence(enum.Enum):
    """Monitor verdict."""

    UNKNOWN = "unknown"  # window not yet full
    FREE = "free"
    TRAPPED = "trapped"


class RecurrenceMonitor:
    """Ring of exec symbols with a periodic RQA verdict.

    Args:
        window: Execs per reading (ring size).
        embed: m-gram length.
        l_min: Shortest deterministic diagonal.
        det_trapped: DET threshold for TRAPPED, in (0, 1].
    """

    def __init__(
        self,
        window: int = _WINDOW,
        embed: int = _EMBED,
        l_min: int = _L_MIN,
        det_trapped: float = _DET_TRAPPED,
    ) -> None:
        if window < _MIN_WINDOW or embed < 1 or l_min < 1 or not 0.0 < det_trapped <= 1.0:
            raise ValueError("recurrence: bad window/embed/l_min/det_trapped")

        self._window = window
        self._embed = embed
        self._l_min = l_min
        self._det_trapped = det_trapped

        self._ring = np.zeros(window, dtype=np.int64)
        self._count = 0
        self._last_novel = -window  # exec index of the latest find
        self._reading: Rqa | None = None
        self.verdict = Recurrence.UNKNOWN
        self.trapped_count = 0  # readings that came out TRAPPED

    def push(self, symbol: int, novelty: Novelty) -> None:
        """Record one execution; re-read every ``window`` pushes."""
        self._ring[self._count % self._window] = symbol
        self._count += 1

        # A find ends a loop at once; no need to wait for the next reading.
        if novelty is Novelty.NEW:
            self._last_novel = self._count
            if self.verdict is Recurrence.TRAPPED:
                self.verdict = Recurrence.FREE

        if self._count % self._window == 0:
            self._evaluate()

    def _evaluate(self) -> None:
        """RQA over the last window, oldest first."""
        start = self._count % self._window
        stream = np.roll(self._ring, -start)
        self._reading = rqa(stream, self._embed, self._l_min)

        quiet = self._count - self._last_novel >= self._window
        if quiet and self._reading.det >= self._det_trapped:
            self.verdict = Recurrence.TRAPPED
            self.trapped_count += 1
            return
        self.verdict = Recurrence.FREE

    def reading(self) -> Rqa | None:
        """Latest RQA reading, None before the first full window."""
        return self._reading
