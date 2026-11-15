"""IJON max-map novelty: keep inputs that push an annotated value higher.

    exec slots ──> IjonMax.observe() ── any slot > campaign max ──> admit

The target marks a value with ``IJON_MAX(x)`` / ``IJON_MIN(x)`` (see
``adapters/ijon.h``); the shim keeps the per-exec maximum per slot. Strict
``>``, unlike HwMax's growth factor: annotated values are deterministic, and
each step up is the progress IJON exists to reward (one more maze cell, one
more matching checksum byte). Reports stay bounded by the value range.
"""

import numpy as np

# Slots in the max-map (AFL++/IJON's MAXMAP_SIZE); a power of two.
IJON_MAP_SLOTS = 512


class IjonMax:
    """Campaign maximum per IJON slot."""

    def __init__(self, slots: int = IJON_MAP_SLOTS) -> None:
        self._best = np.zeros(slots, dtype=np.uint64)
        self.hits = 0

    def observe(self, values: np.ndarray) -> bool:
        """Whether *values* beats the campaign max in any slot; raises it.

        Example: best[3] = 10 -> values[3] = 10 is old, 11 is new.
        """
        if not (values > self._best).any():
            return False

        np.maximum(self._best, values, out=self._best)
        self.hits += 1
        return True

    def best(self) -> np.ndarray:
        """Read-only view of the campaign maxima."""
        view = self._best.view()
        view.flags.writeable = False
        return view

    def to_dict(self) -> dict:
        return {"best": self._best.tobytes()}

    def from_dict(self, state: object) -> None:
        """Restore :meth:`to_dict`; malformed or resized state starts fresh."""
        raw = state.get("best") if isinstance(state, dict) else None
        if not isinstance(raw, bytes) or len(raw) != self._best.nbytes:
            return
        self._best[:] = np.frombuffer(raw, dtype=np.uint64)
