"""IJON max-map novelty: keep inputs that push an annotated value higher.

    exec slots ──> IjonMax.observe() ── any slot > campaign max ──> admit
                                    └── raised slot <- this input (max queue)

The target marks a value with ``IJON_MAX(x)`` / ``IJON_MIN(x)`` (see
``adapters/ijon.h``); the shim keeps the per-exec maximum per slot. Strict
``>``, unlike HwMax's growth factor: annotated values are deterministic, and
each step up is the progress IJON exists to reward (one more maze cell, one
more matching checksum byte). Reports stay bounded by the value range.

Per slot the tracker keeps the input that set the max (the paper's
maximization queue, §IV-A4): a better input replaces it; the old one stays in
the corpus, only the slot stops pointing at it.
"""

from collections.abc import Callable

import numpy as np

from fuzzer_tool.core.hashing import key16

# Slots in the max-map (paper §III-F, "by default 512"); a power of two.
IJON_MAP_SLOTS = 512


class IjonMax:
    """Campaign maximum and best input per IJON slot."""

    def __init__(self, slots: int = IJON_MAP_SLOTS) -> None:
        self._best = np.zeros(slots, dtype=np.uint64)
        self._inputs: dict[int, bytes] = {}
        self.hits = 0

    def observe(self, values: np.ndarray, data: bytes | None = None) -> bool:
        """Whether *values* beats the campaign max in any slot; raises it.

        *data* becomes the input of every raised slot; None records none.
        Example: best[3] = 10 -> values[3] = 10 is old, 11 is new.
        """
        raised = values > self._best
        if not raised.any():
            return False

        np.maximum(self._best, values, out=self._best)
        self.hits += 1
        if data is None:
            return True

        for slot in np.flatnonzero(raised).tolist():
            self._inputs[slot] = data
        return True

    def best(self) -> np.ndarray:
        """Read-only view of the campaign maxima."""
        view = self._best.view()
        view.flags.writeable = False
        return view

    def slot_inputs(self) -> dict[int, bytes]:
        """Slot -> the input holding its max (a copy)."""
        return dict(self._inputs)

    def pick_input(self, choose: Callable[[list[int]], int]) -> bytes | None:
        """The input of the slot *choose* picks, or None with no slot inputs."""
        if not self._inputs:
            return None
        return self._inputs[choose(sorted(self._inputs))]

    def to_dict(self) -> dict:
        keys = {slot: key16(data) for slot, data in self._inputs.items()}
        return {"best": self._best.tobytes(), "inputs": keys}

    def from_dict(
        self, state: object, resolve: Callable[[str], bytes | None] | None = None
    ) -> None:
        """Restore :meth:`to_dict`; malformed or resized state starts fresh.

        *resolve* maps a persisted input key back to corpus bytes; a slot
        whose input left the corpus keeps its max but loses its input.
        None restores the maxima only.
        """
        raw = state.get("best") if isinstance(state, dict) else None
        if not isinstance(raw, bytes) or len(raw) != self._best.nbytes:
            return
        self._best[:] = np.frombuffer(raw, dtype=np.uint64)

        keys = state.get("inputs")
        if resolve is None or not isinstance(keys, dict):
            return
        for slot, key in keys.items():
            data = resolve(key) if isinstance(key, str) else None
            if data is not None and isinstance(slot, int) and 0 <= slot < len(self._best):
                self._inputs[slot] = data
