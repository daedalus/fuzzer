"""Position-selection schedulers: which byte offset a mutation lands on.

Third axis of layer 2 (scheduling), beside operator (``op_*``) and seed
(``seed_*``) selection:

    seed scheduler  -> which corpus entry is fuzzed
    op scheduler    -> which operator mutates it
    pos scheduler   -> where in the buffer that operator lands

A position scheduler proposes an offset, or declines with ``None``.
``services/position_arena.py`` arbitrates between them under ``pos_<name>``
Elo keys; ``services/operators.py::select_position`` is the only caller.

Contract::

    name: str
    propose(data, buf_len) -> int | None   # None = no opinion
    record(data, offsets, outcome, weight) # feedback; may be a no-op

``data`` is the parent seed, ``buf_len`` the live buffer (earlier operators
in the round may have resized it), so proposals are clamped by the caller.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from fuzzer_tool.core.rand_pool import RandPool


class Outcome(enum.Enum):
    """Whether the round found new coverage."""

    GAIN = "gain"
    MISS = "miss"


@runtime_checkable
class PositionScheduler(Protocol):
    name: str

    def propose(self, data: bytes, buf_len: int) -> int | None: ...

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None: ...


class UniformPosition:
    """Uniform offset. The arena's baseline: every other arm must beat it."""

    name = "uniform"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng

    def propose(self, data: bytes, buf_len: int) -> int | None:
        return self._rng.randint(0, max(0, buf_len - 1))

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Stateless."""


class CallablePosition:
    """Adapts an existing proposer (``fn(data, buf_len) -> int | None``).

    The MI/TE/phase/sensitivity/crash-MI/region trackers keep their own
    feedback loops; the arena only borrows their proposals, so ``record``
    is a no-op.
    """

    def __init__(self, name: str, fn: Callable[[bytes, int], int | None]) -> None:
        self.name = name
        self._fn = fn

    def propose(self, data: bytes, buf_len: int) -> int | None:
        return self._fn(data, buf_len)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """The wrapped tracker learns through its own path."""
