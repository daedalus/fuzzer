"""Wall-time limits on z3: per query, and per fuzz round.

Every z3 ``check()`` goes through :data:`BUDGET`:

    query timeout = min(requested, --smt-query-cap, round time left)

    FuzzRound.run ── open_round() ── check() ── check() ── ... ── close_round()
                     remaining=10ms   -3.1ms     -6.9ms    spent: None, no z3

Outside a round only the per-query cap applies, so callers that are not on
the fuzz loop (reports, tests, one-off repairs) never starve. On ffmpeg
--hail-mary z3 was 15-30% of a round with 50-200 ms per-query timeouts.
"""

from __future__ import annotations

import time
from collections.abc import Callable

QUERY_MAX_MS = 30
"""Default --smt-query-cap: no single z3 query runs longer than this. Coupled-section solves take
15-27 ms (structural_constraints), so 10 ms made them always time out."""

ROUND_BUDGET_MS = 10
"""Default --smt-round-budget: all z3 queries of one fuzz round share this."""


class Z3Budget:
    """Grants z3 timeouts and charges the time each query actually took."""

    __slots__ = ("_remaining_ms", "_clock", "_query_cap_ms", "_round_budget_ms")

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._remaining_ms: float | None = None  # None: no round open
        self._clock = clock
        self._query_cap_ms = QUERY_MAX_MS
        self._round_budget_ms = ROUND_BUDGET_MS

    def configure(self, query_cap_ms: int, round_budget_ms: int) -> None:
        """Set both limits (ms). Zero is refused: it would disable z3 silently."""
        if query_cap_ms < 1 or round_budget_ms < 1:
            raise ValueError(f"z3 limits must be >= 1 ms, got {query_cap_ms}, {round_budget_ms}")
        self._query_cap_ms = query_cap_ms
        self._round_budget_ms = round_budget_ms

    def open_round(self) -> None:
        """Start a fuzz round with a full budget."""
        self._remaining_ms = float(self._round_budget_ms)

    def close_round(self) -> None:
        """End the round: only the per-query cap applies until the next one."""
        self._remaining_ms = None

    def spent(self) -> bool:
        """True when a round is open and has no whole millisecond left."""
        return self._remaining_ms is not None and self._remaining_ms < 1

    def grant(self, requested_ms: int) -> int:
        """Timeout for the next query; 0 when the round's budget is spent."""
        cap = min(requested_ms, self._query_cap_ms)
        if self._remaining_ms is None:
            return cap
        return int(min(cap, self._remaining_ms))

    def check(self, solver, requested_ms: int):
        """``solver.check()`` under the budget; None (not run) once spent."""
        timeout_ms = self.grant(requested_ms)
        if timeout_ms < 1:
            return None

        solver.set("timeout", timeout_ms)
        start = self._clock()
        try:
            return solver.check()
        finally:
            if self._remaining_ms is not None:
                self._remaining_ms -= (self._clock() - start) * 1000


BUDGET = Z3Budget()
"""Process-wide budget: z3 work in one round is shared across modules."""
