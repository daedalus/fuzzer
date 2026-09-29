"""Sparse second-order operator transition table: P(next | prev2, prev).

The first-order chain in ``schedulers/op_monte_carlo.py`` conditions on one
predecessor. A dense second-order table is ~150^3 ~ 3.4M counters; this keeps
only observed ``(prev2, prev)`` contexts, capped at ``max_contexts`` (AGENTS
rule 54) by evicting the least-observed context.

Counting matches the first-order rule: only successes count, and a repeat of
the same operator is not a transition.

    (prev2, prev) -> {next: successes}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# 4096 contexts x ~150 ops bounds the table well under a few MB.
DEFAULT_MAX_CONTEXTS = 4096

Context = tuple[str, str]


class SecondOrderChain:
    def __init__(self, max_contexts: int = DEFAULT_MAX_CONTEXTS) -> None:
        if max_contexts < 1:
            raise ValueError(f"max_contexts must be >= 1, got {max_contexts}")
        self._max = max_contexts
        self._counts: dict[Context, dict[str, int]] = {}
        self._totals: dict[Context, int] = {}

    def __len__(self) -> int:
        return len(self._counts)

    def count(self, prev2: str, prev: str, nxt: str) -> int:
        return self._counts.get((prev2, prev), {}).get(nxt, 0)

    def total(self, prev2: str, prev: str) -> int:
        return self._totals.get((prev2, prev), 0)

    def record(self, prev2: str | None, prev: str | None, nxt: str, *, success: bool) -> None:
        if not success or prev2 is None or prev is None or prev == nxt:
            return

        key = (prev2, prev)
        if key not in self._counts:
            self._make_room()
            self._counts[key] = {}
            self._totals[key] = 0
        row = self._counts[key]
        row[nxt] = row.get(nxt, 0) + 1
        self._totals[key] += 1

    def _make_room(self) -> None:
        """Evict the least-observed context (oldest on ties) when at the cap."""
        if len(self._counts) < self._max:
            return
        victim = min(self._totals, key=self._totals.__getitem__)
        del self._counts[victim]
        del self._totals[victim]

    def scores(self, ops: Sequence[str], prev2: str, prev: str) -> dict[str, float] | None:
        """Dirichlet(1)-smoothed P(op | prev2, prev), or None for an unseen context."""
        key = (prev2, prev)
        if key not in self._counts:
            return None

        row = self._counts[key]
        denom = self._totals[key] + len(ops)
        return {op: (row.get(op, 0) + 1) / denom for op in ops}

    def to_state(self) -> dict[str, Any]:
        return {"max": self._max, "counts": {k: dict(v) for k, v in self._counts.items()}}

    @classmethod
    def from_state(cls, state: Any) -> SecondOrderChain:
        """Rebuild from ``to_state``; anything malformed yields an empty chain."""
        if not isinstance(state, dict) or not isinstance(state.get("counts"), dict):
            return cls()

        chain = cls(max_contexts=int(state.get("max", DEFAULT_MAX_CONTEXTS)) or 1)
        for key, row in state["counts"].items():
            if not (isinstance(key, tuple) and len(key) == 2 and isinstance(row, dict)):
                continue
            chain._counts[key] = {str(k): int(v) for k, v in row.items()}
            chain._totals[key] = sum(chain._counts[key].values())
        return chain
