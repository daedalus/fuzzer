"""AFL-style stage: one seed pick serves up to K consecutive execs.

    pick + score ──> begin(seed, perf_score)       exec 1
                     resume(live) -> seed          exec 2 .. k
                     resume(live) -> None          next pick

The main loop picked and scored a seed on every exec. With ``--stage-execs
K`` the pick is reused for ``k = clamp(K * perf_score / 100, 1, K)`` execs,
so better-scored seeds get longer stages (AFL's perf_score sizes havoc the
same way). K = 1 keeps one pick per exec.
"""

from __future__ import annotations

from collections.abc import Container

# AFL's neutral perf_score: a seed scored 100 gets the full stage.
PERF_NEUTRAL = 100.0


class SeedStage:
    """The current stage: its seed and the execs it still has."""

    def __init__(self, execs: int):
        if execs < 1:
            raise ValueError(f"stage execs must be >= 1, got {execs}")
        self._max = execs
        self._seed: bytes | None = None
        self._left = 0

    def begin(self, seed: bytes, perf_score: float) -> None:
        """Start a stage on a fresh pick; this exec is its first."""
        k = round(self._max * perf_score / PERF_NEUTRAL)
        self._seed = seed
        self._left = min(self._max, max(1, k)) - 1

    def resume(self, live: Container[bytes]) -> bytes | None:
        """The stage's seed for this exec, or None when a pick is due.

        A seed no longer in *live* (pruned, minimized away) ends the stage.
        """
        if self._left <= 0 or self._seed not in live:
            self._left = 0
            return None

        self._left -= 1
        return self._seed
