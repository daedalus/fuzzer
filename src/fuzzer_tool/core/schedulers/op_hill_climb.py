"""OpHillClimbScheduler: stochastic hill climbing over operators.

Holds one operator. Each pick draws a uniform neighbour and moves to it only
if its Beta(1, 1) posterior mean is strictly higher; otherwise a random
restart moves anyway with probability ``restart``, so a stale peak is left::

    hold b (.33), neighbour a (.67)  ->  a
    hold a (.67), neighbour b (.33)  ->  a   (b with prob restart)

Unlike ``op_p2c``, the winner persists across picks: the scheduler exploits
one op until a better neighbour shows up. Experimental, Elo-only (absent from
``_FALLBACK_PRECEDENCE``).
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

DEFAULT_RESTART = 0.05


class OpHillClimbScheduler(ArmCounts):
    """Move to a random neighbour when it is uphill, or on a random restart."""

    #: Posterior starts at Beta(1, 1) for every operator; no prior override.
    supports_priors = False

    def __init__(self, rng, restart: float = DEFAULT_RESTART) -> None:
        if rng is None:
            raise ValueError("OpHillClimbScheduler requires a RandPool (Hard Rule 16)")
        if not 0.0 <= restart <= 1.0:
            raise ValueError(f"restart must be in [0, 1], got {restart}")

        super().__init__()
        self._rng = rng
        self._restart = restart
        self._current: str | None = None

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        cand = self._rng.choice(ops)
        cur = self._current
        if cur is None or cur not in ops or self.mean(cand) > self.mean(cur):
            self._current = cand
            return cand

        # Downhill or flat: stay, unless a restart fires.
        if cand != cur and self._restart > 0.0 and self._rng.random() < self._restart:
            self._current = cand
        return self._current
