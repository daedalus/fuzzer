"""RegretMatchingScheduler: bandit regret matching+ over mutation operators.

Regret matching (Hart & Mas-Colell, *Econometrica* 2000) plays each action
in proportion to its positive cumulative regret -- how much better it
would have done than what was actually obtained. RM+ (Tammelin 2014)
clips the cumulative regret at zero every step, which forgets a bad run
instead of having to climb out of it. The bandit form replaces the
counterfactual reward with its importance-weighted estimate::

    p_i  = (1 - mix) R_i / sum R  +  mix / n      (uniform if sum R = 0)
    r^_i = r / p_i  if i was drawn, else 0
    R_i  = max(0, R_i + r^_i - r)                  for every offered arm

Warm start: each offered arm is played once, deterministically, before
the first sampled draw; its importance weight is 1.

No learning rate, no temperature: the only knob is the uniform floor
``mix``, which keeps ``1 / p`` bounded. A family the tree does not have:
game-theoretic, no-regret, and parameter-free.

Only the arms offered in the round are charged ``-r``: an operator the
sniffers did not offer could not have been played, so it accrued no
regret. On-policy only, like ``exp3`` and ``corral``.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Cap on remembered unconsumed draws (see ``op_corral._PENDING_MAX``).
_PENDING_MAX = 256


class RegretMatchingScheduler:
    """Bandit regret matching+.

    Args:
        mix: Uniform floor in [0, 0.5) mixed into every draw. Measured
            (convergence harness, 8 seeds, stationary best-arm tail share
            min / median): 0.000 / 0.439 at 0.01 (oscillation), 0.420 /
            0.566 at 0.05, 0.581 / 0.692 at 0.1, 0.646 / 0.677 at 0.2.
            Playing the average RM+ strategy instead of the current one
            did not help (0.579 / 0.663 at 0.1) and cost recovery. Swept
            before the warm start; with it, 0.649 / 0.679 at 0.1 (12 seeds).
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: Regret starts at zero; there is no prior to seed (Hard Rule 40).
    supports_priors = False

    def __init__(self, mix: float = 0.1, rng: RandPool | None = None) -> None:
        if not (0.0 <= mix < 0.5):
            raise ValueError(f"mix must be in [0, 0.5), got {mix!r}")
        self.mix = float(mix)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed clipped regret; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._regv = np.zeros(0, dtype=np.float64)
        self._drawnv = np.zeros(0, dtype=bool)  # warm start: ever drawn

        # Operator -> (p drawn with, offered arm indices), awaiting its reward.
        self._pending: dict[str, tuple[float, np.ndarray]] = {}
        self._rounds = 0
        self._orphans = 0

    def init_arm(self, name: str) -> None:
        """Register an operator at zero regret (idempotent)."""
        if name in self._idx:
            return
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._regv = np.append(self._regv, 0.0)
        self._drawnv = np.append(self._drawnv, False)

    def _law(self, idx: np.ndarray) -> np.ndarray:
        """Draw distribution over arm indices *idx*."""
        n = len(idx)
        reg = self._regv[idx]
        total = reg.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        return (1.0 - self.mix) * reg / total + self.mix / n

    def probabilities(self, ops: list[str]) -> dict[str, float]:
        """Regret-matching law over *ops*, registering anything new."""
        if not ops:
            return {}
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))
        return dict(zip(ops, self._law(idx).tolist(), strict=True))

    def select_op(self, ops: list[str]) -> str:
        """Draw by inverse CDF; remember p and the ballot for record."""
        if not ops:
            return ""
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))

        # Warm start: an offered arm never drawn is played first, with
        # certainty (p = 1). The uniform floor alone left 1-2 of 252 equal
        # arms unreached in 20k rounds on 3 of 20 seeds.
        fresh = np.flatnonzero(~self._drawnv[idx])
        if fresh.size:
            pos, p_pos = int(fresh[0]), 1.0
        else:
            p = self._law(idx)
            pos = int(np.searchsorted(np.cumsum(p), self._rng.random(), side="right"))
            pos = min(pos, len(ops) - 1)
            p_pos = float(p[pos])
        chosen = ops[pos]
        self._drawnv[idx[pos]] = True

        # Drop the oldest unconsumed draw rather than grow without bound.
        if len(self._pending) >= _PENDING_MAX and chosen not in self._pending:
            self._pending.pop(next(iter(self._pending)))
        self._pending[chosen] = (p_pos, idx)
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """RM+ step over the ballot the arm was drawn from."""
        drawn = self._pending.pop(name, None)
        if drawn is None:
            self._orphans += 1
            return
        p, ballot = drawn
        self._rounds += 1
        r = unit_reward(success, weight)
        if r == 0.0:
            return  # r^ = r = 0: no regret moves

        # Every offered arm is charged r; the drawn one is credited r / p.
        reg = self._regv
        reg[ballot] -= r
        reg[self._idx[name]] += r / p
        np.maximum(reg, 0.0, out=reg)

    def bandit_stats(self) -> dict:
        """Return regret-matching diagnostics."""
        return {
            "regret_matching_pulls": self._rounds,
            "regret_matching_arms": len(self._names),
            "regret_matching_orphans": self._orphans,
            "regret_matching_max_regret": float(self._regv.max()) if self._names else 0.0,
        }
