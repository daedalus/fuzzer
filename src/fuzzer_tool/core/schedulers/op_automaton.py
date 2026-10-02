"""LearningAutomatonScheduler: linear reward-inaction (L_R-I) automaton.

A variable-structure stochastic learning automaton (Narendra & Thathachar,
*Learning Automata*, 1989) keeps a probability vector and nudges it after
each action::

    reward r > 0 on drawn j:  p_j += l r (1 - p_j);  p_i *= 1 - l r  (i != j)
    reward 0:                 nothing ("inaction")

The update preserves ``sum p = 1`` exactly and is O(K). L_R-I is
epsilon-optimal in stationary environments: for small enough ``l`` it
converges to the best action with probability arbitrarily close to 1.
It is the cheapest learner here and the only one whose state *is* the
policy: no estimates, no posterior, no normaliser.

Warm start: each offered arm is played once before the first sampled draw,
so every operator gets one chance at a reward before the walk can absorb.

Its known failure is absorption: the vector reaches a unit vector and the
automaton never explores again, right or wrong. ``mix`` floors every
offered arm at ``mix / n`` at selection so an absorbed automaton can still
see -- and move toward -- an operator that starts paying.

On-policy only, like ``exp3`` and ``corral``: the update is a stochastic
gradient step only for actions sampled from ``p`` itself.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Cap on remembered unconsumed draws (see ``op_corral._PENDING_MAX``).
_PENDING_MAX = 256


class LearningAutomatonScheduler:
    """L_R-I automaton over mutation operators.

    Args:
        rate: Learning step ``l`` in (0, 1]; a reward r moves ``l r`` of mass.
            Measured (convergence harness, 8 seeds; stationary share min,
            regret slope max, DecayingBest late share min): 0.910 / 0.257 /
            0.884 at 0.01, 0.912 / 0.492 / 0.932 at 0.03, 0.914 / 0.693 /
            0.930 at 0.05. The mix floor caps the stationary share near
            1 - mix (K - 1) / K and is also what recovers a dead arm.
            Swept before the warm start; with it, at 0.03 over 12 seeds:
            0.926 / 0.481 / 0.928.
        mix: Uniform floor in [0, 0.5) applied at selection.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: The probability vector starts uniform; no prior to seed (Hard Rule 40).
    supports_priors = False

    def __init__(self, rate: float = 0.03, mix: float = 0.05, rng: RandPool | None = None) -> None:
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"rate must be in (0, 1], got {rate!r}")
        if not (0.0 <= mix < 0.5):
            raise ValueError(f"mix must be in [0, 0.5), got {mix!r}")
        self.rate = float(rate)
        self.mix = float(mix)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Probability vector over every registered arm; _names[i] <-> _idx.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._pv = np.zeros(0, dtype=np.float64)
        self._drawnv = np.zeros(0, dtype=bool)  # warm start: ever drawn

        # Operators drawn by this automaton, awaiting their reward.
        self._pending: dict[str, None] = {}
        self._rounds = 0
        self._orphans = 0

    def init_arm(self, name: str) -> None:
        """Register at the uniform share; incumbents scale by M / (M + 1)."""
        if name in self._idx:
            return
        m = len(self._names)
        self._idx[name] = m
        self._names.append(name)
        share = 1.0 / (m + 1)
        self._pv = np.append(self._pv * (1.0 - share), share)
        self._drawnv = np.append(self._drawnv, False)

    def _law(self, ops: list[str]) -> np.ndarray:
        """The vector restricted to *ops*, renormalised, with the mix floor."""
        for op in ops:
            self.init_arm(op)
        n = len(ops)
        q = self._pv[[self._idx[op] for op in ops]]
        total = q.sum()
        q = q / total if total > 0.0 else np.full(n, 1.0 / n)
        return (1.0 - self.mix) * q + self.mix / n

    def probabilities(self, ops: list[str]) -> dict[str, float]:
        """Draw distribution over *ops*, registering anything new."""
        if not ops:
            return {}
        return dict(zip(ops, self._law(ops).tolist(), strict=True))

    def select_op(self, ops: list[str]) -> str:
        """Draw by inverse CDF and mark the draw as this automaton's."""
        if not ops:
            return ""
        q = self._law(ops)
        drawn = self._drawnv[[self._idx[op] for op in ops]]

        # Warm start: a never-drawn offered arm is played first. With equal
        # arms L_R-I random-walks toward absorption, and the mix floor alone
        # left one of 252 arms unreached in 20k rounds on 4 of 20 seeds.
        fresh = np.flatnonzero(~drawn)
        if fresh.size:
            pos = int(fresh[0])
        else:
            pos = int(np.searchsorted(np.cumsum(q), self._rng.random(), side="right"))
            pos = min(pos, len(ops) - 1)
        chosen = ops[pos]
        self._drawnv[self._idx[chosen]] = True

        # Drop the oldest unconsumed draw rather than grow without bound.
        if len(self._pending) >= _PENDING_MAX and chosen not in self._pending:
            self._pending.pop(next(iter(self._pending)))
        self._pending[chosen] = None
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Reward-inaction step for an arm this automaton drew."""
        if name not in self._pending:
            self._orphans += 1
            return
        del self._pending[name]
        self._rounds += 1
        step = self.rate * unit_reward(success, weight)
        if step <= 0.0:
            return  # inaction

        # p_i *= 1 - step for all, then p_j += step: same as the textbook pair.
        self._pv *= 1.0 - step
        self._pv[self._idx[name]] += step

    def bandit_stats(self) -> dict:
        """Return automaton diagnostics."""
        n = len(self._names)
        p = self._pv
        ent = -float((p[p > 0.0] * np.log(p[p > 0.0])).sum()) if n else 0.0
        return {
            "automaton_pulls": self._rounds,
            "automaton_orphans": self._orphans,
            "automaton_concentration": float(p.max()) if n else 0.0,
            "automaton_entropy": ent / math.log(n) if n > 1 else 0.0,
        }
