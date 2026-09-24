"""OpTPEScheduler: Tree-structured Parzen Estimator over operators (BO-3).

GP-UCB (``op_gp_ucb.py`` / ``op_bo_gp_ucb.py``) models ``p(y | op)``. TPE
(Bergstra et al., 2011) inverts it: split recent outcomes at the ``gamma``
reward quantile and model the operator density on each side.

    l(op) ∝ prior_alpha(op) + good(op)     # density among the top-gamma outcomes
    g(op) ∝ prior_beta(op)  + bad(op)      # density among the rest

Expected improvement is monotone in ``l/g``. For a categorical variable the
Parzen estimator is this smoothed histogram; the prior pseudo-counts are the
kernel's bandwidth. Selection follows hyperopt: draw ``N_CANDIDATES`` from
``l``, keep the one with the largest ``l/g`` (first drawn wins ties). The
draws give exploration; the ratio gives exploitation.

Split, over the last ``window`` outcomes (non-stationarity):

    n_good = ceil(gamma * n)
    good   = the n_good highest rewards among rewards > 0 (newer wins ties)

Rewards are ``weight`` on success, else 0. Non-finite or non-positive
weights count as failures. Example, gamma=0.25, n=8, rewards
[.9, .5, .8, 0, 0, 0, 0, 0]: n_good=2, good = the .9 and .8 outcomes.

``supports_priors = True``: format-operator priors seed ``l``/``g``.
Elo-only like ``op_kruskal_count``; absent from ``_FALLBACK_PRECEDENCE``.
"""

from __future__ import annotations

import bisect
import math
from collections import deque

from fuzzer_tool.core.rand_pool import RandPool

GAMMA = 0.25
WINDOW = 512
N_CANDIDATES = 24
_DEFAULT_PRIOR = (1.0, 1.0)


def _reward(success: bool, weight: float) -> float:
    """Success weight, or 0 for failures and non-finite/non-positive weights."""
    if not success or not math.isfinite(weight) or weight <= 0.0:
        return 0.0
    return weight


class OpTPEScheduler:
    """Categorical TPE operator scheduler.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        gamma: Good-set quantile, in (0, 1).
        window: Outcomes kept for the split.
    """

    supports_priors = True  # init_arm's (alpha, beta) seeds l and g.

    def __init__(self, rng: RandPool | None = None, gamma: float = GAMMA, window: int = WINDOW):
        if rng is None:
            raise ValueError("OpTPEScheduler requires a RandPool (Hard Rule 16)")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self._rng = rng
        self._gamma = gamma
        self._hist: deque[tuple[str, float]] = deque(maxlen=max(1, window))
        self._priors: dict[str, tuple[float, float]] = {}
        self._pulls = 0
        self._split: tuple[dict[str, int], dict[str, int]] | None = None

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *name*; the priors are its l and g pseudo-counts."""
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(f"priors must be > 0, got ({prior_alpha}, {prior_beta})")
        self._priors[name] = (prior_alpha, prior_beta)

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """Append one outcome; the split is recomputed lazily on next use."""
        self._hist.append((op, _reward(success, weight)))
        self._pulls += 1
        self._split = None

    def split_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        """Per-op (good, bad) counts over the window; cached until next record."""
        if self._split is not None:
            return self._split

        hist = self._hist
        n_good = math.ceil(self._gamma * len(hist))

        # Top n_good positive rewards; newer index wins ties.
        pos = [(r, i) for i, (_, r) in enumerate(hist) if r > 0.0]
        if len(pos) > n_good:
            pos.sort(reverse=True)
            pos = pos[:n_good]
        good_idx = {i for _, i in pos}

        good: dict[str, int] = {}
        bad: dict[str, int] = {}
        for i, (op, _) in enumerate(hist):
            side = good if i in good_idx else bad
            side[op] = side.get(op, 0) + 1

        self._split = (good, bad)
        return self._split

    def _masses(self, ops: list[str]) -> tuple[list[float], list[float]]:
        """Unnormalised l and g mass per offered op."""
        good, bad = self.split_counts()
        priors = self._priors
        ls: list[float] = []
        gs: list[float] = []
        for op in ops:
            a, b = priors.get(op, _DEFAULT_PRIOR)
            ls.append(a + good.get(op, 0))
            gs.append(b + bad.get(op, 0))
        return ls, gs

    def ratios(self, ops: list[str]) -> dict[str, float]:
        """``l/g`` per op up to a shared constant (the EI ordering)."""
        ls, gs = self._masses(ops)
        return {op: lv / gv for op, lv, gv in zip(ops, ls, gs, strict=True)}

    def select_op(self, ops: list[str]) -> str:
        """Draw N_CANDIDATES from l; return the one with the largest l/g."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        ls, gs = self._masses(ops)
        cum: list[float] = []
        acc = 0.0
        for v in ls:
            acc += v
            cum.append(acc)

        last = len(ops) - 1
        best, best_ratio = 0, -1.0
        for _ in range(N_CANDIDATES):
            i = bisect.bisect_right(cum, self._rng.random() * acc, 0, last)
            ratio = ls[i] / gs[i]
            if ratio > best_ratio:
                best, best_ratio = i, ratio
        return ops[best]

    def bandit_stats(self) -> dict:
        """TPE diagnostics."""
        good, _ = self.split_counts()
        return {
            "tpe_pulls": self._pulls,
            "tpe_window": len(self._hist),
            "tpe_good": sum(good.values()),
            "tpe_gamma": self._gamma,
            "operators_tracked": len(self._priors),
        }
