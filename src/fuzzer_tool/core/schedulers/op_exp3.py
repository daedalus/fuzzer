"""Exp3Scheduler: adversarial bandit (EXP3)."""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool

# Blowup guard on the relative weights, and a floor for arms scaled below it.
_RENORM_THRESHOLD = 1e9
_WEIGHT_FLOOR = 1e-300
# Largest log-weight step one record may take. A realistic step is <= the
# reward weight (p >= gamma/K bounds r̂); the cap only stops a pathological
# weight from raising OverflowError inside math.exp.
_MAX_LOG_STEP = 100.0
# Fenwick trees kept at once, one per distinct operator list offered to
# select_op() (build_ops() filters REGISTRY per input, so a campaign sees a
# handful of lists). Evicted first-in first-out past this.
_TREE_CACHE_MAX = 16
# Update-log entries retained before every tree is forced to rebuild. Bounds
# the log for a list that is not offered again for a long time.
_UPDATE_LOG_MAX = 4096


class Exp3Scheduler:
    """EXP3 adversarial bandit for operator selection.

    The Exponential-weight algorithm for Exploration and Exploitation (Auer et
    al. 2002) handles non-stationary reward distributions that violate the
    i.i.d. assumption of Beta-Bernoulli Thompson sampling.

    At each round:
        p_i = (1 - gamma) * w_i / sum(w)  +  gamma / K    (mixture)
        sample i ~ p
        receive reward r in [0, 1]
        r̂_i = r / p_i   (importance-weighted estimator)
        w_i = w_i * exp(gamma * r̂_i / K)

    Args:
        gamma: Exploration rate in [0, 1]. Higher = more uniform exploration.
        window_decay: Exponential decay per update (1.0 = no decay).
            Values < 1.0 discount older observations.
        rng: PRNG for random draws. Defaults to RandPool().
    """

    supports_priors = False

    def __init__(
        self, gamma: float = 0.1, window_decay: float = 0.999, rng: RandPool | None = None
    ):
        self.gamma = gamma
        self.window_decay = window_decay
        self._log_window_decay = math.log(window_decay) if 0.0 < window_decay < 1.0 else 0.0
        self._rng = rng if rng is not None else RandPool()
        # Per-arm weights RELATIVE to exp(_log_decay): the actual weight is
        # weights[i] * exp(_log_decay). Decay is folded into that one scalar
        # so record() stays O(1) instead of sweeping every arm. The scalar is
        # kept in log space because 0.999**n underflows to a subnormal and
        # then sticks near 5e-324 after ~745k records.
        #
        # A factor common to every arm cancels in w_i / sum(w), so decay
        # cannot change the sampling law; it only rescales the reported
        # absolute weight. That is also why the blowup guard below must look
        # at the RELATIVE weights: guarding on decay * max_relative let the
        # shrinking factor hide relative growth until it overflowed to inf
        # (NaN probabilities at pull 691,465 with K=20, window_decay=0.999).
        self.weights: dict[str, float] = {}
        self._log_decay: float = 0.0
        # Largest relative weight — only changes on the recorded arm, so
        # the blowup check below stays O(1). Non-decreasing until renorm.
        self._max_relative: float = 1.0
        self._total_pulls: int = 0
        # Selection law of the last select_op(), read by record() for the
        # importance-weighted estimator. Held as a snapshot of the tree that
        # drew it (see _last_probs below) so a draw costs O(log K) rather
        # than an O(K) dict build; an explicitly assigned dict still wins.
        self._probs_dict: dict[str, float] | None = {}
        self._snapshot: tuple | None = None
        # Fenwick sampling state: tuple(ops) -> [index, leaf, tree, epoch,
        # log_pos, capacity]. Trees follow self.weights lazily through
        # _update_log (arm names whose weight changed); _epoch bumps on
        # renormalisation or log overflow and forces a rebuild.
        self._trees: dict[tuple, list] = {}
        self._update_log: list[str] = []
        self._epoch: int = 0

    def init_arm(self, name: str) -> None:
        """Register an operator with initial weight 1.0."""
        if name not in self.weights:
            self.weights[name] = 1.0
            if self._max_relative < 1.0:
                self._max_relative = 1.0

    @property
    def _last_probs(self) -> dict[str, float]:
        """The last selection law as a dict, materialised on demand (O(K))."""
        if self._probs_dict is None:
            ops, index, leaf, scale, uniform = self._snapshot
            self._probs_dict = {op: scale * leaf[index[op]] + uniform for op in ops}
        return self._probs_dict

    @_last_probs.setter
    def _last_probs(self, probs: dict[str, float]) -> None:
        self._probs_dict = probs
        self._snapshot = None

    def _prob_of(self, name: str) -> float:
        """Selection probability *name* had in the last draw, O(1)."""
        snap = self._snapshot
        if snap is not None and self._probs_dict is None:
            ops, index, leaf, scale, uniform = snap
            i = index.get(name)
            if i is None:
                return 1.0 / max(len(ops), 1)
            return scale * leaf[i] + uniform
        probs = self._probs_dict
        return probs.get(name, 1.0 / max(len(probs), 1))

    def last_selection_probs(self) -> dict[str, float]:
        """The mixture this scheduler last sampled from, normalised to 1.

        Exposed for the work functional in ``core/fluctuation.py``, whose
        Rényi identity needs the actual law the trajectory was drawn from.
        EXP3 is the scheduler that can answer: it already retains this
        mixture for its own importance-weighted estimator. Deterministic
        argmax schedulers have no such law and must not synthesise one.
        """
        return dict(self._last_probs)

    def select_op(self, ops: list[str]) -> str:
        """Select operator via EXP3 mixture distribution.

        p_i = (1-γ)·w_i/Σw + γ/K is sampled by descending a Fenwick tree over
        the offered operators: O(log K) per draw plus O(log K) per weight
        changed since this list was last offered. The prefix mass of the
        first c arms is (1-γ)·S_c/Σw + γ·c/K, so the uniform term needs no
        tree of its own. Same law as the linear roulette walk this replaced;
        not bit-identical, because the tree adds weights in a different order.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        entry = self._tree_for(ops)
        if entry is None:
            return self._select_linear(ops)
        index, leaf, tree, _epoch, _pos, cap = entry
        n = len(ops)
        total_w = 0.0
        i = n
        while i > 0:
            total_w += tree[i]
            i -= i & -i
        if not (0.0 < total_w < math.inf):
            return self._select_linear(ops)

        scale = (1.0 - self.gamma) / total_w
        uniform = self.gamma / n
        r = self._rng.random()
        # Largest prefix whose mass is still below r; the arm after it is the
        # first whose cumulative mass reaches r (the walk's `r <= cumulative`).
        pos = 0
        acc = 0.0
        bit = cap
        while bit:
            nxt = pos + bit
            if nxt <= n and scale * (acc + tree[nxt]) + uniform * nxt < r:
                pos = nxt
                acc += tree[nxt]
            bit >>= 1
        if pos >= n:
            pos = n - 1
        self._snapshot = (ops, index, leaf, scale, uniform)
        self._probs_dict = None
        return ops[pos]

    def _select_linear(self, ops: list[str]) -> str:
        """Roulette walk over a dict law: duplicate names or a degenerate total."""
        K = len(ops)
        total_w = sum(self.weights.get(op, 1.0) for op in ops)
        if total_w <= 0 or not math.isfinite(total_w):
            self._last_probs = {}
            return self._rng.choice(ops)
        probs: dict[str, float] = {}
        for op in ops:
            w = self.weights.get(op, 1.0)
            probs[op] = (1.0 - self.gamma) * (w / total_w) + self.gamma / K
        self._last_probs = probs
        r = self._rng.random()
        cumulative = 0.0
        for op in ops:
            cumulative += probs[op]
            if r <= cumulative:
                return op
        return ops[-1]

    def _tree_for(self, ops: list[str]) -> list | None:
        """Fenwick tree over *ops*' weights, built or brought up to date."""
        key = tuple(ops)
        entry = self._trees.get(key)
        if entry is None or entry[3] != self._epoch:
            entry = self._build_tree(ops)
            if entry is None:
                return None
            if key not in self._trees and len(self._trees) >= _TREE_CACHE_MAX:
                del self._trees[next(iter(self._trees))]
            self._trees[key] = entry
            return entry
        log = self._update_log
        pos = entry[4]
        if pos < len(log):
            index, leaf, tree, _epoch, _pos, cap = entry
            weights = self.weights
            for name in log[pos:]:
                i = index.get(name)
                if i is None:
                    continue
                delta = weights[name] - leaf[i]
                if delta:
                    leaf[i] += delta
                    j = i + 1
                    while j <= cap:
                        tree[j] += delta
                        j += j & -j
            entry[4] = len(log)
        return entry

    def _build_tree(self, ops: list[str]) -> list | None:
        """O(K) Fenwick build; None when *ops* repeats a name."""
        n = len(ops)
        index = {op: i for i, op in enumerate(ops)}
        if len(index) != n:
            return None
        cap = 1 << (n - 1).bit_length()
        weights = self.weights
        leaf = [weights.get(op, 1.0) for op in ops]
        tree = [0.0] * (cap + 1)
        tree[1 : n + 1] = leaf
        for i in range(1, cap + 1):
            j = i + (i & -i)
            if j <= cap:
                tree[j] += tree[i]
        return [index, leaf, tree, self._epoch, len(self._update_log), cap]

    def _invalidate_trees(self) -> None:
        """Force every tree to rebuild; call after writing self.weights directly."""
        self._epoch += 1
        self._update_log.clear()

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update EXP3 weights.

        Uses the importance-weighted estimator: reward_estimate = r / p_i,
        where p_i is the probability this operator had when it was selected.

        Exponential decay is folded into ``_log_decay`` (one add per call)
        instead of multiplying every arm's weight; per-arm values in
        ``self.weights`` are relative to ``exp(_log_decay)``.
        """
        self._total_pulls += 1
        reward = weight if success else 0.0

        # Exponential decay, folded into the log-space scalar (see __init__).
        if self.window_decay < 1.0:
            self._log_decay += self._log_window_decay

        # EXP3 weight update: w_i *= exp(gamma * r̂_i / K)  (relative space)
        # r̂_i = reward / p_i  (importance-weighted). An arm record() has not
        # seen starts at relative 1.0, the value init_arm() gives it and the
        # default select_op() reads -- the old 1/decay default disagreed with
        # both and grew without bound as the factor shrank.
        p = self._prob_of(name)
        K = max(len(self.weights), 1)
        estimated_reward = reward / max(p, 1e-9)
        step = min(self.gamma * estimated_reward / max(K, 1), _MAX_LOG_STEP)
        relative = self.weights.get(name, 1.0) * math.exp(step)
        self.weights[name] = relative
        if step:
            if len(self._update_log) >= _UPDATE_LOG_MAX:
                self._invalidate_trees()
            else:
                self._update_log.append(name)
        if relative > self._max_relative:
            self._max_relative = relative

        # Prevent floating-point blowup: renormalize relative weights so the
        # largest is 1.0. The floor keeps an arm that fell below the double
        # range revivable instead of pinned at exactly 0 forever.
        if self._max_relative > _RENORM_THRESHOLD:
            scale = 1.0 / self._max_relative
            for k in self.weights:
                self.weights[k] = max(self.weights[k] * scale, _WEIGHT_FLOOR)
            self._max_relative = 1.0
            self._invalidate_trees()

    def bandit_stats(self) -> dict:
        """Return EXP3 diagnostics."""
        return {
            "exp3_pulls": self._total_pulls,
            "exp3_max_weight": (self._actual_max_weight() if self.weights else 0.0),
        }

    def _actual_max_weight(self) -> float:
        """Largest decay-adjusted weight, computed in log space (inf past range)."""
        log_w = math.log(self._max_relative) + self._log_decay
        return math.exp(log_w) if log_w < 709.0 else math.inf
