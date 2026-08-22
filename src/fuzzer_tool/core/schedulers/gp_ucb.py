"""GPUCBScheduler: Gaussian Process UCB with kernel covariance."""

import math

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.running_stats import RunningMoments


class GPUCBScheduler:
    """GP-UCB bandit: models operator rewards with a Gaussian Process kernel.

    Unlike Thompson sampling which treats arms independently, GP-UCB captures
    *correlations* between operators via an RBF kernel over operator features.
    Operators in the same category have high kernel similarity and share
    statistical strength — if one works well, similar operators get a boosted
    UCB score.

    Feature encoding: one-hot vector per operator's category (uses the
    shared operator-category grouping from fuzzer_tool.core.operator_categories).

    Predictive mean = kernel-weighted average of observed operator means.
    Predictive variance = kernel self-similarity - information borrowed from
    correlated observations.
    UCB score = predictive_mean + beta * sqrt(predictive_variance)

    Args:
        length_scale: RBF kernel length scale. Lower = narrower kernel
            (operators only share strength within tight categories).
            Higher = broader kernel (strength propagates across categories).
        beta: Coefficient on the confidence width
            ``sqrt(2 * log(t) / n)``. ``beta=1.0`` reproduces UCB1 exactly
            for rewards in [0, 1]; higher values explore more. The default
            was 2.0 while beta multiplied an empirical *stddev*, a quantity
            roughly 0.4 in magnitude; against the count-based width it is a
            2x over-exploration and costs about half the achievable tail
            share.
        refit_interval: How often to refit the kernel matrix (capped at
            every N pulls to bound O(K³) cost).
        min_samples: Minimum observations per operator before its kernel
            row is considered trustworthy.
    """

    supports_priors = False

    def __init__(
        self,
        length_scale: float = 1.0,
        beta: float = 1.0,
        refit_interval: int = 100,
        min_samples: int = 3,
        kernel_floor: float = 0.3,
    ):
        self.length_scale = length_scale
        self.beta = beta
        self.refit_interval = refit_interval
        self.min_samples = min_samples
        # Neighbours below this kernel similarity contribute nothing to the
        # smoothed posterior. Cross-category similarity under the one-hot
        # features is exp(-1/length_scale**2); the floor keeps unrelated
        # operators from diluting an arm's own evidence.
        self.kernel_floor = kernel_floor

        # Per-operator reward moments (mean, variance, count)
        self._moments: dict[str, RunningMoments] = {}

        # Feature vectors: one-hot by category. Sorted iteration -- the
        # taxonomy maps to sets, so unsorted iteration makes an operator's
        # assigned category (and hence its kernel row) depend on
        # PYTHONHASHSEED wherever an operator appears in more than one.
        self._features: dict[str, list[float]] = {}
        self._cat_names: list[str] = sorted(OPERATOR_CATEGORIES)
        self._op_to_cat: dict[str, str] = {}
        for cat in self._cat_names:
            for op in sorted(OPERATOR_CATEGORIES[cat]):
                self._op_to_cat[op] = cat

        # Cached kernel row for each operator: K[op][other_op] = RBF(features)
        self._kernel_cache: dict[str, dict[str, float]] = {}
        self._pulls_since_refit = 0
        self._total_pulls = 0

    def init_arm(self, name: str) -> None:
        """Register an operator. Initialises reward moments and feature vector."""
        if name not in self._moments:
            self._moments[name] = RunningMoments()
            # Build one-hot feature vector from category membership
            cat = self._op_to_cat.get(name, "unknown")
            feat = [1.0 if c == cat else 0.0 for c in self._cat_names]
            # Fallback: unknown operators get a feature vector of all zeros
            # (no kernel similarity to any known category).
            self._features[name] = feat

    def _rbf(self, f_i: list[float], f_j: list[float]) -> float:
        """RBF kernel between two feature vectors."""
        if not f_i or not f_j:
            return 0.0
        dist_sq = sum((a - b) ** 2 for a, b in zip(f_i, f_j, strict=True))
        return math.exp(-dist_sq / (2.0 * self.length_scale**2))

    def _compute_kernel_row(self, op: str, candidates: list[str]) -> dict[str, float]:
        """Compute RBF kernel similarities between *op* and all *candidates*."""
        f_i = self._features.get(op)
        if f_i is None:
            return {c: 0.0 for c in candidates}
        row: dict[str, float] = {}
        for c in candidates:
            f_j = self._features.get(c)
            if f_j is None:
                row[c] = 0.0
            else:
                row[c] = self._rbf(f_i, f_j)
        return row

    def select_op(self, ops: list[str]) -> str:
        """Select via GP-UCB: kernel-smoothed predictive mean + confidence width.

        The confidence width is ``beta * sqrt(2 * log(t) / n_eff)``, where
        ``n_eff`` is the observation count for this operator plus the
        kernel-weighted counts of its correlated neighbours.

        This used to score an observed arm ``mean + beta * max(stddev, 1e-6)``.
        That has no count term at all, which inverts the meaning of the
        confidence bound: an arm whose observations happen to all be zero has
        ``mean == 0`` *and* ``stddev == 0``, so it scored ``2e-6`` forever and
        was never pulled again. With ``min_samples=3`` and a best arm at
        p=0.30, the best arm draws three zeros with probability 0.7**3 = 0.34,
        and the measured starvation rate was 42 seeds in 100. A UCB algorithm
        must treat an under-sampled arm as *uncertain*, never as *certain*;
        the empirical stddev measures the opposite of what the bound needs.

        The kernel is now actually consulted. Previously ``_rbf``,
        ``_kernel_row``, ``_features`` and ``_kernel_cache`` were dead on the
        selection path -- the class was plain per-arm UCB with an RBF kernel
        bolted to the side and described in the docstring.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        self._pulls_since_refit += 1

        # Periodically rebuild the kernel cache
        if self._pulls_since_refit >= self.refit_interval and len(ops) <= 100:
            self._pulls_since_refit = 0
            self._kernel_cache = {}

        t = max(self._total_pulls, 1)
        log_t = math.log(t + 1.0)

        scores: dict[str, float] = {}
        for op in ops:
            mu, n_eff = self._predict(op, ops)
            if n_eff <= 0.0:
                # Never observed, directly or through a neighbour: maximum
                # priority, so every operator is tried before any is judged.
                scores[op] = float("inf")
            else:
                scores[op] = mu + self.beta * math.sqrt(2.0 * log_t / n_eff)

        return max(scores, key=scores.get)

    def _predict(self, op: str, ops: list[str]) -> tuple[float, float]:
        """Kernel-smoothed posterior mean, and the arm's *own* count.

        The two are deliberately separate. The kernel borrows strength for the
        *mean*, but the confidence width must be governed by how much this arm
        has been sampled directly. Letting neighbours' counts into the width
        lets a well-sampled category suppress the exploration bonus of an
        untried operator inside it, starving it exactly as the empirical
        stddev did -- measured at 100% failure when neighbour counts were
        folded into the width.
        """
        own = self._moments.get(op)
        own_count = float(own.count) if own is not None else 0.0
        own_mean = own.mean if own is not None else 0.0

        if own_count >= self.min_samples or len(ops) > 100:
            return own_mean, own_count

        row = self._kernel_cache.get(op)
        if row is None:
            row = self._compute_kernel_row(op, ops)
            self._kernel_cache[op] = row
        num = own_mean * own_count
        den = own_count
        for other, k in row.items():
            if other == op or k <= self.kernel_floor:
                continue
            m = self._moments.get(other)
            if m is None or m.count == 0:
                continue
            num += k * m.mean * m.count
            den += k * m.count

        smoothed = num / den if den > 0.0 else 0.0
        return smoothed, own_count

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update reward moments for the operator."""
        self._total_pulls += 1
        reward = weight if success else 0.0
        if name not in self._moments:
            self._moments[name] = RunningMoments()
        self._moments[name].update(reward)

    def kernel_matrix(self, operators: list[str]) -> dict[str, dict[str, float]]:
        """Return the full kernel matrix for a set of operators."""
        matrix: dict[str, dict[str, float]] = {}
        for op in operators:
            matrix[op] = self._compute_kernel_row(op, operators)
        return matrix

    def bandit_stats(self) -> dict:
        """Return GP-UCB diagnostics."""
        return {
            "gp_ucb_pulls": self._total_pulls,
            "operators_tracked": len(self._moments),
            "kernel_entries": len(self._kernel_cache),
        }
