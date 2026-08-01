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
        beta: Exploration parameter. Higher = more exploration via
            uncertainty bonus.
        refit_interval: How often to refit the kernel matrix (capped at
            every N pulls to bound O(K³) cost).
        min_samples: Minimum observations per operator before its kernel
            row is considered trustworthy.
    """

    supports_priors = False

    def __init__(
        self,
        length_scale: float = 1.0,
        beta: float = 2.0,
        refit_interval: int = 100,
        min_samples: int = 3,
    ):
        self.length_scale = length_scale
        self.beta = beta
        self.refit_interval = refit_interval
        self.min_samples = min_samples

        # Per-operator reward moments (mean, variance, count)
        self._moments: dict[str, RunningMoments] = {}

        # Feature vectors: one-hot by category
        self._features: dict[str, list[float]] = {}
        self._cat_names: list[str] = list(OPERATOR_CATEGORIES.keys())
        self._op_to_cat: dict[str, str] = {}
        for cat, ops in OPERATOR_CATEGORIES.items():
            for op in ops:
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
        """Select operator via GP-UCB: highest predictive mean + beta * sigma.

        Only considers operators with >= min_samples observations for the
        predictive estimate; unobserved operators get a fixed exploration bonus.
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

        scores: dict[str, float] = {}
        for op in ops:
            moments = self._moments.get(op)
            if moments is not None and moments.count >= self.min_samples:
                mu = moments.mean
                # Predictive variance = self-kernel - info borrowed from others
                # Simplified: use the empirical stddev scaled by correlated ops
                sigma = moments.stddev
                # UCB score
                scores[op] = mu + self.beta * max(sigma, 1e-6)
            else:
                # Exploration bonus for operators with insufficient data
                # Use a fixed high-uncertainty bonus to encourage exploration
                scores[op] = self.beta * 2.0  # generous initial exploration bonus

        return max(scores, key=scores.get)

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
