"""BOGPUCBScheduler: Bayesian Optimization with Expected Improvement (EI) acquisition.

Full Bayesian Optimization loop:
- Probabilistic model: Gaussian Process with RBF kernel over operator-category one-hot features
- Acquisition: Expected Improvement for maximization (maximize new coverage)
- Noisy observations supported via observation noise parameter
- Posterior updated via Cholesky decomposition (O(n^3) where n = num operators)

Uses the same one-hot category features as GPUCBScheduler for direct compatibility.
"""

import math

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, category_of
from fuzzer_tool.core.running_stats import RunningMoments


def _Phi(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


class BOGPUCBScheduler:
    """Bayesian Optimization GP-UCB with Expected Improvement acquisition.

    Unlike standard GP-UCB which uses UCB as acquisition, BO uses Expected
    Improvement (EI) which directly optimizes the expected gain over the
    current best observation. Supports noisy observations.

    Feature encoding: one-hot vector per operator's category (shared with
    GPUCBScheduler for consistency).

    Posterior:
        mu = K @ (K + sigma_noise^2 * I)^-1 @ y
        var = K_ii - K_i @ (K + sigma_noise^2 * I)^-1 @ K_i

    EI acquisition:
        EI(x) = (mu(x) - f_max) * Phi(z) + sigma(x) * phi(z)
        z = (mu(x) - f_max) / sigma(x)

    Args:
        length_scale: RBF kernel length scale
        noise: Observation noise standard deviation (sqrt of noise variance)
        refit_interval: How often to recompute the posterior (O(n^3) cost)
    """

    # Declare priors support so _register_arms passes informative Beta priors
    supports_priors = True

    def __init__(
        self,
        length_scale: float = 1.0,
        noise: float = 0.01,
        refit_interval: int = 100,
    ):
        self.length_scale = length_scale
        self.noise = noise  # observation noise sigma
        self.refit_interval = refit_interval

        # Per-operator reward moments (mean, variance, count)
        self._moments: dict[str, RunningMoments] = {}

        # Feature vectors: one-hot by category
        self._features: dict[str, list[float]] = {}
        self._cat_names: list[str] = sorted(OPERATOR_CATEGORIES)
        self._op_to_cat: dict[str, str] = {}
        for cat in self._cat_names:
            for op in sorted(OPERATOR_CATEGORIES[cat]):
                self._op_to_cat[op] = cat

        # Posterior cache
        self._kernel_matrix: list[list[float]] | None = None
        self._inv_K: list[list[float]] | None = None
        self._y: list[float] = []
        self._op_list: list[str] = []
        self._needs_refit = True
        self._pulls_since_refit = 0
        self._total_pulls = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an operator. Initialises reward moments and feature vector.

        Args:
            name: Name of the mutation operator.
            prior_alpha: Prior alpha (successes + 1). Must be > 0.
            prior_beta: Prior beta (failures + 1). Must be > 0.
        """
        if name not in self._moments:
            # Initialize with prior pseudo-counts
            self._moments[name] = RunningMoments()
            # Add prior successes/failures as pseudo-observations
            if prior_alpha > 1.0:
                self._moments[name].update(1.0)  # prior successes
                self._moments[name].update(1.0)  # prior successes
            if prior_beta > 1.0:
                self._moments[name].update(0.0)  # prior failures

            # Build one-hot feature vector
            cat = self._op_to_cat.get(name)
            if cat is None:
                cat = category_of(name)
                self._op_to_cat[name] = cat
            self._ensure_category(cat)
            self._features[name] = [1.0 if c == cat else 0.0 for c in self._cat_names]
            self._needs_refit = True

    def _ensure_category(self, cat: str) -> None:
        """Add *cat* to the one-hot basis, padding existing feature vectors."""
        if cat in self._cat_names:
            return
        self._cat_names.append(cat)
        for feat in self._features.values():
            feat.append(0.0)
        self._needs_refit = True

    @property
    def _cross_category_similarity(self) -> float:
        """Off-diagonal RBF value for one-hot features: exp(-1/ls^2)."""
        return math.exp(-1.0 / (self.length_scale**2))

    def _rbf(self, f_i: list[float], f_j: list[float]) -> float:
        """RBF kernel between two one-hot feature vectors."""
        if not f_i or not f_j:
            return 0.0
        # One-hot: distance is 0 (same) or sqrt(2) (different) -> squared = 0 or 2
        dist_sq = sum((a - b) ** 2 for a, b in zip(f_i, f_j, strict=True))
        return math.exp(-dist_sq / (2.0 * self.length_scale**2))

    def _build_kernel_matrix(self, ops: list[str]) -> list[list[float]]:
        """Build full kernel matrix K for given operators."""
        n = len(ops)
        K = [[0.0] * n for _ in range(n)]
        for i, op_i in enumerate(ops):
            fi = self._features.get(op_i)
            for j, op_j in enumerate(ops):
                if i == j:
                    K[i][j] = 1.0  # diagonal is always 1 for RBF
                elif i < j:
                    fj = self._features.get(op_j)
                    val = self._rbf(fi, fj) if fi and fj else 0.0
                    K[i][j] = val
                    K[j][i] = val
        return K

    def _cholesky(self, A: list[list[float]]) -> list[list[float]]:
        """Cholesky decomposition: A = L @ L^T, L lower triangular."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-12))
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    def _solve_triangular(self, L: list[list[float]], b: list[float]) -> list[float]:
        """Solve L @ x = b for lower-triangular L (forward substitution)."""
        n = len(L)
        x = [0.0] * n
        for i in range(n):
            s = sum(L[i][k] * x[k] for k in range(i))
            x[i] = (b[i] - s) / L[i][i]
        return x

    def _solve_triangular_T(self, L: list[list[float]], b: list[float]) -> list[float]:
        """Solve L^T @ x = b for lower-triangular L (backward substitution)."""
        n = len(L)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / L[i][i]
        return x

    def _solve_Ky(self, K: list[list[float]], y: list[float]) -> list[float]:
        """Solve (K + noise^2 I) @ alpha = y via Cholesky."""
        n = len(K)
        # Add noise to diagonal
        for i in range(n):
            K[i][i] += self.noise * self.noise
        L = self._cholesky(K)
        alpha = self._solve_triangular(L, y)
        alpha = self._solve_triangular_T(L, alpha)
        return alpha

    def _update_posterior(self) -> None:
        """Compute GP posterior from current observations."""
        # Get all initialized operators
        self._op_list = [
            op for op in self._features if op in self._moments and self._moments[op].count > 0
        ]
        if not self._op_list:
            self._kernel_matrix = None
            self._inv_K = None
            self._y = []
            self._needs_refit = False
            return

        n = len(self._op_list)
        K = self._build_kernel_matrix(self._op_list)
        self._y = [self._moments[op].mean for op in self._op_list]

        # Solve (K + sigma_n^2 I) alpha = y
        # Keep a copy of K with noise for posterior variance computation
        K_noisy = [row[:] for row in K]
        for i in range(n):
            K_noisy[i][i] += self.noise * self.noise
        L = self._cholesky(K_noisy)
        alpha = self._solve_triangular(L, self._y)
        alpha = self._solve_triangular_T(L, alpha)

        # Compute inverse kernel for posterior variance: K^{-1} = L^{-T} @ L^{-1}
        # Actually we need diag(K - K @ K_noisy^{-1} @ K)
        # Compute v = K_noisy^{-1} @ K via Cholesky
        v = []
        for j in range(n):
            k_j = [K[i][j] for i in range(n)]
            v_j = self._solve_triangular(L, k_j)
            v_j = self._solve_triangular_T(L, v_j)
            v.append(v_j)

        # Posterior variance: diag(K - K @ v)
        self._post_var = []
        for i in range(n):
            var = K[i][i] - sum(K[i][j] * v[j][i] for j in range(n))
            self._post_var.append(max(var, 1e-12))

        self._kernel_matrix = K
        self._alpha = alpha
        self._needs_refit = False

    def _predict_mean(self, op: str) -> float:
        """Predict posterior mean for operator."""
        if op not in self._moments or self._moments[op].count == 0:
            return 0.0
        if self._needs_refit or self._kernel_matrix is None:
            self._update_posterior()
        if not self._op_list:
            return self._moments[op].mean
        try:
            idx = self._op_list.index(op)
            # mean = sum_j alpha_j * K(op, op_j)
            mu = sum(
                self._alpha[j] * self._kernel_matrix[idx][j] for j in range(len(self._op_list))
            )
            return mu
        except ValueError:
            return self._moments[op].mean

    def _posterior_variance(self, op: str) -> float:
        """Predict posterior variance for operator."""
        if op not in self._moments or self._moments[op].count == 0:
            return 1.0  # maximum uncertainty for unobserved
        if self._needs_refit or self._kernel_matrix is None:
            self._update_posterior()
        if not self._op_list:
            return 1.0
        try:
            idx = self._op_list.index(op)
            return self._post_var[idx]
        except ValueError:
            return 1.0

    def _best_mean(self) -> float:
        """Best observed mean reward so far (f_max for EI)."""
        if not self._moments:
            return 0.0
        return max((m.mean for m in self._moments.values()), default=0.0)

    def _expected_improvement(self, op: str) -> float:
        """Compute Expected Improvement for operator."""
        mu = self._predict_mean(op)
        sigma = math.sqrt(self._posterior_variance(op))
        f_max = self._best_mean()

        if sigma <= 0.0:
            return max(mu - f_max, 0.0)

        z = (mu - f_max) / sigma
        return (mu - f_max) * _Phi(z) + sigma * _phi(z)

    def select_op(self, ops: list[str]) -> str:
        """Select operator via Expected Improvement acquisition."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        self._pulls_since_refit += 1
        if self._pulls_since_refit >= self.refit_interval:
            self._pulls_since_refit = 0
            self._needs_refit = True

        # Compute EI for each candidate
        scores = {}
        for op in ops:
            scores[op] = self._expected_improvement(op)

        # Max EI (break ties by name for determinism)
        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update GP posterior."""
        self._total_pulls += 1
        reward = weight if success else 0.0
        if name not in self._moments:
            self._moments[name] = RunningMoments()
        self._moments[name].update(reward)
        self._needs_refit = True

    def bandit_stats(self) -> dict:
        """Return BO-GP-UCB diagnostics."""
        return {
            "bo_gp_ucb_pulls": self._total_pulls,
            "operators_tracked": len(self._moments),
            "posterior_dim": len(self._op_list) if self._op_list else 0,
            "noise": self.noise,
            "length_scale": self.length_scale,
        }
