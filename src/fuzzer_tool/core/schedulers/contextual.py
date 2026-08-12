"""ContextualLinUCBScheduler: per-arm ridge regression over seed features.

Every other scheduler in this package (Thompson/MC, MOpt-PSO, Replicator,
Exp3, epsilon-greedy, hierarchical, GP-UCB) learns a single global ranking
over operators. None of them condition the ranking on the seed being
mutated, even though the best operator for a 40-byte PNG header seed is
not the best for a 60KB IDAT-heavy seed.

The codebase already concedes the context signal is real -- it just
hardcodes it instead of learning it (``_FORMAT_SNIFFERS`` gating,
``format_operator_priors()``, the ``format_lock`` operator). This
scheduler learns the same kind of rule from data instead: LinUCB
(Li et al. 2010) with one independent ridge regressor per arm.

Per arm a:
    A_a <- A_a + x xT           (design matrix, starts at lambda_reg * I)
    b_a <- b_a + r * x          (reward-weighted feature sum)
    theta_a = A_a^-1 b_a
    select argmax_a  theta_a . x + alpha * sqrt(x^T A_a^-1 x)

The inverse is maintained directly via the Sherman-Morrison rank-1 update
instead of being recomputed, so a pull costs O(d^2) with no matrix
inversion on the hot path:

    A_inv <- A_inv - (A_inv x)(A_inv x)^T / (1 + x^T A_inv x)

With d ~= 14 and ~106 operators that's a few hundred KB of state total.
"""

import math

import numpy as np


class ContextualLinUCBScheduler:
    """LinUCB contextual bandit: one ridge regressor per operator.

    Args:
        dim: Dimensionality of the context feature vector.
        alpha: Exploration weight on the confidence-bound term. Higher =
            more exploration of arms with uncertain context coverage.
        lambda_reg: Ridge regularization; also the initial diagonal of
            A_inv is ``1 / lambda_reg``. Higher = stronger shrinkage of
            theta toward zero for under-observed arms.
    """

    supports_priors = False

    def __init__(self, dim: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self._A_inv: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._pulls: dict[str, int] = {}
        self._total_pulls = 0

    def init_arm(self, name: str) -> None:
        """Register an operator: A_inv <- (1/lambda_reg) I, b <- 0."""
        if name not in self._A_inv:
            inv_lambda = 1.0 / self.lambda_reg
            self._A_inv[name] = np.eye(self.dim) * inv_lambda
            self._b[name] = np.zeros(self.dim)
            self._pulls[name] = 0

    def _matvec(self, matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
        return matrix @ vec

    def score(self, name: str, x: list[float] | np.ndarray) -> float:
        """UCB score for one arm given its context vector: theta.x + alpha*sqrt(x^T A^-1 x)."""
        self.init_arm(name)
        a_inv = self._A_inv[name]
        b = self._b[name]
        x_arr = np.asarray(x, dtype=float)
        ainv_x = a_inv @ x_arr
        # theta.x == (A_inv b).x == b.(A_inv x) since A_inv is symmetric --
        # avoids materializing theta separately.
        mean = b @ ainv_x
        variance = x_arr @ ainv_x
        conf = math.sqrt(variance) if variance > 0.0 else 0.0
        return mean + self.alpha * conf

    def select_op(self, ops: list[str], context) -> str:
        """Select the operator with the highest LinUCB score.

        Args:
            ops: Candidate operator names.
            context: Either a single feature vector shared by all arms, or
                a callable ``op -> list[float]`` for per-arm context (used
                here to append each operator's own cost feature to a
                shared seed-context prefix).
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        for op in ops:
            self.init_arm(op)

        xs = [context(op) for op in ops] if callable(context) else [context] * len(ops)

        x_arr = np.asarray(xs, dtype=float)
        A_stack = np.stack([self._A_inv[op] for op in ops])
        b_stack = np.stack([self._b[op] for op in ops])

        ainv_x = np.einsum("nij,nj->ni", A_stack, x_arr)
        mean = np.einsum("nd,nd->n", b_stack, ainv_x)
        variance = np.einsum("nd,nd->n", x_arr, ainv_x)
        scores = mean + self.alpha * np.sqrt(variance)
        best_idx = int(np.argmax(scores))
        return ops[best_idx]

    def record(self, name: str, x: list[float] | np.ndarray, reward: float) -> None:
        """Update arm *name*'s ridge regressor with observed (x, reward).

        Uses the Sherman-Morrison identity so no matrix is ever inverted:
        A_inv <- A_inv - (A_inv x)(A_inv x)^T / (1 + x^T A_inv x)
        """
        self.init_arm(name)
        a_inv = self._A_inv[name]
        x_arr = np.asarray(x, dtype=float)
        ainv_x = a_inv @ x_arr
        denom = 1.0 + x_arr @ ainv_x
        if denom <= 0.0:
            # Should not happen (A_inv is PSD), but guard against numerical
            # drift over a long run rather than injecting a NaN into state.
            denom = 1e-9
        a_inv -= np.outer(ainv_x, ainv_x) / denom

        self._b[name] += reward * x_arr

        self._pulls[name] = self._pulls.get(name, 0) + 1
        self._total_pulls += 1

    def bandit_stats(self) -> dict:
        """Return LinUCB diagnostics."""
        return {
            "contextual_pulls": self._total_pulls,
            "operators_tracked": len(self._A_inv),
            "dim": self.dim,
        }
