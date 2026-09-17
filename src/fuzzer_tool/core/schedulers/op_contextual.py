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

        # Arm state lives in two contiguous backing arrays -- (n, dim, dim)
        # for A_inv and (n, dim) for b -- so select_op can slice a batch
        # without restaging it. The dicts below hold *views* into those
        # arrays, keeping score()/record()'s per-arm access (and their
        # in-place updates) working unchanged.
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        self._cap = 0
        self._A_inv_arr = np.empty((0, dim, dim), dtype=float)
        self._b_arr = np.empty((0, dim), dtype=float)

        self._A_inv: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._pulls: dict[str, int] = {}
        self._total_pulls = 0

        # select_op candidate-list memo (see _resolve_idx)
        self._sel_ops: tuple[str, ...] | None = None
        self._sel_idx: np.ndarray | None = None
        self._sel_contig = False
        # rank-1 scratch for record()'s Sherman-Morrison update
        self._outer_buf = np.empty((dim, dim), dtype=float)

    def _grow(self, need: int) -> None:
        """Ensure capacity for *need* arms, reallocating geometrically.

        Reallocation invalidates every outstanding view, so the view dicts
        are rebuilt here. This is amortized O(1) per arm and in practice
        runs a handful of times during warmup and never again.
        """
        if need <= self._cap:
            return
        newcap = max(16, self._cap * 2)
        while newcap < need:
            newcap *= 2

        new_a = np.empty((newcap, self.dim, self.dim), dtype=float)
        new_b = np.zeros((newcap, self.dim), dtype=float)
        n = len(self._names)
        if n:
            new_a[:n] = self._A_inv_arr[:n]
            new_b[:n] = self._b_arr[:n]
        self._A_inv_arr = new_a
        self._b_arr = new_b
        self._cap = newcap

        for i, nm in enumerate(self._names):
            self._A_inv[nm] = self._A_inv_arr[i]
            self._b[nm] = self._b_arr[i]
        # Views are stale for the memoized batch too.
        self._sel_ops = None

    def init_arm(self, name: str) -> None:
        """Register an operator: A_inv <- (1/lambda_reg) I, b <- 0."""
        if name in self._index:
            return
        i = len(self._names)
        self._grow(i + 1)
        self._names.append(name)
        self._index[name] = i

        self._A_inv_arr[i] = np.eye(self.dim) * (1.0 / self.lambda_reg)
        self._b_arr[i] = 0.0
        self._A_inv[name] = self._A_inv_arr[i]
        self._b[name] = self._b_arr[i]
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
        A_stack, b_stack = self._resolve_batch(ops)

        ainv_x = np.einsum("nij,nj->ni", A_stack, x_arr)
        mean = np.einsum("nd,nd->n", b_stack, ainv_x)
        variance = np.einsum("nd,nd->n", x_arr, ainv_x)
        scores = mean + self.alpha * np.sqrt(variance)
        best_idx = int(np.argmax(scores))
        return ops[best_idx]

    def _resolve_batch(self, ops: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (A_inv, b) batches for *ops* without restaging per-arm state.

        The candidate list is near-constant across calls, so the resolved
        row indices are memoized. When the candidates are exactly the first
        ``len(ops)`` arms in registration order -- the overwhelmingly common
        case, since arms are registered in candidate order -- the batches are
        plain slice *views* and cost no allocation at all. Otherwise a single
        fancy-index gather replaces what used to be two ``np.stack`` calls
        over ``len(ops)`` separate arrays.
        """
        n = len(ops)
        ops_t = tuple(ops)
        if ops_t != self._sel_ops:
            idx = np.fromiter((self._index[op] for op in ops), dtype=np.intp, count=n)
            self._sel_ops = ops_t
            self._sel_idx = idx
            self._sel_contig = bool(n <= len(self._names) and np.array_equal(idx, np.arange(n)))

        if self._sel_contig:
            return self._A_inv_arr[:n], self._b_arr[:n]

        # Subset/reordered candidates need a real gather. Plain fancy
        # indexing is used deliberately: np.take(out=) into a reused scratch
        # buffer was measured at ~2x the wall time for identical allocation
        # volume (its wrapper allocates regardless), so the buffer bought
        # nothing. This is one allocation per call, down from the 2*len(ops)
        # that the previous np.stack restaging cost.
        return self._A_inv_arr[self._sel_idx], self._b_arr[self._sel_idx]

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
        # np.outer allocates a (dim, dim) temporary on every exec; write the
        # rank-1 term into reusable scratch instead.
        np.multiply(ainv_x[:, None], ainv_x[None, :], out=self._outer_buf)
        self._outer_buf /= denom
        a_inv -= self._outer_buf

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
