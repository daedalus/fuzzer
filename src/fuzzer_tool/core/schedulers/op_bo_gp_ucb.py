"""BOGPUCBScheduler: Bayesian Optimization with Expected Improvement (EI) acquisition.

Full Bayesian Optimization loop:
- Probabilistic model: Gaussian Process with RBF kernel over operator-category one-hot features
- Acquisition: Expected Improvement for maximization (maximize new coverage)
- Noisy observations supported via observation noise parameter
- Posterior updated via Cholesky decomposition (O(n^3) where n = num operators)

Uses the same one-hot category features as GPUCBScheduler for direct compatibility.

Performance note (fixed 2026-09-20, two independent bugs): this scheduler's
posterior refit is O(n^3) in the operator count by nature (Gaussian Process
posterior over n arms), and was hanging
``tests/test_regression_scheduler_operator_reach.py``'s 20,000-round
reachability sweep at this project's current operator-registry size (218;
it was 197 not long ago and keeps growing) via two compounding causes,
both fixed here:

1. The O(n^3) refit itself was a hand-rolled Cholesky decomposition plus
   forward/backward substitution over plain ``list[list[float]]``, done
   with Python-level loops and generator-expression ``sum()`` calls --
   genuinely O(n^3) *Python bytecode*, not O(n^3) BLAS calls (~218^3 ~=
   10.3M scalar Python operations per refit). Refactored onto
   ``numpy.linalg`` (LAPACK-backed, the same tier of fix ``op_katz.py`` and
   ``core/kuramoto.py`` already use for their own eigenvalue/matrix work),
   with a small numerical nugget (``_JITTER``) added to the solved matrix's
   diagonal, replacing the old Cholesky's implicit regularization
   (``max(diagonal_term, 1e-12)`` inside the sqrt) that let it proceed on a
   singular/rank-deficient K -- the common case here, since many operators
   share a category and therefore an identical one-hot feature vector,
   making K exactly rank-deficient whenever ``noise=0.0``.
   ``numpy.linalg.solve`` raises on an exactly singular matrix where the old
   loop-based Cholesky degraded gracefully instead; the nugget keeps that
   same "proceed anyway" behavior. ``_predict_mean``/``_posterior_variance``
   also did an O(n) ``list.index(op)`` lookup per call -- individually
   cheap, but with every candidate in a 218-op ballot queried twice per
   round (mean + variance) that was itself an O(n^2)-per-round cost sitting
   on top of the refit; replaced with an O(1) ``_op_index`` dict built
   alongside ``_op_list`` at refit time.

2. Independently of (1)'s constant-factor fix, ``record()`` set
   ``_needs_refit = True`` unconditionally on *every* call, silently
   defeating the ``refit_interval`` cadence ``select_op`` already tracks via
   ``_pulls_since_refit`` -- so even with (1) fixed, a full O(n^3) refit
   still ran on nearly every round instead of every ``refit_interval``
   rounds as the constructor's own docstring promises. This was the larger
   of the two costs once (1) made each individual refit cheap: measured
   ~3ms/refit at n=218 (down from tens of milliseconds of pure-Python work),
   but 20,000 refits/rounds at 3ms each is still a full minute. Removed
   from ``record()``; the posterior is now refreshed only by the cadence
   ``select_op`` already tracked, or immediately the first time it's needed
   (``_kernel_matrix is None``). No existing test exercises the
   record-then-immediately-read-a-stale-posterior distinction this changes
   (every test either never interleaves `record()` with a posterior read
   mid-sequence, or its first read is the always-forced initial refit), so
   this is a pure latent-bug fix, not an observable behavior change for
   anything this module's own regression tests pin down.

Same math throughout, no behavioral change to any of the tested
posterior-mean/-variance/EI *values* -- only to how often they're
recomputed, which was already meant to be governed by ``refit_interval``.
"""

import math

import numpy as np

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, category_of
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.running_stats import RunningMoments

# Diagonal regularization added before solving, independent of the model's
# own `noise` parameter. Keeps a rank-deficient K (common here -- see module
# docstring) numerically solvable instead of raising LinAlgError, mirroring
# the old hand-rolled Cholesky's `max(diagonal_term, 1e-12)` floor without
# reintroducing its O(n^3)-in-pure-Python cost.
_JITTER = 1e-10


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
        rng: RandPool | None = None,
    ):
        self.length_scale = length_scale
        self.noise = noise  # observation noise sigma
        self.refit_interval = refit_interval
        # Hard Rule 16 (shared RandPool for reproducibility): same
        # rng-or-default idiom UCBBase/DUCBScheduler already use -- see
        # select_op's tie-break fix below (module docstring, bug 3) for
        # why this scheduler needs one now.
        self._rng = rng if rng is not None else get_default_rand_pool()

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
        self._op_index: dict[str, int] = {}
        self._needs_refit = True
        self._pulls_since_refit = 0
        self._total_pulls = 0
        # Incremental cache for _best_mean() -- see _note_mean's docstring
        # for why a full max() scan on every call was itself an
        # O(n)-per-EI-candidate (== O(n^2)-per-round) cost.
        self._best_op: str | None = None
        self._best_mean_value: float = 0.0

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
            if prior_alpha > 1.0 or prior_beta > 1.0:
                self._note_mean(name, self._moments[name].mean)

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

    def _build_kernel_matrix(self, ops: list[str]) -> np.ndarray:
        """Build full kernel matrix K for given operators.

        Vectorized pairwise-squared-distance form of the same RBF this
        module always computed (see module docstring's performance note):
        ``||f_i - f_j||^2 = ||f_i||^2 + ||f_j||^2 - 2 f_i.f_j``, via one
        matmul instead of an O(n^2) Python double loop each calling the
        old per-pair ``_rbf``. Diagonal is forced to exactly 1.0 (as the
        old code did explicitly) rather than trusted to the
        ``dist_sq=0 -> exp(0)=1`` identity, so floating-point noise in the
        squared-norm subtraction can never move it off 1.0.
        """
        n = len(ops)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        feats = [self._features.get(op) for op in ops]
        if any(f is None or len(f) == 0 for f in feats):
            # Mirrors the old _rbf's "missing/empty feature -> 0.0"
            # fallback: shouldn't happen once init_arm has run for every
            # offered op, but degrade the same way rather than raising.
            X = np.zeros((n, len(self._cat_names)), dtype=np.float64)
            for i, f in enumerate(feats):
                if f:
                    X[i, : len(f)] = f
        else:
            X = np.asarray(feats, dtype=np.float64)
        sq_norms = np.sum(X * X, axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        np.clip(dist_sq, 0.0, None, out=dist_sq)  # guard tiny fp negatives
        K = np.exp(-dist_sq / (2.0 * self.length_scale**2))
        np.fill_diagonal(K, 1.0)
        return K

    def _update_posterior(self) -> None:
        """Compute GP posterior from current observations.

        Same equations the old hand-rolled Cholesky computed (see class
        docstring): ``alpha = (K + sigma_n^2 I)^-1 y`` and posterior
        variance ``diag(K - K @ (K + sigma_n^2 I)^-1 @ K)`` -- now via
        ``numpy.linalg.solve`` (LAPACK) instead of Python-level forward/
        backward substitution. See module docstring's performance note
        for why this was necessary and for ``_JITTER``'s role.
        """
        # Get all initialized operators
        self._op_list = [
            op for op in self._features if op in self._moments and self._moments[op].count > 0
        ]
        if not self._op_list:
            self._kernel_matrix = None
            self._inv_K = None
            self._y = []
            self._op_index = {}
            self._needs_refit = False
            return

        n = len(self._op_list)
        K = self._build_kernel_matrix(self._op_list)
        y = np.array([self._moments[op].mean for op in self._op_list], dtype=np.float64)
        self._y = y.tolist()

        K_noisy = K + (self.noise * self.noise + _JITTER) * np.eye(n, dtype=np.float64)
        try:
            alpha = np.linalg.solve(K_noisy, y)
            # K_noisy^{-1} @ K in one batched solve (all n columns of K as
            # right-hand sides) instead of the old per-column Python loop.
            v = np.linalg.solve(K_noisy, K)
        except np.linalg.LinAlgError:
            # _JITTER is chosen to make this unreachable in practice (see
            # module docstring); kept as a defensive, inspectable fallback
            # rather than letting a pathological K crash the scheduler.
            alpha = np.zeros(n, dtype=np.float64)
            v = np.eye(n, dtype=np.float64)

        # Posterior variance: diag(K - K @ v). diag(K) is exactly 1.0 by
        # construction (see _build_kernel_matrix), matching the old code's
        # K[i][i] term directly rather than re-reading it from K.
        post_var = 1.0 - np.sum(K * v.T, axis=1)
        self._post_var = np.maximum(post_var, 1e-12).tolist()

        self._kernel_matrix = K
        self._alpha = alpha
        # O(1) op -> row/col index, replacing the old list.index(op) lookup
        # _predict_mean/_posterior_variance did per call: that was O(n) each,
        # and with every candidate in a 218-operator ballot queried twice
        # per round (mean + variance) it was itself an O(n^2)-per-round cost
        # left over after the O(n^3) refit above was fixed (see module
        # docstring) -- dict lookup removes it entirely.
        self._op_index = {op: i for i, op in enumerate(self._op_list)}
        self._needs_refit = False

    def _predict_mean(self, op: str) -> float:
        """Predict posterior mean for operator."""
        if op not in self._moments or self._moments[op].count == 0:
            return 0.0
        if self._needs_refit or self._kernel_matrix is None:
            self._update_posterior()
        if not self._op_list:
            return self._moments[op].mean
        idx = self._op_index.get(op)
        if idx is None:
            return self._moments[op].mean
        # mean = sum_j alpha_j * K(op, op_j) = row `idx` of K dotted with alpha
        return float(self._kernel_matrix[idx] @ self._alpha)

    def _posterior_variance(self, op: str) -> float:
        """Predict posterior variance for operator."""
        if op not in self._moments or self._moments[op].count == 0:
            return 1.0  # maximum uncertainty for unobserved
        if self._needs_refit or self._kernel_matrix is None:
            self._update_posterior()
        if not self._op_list:
            return 1.0
        idx = self._op_index.get(op)
        return self._post_var[idx] if idx is not None else 1.0

    def _note_mean(self, name: str, new_mean: float) -> None:
        """Incrementally maintain (`_best_op`, `_best_mean_value`) so
        `_best_mean()` is O(1) instead of the O(n) `max(...)` scan it used
        to do on every call. That scan was itself hot: `_expected_improvement`
        calls it once per EI candidate, and with every operator in a
        218-strong ballot queried every round, 218 O(n) scans/round is an
        O(n^2)-per-round cost -- measured as the dominant remaining cost
        (~28s of a 42.6s/3000-round profile) even after fixing this
        module's O(n^3) refit itself (see module docstring).

        `RunningMoments.mean` is a sliding-window mean here, not monotonic,
        so a champion's own mean can *drop* below the cached value; that
        one case needs an O(n) rescan (nothing cheaper is possible without
        tracking the full order), but it only fires when the current
        best-mean operator's own record() causes a decrease -- far rarer
        than "any operator's record() happens", which is what the old code
        rescanned on every EI candidate for regardless of which operator
        actually changed.
        """
        if self._best_op is None or new_mean > self._best_mean_value:
            self._best_op = name
            self._best_mean_value = new_mean
        elif name == self._best_op and new_mean < self._best_mean_value:
            self._best_op, self._best_mean_value = max(
                ((op, m.mean) for op, m in self._moments.items()), key=lambda t: t[1]
            )

    def _best_mean(self) -> float:
        """Best observed mean reward so far (f_max for EI)."""
        return self._best_mean_value if self._moments else 0.0

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

        # Bug fix (2026-09-20, see module docstring's bug 3): every
        # never-observed operator gets the identical default (mu=0,
        # sigma=1) and therefore the identical EI value -- phi(0) with
        # f_max=0 at the very start of a run, exp(-1/ls^2)-shaped values
        # later, but always exactly tied within the whole never-observed
        # group. The old tie-break, `max(kv[1], kv[0])`, picks the
        # lexicographically *largest* name on a tie, every time, which
        # systematically starves alphabetically-early operators forever
        # once any later-sorting operator enters that same tied group
        # (reproduced directly: 'bit_flip' never selected in 20,000 rounds
        # against a 218-operator ballot). Uniform-random pick among the
        # argmax set via the shared RandPool -- the same
        # tied/unpulled-arm idiom UCBBase.select_op already uses
        # (`self._rng.choice(unpulled)`) -- gives every tied candidate,
        # including the never-observed ones, a fair shot each round
        # instead of a fixed alphabetical winner.
        best_score = max(scores.values())
        tied = [op for op, s in scores.items() if s == best_score]
        if len(tied) == 1:
            return tied[0]
        return self._rng.choice(tied)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome. Does *not* force an immediate refit -- see module
        docstring's performance note: this method used to set
        ``_needs_refit = True`` unconditionally, silently defeating the
        ``refit_interval`` cadence ``select_op`` already tracks via
        ``_pulls_since_refit`` (a refit ran on nearly every round instead of
        every ``refit_interval`` rounds as documented). The posterior is
        refreshed lazily by that existing cadence, or immediately the first
        time it's needed (``_kernel_matrix is None`` in `_predict_mean`/
        `_posterior_variance`), never by `record()` itself.
        """
        self._total_pulls += 1
        reward = weight if success else 0.0
        if name not in self._moments:
            self._moments[name] = RunningMoments()
        self._moments[name].update(reward)
        self._note_mean(name, self._moments[name].mean)

    def bandit_stats(self) -> dict:
        """Return BO-GP-UCB diagnostics."""
        return {
            "bo_gp_ucb_pulls": self._total_pulls,
            "operators_tracked": len(self._moments),
            "posterior_dim": len(self._op_list) if self._op_list else 0,
            "noise": self.noise,
            "length_scale": self.length_scale,
        }
