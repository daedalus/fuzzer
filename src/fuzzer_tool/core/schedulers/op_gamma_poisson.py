"""GammaPoissonScheduler: Thompson sampling over discounted Gamma-Poisson rates.

Each operator's yield is a Poisson rate with a conjugate Gamma posterior,
discounted per global round so stale evidence fades (West & Harrison's
power discount; Smith & Miller 1986)::

    predict:  f = max(discount ** (t - last_i), min(1, shape_min / a_i))
              a_i *= f;  b_i *= f                 mean a/b kept, shape shrinks
    record:   a_i += y;  b_i += 1                 y = reward in [0, 1]
    select:   theta_i ~ Gamma(a_i, rate=b_i)  ->  argmax

Why not ``kalman_ts``? Its Gaussian draw is the De Moivre-Laplace limit,
valid when n p is large. Operator yields sit near 1e-3: there a binomial
count is Poisson (the law of rare events), the posterior is right-skewed,
and the Bernoulli variance p (1 - p) falls under Kalman-TS's ``obs_floor``.
Convergence harness, rates x0.05 (best 0.015), 60k rounds, 4 seeds:

    ==================  ===========  ===========
    env                 gamma_pois.  kalman_ts
    ==================  ===========  ===========
    stationary          0.863 min    0.461 max
    decaying best       0.796 min    0.246 max
    ==================  ===========  ===========

At the harness's default rates (best 0.30) Kalman-TS wins (0.990 vs 0.948
stationary, 0.984 vs 0.913 decay): the normal limit holds there.

The discount is mean-preserving: a neglected arm keeps its rate estimate
and loses confidence, like the Kalman predict step. ``shape_min`` stops a
long gap from collapsing the shape toward 0, where Gamma draws concentrate
near 0 and the arm would never be re-explored.

Off-policy safe: a conjugate posterior accepts any arm's outcome, so it
records every round like ``kalman_ts``.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Floor on a Beta prior parameter before mapping it to a Gamma prior.
PRIOR_MIN = 1e-6


class GammaPoissonScheduler:
    """Thompson sampling over per-arm discounted Gamma-Poisson posteriors.

    Args:
        discount: Per-global-round evidence retention, in (0, 1]; 1 is the
            static conjugate model. Effective memory ~1 / (1 - discount)
            rounds. Harness tail share min over 4 seeds (stationary /
            decay / rare / rare decay):

            ======== ===== ===== ===== ==========
            discount stat. decay rare  rare decay
            ======== ===== ===== ===== ==========
            0.999    0.927 0.915 0.784 0.732
            0.9995   0.938 0.936 0.669 0.815
            0.9998   0.948 0.913 0.863 0.796
            0.9999   0.955 0.856 0.852 0.866
            1.0      0.983 0.191 0.964 n/a
            ======== ===== ===== ===== ==========
        shape_min: Floor the discount cannot push a shape below, > 0.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm maps Beta(a, b) to Gamma(a, a + b): same mean, a + b
    #: pseudo-pulls (Hard Rule 40).
    supports_priors = True

    def __init__(
        self,
        discount: float = 0.9998,
        shape_min: float = 1.0,
        rng: RandPool | None = None,
    ) -> None:
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {discount!r}")
        if shape_min <= 0.0:
            raise ValueError(f"shape_min must be > 0, got {shape_min!r}")
        self.discount = float(discount)
        self.shape_min = float(shape_min)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed per-arm posterior; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._shapev = np.zeros(0, dtype=np.float64)
        self._ratev = np.zeros(0, dtype=np.float64)
        self._lastv = np.zeros(0, dtype=np.int64)  # round of last discount
        self._pullv = np.zeros(0, dtype=np.int64)

        # Global record clock: discount accrues per round, not per own pull.
        self._t = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm at the Beta prior's mean (idempotent)."""
        if name in self._idx:
            return
        a = max(float(prior_alpha), PRIOR_MIN)
        b = max(float(prior_beta), PRIOR_MIN)
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._shapev = np.append(self._shapev, a)
        self._ratev = np.append(self._ratev, a + b)
        self._lastv = np.append(self._lastv, self._t)
        self._pullv = np.append(self._pullv, 0)

    def _predict(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Discounted (shape, rate) of arms *idx* at the current round."""
        shape = self._shapev[idx]
        rate = self._ratev[idx]

        # Mean-preserving shrink, floored at shape_min (never raised to it).
        f = self.discount ** (self._t - self._lastv[idx]).astype(np.float64)
        f = np.maximum(f, np.minimum(1.0, self.shape_min / shape))
        return shape * f, rate * f

    def posterior(self, name: str) -> tuple[float, float]:
        """Predicted Gamma (shape, rate) of *name*'s rate at the current round."""
        self.init_arm(name)
        shape, rate = self._predict(np.array([self._idx[name]]))
        return float(shape[0]), float(rate[0])

    def select_op(self, ops: list[str]) -> str:
        """Thompson draw from every offered arm's discounted posterior; argmax."""
        if not ops:
            return ""
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))

        shape, rate = self._predict(idx)
        return ops[int(np.argmax(self._rng.gammavariate_array(shape, rate)))]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Discount *name* to now, then add one exposure and its reward."""
        self.init_arm(name)
        self._t += 1
        j = self._idx[name]

        # Scalar _predict: a 1-element numpy round trip costs ~3x on this path.
        shape = float(self._shapev[j])
        f = self.discount ** (self._t - int(self._lastv[j]))
        f = max(f, min(1.0, self.shape_min / shape))
        self._shapev[j] = shape * f + unit_reward(success, weight)
        self._ratev[j] = float(self._ratev[j]) * f + 1.0
        self._lastv[j] = self._t
        self._pullv[j] += 1

    def bandit_stats(self) -> dict:
        """Return Gamma-Poisson diagnostics."""
        n = len(self._names)
        means = self._shapev / self._ratev if n else self._shapev
        return {
            "gamma_poisson_pulls": self._t,
            "gamma_poisson_arms": n,
            "gamma_poisson_max_mean": float(means.max()) if n else 0.0,
        }
