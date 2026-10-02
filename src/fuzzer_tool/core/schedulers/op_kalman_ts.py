"""KalmanTSScheduler: Thompson sampling over Kalman-tracked operator yields.

Each operator's success rate is a drifting latent level (a local-level
state-space model), filtered per arm and sampled Thompson-style::

    mu_t = mu_{t-1} + w,   w ~ N(0, q_i)        drift, per global round
    y_t  = mu_t + v,       v ~ N(0, r_i)        observed reward in [0, 1]

    select:  theta_i ~ N(m_i, P_i + q_i * (t - last_i))   ->  argmax
    record:  P = P + q dt;  S = P + r;  K = P / S
             m += K (y - m);  P *= 1 - K
             e = (y - m) / sqrt(S)
             q *= exp(rate * e * e_prev)               innovation whiteness

Why not ``monte_carlo``'s Beta posterior with ``arm_decay``? That forgets
at one fixed global rate for every arm. Here the drift ``q_i`` is learned
per arm from the whiteness of its own innovations (Mehra 1970): a correctly
tuned filter's normalised innovations are uncorrelated. A sluggish one
(``q`` too small) lags a moving level, so consecutive innovations share a
sign and their product raises ``q``; an over-reactive one chases noise, so
they alternate and ``q`` falls. An arm whose yield collapses is abandoned
in ~15 failures instead of the ~200 a saturated Beta(201, 1) needs.

Why not normalised innovation squared (NIS, ``E[innov^2 / S] = 1``)? With
a plug-in Bernoulli variance ``m (1 - m)`` it is biased upward by ~8 d^2
(d the level error) even with zero drift: measured on a stationary p = 0.3
arm, NIS ran ``q`` to the 1e-2 cap in 20k rounds while whiteness drove it
to 2e-6 and estimated the level at 0.301. The lag-1 product has no such
bias because the observation noise is independent across rounds.

``r_i = max(m (1 - m), obs_floor)``: the Bernoulli variance at the current
level, floored so a level near 0 or 1 still admits evidence.

Unobserved arms are predicted lazily: the variance grows by ``q_i`` per
elapsed round, computed at selection, capped at ``VAR_MAX`` (the largest
variance of a [0, 1] variable). Selection is one vectorised draw.

Off-policy safe: a Bayesian filter accepts any arm's outcome, so it records
every round like ``monte_carlo``.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Largest variance of a random variable on [0, 1]; the predict cap.
VAR_MAX = 0.25

#: Clamp on the learned per-round drift. Q_MAX lets a level traverse [0, 1]
#: in ~10 rounds; Q_MIN keeps a quiet arm's variance from freezing.
Q_MIN = 1e-8
Q_MAX = 1e-2


class KalmanTSScheduler:
    """Thompson sampling over per-arm local-level Kalman filters.

    Args:
        q0: Initial per-round drift variance of every arm, >= 0. Small on
            measurement: an unpulled arm's variance grows by ``q`` per
            global round, so at 1e-4 every neglected arm reaches the 0.25
            cap within ~2500 rounds and Thompson re-explores all of them.
            Convergence harness, 6 seeds (stationary best-arm tail share
            min / DecayingBest late-arm share min):

            ====== ===== ============= =============
            q0     rate  stationary    recovery
            ====== ===== ============= =============
            1e-5   0.05  0.700         0.131
            1e-6   0.05  0.887         0.755
            1e-7   0.01  0.964         0.924
            1e-8   0     0.992         0.940
            1e-8   0.02  0.990         0.946
            1e-8   0.05  0.983         0.751
            ====== ===== ============= =============
        drift_rate: Step of the multiplicative whiteness drift update,
            >= 0; 0 freezes ``q`` at ``q0``. It is what drops a collapsed
            arm: after 200 successes, mean < 0.5 takes 27 failures at 0.02
            and over 400 at 0 (a Beta posterior needs 200).
        obs_floor: Floor on the observation noise, in (0, 0.25].
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm maps a Beta(alpha, beta) prior to the filter's initial
    #: mean and variance by moment matching (Hard Rule 40).
    supports_priors = True

    def __init__(
        self,
        q0: float = 1e-8,
        drift_rate: float = 0.02,
        obs_floor: float = 0.01,
        rng: RandPool | None = None,
    ) -> None:
        if q0 < 0.0:
            raise ValueError(f"q0 must be >= 0, got {q0!r}")
        if drift_rate < 0.0:
            raise ValueError(f"drift_rate must be >= 0, got {drift_rate!r}")
        if not 0.0 < obs_floor <= VAR_MAX:
            raise ValueError(f"obs_floor must be in (0, {VAR_MAX}], got {obs_floor!r}")
        self.q0 = float(q0)
        self.drift_rate = float(drift_rate)
        self.obs_floor = float(obs_floor)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed per-arm filter state; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._meanv = np.zeros(0, dtype=np.float64)
        self._varv = np.zeros(0, dtype=np.float64)
        self._qv = np.zeros(0, dtype=np.float64)
        self._lastv = np.zeros(0, dtype=np.int64)  # round of last update
        self._innv = np.zeros(0, dtype=np.float64)  # last normalised innovation
        self._pullv = np.zeros(0, dtype=np.int64)

        # Global record clock: drift accrues per round, not per own pull.
        self._t = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm at the Beta prior's mean and variance (idempotent)."""
        if name in self._idx:
            return
        a = max(float(prior_alpha), 1e-6)
        b = max(float(prior_beta), 1e-6)
        n = a + b
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._meanv = np.append(self._meanv, a / n)
        self._varv = np.append(self._varv, min(a * b / (n * n * (n + 1.0)), VAR_MAX))
        q = min(self.q0, Q_MAX) if self.drift_rate == 0.0 else min(max(self.q0, Q_MIN), Q_MAX)
        self._qv = np.append(self._qv, q)
        self._lastv = np.append(self._lastv, self._t)
        self._innv = np.append(self._innv, 0.0)
        self._pullv = np.append(self._pullv, 0)

    def posterior(self, name: str) -> tuple[float, float]:
        """Predicted (mean, variance) of *name*'s level at the current round."""
        self.init_arm(name)
        j = self._idx[name]
        dt = self._t - int(self._lastv[j])
        var = min(float(self._varv[j]) + float(self._qv[j]) * dt, VAR_MAX)
        return float(self._meanv[j]), var

    def select_op(self, ops: list[str]) -> str:
        """Thompson draw from every offered arm's predicted level; argmax."""
        if not ops:
            return ""
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))

        # Lazy predict: variance grown by drift since each arm's last update.
        dt = self._t - self._lastv[idx]
        var = np.minimum(self._varv[idx] + self._qv[idx] * dt, VAR_MAX)
        z = np.asarray(self._rng.gauss_list(0.0, 1.0, len(ops)), dtype=np.float64)
        return ops[int(np.argmax(self._meanv[idx] + np.sqrt(var) * z))]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """One predict/update step for *name* plus its drift adaptation."""
        self.init_arm(name)
        self._t += 1
        j = self._idx[name]
        y = unit_reward(success, weight)

        # Predict over the rounds elapsed since this arm's last update.
        m = float(self._meanv[j])
        q = float(self._qv[j])
        p = min(float(self._varv[j]) + q * (self._t - int(self._lastv[j])), VAR_MAX)

        # Update: Bernoulli observation noise at the current level.
        s = p + max(m * (1.0 - m), self.obs_floor)
        innov = y - m
        gain = p / s
        self._meanv[j] = m + gain * innov
        self._varv[j] = p * (1.0 - gain)

        # Whiteness: same-sign consecutive innovations mean the filter lags.
        e = innov / math.sqrt(s)
        if self.drift_rate > 0.0:
            q *= math.exp(self.drift_rate * e * float(self._innv[j]))
            self._qv[j] = min(max(q, Q_MIN), Q_MAX)
        self._innv[j] = e
        self._lastv[j] = self._t
        self._pullv[j] += 1

    def bandit_stats(self) -> dict:
        """Return Kalman-TS diagnostics."""
        n = len(self._names)
        return {
            "kalman_ts_pulls": self._t,
            "kalman_ts_arms": n,
            "kalman_ts_mean_q": float(self._qv.mean()) if n else 0.0,
            "kalman_ts_max_mean": float(self._meanv.max()) if n else 0.0,
        }
