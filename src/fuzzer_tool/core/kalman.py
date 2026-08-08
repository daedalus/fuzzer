"""Kalman filter variants for online state estimation.

Provides a vanilla KalmanFilter base class (1D and 2D constant-velocity
models) and a RobustKF subclass with Huber innovation gating and adaptive
measurement-noise covariance estimation.

These are used by the fuzzer for three specific applications:
  1. **Adaptive network settle time** — 1D KF over target processing
     latency, replacing the hardcoded ``net_settle_ms`` guess.
  2. **Denoised discovery rate** — 2D KF upstream of
     :class:`CriticalSlowingDown` / :class:`AllanVarianceDetector`,
     providing a causal, single-pass smoothed rate with uncertainty.
  3. **Execs/sec estimation** — 2D KF over EPS with irregular-time
     handling, used for ETA, dictionary-cap tuning, and budget allocation.
"""

from __future__ import annotations

import copy
import logging
import math

log = logging.getLogger(__name__)

# Default innovation gate: observations beyond 3 sigma are downweighted
_HUBER_THRESHOLD = 3.0


class KalmanFilter:
    """Vanilla Kalman filter — 1D or 2D constant-velocity model.

    Args:
        dim: State dimension (1 or 2).  1D tracks only the value;
            2D tracks [value, derivative].
        process_noise: Per-step process noise standard deviation
            (``sqrt(Q)`` for the value component).
        measurement_noise: Measurement noise standard deviation (``sqrt(R)``).
        initial_state: Optional  [state] or [state, derivative] seed.
            If ``None``, the first ``update()`` snap-initializes.
        initial_covariance: Optional P₀ diagonal values.

    The 2D model uses the state-transition matrix::

        F = [[1, dt],    H = [[1, 0]],
             [0, 1]]

    so ``update()`` always receives a **direct scalar observation** of the
    value, never of the derivative.
    """

    def __init__(
        self,
        dim: int = 1,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_state: tuple[float, ...] | None = None,
        initial_covariance: float | None = None,
    ):
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        self._dim = dim
        self._q_base = process_noise  # per-step Q scalar for the value
        self._r_base = measurement_noise  # measurement noise stddev

        # State vector x (len dim)
        self._x: list[float] = list(initial_state) if initial_state is not None else [0.0] * dim
        # Covariance matrix P (dim×dim, flattened row-major)
        init_cov = initial_covariance if initial_covariance is not None else 1.0
        self._p: list[list[float]] = [
            [init_cov if i == j else 0.0 for j in range(dim)] for i in range(dim)
        ]
        self._initialized: bool = initial_state is not None

        # Last innovation (for diagnostics)
        self._last_innovation: float = 0.0
        self._last_innovation_variance: float = 0.0

    # ── properties ────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def state(self) -> list[float]:
        """Current state estimate [value, ...]."""
        return list(self._x)

    @property
    def covariance(self) -> list[list[float]]:
        """Current covariance matrix (row-major)."""
        return copy.deepcopy(self._p)

    @property
    def estimate(self) -> float:
        """Point estimate of the tracked value (first state component)."""
        return self._x[0]

    @property
    def derivative(self) -> float:
        """Rate-of-change estimate (0 for 1D filters)."""
        return self._x[1] if self._dim >= 2 else 0.0

    @property
    def innovation(self) -> float:
        """Most recent innovation (measurement residual)."""
        return self._last_innovation

    @property
    def innovation_variance(self) -> float:
        """Most recent innovation covariance S (scalar)."""
        return self._last_innovation_variance

    @property
    def uncertainty(self) -> float:
        """Posterior standard deviation of the value estimate."""
        return math.sqrt(self._p[0][0])

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── core cycle ────────────────────────────────────────────────────────

    def predict(self, dt: float = 1.0) -> None:
        """Predict the state at time *dt* in the future.

        Args:
            dt: Time step (must be > 0).
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if not self._initialized:
            return  # no prediction until we have a first observation

        if self._dim == 1:
            # x = x (no change for 1D)
            # P = P + Q
            self._p[0][0] += self._q_base * self._q_base
        else:
            # 2D: F = [[1, dt], [0, 1]]
            x0 = self._x[0] + dt * self._x[1]
            self._x[1] = self._x[1]  # derivative unchanged in prediction
            self._x[0] = x0

            # P = F @ P @ F^T + Q
            # Q for 2D: process_noise^2 * [[dt³/3, dt²/2], [dt²/2, dt]]
            # P is kept symmetric (see the writes below), so p10 == p01 and
            # only p01 is read here.
            p00 = self._p[0][0]
            p01 = self._p[0][1]
            p11 = self._p[1][1]

            q_base = self._q_base
            q00 = q_base * q_base * dt * dt * dt / 3.0
            q01 = q_base * q_base * dt * dt / 2.0
            q11 = q_base * q_base * dt

            # F @ P @ F^T  (then P = result + Q)
            fp00 = p00 + 2.0 * dt * p01 + dt * dt * p11
            fp01 = p01 + dt * p11
            fp11 = p11

            self._p[0][0] = fp00 + q00
            self._p[0][1] = fp01 + q01
            self._p[1][0] = fp01 + q01  # symmetric
            self._p[1][1] = fp11 + q11

    def update(self, measurement: float) -> float:
        """Incorporate a scalar measurement.

        Performs a full predict-then-update cycle for a direct observation
        of the first state component (H = [1, 0] for 2D, [1] for 1D).

        Args:
            measurement: Noisy observation of the tracked value.

        Returns:
            Innovation (measurement residual ``z - H·x``), positive if the
            observation was above the prediction.
        """
        if not self._initialized:
            # Snap-initialise on first observation.
            self._x[0] = measurement
            self._initialized = True
            self._last_innovation = 0.0
            self._last_innovation_variance = self._p[0][0] + self._r_base * self._r_base
            return 0.0

        # Innovation / residual
        y = measurement - self._x[0]
        # Innovation covariance S = H @ P @ H^T + R
        s = self._p[0][0] + self._r_base * self._r_base
        self._last_innovation = y
        self._last_innovation_variance = s

        # Kalman gain K = P @ H^T / S
        if s < 1e-30:
            k = [0.0] * self._dim
        elif self._dim == 1:
            k = [self._p[0][0] / s]
        else:
            k0 = self._p[0][0] / s
            k1 = self._p[1][0] / s
            k = [k0, k1]

        # State update: x += K * y
        for i in range(self._dim):
            self._x[i] += k[i] * y

        # Covariance update: P = (I - K @ H) @ P
        #                    = [[1-k0, 0], [-k1, 1]] @ P  (for 2D)
        if self._dim == 1:
            self._p[0][0] = (1.0 - k[0]) * self._p[0][0]
        else:
            p00 = self._p[0][0]
            p01 = self._p[0][1]
            p10 = self._p[1][0]
            p11 = self._p[1][1]
            k0, k1 = k[0], k[1]

            self._p[0][0] = (1.0 - k0) * p00
            self._p[0][1] = (1.0 - k0) * p01
            self._p[1][0] = p10 - k1 * p00
            self._p[1][1] = p11 - k1 * p01

        return y

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(
        self,
        state: tuple[float, ...] | None = None,
        covariance: float | None = None,
    ) -> None:
        """Reset filter to initial state.

        Args:
            state: Optional [value] or [value, derivative]. Use ``None``
                for lazy initialisation on first ``update()``.
            covariance: Diagonal value for the covariance matrix.
        """
        self._x = list(state) if state is not None else [0.0] * self._dim
        init_cov = covariance if covariance is not None else 1.0
        self._p = [
            [init_cov if i == j else 0.0 for j in range(self._dim)] for i in range(self._dim)
        ]
        self._initialized = state is not None
        self._last_innovation = 0.0
        self._last_innovation_variance = 0.0

    # ── serialisation ─────────────────────────────────────────────────────

    def save(self) -> dict:
        """Serialize filter state for persistence."""
        return {
            "dim": self._dim,
            "q_base": self._q_base,
            "r_base": self._r_base,
            "x": list(self._x),
            "p": [list(row) for row in self._p],
            "initialized": self._initialized,
        }

    def load(self, data: dict) -> None:
        """Restore filter state from a ``save()`` dict."""
        self._dim = data.get("dim", self._dim)
        self._q_base = data.get("q_base", self._q_base)
        self._r_base = data.get("r_base", self._r_base)
        self._x = list(data.get("x", self._x))
        self._p = [list(row) for row in data.get("p", self._p)]
        self._initialized = data.get("initialized", self._initialized)
        self._last_innovation = 0.0
        self._last_innovation_variance = 0.0


class RobustKF(KalmanFilter):
    """Kalman filter with Huber innovation gating and adaptive R estimation.

    Extends :class:`KalmanFilter` with two modifications:

    1. **Huber gating** — when the Mahalanobis distance of the innovation
       exceeds ``huber_threshold``, the measurement noise is inflated
       (rather than fully rejecting the observation), making the update
       robust to outliers without ignoring them entirely.

    2. **Adaptive R** — running estimate of the measurement noise
       covariance via exponentially-weighted innovation covariance
       matching.  This lets the filter tune its own ``R`` online when
       the true observation noise is unknown or non-stationary.

    Args:
        huber_threshold: Mahalanobis distance above which to inflate R.
        adaptive_r_gain: Gain for the adaptive R update (0 = fixed R).
            Typical values: 0.01 – 0.05.
        max_r_inflation: Maximum multiplier for R under Huber gating.
        **kwargs: Passed through to :class:`KalmanFilter`.
    """

    def __init__(
        self,
        huber_threshold: float = _HUBER_THRESHOLD,
        adaptive_r_gain: float = 0.02,
        max_r_inflation: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._huber_threshold = huber_threshold
        self._adaptive_r_gain = adaptive_r_gain
        self._max_r_inflation = max_r_inflation

        # Effective R (can grow beyond R_base under gating)
        self._r_eff = self._r_base

        # Running innovation RMS for adaptive R (exponential window)
        self._innov_rms: float = 0.0
        self._innov_count: int = 0

    # ── properties ────────────────────────────────────────────────────────

    @property
    def effective_measurement_noise(self) -> float:
        """Current measurement noise stddev (may be inflated by Huber gate)."""
        return self._r_eff

    @property
    def huber_threshold(self) -> float:
        return self._huber_threshold

    # ── overrides ─────────────────────────────────────────────────────────

    def update(self, measurement: float) -> float:
        """Robust update with Huber gating and adaptive R.

        Huber gating is **one-shot**: the measurement noise is inflated
        only for the current step if the Mahalanobis distance exceeds
        the threshold.  The effective R stored on the filter is only
        updated by the slower **adaptive R** mechanism (innovation RMS
        matching) — gating does not persist.

        Returns the raw innovation (``z - H·x`` before gating).
        """
        if not self._initialized:
            self._innov_rms = 0.0
            self._innov_count = 0
            return super().update(measurement)

        # 1. Compute the Mahalanobis distance using BASE R.
        y = measurement - self._x[0]
        s_base = self._p[0][0] + self._r_base * self._r_base
        m_dist = abs(y) / (math.sqrt(s_base) if s_base > 1e-30 else 1.0)

        # 2. Decide whether to gate this observation (one-shot).
        if m_dist > self._huber_threshold:
            inflation = min(
                self._max_r_inflation,
                1.0 + (m_dist - self._huber_threshold) / self._huber_threshold,
            )
            r_step = self._r_base * inflation
        else:
            r_step = self._r_base

        # 3. Perform the update with the (potentially inflated) step R.
        saved_r = self._r_base
        self._r_base = r_step
        innovation = super().update(measurement)
        self._r_base = saved_r

        # 4. Adaptive R via innovation RMS matching (slow timescale).
        if self._adaptive_r_gain > 0:
            w = self._adaptive_r_gain
            # Use the gated innovation from the step above (reflects
            # what the filter actually saw, not the raw residual).
            gated_y = self._last_innovation
            if self._innov_count < 50:
                # Burn-in: accumulate.
                self._innov_count += 1
                self._innov_rms = (1.0 - w) * self._innov_rms + w * (gated_y * gated_y)
            else:
                self._innov_rms = (1.0 - w) * self._innov_rms + w * (gated_y * gated_y)
                # Nudge effective R to track observed innovation RMS,
                # clamped to a plausible band.
                target_r = max(math.sqrt(self._innov_rms), self._r_base * 0.3)
                self._r_eff = min(target_r, self._r_base * self._max_r_inflation)
        else:
            # Without adaptive gain, effective R stays at base.
            self._r_eff = self._r_base

        # Return the raw (pre-gating) innovation for diagnostics.
        return innovation

    # ── serialisation ─────────────────────────────────────────────────────

    def save(self) -> dict:
        data = super().save()
        data.update(
            {
                "huber_threshold": self._huber_threshold,
                "adaptive_r_gain": self._adaptive_r_gain,
                "max_r_inflation": self._max_r_inflation,
                "r_eff": self._r_eff,
                "innov_rms": self._innov_rms,
                "innov_count": self._innov_count,
            }
        )
        return data

    def load(self, data: dict) -> None:
        super().load(data)
        self._huber_threshold = data.get("huber_threshold", self._huber_threshold)
        self._adaptive_r_gain = data.get("adaptive_r_gain", self._adaptive_r_gain)
        self._max_r_inflation = data.get("max_r_inflation", self._max_r_inflation)
        self._r_eff = data.get("r_eff", self._r_base)
        self._innov_rms = data.get("innov_rms", 0.0)
        self._innov_count = data.get("innov_count", 0)
