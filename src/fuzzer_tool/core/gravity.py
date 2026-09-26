"""Gravity-model splice donor selection.

Trade gravity model, applied to splice pairs (base i, donor j)::

    T_ij = G · (1+M_i)^α · (1+M_j)^β / (d_ij² + ε²)^(γ/2)

    M_j   edges the donor covers that the base lacks  (|E_j \\ E_i|)
    M_i   edges the base covers that the donor lacks  (|E_i \\ E_j|)
    d_ij  1 - Jaccard(E_i, E_j), estimated by MinHash
    ε     Plummer softening: keeps d=0 finite

    corpus            pick(i)            splice(i, j)       new edges y
    ──────  k cands ─► weight(i, j) ─► ─────────────── ─► observe(y)
                            ▲                                   │
                            └──── (α, β, γ) ◄── PPML refit ◄────┘

A donor with M_j = 0 has nothing to give and weighs 0: plain 1/d² would
otherwise make near-clones the heaviest bodies in the corpus.

The exponents are learned, not assumed: log T is linear in the features,
so the yield of each splice round is one observation of a Poisson GLM
(PPML, Santos Silva & Tenreyro 2006), which takes the many zero-yield
rounds as data instead of dropping them the way log-OLS must. A fitted
γ ≈ 0 says distance carries no signal on this target.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

# Plummer softening in Jaccard-distance units. MinHash with 64 permutations
# resolves J in steps of 1/64 ≈ 0.016, so ε spans ~3 steps.
SOFTENING = 0.05

# Candidates sampled per donor pick: O(k · num_perm) per splice, not O(N).
DONOR_CANDIDATES = 8

# Ring buffer of (features, yield) rows the fit sees: bounds memory
# (2048 x 5 float64 = 80 KiB). Yields are sparse, so the window is wide.
WINDOW = 2048

# Observations between refits, and the minimum the fit needs.
REFIT_EVERY = 64
MIN_ROWS = 64

# Evidence gate: positive-yield rows required before any refit. Measured on
# fuzzgoat: 6 hits in 512 rows drove γ to the clamp (quasi-separation), so
# below the gate the prior exponents stand.
MIN_POSITIVES = 32

# IRLS settings. Ridge keeps XᵀWX invertible when a feature is constant.
IRLS_ITERS = 8
RIDGE = 1e-3

# Exponent clamp: a runaway fit must not turn the pick into an argmax.
EXP_LIMIT = 4.0

# Feature layout: (intercept, log1p M_i, log1p M_j, -½·log(d²+ε²)).
_N_FEATURES = 4


class SpliceDonor(Enum):
    """How splice-family operators choose their donor."""

    UNIFORM = "uniform"
    GRAVITY = "gravity"


def pair_terms(size_i: int, size_j: int, jac: float) -> tuple[float, float, float]:
    """Complement masses and distance from edge-set sizes and a Jaccard estimate.

    |E_i ∩ E_j| = J·(|E_i| + |E_j|) / (1 + J). Example: sizes 60/60, J=0.2
    → intersection 20, each side owns 40 the other lacks, d = 0.8.
    """
    inter = jac * (size_i + size_j) / (1.0 + jac)
    return max(0.0, size_i - inter), max(0.0, size_j - inter), 1.0 - jac


def pick_index(weights: list[float], rng) -> int:
    """CDF draw over ``weights``; -1 (no draw) when they carry no mass."""
    total = sum(weights)
    if total <= 0.0:
        return -1

    r = rng.random() * total
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += w
        if acc > r:
            return idx
    return len(weights) - 1


def _irls_step(rows: np.ndarray, ys: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """One Newton step of the Poisson log-link likelihood (ridge-damped)."""
    eta = np.clip(rows @ theta, -30.0, 30.0)
    mu = np.exp(eta)
    z = eta + (ys - mu) / mu
    wx = rows * mu[:, None]
    lhs = rows.T @ wx + RIDGE * np.eye(_N_FEATURES)
    return np.linalg.solve(lhs, wx.T @ z)


def fit_ppml(rows: np.ndarray, ys: np.ndarray, theta0: np.ndarray) -> np.ndarray | None:
    """Poisson pseudo-maximum-likelihood fit of log E[y] = rows · θ.

    Returns None when there is too little to learn from (rows, or
    positive-yield rows) or the system is singular; callers keep their old θ.
    """
    if len(ys) < MIN_ROWS or np.count_nonzero(ys > 0) < MIN_POSITIVES:
        return None

    theta = theta0.copy()
    # Start the intercept at the mean yield so the first step is well scaled.
    theta[0] = math.log(float(np.mean(ys)))
    try:
        for _ in range(IRLS_ITERS):
            theta = _irls_step(rows, ys, theta)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(theta)):
        return None
    return theta


class GravityModel:
    """Donor weights plus the online PPML fit of their exponents."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self._theta = np.array([0.0, alpha, beta, gamma])
        self._rows = np.zeros((WINDOW, _N_FEATURES))
        self._ys = np.zeros(WINDOW)
        self._filled = 0
        self._head = 0
        self._since_fit = 0
        self._refits = 0
        self._pending: list[np.ndarray] = []

    @staticmethod
    def features(m_i: float, m_j: float, d: float) -> np.ndarray:
        """Log-linear design row for one (base, donor) pair."""
        return np.array(
            [1.0, math.log1p(m_i), math.log1p(m_j), -0.5 * math.log(d * d + SOFTENING**2)]
        )

    @property
    def exponents(self) -> tuple[float, float, float]:
        """Current (α, β, γ)."""
        return float(self._theta[1]), float(self._theta[2]), float(self._theta[3])

    @property
    def pending(self) -> int:
        """Pairs staged this round and not yet observed."""
        return len(self._pending)

    def weight(self, m_i: float, m_j: float, d: float) -> float:
        """Unnormalised pick weight; 0 for a donor with nothing the base lacks."""
        if m_j <= 0.0:
            return 0.0

        alpha, beta, gamma = self.exponents
        log_w = (
            alpha * math.log1p(m_i)
            + beta * math.log1p(m_j)
            - 0.5 * gamma * math.log(d * d + SOFTENING**2)
        )
        return math.exp(log_w)

    def stage(self, m_i: float, m_j: float, d: float) -> None:
        """Remember a pair picked this round; its yield arrives via observe()."""
        self.stage_features(self.features(m_i, m_j, d))

    def stage_features(self, row: np.ndarray) -> None:
        """Stage a prebuilt design row (bounded: extra pairs are dropped)."""
        if len(self._pending) < WINDOW:
            self._pending.append(row)

    def discard(self) -> None:
        """Drop staged pairs whose round never reported a yield."""
        self._pending.clear()

    def observe(self, new_edges: int) -> None:
        """Credit this round's new edges, split evenly, to the staged pairs."""
        if not self._pending:
            return

        share = new_edges / len(self._pending)
        for row in self._pending:
            self._push(row, share)
        self._pending.clear()

        if self._since_fit >= REFIT_EVERY:
            self._refit()

    def _push(self, row: np.ndarray, y: float) -> None:
        """Append to the ring buffer, overwriting the oldest row when full."""
        self._rows[self._head] = row
        self._ys[self._head] = y
        self._head = (self._head + 1) % WINDOW
        self._filled = min(self._filled + 1, WINDOW)
        self._since_fit += 1

    def _refit(self) -> None:
        """Refit θ on the window; clamp exponents so no arm becomes argmax."""
        self._since_fit = 0
        n = self._filled
        theta = fit_ppml(self._rows[:n], self._ys[:n], self._theta)
        if theta is None:
            return

        theta[1:] = np.clip(theta[1:], -EXP_LIMIT, EXP_LIMIT)
        self._theta = theta
        self._refits += 1

    def summary(self) -> dict:
        """Diagnostics for the exit summary."""
        alpha, beta, gamma = self.exponents
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "observations": self._filled,
            "positives": int(np.count_nonzero(self._ys[: self._filled] > 0)),
            "refits": self._refits,
        }

    def state_dict(self) -> dict:
        """Plain-container state for StateStore (pickle-safe types only)."""
        # Oldest-first, so a reload replays rows in arrival order.
        order = np.roll(np.arange(self._filled), -self._head if self._filled == WINDOW else 0)
        return {
            "theta": [float(v) for v in self._theta],
            "rows": self._rows[order].tolist(),
            "ys": self._ys[order].tolist(),
            "refits": self._refits,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state_dict(); missing keys keep the constructor defaults."""
        theta = state.get("theta")
        if theta is not None and len(theta) == _N_FEATURES:
            self._theta = np.array(theta, dtype=float)

        rows = state.get("rows") or []
        ys = state.get("ys") or []
        for row, y in zip(rows[-WINDOW:], ys[-WINDOW:], strict=False):
            self._push(np.asarray(row, dtype=float), float(y))
        self._since_fit = 0
        self._refits = int(state.get("refits", 0))
