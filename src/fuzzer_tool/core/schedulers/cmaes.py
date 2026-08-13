"""CMAESScheduler: Covariance Matrix Adaptation Evolution Strategy.

Adapts a multivariate normal over operator-selection logits. After each
generation of ``generation_size`` evaluations the mean, step size, and
covariance matrix are updated from the ranked rewards of the candidates
that were drawn. This gives CMA-ES its well-known advantage on
ill-conditioned or correlated landscapes: correlated operator pairs
(e.g. bit_flip/bit_offset_flip, block_insert/length_grow) get a
shared direction in covariance space instead of being tuned
independently like Thompson arms.

The fuzzer calls ``select_op()`` once per execution. To reconcile that
with CMA-ES's batch update cadence we keep a *candidate pool*: each
``select_op()`` consumes one pre-sampled candidate and returns the
operator drawn from its softmax'd probability vector. ``record()``
assigns the outcome to that candidate and triggers the rank-μ update
once the generation is full.

Interface matches the other operator schedulers:
  - ``init_arm(name)``
  - ``select_op(ops) -> str``
  - ``record(name, success, weight=1.0)``
  - ``bandit_stats() -> dict``
  - ``to_dict() / from_dict()``
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NU_MAX = 2.0  # cap on effective μ_eff/λ to keep weights well-behaved


def _softmax(logits: list[float], floor: float = 0.005) -> list[float]:
    """Stable softmax with a minimum exploration floor per arm."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if total <= 0:
        n = len(logits)
        return [1.0 / n] * n
    probs = [e / total for e in exps]
    # Enforce floor and renormalise
    floored = [max(p, floor) for p in probs]
    total = sum(floored)
    return [p / total for p in floored]


def _sample_operator(probs: list[float], ops: list[str], rng: RandPool) -> str:
    """Weighted draw from *probs* over *ops*."""
    r = rng.random() * sum(probs)
    cumulative = 0.0
    for op, p in zip(ops, probs, strict=False):
        cumulative += p
        if r <= cumulative:
            return op
    return ops[-1]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class CMAESScheduler:
    """Covariance Matrix Adaptation Evolution Strategy over operator logits.

    Args:
        pop_size: CMA-ES population size λ.
        generation_size: Number of evaluations per generation. When this
            many ``record()`` calls have been received the rank-μ update
            runs and a new population is sampled.
        step_size: Initial σ.
        elite_frac: μ/λ fraction for weighted mean update. Clamped so
            1 ≤ μ ≤ λ.
        learning_rate_mean: CMA-ES learning rate for the mean update.
        learning_rate_cov: CMA-ES learning rate for the covariance update.
        learning_rate_sigma: CSA learning rate.
        cov_diag_min: Minimum diagonal entry in C to keep the density
            matrix well-conditioned.
        max_ops_before_diag_fallback: If the number of registered arms
            exceeds this threshold the covariance update is skipped and
            a diagonal C is kept, bounding memory at O(n).
    """

    supports_priors = False

    def __init__(
        self,
        pop_size: int = 8,
        generation_size: int = 200,
        step_size: float = 0.3,
        elite_frac: float = 0.5,
        learning_rate_mean: float = 1.0,
        learning_rate_cov: float = 0.5,
        learning_rate_sigma: float = 0.3,
        cov_diag_min: float = 1e-6,
        max_ops_before_diag_fallback: int = 64,
        rng: RandPool | None = None,
    ):
        self.pop_size = max(2, int(pop_size))
        self.generation_size = max(self.pop_size, int(generation_size))
        self.step_size = float(step_size)
        self.elite_frac = max(0.0, min(1.0, float(elite_frac)))
        self.learning_rate_mean = float(learning_rate_mean)
        self.learning_rate_cov = float(learning_rate_cov)
        self.learning_rate_sigma = float(learning_rate_sigma)
        self.cov_diag_min = float(cov_diag_min)
        self.max_ops_before_diag_fallback = int(max_ops_before_diag_fallback)
        self._rng = rng if rng is not None else RandPool()

        self.operators: list[str] = []
        self.op_index: dict[str, int] = {}

        # CMA-ES state
        self._mean: np.ndarray | None = None
        self._sigma: float = self.step_size
        self._C: np.ndarray | None = None
        self._pc: np.ndarray | None = None  # evolution path for C
        self._ps: np.ndarray | None = None  # evolution path for σ
        self._generation: int = 0
        self._eval_count: int = 0

        # Current generation's candidates and assigned rewards
        self._candidates: list[dict[str, Any]] = []
        self._next_candidate_idx: int = 0

        # Discovery bookkeeping
        self._total_execs: int = 0
        self._total_discoveries: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register a mutation operator arm."""
        if name in self.op_index:
            return
        idx = len(self.operators)
        self.operators.append(name)
        self.op_index[name] = idx
        n = len(self.operators)
        # Initialise CMA-ES state on first arm
        if self._mean is None:
            self._mean = np.zeros(n, dtype=float)
            self._C = np.eye(n, dtype=float)
            self._pc = np.zeros(n, dtype=float)
            self._ps = np.zeros(n, dtype=float)
        else:
            # Grow mean / C / paths to match new arm count
            new_mean = np.zeros(n, dtype=float)
            new_mean[: self._mean.shape[0]] = self._mean
            self._mean = new_mean
            new_C = np.eye(n, dtype=float)
            old = self._C.shape[0]
            new_C[:old, :old] = self._C
            self._C = new_C
            new_pc = np.zeros(n, dtype=float)
            new_pc[: self._pc.shape[0]] = self._pc
            self._pc = new_pc
            new_ps = np.zeros(n, dtype=float)
            new_ps[: self._ps.shape[0]] = self._ps
            self._ps = new_ps
        # Start a fresh generation when arms change
        self._new_generation()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Sample an operator from the current CMA-ES candidate distribution.

        If the candidate pool is exhausted a new generation of ``pop_size``
        candidates is sampled from N(mean, σ²C) and each is softmax'd into
        a probability vector over *ops*.
        """
        if not self.operators or not ops:
            return ops[0] if ops else ""

        if not self._candidates or self._next_candidate_idx >= len(self._candidates):
            self._new_generation()

        candidate = self._candidates[self._next_candidate_idx]
        self._next_candidate_idx += 1
        op = _sample_operator(candidate["probs"], ops, self._rng)
        candidate["selected_op"] = op
        return op

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        success: bool,
        weight: float = 1.0,
    ) -> None:
        """Record the outcome for the most recently returned operator.

        Args:
            name: Operator name returned by ``select_op()``.
            success: Whether the mutation produced new coverage.
            weight: Surprisal-weighted reward in ``(0, 1]``.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        # Assign reward to the candidate that produced this op
        candidate = self._current_candidate()
        if candidate is not None and candidate.get("selected_op") == name:
            candidate["reward"] += weight if success else 0.0
            candidate["count"] += 1

        self._eval_count += 1
        if self._eval_count >= self.generation_size:
            self._update_cmaes()

    # ------------------------------------------------------------------
    # CMA-ES update
    # ------------------------------------------------------------------

    def _randn(self, n: int) -> np.ndarray:
        """Standard-normal samples via Box-Muller from the RandPool."""
        u = self._rng.random_list(2 * n)
        u1 = np.array(u[0::2], dtype=float)
        u2 = np.array(u[1::2], dtype=float)
        # Guard log(0) from the pool's uint32-backed uniform samples.
        u1 = np.maximum(u1, 1e-300)
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)

    def _new_generation(self) -> None:
        """Sample a fresh population of candidates from the current mean/C.

        Does NOT advance ``self._generation`` or reset ``self._eval_count``;
        those are advanced/reset in ``_update_cmaes()`` after the current
        batch has been fully evaluated and ranked.
        """
        n = len(self.operators)
        self._candidates = []
        self._next_candidate_idx = 0
        if self._mean is None or self._C is None:
            return
        # Cholesky factorisation of σ²C; fall back to σI on failure
        scale = self._sigma * math.sqrt(n)
        try:
            L = np.linalg.cholesky(self._C)
        except np.linalg.LinAlgError:
            L = np.eye(n, dtype=float)
        for _ in range(self.pop_size):
            z = self._randn(n)
            x = self._mean + scale * (L @ z)
            logits = x.tolist()
            probs = _softmax(logits)
            self._candidates.append(
                {
                    "logits": logits,
                    "probs": probs,
                    "reward": 0.0,
                    "count": 0,
                    "selected_op": None,
                }
            )

    def _current_candidate(self) -> dict[str, Any] | None:
        """Return the candidate that produced the most recent op, if any."""
        idx = self._next_candidate_idx - 1
        if 0 <= idx < len(self._candidates):
            return self._candidates[idx]
        return None

    def _update_cmaes(self) -> None:
        """Rank-μ CMA-ES update from the completed generation."""
        if self._mean is None or self._C is None:
            return
        n = len(self.operators)
        if n == 0:
            return
        self._eval_count = 0

        # Sort candidates by reward descending
        candidates = sorted(
            [c for c in self._candidates if c["count"] > 0],
            key=lambda c: c["reward"],
            reverse=True,
        )
        if not candidates:
            self._new_generation()
            return

        # μ elites, clamped
        mu = max(1, min(len(candidates), int(self.pop_size * self.elite_frac)))
        elites = candidates[:mu]
        mu_eff = max(1.0, sum(1.0 for _ in elites) ** 2 / max(1.0, sum(1.0 for _ in elites)))
        # Clamp effective selection intensity
        if mu_eff < 1.0:
            mu_eff = 1.0

        # Rank-μ weights: positive, sum to 1
        weights = [math.log(mu_eff + 0.5) - math.log(i + 1) for i in range(mu)]
        weights_sum = sum(weights)
        if weights_sum <= 0:
            weights = [1.0 / mu] * mu
            weights_sum = 1.0
        weights = [w / weights_sum for w in weights]

        # Mean update
        delta = np.zeros(n, dtype=float)
        for w, c in zip(weights, elites, strict=False):
            delta += w * (np.array(c["logits"], dtype=float) - self._mean)
        step = self.learning_rate_mean * self._sigma
        self._mean += step * delta

        # Evolution path for covariance
        self._pc = (1.0 - self.learning_rate_cov) * self._pc + math.sqrt(
            self.learning_rate_cov * (2.0 - self.learning_rate_cov) * mu_eff
        ) * (self._sigma * delta)

        # Covariance update (diagonal fallback for large n)
        use_diag = n > self.max_ops_before_diag_fallback
        if use_diag:
            self._C[np.diag_indices(n)] = np.maximum(
                self._C[np.diag_indices(n)]
                + self.learning_rate_cov * (self._pc**2 - self._C[np.diag_indices(n)]),
                self.cov_diag_min,
            )
        else:
            rank_mu = np.zeros((n, n), dtype=float)
            for w, c in zip(weights, elites, strict=False):
                y = np.array(c["logits"], dtype=float) - self._mean
                rank_mu += w * np.outer(y, y)
            self._C = (1.0 - self.learning_rate_cov) * self._C + self.learning_rate_cov * rank_mu
            # Ensure symmetric positive definite-ish
            self._C = (self._C + self._C.T) / 2.0
            self._C[np.diag_indices(n)] = np.maximum(self._C[np.diag_indices(n)], self.cov_diag_min)

        # CSA step-size update
        hsigma = 0.0
        if (
            np.linalg.norm(self._ps)
            / math.sqrt(1.0 - (1.0 - self.learning_rate_sigma) ** (2 * (self._eval_count or 1)))
            < (1.4 + 2.0 / (n + 1)) * n**0.5
        ):
            hsigma = 1.0
        self._ps = (1.0 - self.learning_rate_sigma) * self._ps + math.sqrt(
            self.learning_rate_sigma * (2.0 - self.learning_rate_sigma) * mu_eff
        ) * (delta / (self._sigma + 1e-12))
        self._sigma *= math.exp(
            (self.learning_rate_sigma / (1.0 - self.learning_rate_sigma))
            * (np.linalg.norm(self._ps) ** 2 / n - 1.0)
            * hsigma
        )
        self._sigma = max(self._sigma, 1e-12)

        self._generation += 1
        self._new_generation()

    # ------------------------------------------------------------------
    # Diagnostics / compatibility
    # ------------------------------------------------------------------

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Compatibility shim matching ``MonteCarloScheduler.bandit_stats``."""
        return {
            "_cmaes_global": (
                float(self._total_discoveries),
                float(self._total_execs - self._total_discoveries),
            )
        }

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_fitness(self) -> float:
        """Highest reward seen in the current or most recent generation."""
        if not self._candidates:
            return 0.0
        return max((c["reward"] for c in self._candidates), default=0.0)

    def convergence_stats(self) -> dict[str, Any]:
        """Return diagnostics for logging / reporting."""
        top_op = ""
        top_prob = 0.0
        if self.operators and self._mean is not None and self._C is not None:
            probs = _softmax(self._mean.tolist())
            idx = max(range(len(probs)), key=lambda i: probs[i])
            top_op = self.operators[idx] if idx < len(self.operators) else ""
            top_prob = probs[idx]
        return {
            "generation": self._generation,
            "sigma": round(self._sigma, 6),
            "top_op": top_op,
            "top_prob": round(top_prob, 4),
            "eval_count": self._eval_count,
            "total_execs": self._total_execs,
            "total_discoveries": self._total_discoveries,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise scheduler state for the fuzzer's state store."""
        return {
            "pop_size": self.pop_size,
            "generation_size": self.generation_size,
            "step_size": self.step_size,
            "elite_frac": self.elite_frac,
            "learning_rate_mean": self.learning_rate_mean,
            "learning_rate_cov": self.learning_rate_cov,
            "learning_rate_sigma": self.learning_rate_sigma,
            "cov_diag_min": self.cov_diag_min,
            "max_ops_before_diag_fallback": self.max_ops_before_diag_fallback,
            "operators": self.operators,
            "op_index": self.op_index,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "sigma": self._sigma,
            "C": self._C.tolist() if self._C is not None else None,
            "pc": self._pc.tolist() if self._pc is not None else None,
            "ps": self._ps.tolist() if self._ps is not None else None,
            "generation": self._generation,
            "eval_count": self._eval_count,
            "total_execs": self._total_execs,
            "total_discoveries": self._total_discoveries,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore scheduler state from the fuzzer's state store."""
        self.pop_size = int(data.get("pop_size", self.pop_size))
        self.generation_size = int(data.get("generation_size", self.generation_size))
        self.step_size = float(data.get("step_size", self.step_size))
        self.elite_frac = float(data.get("elite_frac", self.elite_frac))
        self.learning_rate_mean = float(data.get("learning_rate_mean", self.learning_rate_mean))
        self.learning_rate_cov = float(data.get("learning_rate_cov", self.learning_rate_cov))
        self.learning_rate_sigma = float(data.get("learning_rate_sigma", self.learning_rate_sigma))
        self.cov_diag_min = float(data.get("cov_diag_min", self.cov_diag_min))
        self.max_ops_before_diag_fallback = int(
            data.get("max_ops_before_diag_fallback", self.max_ops_before_diag_fallback)
        )
        self.operators = list(data.get("operators", []))
        self.op_index = {k: int(v) for k, v in data.get("op_index", {}).items()}
        mean = data.get("mean")
        self._mean = np.array(mean, dtype=float) if mean is not None else None
        self._sigma = float(data.get("sigma", self._sigma))
        C = data.get("C")
        self._C = np.array(C, dtype=float) if C is not None else None
        pc = data.get("pc")
        self._pc = np.array(pc, dtype=float) if pc is not None else None
        ps = data.get("ps")
        self._ps = np.array(ps, dtype=float) if ps is not None else None
        self._generation = int(data.get("generation", 0))
        self._eval_count = int(data.get("eval_count", 0))
        self._total_execs = int(data.get("total_execs", 0))
        self._total_discoveries = int(data.get("total_discoveries", 0))
        self._candidates = []
        self._next_candidate_idx = 0
        # Pre-sample next generation so select_op is immediately usable
        self._new_generation()
