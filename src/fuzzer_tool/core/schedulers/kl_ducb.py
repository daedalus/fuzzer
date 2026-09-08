"""KL_DUCBScheduler: KL-UCB over discounted statistics.

Same discounted structure as ``DUCBScheduler`` (Garivier & Moulines),
but the confidence width is the empirical-Bernoulli KL upper bound
instead of the Gaussian form. For Bernoulli rewards the KL bound is
tighter, so it explores less and exploits the best arm more aggressively.
"""

import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound

RENORM_FLOOR = 1e-12
MIN_LOG_ARG = 1.0 + 1e-9


class KL_DUCBScheduler:
    """Discounted UCB with KL-UCB confidence bound.

    Args:
        gamma: Discount factor per record, in (0, 1].
        xi: Exploration constant inside the KL budget
            ``xi * log(n) / N``.
        b: Reward range (unused by the KL width, kept for API parity).
        exploration: Multiplier on the whole confidence width.
            Applied to the KL bound as-is -- the KL theorem uses a
            leading constant of 1, so ``exploration`` is the tuning
            knob here.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        gamma: float = 0.9999,
        xi: float = 0.6,
        b: float = 1.0,
        exploration: float = 0.25,
        rng: RandPool | None = None,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")
        if exploration <= 0.0:
            raise ValueError(f"exploration must be positive, got {exploration!r}")

        self.gamma = gamma
        self.xi = xi
        self.b = b
        self.exploration = exploration
        self._rng = rng if rng is not None else RandPool()

        self._n_rel: dict[str, float] = {}
        self._x_rel: dict[str, float] = {}
        self._discount: float = 1.0
        self._total_pulls: int = 0

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator with zero discounted count and reward."""
        self._n_rel.setdefault(name, 0.0)
        self._x_rel.setdefault(name, 0.0)

    def _renormalise(self) -> None:
        """Fold the accumulated discount back into the per-arm statistics."""
        d = self._discount
        for k in self._n_rel:
            self._n_rel[k] *= d
        for k in self._x_rel:
            self._x_rel[k] *= d
        self._discount = 1.0

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the operator with the highest discounted-KL-UCB index."""
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        d = self._discount

        n_total = 0.0
        unpulled = []
        for op in ops:
            n = self._n_rel.get(op, 0.0) * d
            if n <= 0.0:
                unpulled.append(op)
                continue
            n_total += n

        if unpulled:
            return self._rng.choice(unpulled)

        log_n = math.log(max(n_total, MIN_LOG_ARG))
        width_scale = self.exploration * log_n

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._n_rel.get(op, 0.0) * d
            mean = (self._x_rel.get(op, 0.0) * d) / n
            score = mean + self._width(mean, n, log_n, width_scale)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    def _width(self, mean: float, n: float, log_n: float, width_scale: float) -> float:
        """KL-UCB confidence width: smallest q >= mean with KL(mean||q) >= xi*log(n)/n."""
        return kl_upper_bound(mean, self.xi * log_n / n) - mean

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Discount every arm, then credit *name* with this round's reward."""
        self._total_pulls += 1
        reward = weight if success else 0.0

        if self.gamma < 1.0:
            self._discount *= self.gamma
            if self._discount < RENORM_FLOOR:
                self._renormalise()

        inv = 1.0 / self._discount
        self._n_rel[name] = self._n_rel.get(name, 0.0) + inv
        if reward:
            self._x_rel[name] = self._x_rel.get(name, 0.0) + reward * inv

    # -- diagnostics ------------------------------------------------------

    def discounted_counts(self) -> dict[str, float]:
        """Per-arm discounted pull count N_t(i), in absolute units."""
        d = self._discount
        return {k: v * d for k, v in self._n_rel.items()}

    def discounted_means(self) -> dict[str, float]:
        """Per-arm discounted empirical mean X_t(i)/N_t(i)."""
        d = self._discount
        out = {}
        for k, n_rel in self._n_rel.items():
            n = n_rel * d
            out[k] = (self._x_rel.get(k, 0.0) * d / n) if n > 0 else 0.0
        return out

    def bandit_stats(self) -> dict:
        """Return KL-D-UCB diagnostics."""
        counts = self.discounted_counts()
        return {
            "kl_ducb_pulls": self._total_pulls,
            "kl_ducb_effective_n": round(sum(counts.values()), 3),
            "kl_ducb_arms": len(self._n_rel),
        }
