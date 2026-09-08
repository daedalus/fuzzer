"""FPLScheduler: Follow Perturbed Leader bandit algorithm.

The Follow Perturbed Leader algorithm selects arms by adding random
perturbations to empirical mean rewards, then choosing the arm with
the highest perturbed score. With a time-decaying perturbation
schedule, FPL achieves O(log T) regret for stochastic bandits.

At each round t:
    For each arm i: score_i = empirical_mean_i + Z_i / sqrt(t)
    where Z_i ~ Exp(1) (exponential distribution)
    Select arm with highest score_i

The perturbation decays as 1/sqrt(t), ensuring exploration early
and exploitation later -- the standard schedule for FPL in the
stochastic bandit setting.

References:
- Kalai and Vempala, "Efficient Algorithms for Online Decision Problems"
  (Journal of Computer and System Sciences, 2005)
- Abernethy, Hazan, Rakhlin, "Competing in the Dark: An Efficient
  Algorithm for Bandit Linear Optimization" (COLT 2008)
"""

import math

from fuzzer_tool.core.rand_pool import RandPool


class FPLScheduler:
    """Follow Perturbed Leader (FPL) bandit for mutation operators.

    Args:
        epsilon: Perturbation scale multiplier. Default 1.0 uses the
                 standard Exp(1)/sqrt(t) schedule. Higher values
                 increase early exploration; lower values focus on
                 exploitation sooner.
        rng: Shared ``RandPool`` (Hard Rule 16). Used for generating
             exponential perturbations and tie-breaking.
    """

    # FPL does not use Beta-Bernoulli priors, so it does not support priors
    supports_priors = False

    def __init__(
        self,
        epsilon: float = 1.0,
        rng: RandPool | None = None,
    ):
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}")

        self.epsilon = epsilon
        # Hard Rule 16: all randomness comes from RandPool for reproducibility
        self._rng = rng if rng is not None else RandPool()

        # Empirical mean rewards and pull counts for each arm
        self._mu: dict[str, float] = {}  # empirical mean reward
        self._n: dict[str, int] = {}  # pull count
        self._total_pulls: int = 0

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator with zero empirical mean and count."""
        self._mu.setdefault(name, 0.0)
        self._n.setdefault(name, 0)

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select operator via Follow Perturbed Leader.

        Arms with zero count are opened first to ensure initial
        exploration. For arms with data, we add decaying exponential
        perturbations to empirical means and select the highest-
        scoring arm.
        """
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        # First, try any unpulled arms (standard bandit initialization)
        for op in ops:
            if self._n.get(op, 0) == 0:
                return op

        # All arms have been pulled at least once - apply decaying perturbations
        # Perturbation scale decays as epsilon / sqrt(total_pulls)
        # This ensures exploration early and convergence later
        scale = self.epsilon / math.sqrt(max(self._total_pulls, 1))

        best_op = ops[0]
        best_score = -math.inf

        for op in ops:
            # Empirical mean (guaranteed to exist since n[op] > 0)
            mean = self._mu[op]

            # Generate exponential perturbation: scale * Exp(1)
            # Using RandPool for reproducibility (Hard Rule 16)
            u = self._rng.random()
            # Exponential distribution: -ln(U) where U ~ Uniform(0,1)
            perturbation = -math.log(max(u, 1e-100)) * scale

            score = mean + perturbation

            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update empirical mean incrementally.

        Uses sample-average update:
            mu_new = mu_old + (reward - mu_old) / (n + 1)
        """
        self._total_pulls += 1
        reward = weight if success else 0.0

        n = self._n.get(name, 0)
        mu_current = self._mu.get(name, 0.0)

        # Incremental update: mu = mu + (reward - mu) / (n + 1)
        self._mu[name] = mu_current + (reward - mu_current) / (n + 1)
        self._n[name] = n + 1

    # -- diagnostics ------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Return FPL diagnostics."""
        return {
            "fpl_pulls": self._total_pulls,
            "fpl_arms": len(self._mu),
        }
