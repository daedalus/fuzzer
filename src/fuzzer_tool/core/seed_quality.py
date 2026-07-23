"""Bayesian seed quality estimation via Beta-Bernoulli posterior.

Each seed maintains a Beta(alpha, beta) posterior over its probability of
generating new coverage (or a crash) when mutated. Selection uses Thompson
sampling — draw from each seed's posterior and pick the highest draw — which
naturally balances exploration (seeds with few observations have wide
posteriors) and exploitation (seeds with high historical success rates have
posteriors concentrated at high values).

Optional hierarchical pooling shrinks individual posteriors toward the
population mean, sharing statistical strength across seeds with few
observations.

Usage:
    bsq = BayesianSeedQuality()
    bsq.init_seed("seed_a")
    bsq.record_outcome("seed_a", discovered=True)
    bsq.record_outcome("seed_a", discovered=False)
    sample = bsq.posterior_sample("seed_a")  # ~Beta(2,2) → ≈0.5
    chosen = bsq.select_seed(["seed_a", "seed_b"])  # Thompson draw
"""

from __future__ import annotations

import random


# Minimum parameter floor to avoid degenerate Beta(0, 0)
MIN_BETA_PARAM = 1e-6


class BayesianSeedQuality:
    """Beta-Bernoulli posterior for seed quality with optional hierarchical pooling.

    Each seed's success probability θ is modeled as:
        θ ~ Beta(alpha, beta)
        y_i | θ ~ Bernoulli(θ)

    With the default Beta(1, 1) prior, the posterior is:
        θ | data ~ Beta(1 + successes, 1 + failures)

    When hierarchical_pooling > 0, individual posteriors are shrunk toward a
    population-level prior estimated from all seeds' aggregated outcomes.

    Args:
        prior_alpha: Prior alpha (pseudocount of successes). Default 1.0.
        prior_beta: Prior beta (pseudocount of failures). Default 1.0.
        decay: Exponential decay factor applied periodically (< 1.0 for
            non-stationary). 1.0 = no decay (fully stationary). Default 1.0.
        decay_interval: Number of observations between decay applications.
            Default 500.
        hierarchical_pooling: Shrinkage strength (0.0 = no pooling, 1.0 = full
            pooling toward population mean). Default 0.0.
    """

    # Declares that this class supports informative priors, matching the
    # convention used by MonteCarloScheduler.
    supports_priors = True

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay: float = 1.0,
        decay_interval: int = 500,
        hierarchical_pooling: float = 0.0,
    ):
        self._prior_alpha = max(prior_alpha, MIN_BETA_PARAM)
        self._prior_beta = max(prior_beta, MIN_BETA_PARAM)
        self._decay = decay
        self._decay_interval = decay_interval

        # Clamp hierarchical pooling to [0, 1]
        self._hierarchical_pooling = max(0.0, min(1.0, hierarchical_pooling))

        # Per-seed posterior parameters
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}

        # Total observations for decay scheduling
        self._total_observations = 0

        # Pooled counts for hierarchical shrinkage
        self._pooled_successes = 0
        self._pooled_failures = 0

    def init_seed(
        self,
        seed_id: str,
        prior_alpha: float | None = None,
        prior_beta: float | None = None,
    ) -> None:
        """Register a seed, optionally overriding the default prior.

        The prior only applies at first registration — subsequent calls for
        an already-registered seed are no-ops (idempotent). This lets seeds
        discovered with high confidence (e.g., format-valid seeds) start with
        a more informative prior.

        Args:
            seed_id: Unique identifier for the seed (typically a content hash).
            prior_alpha: Override prior alpha. None = use instance default.
            prior_beta: Override prior beta. None = use instance default.
        """
        if seed_id in self._alpha:
            return
        self._alpha[seed_id] = max(
            prior_alpha if prior_alpha is not None else self._prior_alpha,
            MIN_BETA_PARAM,
        )
        self._beta[seed_id] = max(
            prior_beta if prior_beta is not None else self._prior_beta,
            MIN_BETA_PARAM,
        )

    def record_outcome(self, seed_id: str, discovered: bool, weight: float = 1.0) -> None:
        """Record whether mutating this seed produced new coverage (or a crash).

        Optionally applies exponential decay to all seeds periodically,
        giving recent evidence more weight (non-stationary bandit).

        Args:
            seed_id: Seed identifier (must already be registered).
            discovered: True if the mutation produced new coverage or a crash.
            weight: Reward weight (default 1.0). Surprisal-weighted calls
                pass a value in (0, 1] proportional to discovery rarity.
        """
        if seed_id not in self._alpha:
            # Auto-register with default prior if not yet known
            self.init_seed(seed_id)

        self._total_observations += 1

        # Periodic decay for non-stationarity
        if (
            self._decay < 1.0
            and self._decay_interval > 0
            and self._total_observations % self._decay_interval == 0
        ):
            for k in list(self._alpha):
                self._alpha[k] *= self._decay
                self._beta[k] *= self._decay

        # Update posterior
        if discovered:
            self._alpha[seed_id] += weight
            self._pooled_successes += weight
        else:
            self._beta[seed_id] += 1.0
            self._pooled_failures += 1.0

    def _get_pooled_params(self, seed_id: str) -> tuple[float, float]:
        """Get posterior parameters with optional hierarchical shrinkage applied.

        When hierarchical_pooling > 0, shrinks the individual seed's posterior
        toward the population mean, sharing statistical strength across seeds.

        Returns (alpha_eff, beta_eff) for the given seed.
        """
        if seed_id not in self._alpha:
            return self._prior_alpha, self._prior_beta

        alpha_i = self._alpha[seed_id]
        beta_i = self._beta[seed_id]

        if self._hierarchical_pooling > 0:
            h = self._hierarchical_pooling
            pooled_total = self._pooled_successes + self._pooled_failures
            if pooled_total > 0:
                pooled_alpha = self._prior_alpha + self._pooled_successes
                pooled_beta = self._prior_beta + self._pooled_failures
                alpha_i = (1 - h) * alpha_i + h * pooled_alpha
                beta_i = (1 - h) * beta_i + h * pooled_beta

        return alpha_i, beta_i

    def posterior_sample(self, seed_id: str) -> float:
        """Draw a single Thompson sample from the seed's posterior.

        Returns a random sample from Beta(alpha, beta), which represents a
        plausible value for the seed's true success probability given observed
        data. Seeds with wide posteriors (few observations) produce a wider
        range of samples, naturally driving exploration.

        When hierarchical_pooling > 0, the individual posterior is shrunk
        toward the population mean — see _get_pooled_params() for details.

        Args:
            seed_id: Seed identifier.

        Returns:
            A float in (0, 1) drawn from the posterior.
        """
        a, b = self._get_pooled_params(seed_id)
        return random.betavariate(a, b)

    def select_seed(self, seed_ids: list[str]) -> str:
        """Select a seed via Thompson sampling.

        Draws one sample from each seed's posterior and returns the seed with
        the highest draw. This is the standard Thompson sampling policy for
        the multi-armed bandit formulation of seed selection.

        Args:
            seed_ids: List of candidate seed identifiers.

        Returns:
            The selected seed identifier.
        """
        if not seed_ids:
            msg = "Cannot select from empty seed list"
            raise ValueError(msg)
        if len(seed_ids) == 1:
            return seed_ids[0]

        samples = [(sid, self.posterior_sample(sid)) for sid in seed_ids]
        return max(samples, key=lambda x: x[1])[0]

    def posterior_mean(self, seed_id: str) -> float:
        """Return the posterior mean (expected success probability).

        This is a deterministic point estimate: alpha / (alpha + beta).
        When hierarchical_pooling > 0, includes shrinkage toward the
        population mean.

        Useful for diagnostics and logging. NOT used by Thompson sampling
        (which draws a random sample to preserve exploration).

        Args:
            seed_id: Seed identifier.

        Returns:
            Float in (0, 1) — the Beta distribution mean.
        """
        a, b = self._get_pooled_params(seed_id)
        return a / (a + b)

    def posterior_variance(self, seed_id: str) -> float:
        """Return the posterior variance of the success probability.

        Measures uncertainty: higher variance = less evidence about this seed.
        Useful for diagnostics (which seeds are most uncertain).

        Args:
            seed_id: Seed identifier.

        Returns:
            Float — Beta distribution variance.
        """
        a, b = self._get_pooled_params(seed_id)
        total = a + b
        return (a * b) / (total * total * (total + 1))

    @property
    def population_mean(self) -> float:
        """Population-level mean success probability across all seeds."""
        total_alpha = self._pooled_successes + self._prior_alpha * max(len(self._alpha), 1)
        total_beta = self._pooled_failures + self._prior_beta * max(len(self._alpha), 1)
        if total_alpha + total_beta == 0:
            return 0.5
        return total_alpha / (total_alpha + total_beta)

    @property
    def n_seeds(self) -> int:
        """Number of registered seeds."""
        return len(self._alpha)

    @property
    def total_observations(self) -> int:
        """Total number of record_outcome calls across all seeds."""
        return self._total_observations

    def state_dict(self) -> dict:
        """Serialize state for persistence (e.g., in state.json)."""
        return {
            "version": 1,
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "alpha": dict(self._alpha),
            "beta": dict(self._beta),
            "total_observations": self._total_observations,
            "pooled_successes": self._pooled_successes,
            "pooled_failures": self._pooled_failures,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore serialized state."""
        self._prior_alpha = state.get("prior_alpha", self._prior_alpha)
        self._prior_beta = state.get("prior_beta", self._prior_beta)
        self._alpha.update(state.get("alpha", {}))
        self._beta.update(state.get("beta", {}))
        self._total_observations = state.get("total_observations", 0)
        self._pooled_successes = state.get("pooled_successes", 0)
        self._pooled_failures = state.get("pooled_failures", 0)
