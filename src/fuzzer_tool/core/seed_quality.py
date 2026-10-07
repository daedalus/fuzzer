"""Bayesian seed quality estimation via Beta-Bernoulli posterior.

Each seed maintains a Beta(alpha, beta) posterior over its probability of
generating new coverage (or a crash) when mutated. Selection uses Thompson
sampling — draw from each seed's posterior and pick the highest draw — which
naturally balances exploration (seeds with few observations have wide
posteriors) and exploitation (seeds with high historical success rates have
posteriors concentrated at high values).

Optional hierarchical pooling adds an empirical-Bayes (BLUP) prior fitted
from the spread of all seeds' rates (core/blup.py), so seeds with few
observations borrow strength from the population and well-observed ones
keep their own rate.

Usage:
    bsq = BayesianSeedQuality()
    bsq.init_seed("seed_a")
    bsq.record_outcome("seed_a", discovered=True)
    bsq.record_outcome("seed_a", discovered=False)
    sample = bsq.posterior_sample("seed_a")  # ~Beta(2,2) → ≈0.5
    chosen = bsq.select_seed(["seed_a", "seed_b"])  # Thompson draw
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.blup import POOL_MAX_STRENGTH, PoolCache
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

# Minimum parameter floor to avoid degenerate Beta(0, 0)
MIN_BETA_PARAM = 1e-6

# Parameter value at which a Beta has a closed-form inverse CDF (see below).
_UNIT_PARAM = 1.0

# Observations between refits of the pooled prior (one pass over all seeds).
_POOL_REFIT_EVERY = 64


def _beta_sample(alpha: float, beta: float, rng: RandPool) -> float:
    """Draw from Beta(*alpha*, *beta*), by inverse CDF when one side is 1.

    Beta(a, 1) and Beta(1, b) invert in closed form::

        Beta(a, 1) == U**(1/a)            # max of a uniforms, integer a
        Beta(1, b) == 1 - (1-U)**(1/b)    # min of b uniforms, integer b

    which is one uniform draw and a pow, against the two gammavariate draws
    ``random.betavariate`` needs. It is not a general win -- the guard costs
    ~6% when it misses -- but a seed's alpha stays at the prior until that
    seed first discovers coverage, and per-seed discovery is rare: 83% of
    arms degenerate at 500 seeds, 95% at 2000, 99% at 8000.

    Entropy comes from ``rng`` on both branches, so ``--seed`` keeps
    determining the draw.
    """
    if alpha == _UNIT_PARAM:
        return 1.0 - (1.0 - rng.random()) ** (1.0 / beta)

    if beta == _UNIT_PARAM:
        return rng.random() ** (1.0 / alpha)

    return rng.betavariate(alpha, beta)


class BayesianSeedQuality:
    """Beta-Bernoulli posterior for seed quality with optional hierarchical pooling.

    Each seed's success probability θ is modeled as:
        θ ~ Beta(alpha, beta)
        y_i | θ ~ Bernoulli(θ)

    With the default Beta(1, 1) prior, the posterior is:
        θ | data ~ Beta(1 + successes, 1 + failures)

    When hierarchical_pooling > 0, each posterior gets an extra
    Beta(h m mu, h m (1 - mu)) prior: mu the population rate, m the strength
    the seeds' dispersion implies (core/blup.py).

    Args:
        prior_alpha: Prior alpha (pseudocount of successes). Default 1.0.
        prior_beta: Prior beta (pseudocount of failures). Default 1.0.
        decay: Exponential decay factor applied periodically (< 1.0 for
            non-stationary). 1.0 = no decay (fully stationary). Default 1.0.
            Decay pulls each posterior toward the prior, so a seed with
            stale evidence returns to "unknown" rather than to a degenerate
            Beta(eps, eps).
        decay_interval: Number of observations between decay applications.
            Default 500.
        hierarchical_pooling: Scale h on the fitted prior strength (0.0 = no
            pooling, 1.0 = the full BLUP prior). Default 0.0.
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
        rng: RandPool | None = None,
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
        self._pool = PoolCache(POOL_MAX_STRENGTH, _POOL_REFIT_EVERY)
        self._rng = rng or get_default_rand_pool()

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
            # Decay toward the *prior*, not toward zero.
            #
            # ``alpha *= decay`` shrinks the pseudocounts without bound: at
            # decay=0.99 and a 500-observation interval, a long campaign
            # drives every alpha and beta below the Beta(1,1) prior and on
            # toward 0. Beta(eps, eps) is not an uninformative posterior --
            # it is a nearly degenerate one, with essentially all its mass
            # at 0 and 1, so posterior_sample() returns a coin flip and
            # Thompson selection becomes uniform noise over the corpus.
            # Forgetting evidence should return a seed to "unknown", which
            # is the prior, not to "arbitrarily confident in both
            # directions".
            pa, pb = self._prior_alpha, self._prior_beta
            d = self._decay
            for k in list(self._alpha):
                self._alpha[k] = pa + (self._alpha[k] - pa) * d
                self._beta[k] = pb + (self._beta[k] - pb) * d
            # Pooled counts have no prior to fall back to and are only ever
            # used as a ratio, so they decay multiplicatively; this keeps
            # the population-mean target as responsive as the individual
            # posteriors it shrinks toward.
            self._pooled_successes *= d
            self._pooled_failures *= d

        # Update posterior
        if discovered:
            self._alpha[seed_id] += weight
            self._pooled_successes += weight
        else:
            self._beta[seed_id] += 1.0
            self._pooled_failures += 1.0

    def _get_pooled_params(self, seed_id: str) -> tuple[float, float]:
        """Get posterior parameters with optional hierarchical shrinkage applied.

        When hierarchical_pooling > 0, adds the population BLUP prior: a
        seed with n observations keeps weight n / (n + h m) on its own rate.
        No pooling while the fit is undefined (under two observed seeds, or
        a population rate of 0 or 1).

        Returns (alpha_eff, beta_eff) for the given seed.
        """
        if seed_id not in self._alpha:
            return self._prior_alpha, self._prior_beta

        alpha_i = self._alpha[seed_id]
        beta_i = self._beta[seed_id]
        if self._hierarchical_pooling <= 0:
            return alpha_i, beta_i

        prior = self._pool.prior(self._total_observations, self._evidence)
        if prior is None:
            return alpha_i, beta_i

        mu, m = prior
        hm = self._hierarchical_pooling * m
        return alpha_i + hm * mu, beta_i + hm * (1.0 - mu)

    def _evidence(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-seed (successes, observations), net of the default prior."""
        n = len(self._alpha)
        a = np.fromiter(self._alpha.values(), dtype=float, count=n)
        b = np.fromiter((self._beta[k] for k in self._alpha), dtype=float, count=n)
        succ = np.maximum(a - self._prior_alpha, 0.0)
        fail = np.maximum(b - self._prior_beta, 0.0)
        return succ, succ + fail

    def posterior_sample(self, seed_id: str) -> float:
        """Draw a single Thompson sample from the seed's posterior.

        Returns a random sample from Beta(alpha, beta), which represents a
        plausible value for the seed's true success probability given observed
        data. Seeds with wide posteriors (few observations) produce a wider
        range of samples, naturally driving exploration.

        When hierarchical_pooling > 0, the population BLUP prior is added —
        see _get_pooled_params() for details.

        Args:
            seed_id: Seed identifier.

        Returns:
            A float in (0, 1) drawn from the posterior.
        """
        a, b = self._get_pooled_params(seed_id)
        return _beta_sample(a, b, self._rng)

    def select_index(self, seed_ids: list[str]) -> int:
        """Return the *position* of the Thompson winner in *seed_ids*.

        Callers hold the corpus in the same order they built the id list, so
        the position is what they need. Returning the id instead forced them
        to hash the corpus a second time to find it again.

        Args:
            seed_ids: List of candidate seed identifiers.

        Returns:
            Index into *seed_ids* of the seed with the highest draw.
        """
        if not seed_ids:
            msg = "Cannot select from empty seed list"
            raise ValueError(msg)
        if len(seed_ids) == 1:
            return 0

        if self._hierarchical_pooling > 0:
            return self._pooled_index(seed_ids)

        best_i, best_v = 0, -1.0
        for i, sid in enumerate(seed_ids):
            v = self.posterior_sample(sid)
            if v > best_v:
                best_i, best_v = i, v

        return best_i

    def _pooled_index(self, seed_ids: list[str]) -> int:
        """Vectorised Thompson draw over the pooled posteriors.

        Same parameters as ``_get_pooled_params`` per seed, one prior lookup
        and one array draw instead of a Python loop of scalar draws.
        """
        n = len(seed_ids)
        alpha, beta = self._alpha, self._beta
        a = np.fromiter((alpha.get(s, self._prior_alpha) for s in seed_ids), dtype=float, count=n)
        b = np.fromiter((beta.get(s, self._prior_beta) for s in seed_ids), dtype=float, count=n)

        prior = self._pool.prior(self._total_observations, self._evidence)
        if prior is not None:
            # Unregistered seeds keep the bare default prior, as in _get_pooled_params.
            known = np.fromiter((s in alpha for s in seed_ids), dtype=bool, count=n)
            mu, m = prior
            hm = self._hierarchical_pooling * m
            a = a + known * (hm * mu)
            b = b + known * (hm * (1.0 - mu))

        draws = self._rng.betavariate_array(a, b)
        return int(np.argmax(draws))

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
        return seed_ids[self.select_index(seed_ids)]

    def posterior_mean(self, seed_id: str) -> float:
        """Return the posterior mean (expected success probability).

        This is a deterministic point estimate: alpha / (alpha + beta).
        When hierarchical_pooling > 0, includes the population BLUP prior.

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
