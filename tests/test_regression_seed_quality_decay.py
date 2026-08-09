"""Decay must return a stale seed to the prior, not to a degenerate Beta.

``alpha *= decay`` shrinks pseudocounts without bound. Below the prior,
Beta(eps, eps) puts nearly all its mass at 0 and 1, so Thompson sampling
stops discriminating between seeds -- the opposite of what decaying stale
evidence is supposed to buy.
"""

from __future__ import annotations

from fuzzer_tool.core.seed_quality import BayesianSeedQuality


class TestDecayFloor:
    def test_decay_converges_to_prior_not_zero(self):
        bsq = BayesianSeedQuality(prior_alpha=1.0, prior_beta=1.0, decay=0.5, decay_interval=1)
        bsq.init_seed("A")
        for _ in range(20):
            bsq.record_outcome("A", discovered=True)
        for _ in range(400):
            bsq.record_outcome("B", discovered=False)

        assert bsq._alpha["A"] >= 1.0
        assert bsq._beta["A"] >= 1.0
        # And it actually converges there rather than sitting on old evidence.
        assert bsq._alpha["A"] < 1.01

    def test_custom_prior_is_the_floor(self):
        bsq = BayesianSeedQuality(prior_alpha=2.0, prior_beta=5.0, decay=0.5, decay_interval=1)
        bsq.init_seed("A")
        for _ in range(200):
            bsq.record_outcome("A", discovered=False)
        assert bsq._alpha["A"] >= 2.0
        assert bsq._beta["A"] >= 5.0

    def test_decay_still_discounts_old_evidence(self):
        """The point of decay is preserved: recent evidence dominates."""
        bsq = BayesianSeedQuality(decay=0.9, decay_interval=10)
        bsq.init_seed("A")
        for _ in range(50):
            bsq.record_outcome("A", discovered=True)
        hot = bsq.posterior_mean("A")
        for _ in range(200):
            bsq.record_outcome("A", discovered=False)
        assert bsq.posterior_mean("A") < hot

    def test_posterior_remains_usable_after_long_decay(self):
        """A decayed seed must still lose to one with fresh good evidence.

        Under the old multiplicative decay both posteriors collapse toward
        Beta(eps, eps) and their means converge, so selection degenerates.
        """
        bsq = BayesianSeedQuality(decay=0.9, decay_interval=10)
        bsq.init_seed("stale")
        bsq.init_seed("fresh")
        for _ in range(500):
            bsq.record_outcome("stale", discovered=False)
        for _ in range(20):
            bsq.record_outcome("fresh", discovered=True)
        assert bsq.posterior_mean("fresh") > bsq.posterior_mean("stale") + 0.2
