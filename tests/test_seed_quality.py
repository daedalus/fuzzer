"""Tests for BayesianSeedQuality — Beta-Bernoulli posterior for seed selection."""

from fuzzer_tool.core.seed_quality import BayesianSeedQuality, MIN_BETA_PARAM


class TestBayesianSeedQuality:
    def test_init_defaults(self):
        bsq = BayesianSeedQuality()
        assert bsq.n_seeds == 0
        assert bsq.total_observations == 0
        assert bsq._prior_alpha == 1.0
        assert bsq._prior_beta == 1.0
        assert bsq._decay == 1.0
        assert bsq._hierarchical_pooling == 0.0

    def test_init_custom_prior(self):
        bsq = BayesianSeedQuality(prior_alpha=2.0, prior_beta=5.0)
        assert bsq._prior_alpha == 2.0
        assert bsq._prior_beta == 5.0

    def test_init_seed(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        assert bsq.n_seeds == 1
        assert bsq._alpha["seed_a"] == 1.0
        assert bsq._beta["seed_a"] == 1.0

    def test_init_seed_custom_prior(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a", prior_alpha=5.0, prior_beta=1.0)
        assert bsq._alpha["seed_a"] == 5.0
        assert bsq._beta["seed_a"] == 1.0

    def test_init_seed_idempotent(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a", prior_alpha=5.0, prior_beta=1.0)
        bsq.init_seed("seed_a")  # second call with different defaults — no-op
        assert bsq._alpha["seed_a"] == 5.0
        assert bsq._beta["seed_a"] == 1.0

    def test_record_outcome_auto_registers(self):
        bsq = BayesianSeedQuality()
        bsq.record_outcome("seed_a", discovered=True)
        assert bsq.n_seeds == 1
        assert bsq._alpha["seed_a"] == 2.0  # 1 (prior) + 1 (success)

    def test_record_success(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=True)
        assert bsq._alpha["seed_a"] == 2.0
        assert bsq._beta["seed_a"] == 1.0

    def test_record_failure(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=False)
        assert bsq._alpha["seed_a"] == 1.0
        assert bsq._beta["seed_a"] == 2.0

    def test_record_multiple_outcomes(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        for _ in range(3):
            bsq.record_outcome("seed_a", discovered=True)
        bsq.record_outcome("seed_a", discovered=False)
        # Posterior: Beta(1 + 3, 1 + 1) = Beta(4, 2)
        assert bsq._alpha["seed_a"] == 4.0
        assert bsq._beta["seed_a"] == 2.0

    def test_posterior_mean(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        for _ in range(3):
            bsq.record_outcome("seed_a", discovered=True)
        bsq.record_outcome("seed_a", discovered=False)
        # Beta(4, 2) mean = 4/6 ≈ 0.667
        mean = bsq.posterior_mean("seed_a")
        assert abs(mean - 4.0 / 6.0) < 1e-10

    def test_posterior_mean_unregistered_seed(self):
        bsq = BayesianSeedQuality(prior_alpha=2.0, prior_beta=2.0)
        mean = bsq.posterior_mean("unknown")
        assert mean == 0.5  # 2 / (2+2)

    def test_posterior_variance(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=True)
        # Beta(2, 1) variance = (2*1) / (3^2 * 4) = 2/36 ≈ 0.0556
        var = bsq.posterior_variance("seed_a")
        expected = (2.0 * 1.0) / (3.0 * 3.0 * 4.0)
        assert abs(var - expected) < 1e-10

    def test_posterior_sample_in_range(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=True)
        bsq.record_outcome("seed_a", discovered=False)
        for _ in range(100):
            sample = bsq.posterior_sample("seed_a")
            assert 0 < sample < 1

    def test_select_seed(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.init_seed("seed_b")
        bsq.record_outcome("seed_a", discovered=True)  # alpha=2, beta=1
        bsq.record_outcome("seed_b", discovered=False)  # alpha=1, beta=2
        result = bsq.select_seed(["seed_a", "seed_b"])
        assert result in ("seed_a", "seed_b")

    def test_select_seed_single(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("only_one")
        assert bsq.select_seed(["only_one"]) == "only_one"

    def test_select_seed_empty_raises(self):
        bsq = BayesianSeedQuality()
        try:
            bsq.select_seed([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_exploration_via_variance(self):
        """Seeds with fewer observations should have higher posterior variance."""
        bsq = BayesianSeedQuality()
        bsq.init_seed("heavily_tested")
        for _ in range(100):
            bsq.record_outcome("heavily_tested", discovered=True)
        bsq.init_seed("lightly_tested")
        bsq.record_outcome("lightly_tested", discovered=True)

        var_heavy = bsq.posterior_variance("heavily_tested")
        var_light = bsq.posterior_variance("lightly_tested")
        assert var_light > var_heavy  # less evidence → more uncertainty

    def test_surprisal_weighted_reward(self):
        """Weighted successes should have proportional effect on posterior."""
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=True, weight=0.3)
        # Posterior: Beta(1 + 0.3, 1) — fractional alpha update
        assert abs(bsq._alpha["seed_a"] - 1.3) < 1e-10

    def test_hierarchical_pooling_shrinks_toward_population(self):
        bsq = BayesianSeedQuality(
            prior_alpha=1.0,
            prior_beta=1.0,
            hierarchical_pooling=0.5,
        )
        # Seed A: 0 successes, 10 failures → poor performer
        bsq.init_seed("seed_a")
        for _ in range(10):
            bsq.record_outcome("seed_a", discovered=False)
        # Seed B-E: mostly successful
        for s in ("seed_b", "seed_c", "seed_d", "seed_e"):
            bsq.init_seed(s)
            for _ in range(10):
                bsq.record_outcome(s, discovered=True)

        # Pooled success rate is high (~50/54 ≈ 0.93)
        # With pooling, seed A's posterior should be higher than its raw Beta(1, 11) ≈ 0.083
        raw_mean = 1.0 / (1.0 + 11.0)  # without pooling
        pooled_mean = bsq.posterior_mean("seed_a")
        assert pooled_mean > raw_mean  # shrinkage toward high population mean
        assert pooled_mean < bsq.population_mean  # but not all the way

    def test_decay_reduces_old_evidence(self):
        bsq = BayesianSeedQuality(decay=0.5, decay_interval=10)
        bsq.init_seed("A")
        for _ in range(10):
            bsq.record_outcome("A", discovered=True)
        alpha_before = bsq._alpha["A"]
        # Next record triggers decay (total_observations = 11; 11 % 10 == 1)
        # Actually decay fires at interval boundaries — let's do 9 more to hit 20
        for _ in range(9):
            bsq.record_outcome("A", discovered=False)
        # Decay should have applied at total_observations = 10 and 20
        # 10 alpha records: 1 + 10 = 11 → *0.5 → 5.5
        # Then 10 beta records: +10 = 15.5 → *0.5 → 7.75
        # So alpha should have decayed
        assert bsq._alpha["A"] < 11.0, f"Expected alpha < 11.0, got {bsq._alpha['A']}"

    def test_state_dict_roundtrip(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("seed_a")
        bsq.record_outcome("seed_a", discovered=True)
        bsq.record_outcome("seed_a", discovered=False)
        bsq.record_outcome("seed_b", discovered=True)

        state = bsq.state_dict()
        assert state["version"] == 1
        assert "seed_a" in state["alpha"]
        assert "seed_b" in state["alpha"]
        assert state["total_observations"] == 3

        # Load into a fresh instance
        bsq2 = BayesianSeedQuality()
        bsq2.load_state_dict(state)
        assert bsq2.n_seeds == 2
        assert bsq2.posterior_mean("seed_a") == bsq.posterior_mean("seed_a")
        assert bsq2.total_observations == 3

    def test_init_seed_prior_clamped_positive(self):
        bsq = BayesianSeedQuality()
        bsq.init_seed("op", prior_alpha=0.0, prior_beta=-1.0)
        assert bsq._alpha["op"] >= MIN_BETA_PARAM
        assert bsq._beta["op"] >= MIN_BETA_PARAM

    def test_population_mean(self):
        bsq = BayesianSeedQuality()
        assert abs(bsq.population_mean - 0.5) < 1e-10  # no data → prior mean
        bsq.init_seed("a")
        bsq.record_outcome("a", discovered=True)  # pooled: 1 success
        bsq.record_outcome("a", discovered=True)  # pooled: 2 successes
        bsq.record_outcome("a", discovered=False)  # pooled: 1 failure
        assert bsq.population_mean > 0.5  # more successes than failures
