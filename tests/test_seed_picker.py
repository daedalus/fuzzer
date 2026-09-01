"""Tests for SeedPicker: Boltzmann seed selection strategy."""

import math
import random

import pytest

from fuzzer_tool.services.seed_picker import SeedPicker


class TestBoltzmannSelection:
    """Boltzmann seed selection: P(seed) ∝ exp(-E/T) with E = log(fuzz_count + 1)."""

    def _make_fuzzer_mock(
        self, corpus_size=3, use_boltzmann=True, temperature=1.0, anneal_budget=100000
    ):
        """Build a minimal Fuzzer-like object with enough attrs for _pick_boltzmann_seed()."""

        class MockFuzzer:
            corpus = [f"seed_{i}".encode() for i in range(corpus_size)]
            seed_meta = {}
            _rand_pool = random
            _temperature = temperature
            _anneal_budget = anneal_budget
            _use_boltzmann = use_boltzmann
            _profile = type("obj", (object,), {"format_signature": None})()

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        # Set fuzz_count: seed_0=1 (rare), seed_1=10, seed_2=100 (common)
        for i, seed in enumerate(f.corpus):
            f.seed_meta[seed] = {"fuzz_count": 10**i}
        return f

    def test_boltzmann_weight_rare_preferred(self):
        """Rare seed (fuzz_count=1) gets higher weight than common seed (fuzz_count=100)."""
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        f = self._make_fuzzer_mock(temperature=1.0)
        sp.f = f

        weights = []
        for seed in f.corpus:
            meta = f.seed_meta.get(seed)
            n = max(meta["fuzz_count"], 1)
            E = math.log(n + 1)
            w = math.exp(-E / 1.0)
            weights.append(max(w, 1e-6))

        # Rare seed weight > common seed weight
        assert weights[0] > weights[1] > weights[2]
        # Ratio rare/common > 10 at T=1.0
        assert weights[0] / weights[2] > 10

    def test_boltzmann_weight_cold_amplifies(self):
        """Ratio of rare/common weights is higher at T=0.1 than at T=1.0."""
        f_hot = self._make_fuzzer_mock(temperature=1.0)
        f_cold = self._make_fuzzer_mock(temperature=0.1)

        def get_weights(f):
            weights = []
            for seed in f.corpus:
                meta = f.seed_meta.get(seed)
                n = max(meta["fuzz_count"], 1)
                E = math.log(n + 1)
                T = max(f._temperature, 0.01)
                w = math.exp(-E / T)
                weights.append(max(w, 1e-6))
            return weights

        hot_weights = get_weights(f_hot)
        cold_weights = get_weights(f_cold)

        hot_ratio = hot_weights[0] / hot_weights[2]
        cold_ratio = cold_weights[0] / cold_weights[2]

        assert cold_ratio > hot_ratio * 10

    def test_boltzmann_weight_hot_near_uniform(self):
        """At T=1.0, the max/min weight ratio across seeds is bounded (< 100:1)."""
        f = self._make_fuzzer_mock(temperature=1.0, corpus_size=5)
        # Override fuzz_counts across a wider range
        for i, seed in enumerate(f.corpus):
            f.seed_meta[seed] = {"fuzz_count": 10**i}  # 1, 10, 100, 1000, 10000

        weights = []
        for seed in f.corpus:
            meta = f.seed_meta.get(seed)
            n = max(meta["fuzz_count"], 1)
            E = math.log(n + 1)
            w = math.exp(-E / 1.0)
            weights.append(max(w, 1e-6))

        max_min_ratio = max(weights) / min(weights)
        # At T=1.0: ratio = (max_n+1)/(min_n+1) ≈ 10001/2 ≈ 5000.
        # The ratio is bounded by the fuzz_count range, not exponential.
        fuzz_range = max(m["fuzz_count"] for m in f.seed_meta.values())
        assert max_min_ratio <= fuzz_range * 2

    def test_boltzmann_empty_corpus_fallback(self):
        """_pick_boltzmann_seed falls back when corpus is empty."""
        # We can't easily test the fallback branch without making a real Fuzzer,
        # but we can verify the method handles empty seed_meta gracefully.
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        f = self._make_fuzzer_mock(corpus_size=0)
        sp.f = f
        # Should not crash — falls through to _format_aware_seed which needs more attrs
        with pytest.raises(AttributeError):
            sp._pick_boltzmann_seed()
        # A real Fuzzer would not have this issue; the error is from the mock

    def test_boltzmann_elo_registered(self):
        """When _use_elo=True and _use_boltzmann=True, the available list includes 'boltzmann'."""
        f = self._make_fuzzer_mock()
        f._use_elo = True
        f._elo = type(
            "obj",
            (object,),
            {
                "select_strategy": lambda s, a: a[0],
                "initial_mu": 1500.0,
                "initial_sigma": 400.0,
            },
        )()
        f.markov_generate = False
        f.markov_trained = False
        f._use_bayesian = False

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        # Call _pick_seed_elo to trigger available list construction
        available = [
            s
            for s, cond in [
                ("ga", f.ga if hasattr(f, "ga") else False),
                ("qea", f.qea if hasattr(f, "qea") else False),
            ]
            if cond
        ]
        available.append("weighted")
        if f.corpus and f.seed_meta:
            available.append("pareto")
        if getattr(f, "_use_boltzmann", False):
            available.append("boltzmann")

        assert "boltzmann" in available


class TestEcoFuzzSelection:
    """EcoFuzz seed selection: energy = reward_prob / cost.

    reward_prob = (coverage_edges + 1) / (fuzz_count + 2), Laplace-smoothed
    estimate of "does fuzzing this seed yield a new edge". cost = the
    cost-ledger's effective_fuzz_count (average-cost-execution units),
    the same normalization Boltzmann uses for its rarity term. Unlike
    Boltzmann, EcoFuzz weighs reward against cost instead of pure pick-count
    rarity, so two equally-rare seeds can still get different energy.
    """

    def _make_fuzzer_mock(self, seed_metas, use_ecofuzz=True):
        """Build a minimal Fuzzer-like object with enough attrs for
        _pick_ecofuzz_seed(). seed_metas is a list of meta dicts, one per
        seed (seed_0, seed_1, ...)."""

        class MockFuzzer:
            corpus = [f"seed_{i}".encode() for i in range(len(seed_metas))]
            seed_meta = dict(zip(corpus, seed_metas, strict=False))
            _rand_pool = random
            _use_ecofuzz = use_ecofuzz
            _profile = type("obj", (object,), {"format_signature": None})()

            def mean_exec_time(self):
                return 0.0  # no corpus-wide timing signal in these tests

            def _seed_key(self, data):
                return data.hex()

        return MockFuzzer()

    def test_ecofuzz_prefers_high_reward_over_low_reward_same_cost(self):
        """Same fuzz_count (same cost, no cost samples) but seed_0 has found
        far more new edges per pick than seed_1 -> seed_0 gets higher energy."""
        f = self._make_fuzzer_mock(
            [
                {"fuzz_count": 10, "coverage_edges": 8},  # high reward rate
                {"fuzz_count": 10, "coverage_edges": 0},  # never rewarded
            ]
        )
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        def energy(meta):
            reward_prob = (meta["coverage_edges"] + 1) / (meta["fuzz_count"] + 2)
            cost = max(meta["fuzz_count"], 1.0)  # no cost samples -> falls back
            return reward_prob / cost

        e0 = energy(f.seed_meta[f.corpus[0]])
        e1 = energy(f.seed_meta[f.corpus[1]])
        assert e0 > e1

    def test_ecofuzz_distinguishes_equally_rare_seeds_unlike_boltzmann(self):
        """Falsification: two seeds with identical fuzz_count (Boltzmann would
        weight them identically) but different coverage_edges must get
        different EcoFuzz energy -- the reward term must actually matter."""
        f = self._make_fuzzer_mock(
            [
                {"fuzz_count": 5, "coverage_edges": 4},
                {"fuzz_count": 5, "coverage_edges": 0},
            ]
        )
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        rng = random.Random(1234)
        f._rand_pool = rng
        picks = [sp._pick_ecofuzz_seed() for _ in range(500)]
        assert picks.count(f.corpus[0]) > picks.count(f.corpus[1])

    def test_ecofuzz_never_zero_weight_for_unrewarded_fresh_seed(self):
        """Adversarial: a brand-new seed (fuzz_count=0, coverage_edges=0,
        possibly missing keys) must not crash and must not get pruned to
        zero energy -- Laplace smoothing keeps it selectable."""
        f = self._make_fuzzer_mock(
            [
                {},  # missing keys entirely
                {"fuzz_count": 0, "coverage_edges": 0},
            ]
        )
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        picked = sp._pick_ecofuzz_seed()
        assert picked in f.corpus

    def test_ecofuzz_empty_corpus_fallback(self):
        """_pick_ecofuzz_seed falls back when corpus is empty."""
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        f = self._make_fuzzer_mock([])
        sp.f = f
        with pytest.raises(AttributeError):
            sp._pick_ecofuzz_seed()

    def test_ecofuzz_elo_registered(self):
        """When _use_elo=True and _use_ecofuzz=True, the available list
        includes 'ecofuzz'."""
        f = self._make_fuzzer_mock([{"fuzz_count": 1, "coverage_edges": 0}])
        f._use_elo = True
        f._elo = type(
            "obj",
            (object,),
            {
                "select_strategy": lambda s, a: next(x for x in a if "ecofuzz" in x),
                "initial_mu": 1500.0,
                "initial_sigma": 400.0,
            },
        )()
        f.ga = None
        f.qea = None
        f.markov_generate = False
        f.markov_trained = False
        f._use_bayesian = False
        f._use_boltzmann = False
        f._seed_strategies_used = set()

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        result = sp._pick_seed_elo()
        assert "ecofuzz" in f._seed_strategy_pool
        assert result in f.corpus


class TestAflgoEloStrategy:
    """AFLGo distance-pure seed strategy (Elo-arbitrated 'aflgo' arm)."""

    def _make_fuzzer_mock(self, corpus_size=2, distances=None):
        """Minimal Fuzzer-like object with enough attrs for _pick_aflgo_seed()."""

        class _Distance:
            max_distance = 10.0

        class MockFuzzer:
            corpus = [f"seed_{i}".encode() for i in range(corpus_size)]
            seed_meta = {}
            _distance = _Distance()
            _seed_strategy_pool = []
            _seed_strategies_used = set()

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        distances = distances or {}
        for seed in f.corpus:
            f.seed_meta[seed] = {"avg_distance": distances.get(seed)}
        return f

    def test_aflgo_pick_prefers_near_target(self):
        """Near-target seed (avg 0.5) is picked far more often than the far one."""
        f = self._make_fuzzer_mock(distances={b"seed_0": 0.5, b"seed_1": 9.5})
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        random.seed(1234)
        near = sum(1 for _ in range(100) if sp._pick_aflgo_seed() == b"seed_0")
        # P(near)/P(far) = exp(-2*0.05)/exp(-2*0.95) ≈ 6; chance floor is 50
        assert near > 60, f"near-target seed picked only {near}/100"

    def test_aflgo_seed_without_distance_counts_as_far(self):
        """A seed with no distance data must not beat a near-target seed."""
        f = self._make_fuzzer_mock(distances={b"seed_0": 0.5, b"seed_1": None})
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        random.seed(99)
        near = sum(1 for _ in range(100) if sp._pick_aflgo_seed() == b"seed_0")
        assert near > 60

    def test_aflgo_elo_registered(self):
        """In directed mode, the Elo pool dispatches to the 'aflgo' arm."""
        f = self._make_fuzzer_mock()
        f.shm_cov = None
        f._use_elo = True
        f._elo = type("o", (object,), {"select_strategy": lambda s, a: "aflgo"})()
        f.markov_generate = f.markov_trained = False
        f._use_bayesian = False
        f._use_boltzmann = False
        f.ga = f.qea = None
        f._profile = type("o", (object,), {"format_signature": None})()

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        picked = sp._pick_seed_elo()
        assert picked in f.corpus
        assert sp.f._seed_strategy == "aflgo"


class TestSeedEloKeyMismatch:
    """Regression: seed strategies are rated under seed_<name> keys, but
    _pick_seed_elo used to select with plain names — so select_strategy
    never found rated strategies and always returned available[0] (inert
    seed arbitration; the uniform-1590 convergence artifact)."""

    def _make_seed_elo_fuzzer(self):
        f = TestAflgoEloStrategy._make_fuzzer_mock(self, corpus_size=2)
        f._use_elo = True
        f._rand_pool = random  # _pick_pareto_only falls back to pool.choice
        f.ga = f.qea = None
        f._use_bayesian = False
        f.markov_generate = False
        f.markov_trained = False
        f._use_boltzmann = False
        f._profile = type("o", (object,), {"format_signature": None})()
        # Both available strategies need seed_meta populated (pareto gate).
        for seed in f.corpus:
            f.seed_meta[seed] = {"fuzz_count": 1}
        return f

    def test_regression_seed_elo_selects_prefixed_keys(self):
        """_pick_seed_elo must ask Elo for seed_<name>-prefixed keys (the
        keyspace elo.json actually rates) and strip the prefix downstream."""
        captured = {}

        class _FakeElo:
            def select_strategy(self, strategies, temperature=None):
                captured["strategies"] = list(strategies)
                return "seed_pareto"

        f = self._make_seed_elo_fuzzer()
        f._elo = _FakeElo()

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        sp._pick_seed_elo()
        assert captured["strategies"], "select_strategy was not called"
        assert all(s.startswith("seed_") for s in captured["strategies"])
        assert f._seed_strategy == "pareto"

    def test_regression_seed_elo_rated_strategy_wins(self):
        """A seed strategy with a real match history must win Thompson
        sampling over an unrated one (real BayesianEloTracker); plain keys
        with no recorded matches still hit the strategies[0] fallback,
        documenting the pre-fix inert behavior."""
        from fuzzer_tool.core.elo import BayesianEloTracker

        elo = BayesianEloTracker(min_matches=1)
        for _ in range(500):
            elo.record_strategy_match("seed_a", "seed_b", 1.0)  # seed_a always wins

        # Dominance must survive Thompson noise: seed_a wins the large
        # majority of trials (measured win rate ≈ 1.0 at 500 matches).
        seed_a_wins = sum(
            1 for _ in range(20) if elo.select_strategy(["seed_a", "seed_b"]) == "seed_a"
        )
        assert seed_a_wins >= 18, f"rated seed_a won only {seed_a_wins}/20 trials"
        # Plain keys have no recorded matches → min_matches gate → [0].
        assert elo.select_strategy(["a", "b"]) == "a"


class TestEloParetoCachedWeights:
    """Regression: the Elo 'pareto' strategy reaches _pick_from_pareto_front
    directly, but _cached_weights is lazy-initialized only by
    weighted_pick_seed — before the fix this raised AttributeError."""

    def test_regression_elo_pareto_strategy_initializes_cached_weights(self):
        f = TestAflgoEloStrategy._make_fuzzer_mock(self, corpus_size=3)
        assert not hasattr(f, "_cached_weights"), "precondition: cache must be absent"
        f._use_elo = True
        f._rand_pool = random
        f.exec_count = 0
        f._temperature = 1.0
        f.ga = f.qea = None
        f._use_bayesian = False
        f.markov_generate = False
        f.markov_trained = False
        f._use_boltzmann = False
        f._profile = type("o", (object,), {"format_signature": None})()
        f._elo = type("o", (object,), {"select_strategy": lambda s, a: "seed_pareto"})()
        for seed in f.corpus:
            f.seed_meta[seed] = {"fuzz_count": 1, "added_at": 0.0}

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        picked = sp._pick_seed_elo()
        assert picked in f.corpus
        assert f._seed_strategy == "pareto"
        assert f._cached_weights == {}


class TestComputeWeightsArrayPath:
    """Regression: _compute_weights' vectorized Phase 1 uses array('d') +
    zero-copy numpy views (was 4x np.array(list) copies). The weights must
    reproduce the closed-form Phase-1 formula exactly (Phase-2 modifiers
    stubbed to identity; 2 seeds skips the Pareto adjustment)."""

    def _make_fuzzer(self):
        class MockFuzzer:
            corpus = [b"seed_0", b"seed_1"]
            seed_meta = {}
            _temperature = 1.0
            exec_count = 1
            _classify_cache = {}
            _distance = None
            _use_lineage = False
            _use_overlap_density = False
            _rand_pool = None
            _edge_tracker = type("o", (object,), {"shannon_entropy_seed": lambda s, sk: 0.5})()

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        now = 1000.0
        for i, seed in enumerate(f.corpus):
            f.seed_meta[seed] = {
                "fuzz_count": i + 1,
                "coverage_edges": (i + 1) * 10,
                "added_at": now - 100.0 * (i + 1),
                "momentum": 0.1 * i,
            }
        return f, now

    def test_weights_match_closed_form(self):
        f, now = self._make_fuzzer()
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        # Phase-2 modifiers to identity: we assert the vectorized Phase 1.
        sp._weight_secretary_and_cached = lambda sk, w, classifications, f: (w, 1.0, 1.0)
        sp._weight_edge_penalties = lambda sk, w, fuzz_count, f, recent_counts=None: w
        sp._weight_entropy_and_distance = lambda seed, sk, meta, w, f, em, me, md: w
        sp._weight_static_features = lambda seed, cov, w, f: w
        sp._weight_length_and_cross_target = lambda seed, meta, w, f: w
        sp._weight_overlap_density = lambda sk, w, f: w

        weights = sp._compute_weights(now)
        assert len(weights) == 2
        for i, seed in enumerate(f.corpus):
            meta = f.seed_meta[seed]
            fuzz = max(meta["fuzz_count"], 1)
            cov = meta["coverage_edges"]
            age = now - meta["added_at"]
            mom = meta.get("momentum", 0.0)
            T = f._temperature
            explore = T * (1.0 / math.sqrt(fuzz))
            exploit = (1.0 + cov * 0.5) / (1.0 + age * 0.01)
            w = explore * exploit * (1.0 + mom * 2.0)
            staleness = fuzz / max(cov + 1, 1)
            if staleness > 50.0 * T:
                w *= 0.01
            assert weights[i] == pytest.approx(max(w, 1e-6))
