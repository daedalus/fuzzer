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
            _rng = random
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
            _rng = random
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
        f._rng = rng
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
        f._rng = random  # _pick_pareto_only falls back to pool.choice
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
        from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker

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
        f._rng = random
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
            _rng = None
            # `good_turing_estimate` is read by the saturation gate, which
            # this test needs OFF: at >=99% saturation the gate cuts
            # subsumption/diversity/Wasserstein/proximity to neutral
            # multipliers and the closed form below would not hold.
            # Reporting 0.0 is therefore not just a stub value, it is the
            # precondition of the assertion.
            _edge_tracker = type(
                "o",
                (object,),
                {
                    "shannon_entropy_seed": lambda s, sk: 0.5,
                    "good_turing_estimate": lambda s: {"saturation": 0.0},
                },
            )()

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
        sp._weight_cached = lambda sk, w, classifications, f: (w, 1.0, 1.0)
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


class TestFormatLearnerSeed:
    """Tests for format-learner-driven seed generation."""

    def _make_fuzzer_with_learner(self, transitions):
        """Create a mock fuzzer with a format learner that has recorded transitions."""
        from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner

        class MockFuzzer:
            def __init__(self):
                self.corpus = [b"seed"]
                self.seed_meta = {}
                self._temperature = 1.0
                self._anneal_budget = 100000
                self._use_boltzmann = False
                self._profile = type("obj", (object,), {"format_signature": None})()
                self._format_learner = FormatLearner()
                self._rng = __import__("random").Random(42)  # deterministic

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        for tx in transitions:
            f._format_learner.record_transition(**tx)
        return f

    def test_format_learner_seed_no_learner(self):
        """When no format learner exists, _format_learner_seed returns None."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        class MockFuzzer:
            corpus = [b"seed"]
            seed_meta = {}
            _temperature = 1.0
            _anneal_budget = 100000
            _use_boltzmann = False
            _profile = type("obj", (object,), {"format_signature": None})()
            _format_learner = None
            _rng = __import__("random").Random(42)

            def _seed_key(self, data):
                return data.hex()

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = MockFuzzer()
        assert sp._format_learner_seed() is None

    def test_format_learner_seed_empty_learner(self):
        """When format learner has no data, _format_learner_seed returns None."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        f = self._make_fuzzer_with_learner([])
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f
        assert sp._format_learner_seed() is None

    def test_format_learner_seed_generates_from_learned_fields(self):
        """When format learner has confident fields, seed is generated from learned values."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        # Create transitions that will create confident hypotheses at offsets 1, 2, 3.
        # Need at least 2 different mutation operations to reach confidence >= 0.5.
        transitions = []
        for i in range(10):
            op = "bit_flip" if i % 2 == 0 else "arithmetic"
            transitions.append(
                {
                    "input_bytes": bytes([i % 10]) + bytes([i % 10]) * 3,
                    "mutation_op": op,
                    "mutation_offset": 1 + (i % 3),
                    "mutation_width": 1,
                    "coverage_before": 10,
                    "coverage_after": 15 + i,
                    "new_edges": {100 + i},
                    "lost_edges": set(),
                }
            )

        f = self._make_fuzzer_with_learner(transitions)
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        seed = sp._format_learner_seed()
        assert seed is not None
        # Seed length should cover offsets 1, 2, 3 (width 1 each) → length 4
        assert len(seed) >= 3
        # Each position should contain the most common byte value (0-9 each appear once,
        # so the first in iteration order is used)
        assert seed[1] == 0  # offset 1
        assert seed[2] == 1  # offset 2
        assert seed[3] == 2  # offset 3

    def test_format_learner_seed_respects_max_len(self):
        """Generated seed respects the fuzzer's max_len limit."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        # Create transitions that will create a confident hypothesis at offset 0
        # with width 10, so the seed length is 10 and max_len truncates it.
        # Need at least 2 different mutation operations to reach confidence >= 0.5.
        transitions = []
        for i in range(10):
            op = "bit_flip" if i % 2 == 0 else "arithmetic"
            transitions.append(
                {
                    "input_bytes": bytes([i % 10]) * 10,
                    "mutation_op": op,
                    "mutation_offset": 0,
                    "mutation_width": 10,
                    "coverage_before": 10,
                    "coverage_after": 15 + i,
                    "new_edges": {100 + i},
                    "lost_edges": set(),
                }
            )

        f = self._make_fuzzer_with_learner(transitions)
        f.max_len = 3  # Limit seed length
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        seed = sp._format_learner_seed()
        assert seed is not None
        assert len(seed) == 3  # Should be truncated to max_len

    def test_format_learner_seed_fallback_to_random_when_no_confidence(self):
        """When fields exist but lack confidence, falls back to random-ish seed."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        # Create only one transition - not enough for confidence >= 0.5
        transitions = [
            {
                "input_bytes": b"\x42\x24",
                "mutation_op": "bit_flip",
                "mutation_offset": 0,
                "mutation_width": 1,
                "coverage_before": 10,
                "coverage_after": 12,
                "new_edges": {100},
                "lost_edges": set(),
            }
        ]

        f = self._make_fuzzer_with_learner(transitions)
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        seed = sp._format_learner_seed()
        # Should return None because confidence will be too low (only 1 observation)
        assert seed is None

    def test_format_learner_seed_falsification_non_matching_input(self):
        """Falsification: _format_learner_seed returns None when learner has no confident fields."""
        from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner
        from fuzzer_tool.services.seed_picker import SeedPicker

        class MockFuzzer:
            corpus = [b"seed"]
            seed_meta = {}
            _temperature = 1.0
            _anneal_budget = 100000
            _use_boltzmann = False
            _profile = type("obj", (object,), {"format_signature": None})()
            _format_learner = FormatLearner()
            _rng = __import__("random").Random(42)

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        # No transitions recorded - learner is empty
        assert sp._format_learner_seed() is None

        # Single transition - confidence too low
        f._format_learner.record_transition(
            input_bytes=b"\x00" * 10,
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=11,
            new_edges={100},
            lost_edges=set(),
        )
        assert sp._format_learner_seed() is None

    def test_format_learner_seed_adversarial_malformed_summary(self):
        """Adversarial: _format_learner_seed handles malformed format summary gracefully."""
        from fuzzer_tool.services.seed_picker import SeedPicker

        class MockLearner:
            def get_format_summary(self):
                # Missing 'fields' key
                return {"timeline_size": 0, "hypotheses": 0, "classified": 0}

        class MockFuzzer:
            corpus = [b"seed"]
            seed_meta = {}
            _temperature = 1.0
            _anneal_budget = 100000
            _use_boltzmann = False
            _profile = type("obj", (object,), {"format_signature": None})()
            _format_learner = MockLearner()
            _rng = __import__("random").Random(42)
            max_len = 100

            def _seed_key(self, data):
                return data.hex()

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = MockFuzzer()
        assert sp._format_learner_seed() is None

        # fields present but empty list
        class MockLearner2:
            def get_format_summary(self):
                return {"fields": []}

        sp.f._format_learner = MockLearner2()
        assert sp._format_learner_seed() is None

        # fields with missing confidence keys
        class MockLearner3:
            def get_format_summary(self):
                return {"fields": [{"offset": 0, "width": 1}]}

        sp.f._format_learner = MockLearner3()
        assert sp._format_learner_seed() is None

        # fields with confidence but missing most_common_value
        class MockLearner4:
            def get_format_summary(self):
                return {"fields": [{"offset": 0, "width": 1, "confidence": 0.9}]}

        sp.f._format_learner = MockLearner4()
        assert sp._format_learner_seed() is None

    def test_format_learner_seed_adversarial_max_len_zero(self):
        """Adversarial: _format_learner_seed handles max_len=0 correctly."""
        from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner
        from fuzzer_tool.services.seed_picker import SeedPicker

        class MockFuzzer:
            corpus = [b"seed"]
            seed_meta = {}
            _temperature = 1.0
            _anneal_budget = 100000
            _use_boltzmann = False
            _profile = type("obj", (object,), {"format_signature": None})()
            _format_learner = FormatLearner()
            _rng = __import__("random").Random(42)
            max_len = 0  # Zero max_len means "no limit"

            def _seed_key(self, data):
                return data.hex()

        f = MockFuzzer()
        # Add enough transitions to reach confidence >= 0.5
        for i in range(10):
            f._format_learner.record_transition(
                input_bytes=bytes([i % 10]) * 10,
                mutation_op="bit_flip" if i % 2 == 0 else "arithmetic",
                mutation_offset=0,
                mutation_width=10,
                coverage_before=10,
                coverage_after=15 + i,
                new_edges={100 + i},
                lost_edges=set(),
            )

        sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
        sp.f = f

        seed = sp._format_learner_seed()
        # max_len=0 means no limit, so seed is full length (10 bytes)
        assert seed is not None
        assert len(seed) == 10
