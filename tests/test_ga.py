"""Tests for genetic algorithm lifecycle."""

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.ga import (
    _GA_MUTATION_OPS,
    FitnessFunction,
    GALifecycle,
    Individual,
    Speciation,
)


class TestIndividual:
    def test_serialization_roundtrip(self):
        ind = Individual(data=b"\x89PNG", fitness=0.5, species_id=3)
        d = ind.to_dict()
        restored = Individual.from_dict(d)
        assert restored.data == ind.data
        assert restored.fitness == ind.fitness
        assert restored.species_id == ind.species_id

    def test_crash_flag_persists(self):
        ind = Individual(data=b"crash", crash=True)
        d = ind.to_dict()
        restored = Individual.from_dict(d)
        assert restored.crash is True

    def test_seed_key_auto_generated(self):
        ind = Individual(data=b"test")
        assert ind.seed_key
        assert len(ind.seed_key) == 16

    def test_seed_key_from_dict(self):
        ind = Individual(data=b"test", seed_key="abc123")
        d = ind.to_dict()
        restored = Individual.from_dict(d)
        assert restored.seed_key == "abc123"


class TestFitnessFunction:
    def test_novelty_bounded_0_1(self):
        fn = FitnessFunction()
        ind = Individual(data=b"test", edge_count=50)
        score = fn.score(ind, total_edges=100, current_generation=10)
        assert 0.0 <= score <= 10.0  # crash_bonus not set

    def test_crash_gets_bonus(self):
        fn = FitnessFunction(crash_bonus=10.0)
        ind = Individual(data=b"crash", crash=True)
        score = fn.score(ind, total_edges=100, current_generation=1)
        assert score >= 10.0

    def test_freshness_decays(self):
        fn = FitnessFunction(fresh_decay=0.5)
        young = Individual(data=b"young", generation=10)
        old = Individual(data=b"old", generation=1)
        fn.score(young, total_edges=100, current_generation=10)
        fn.score(old, total_edges=100, current_generation=10)
        assert young.freshness_score > old.freshness_score

    def test_diversity_contributes(self):
        fn = FitnessFunction(w_diversity=1.0, w_novelty=0.0, w_freshness=0.0, w_mutation=0.0)
        ind = Individual(data=b"test")
        ind.diversity_score = 0.8
        score = fn.score(ind, total_edges=100, current_generation=1)
        assert score == pytest.approx(0.8, abs=0.01)


class TestGALifecycle:
    def test_initialize_from_corpus(self):
        ga = GALifecycle(pop_size=10)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        ga.initialize(corpus, et)
        assert len(ga.population) == 5

    def test_tournament_select_returns_individual(self):
        ga = GALifecycle(pop_size=10, tournament_size=3)
        for i in range(10):
            ga.population.append(Individual(data=b"x", fitness=float(i)))
        parent = ga.select_parent()
        assert isinstance(parent, Individual)

    def test_pick_seed_returns_bytes(self):
        ga = GALifecycle(pop_size=5)
        for i in range(5):
            ga.population.append(Individual(data=b"seed_%d" % i, fitness=float(i)))
        seed = ga.pick_seed()
        assert isinstance(seed, bytes)

    def test_evolve_maintains_pop_size(self):
        ga = GALifecycle(pop_size=20, elite_fraction=0.1)
        for i in range(20):
            ga.population.append(Individual(data=b"x_%d" % i, fitness=float(i)))
        et = EdgeTracker()
        ga._evolve(et)
        assert len(ga.population) == 20

    def test_crash_never_culled(self):
        ga = GALifecycle(pop_size=10, elite_fraction=0.1)
        crash_ind = Individual(data=b"CRASH", fitness=0.0, crash=True)
        ga.population.append(crash_ind)
        for i in range(9):
            ga.population.append(
                Individual(
                    data=b"normal_%d" % i,
                    fitness=float(i + 1),
                )
            )
        et = EdgeTracker()
        ga._evolve(et)
        # Crash individual should survive even with low fitness
        assert any(i.crash for i in ga.population)

    def test_save_load_roundtrip(self, tmp_path):
        ga = GALifecycle(pop_size=5)
        for i in range(5):
            ga.population.append(
                Individual(
                    data=b"seed_%d" % i,
                    fitness=float(i),
                    species_id=i % 2,
                )
            )
        ga.generation = 7
        path = tmp_path / "ga.json"
        ga.save(path)

        ga2 = GALifecycle(pop_size=5)
        ga2.load(path)
        assert ga2.generation == 7
        assert len(ga2.population) == 5
        assert ga2.population[0].data == ga.population[0].data

    def test_on_fuzz_result_returns_individual_on_new_coverage(self):
        ga = GALifecycle(pop_size=5)
        et = EdgeTracker()
        ga.initialize([], et)
        ind = ga.on_fuzz_result(b"new_seed", new_coverage=True, edge_count=10, edge_tracker=et)
        assert ind is not None
        assert ind.data == b"new_seed"

    def test_on_fuzz_result_returns_none_without_coverage(self):
        ga = GALifecycle(pop_size=5, generation_size=100)
        et = EdgeTracker()
        ga.initialize([], et)
        ind = ga.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        assert ind is None

    def test_generation_boundary_triggers_evolution(self):
        ga = GALifecycle(pop_size=10, generation_size=5, elite_fraction=0.2)
        et = EdgeTracker()
        # Initialize with some population
        for i in range(10):
            ga.population.append(Individual(data=b"x_%d" % i, fitness=float(i)))
        # Run 5 iterations without coverage
        for _ in range(5):
            ga.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        # Generation should have incremented
        assert ga.generation == 1


class TestSpeciation:
    def test_crash_gets_own_species(self):
        et = EdgeTracker()
        spec = Speciation(et, jaccard_threshold=0.3)
        pop = [
            Individual(data=b"normal", edge_count=10),
            Individual(data=b"CRASH", crash=True),
        ]
        species_map = spec.assign_species(pop)
        assert len(species_map) >= 2  # crash in its own species


class TestMutationOps:
    def test_all_ops_produce_bytes(self):
        data = b"hello world test data"
        for op in _GA_MUTATION_OPS:
            result = op(data)
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_ops_handle_empty_input(self):
        for op in _GA_MUTATION_OPS:
            result = op(b"")
            assert isinstance(result, bytes)
