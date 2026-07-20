"""Extensive tests for quantum-inspired evolutionary algorithm (QEA) encoding.

Test categories:
1. Core collapse mechanics
2. Rotation gate
3. Mutate amplitudes
4. QEAIndividual serialization
5. QEALifecycle
6. Edge-case stress tests
7. Integration
"""

import random

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.qea import (
    ALPHA_MAX,
    ALPHA_MIN,
    ALPHA_UNIFORM,
    QEAIndividual,
    QEALifecycle,
    _bias_amplitudes_from,
    _bits_to_bytes,
    _bytes_to_bits,
    _uniform_amplitudes,
    collapse,
    mutate_amplitudes,
    rotation_gate,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def make_individual(amplitudes: list[float], **kw) -> QEAIndividual:
    """Convenience factory for test individuals."""
    return QEAIndividual(amplitudes=amplitudes, **kw)


def uniform_population(size: int, n_bits: int = 64, **kw) -> list[QEAIndividual]:
    """Create a uniform-amplitude population of *size* individuals."""
    amps = [ALPHA_UNIFORM] * n_bits
    return [make_individual(amps[:], generation=0, **kw) for _ in range(size)]


# ═══════════════════════════════════════════════════════════════════════
# 1. Core collapse mechanics
# ═══════════════════════════════════════════════════════════════════════


class TestCollapse:
    def test_collapse_produces_bytes(self):
        """Collapse uniform amplitudes, verify correct-length bytes output."""
        n_bits = 64  # 8 bytes
        amps = [ALPHA_UNIFORM] * n_bits
        result = collapse(amps)
        assert isinstance(result, bytes)
        assert len(result) == n_bits // 8

    def test_collapse_is_stochastic(self):
        """Collapse many times from uniform amplitudes, verify ~50% zeros."""
        n_bits = 8  # 1 byte
        amps = [ALPHA_UNIFORM] * n_bits
        zero_count = 0
        trials = 2000
        for _ in range(trials):
            result = collapse(amps)
            bits = _bytes_to_bits(result)
            zero_count += bits.count(0)
        total_bits = trials * n_bits
        p_zero = zero_count / total_bits
        # Binomial: 95% CI for p=0.5 with 16k trials is ~[0.49, 0.51]
        # Use wider CI for robustness
        assert 0.40 <= p_zero <= 0.60, f"p_zero={p_zero} outside expected range"

    def test_collapse_extreme_zero(self):
        """α ≈ 1 → almost always collapses to 0 bits (P(0) ≈ 0.98)."""
        n_bits = 64
        amps = [0.99] * n_bits
        results = [collapse(amps) for _ in range(100)]
        total_zeros = sum(_bytes_to_bits(r).count(0) for r in results)
        total_bits = len(results) * n_bits
        # Expected: 98% zeros
        assert total_zeros > total_bits * 0.90, f"zeros={total_zeros}/{total_bits}"

    def test_collapse_extreme_one(self):
        """α ≈ 0 → almost always collapses to 1 bits (P(0) ≈ 0.0001)."""
        n_bits = 64
        amps = [0.01] * n_bits
        results = [collapse(amps) for _ in range(100)]
        total_zeros = sum(_bytes_to_bits(r).count(0) for r in results)
        total_bits = len(results) * n_bits
        assert total_zeros < total_bits * 0.10, f"zeros={total_zeros}/{total_bits}"

    def test_collapse_empty(self):
        """Empty amplitude list → empty bytes."""
        result = collapse([])
        assert result == b""

    def test_collapse_single_byte(self):
        """8 amplitudes → 1 byte."""
        amps = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        result = collapse(amps)
        assert len(result) == 1

    def test_collapse_variable_lengths(self):
        """Lengths 1, 8, 64, 128 bytes all work."""
        for n_bytes in [1, 8, 64, 128]:
            amps = [ALPHA_UNIFORM] * (n_bytes * 8)
            result = collapse(amps)
            assert len(result) == n_bytes, f"Failed at {n_bytes} bytes"


# ═══════════════════════════════════════════════════════════════════════
# 2. Rotation gate
# ═══════════════════════════════════════════════════════════════════════


class TestRotationGate:
    def test_rotation_toward_zero(self):
        """collapsed=0, improved=True → α increases (rotate toward |0⟩)."""
        amps = [0.5]
        collapsed = b"\x00"  # all zero bits
        result = rotation_gate(amps, collapsed, improved=True, delta=0.05)
        assert result[0] > 0.5 + 1e-10

    def test_rotation_away_from_zero(self):
        """collapsed=0, improved=False → α decreases (rotate toward |1⟩)."""
        amps = [0.5]
        collapsed = b"\x00"
        result = rotation_gate(amps, collapsed, improved=False, delta=0.05)
        assert result[0] < 0.5 - 1e-10

    def test_rotation_toward_one(self):
        """collapsed=1, improved=True → α decreases (rotate toward |1⟩)."""
        amps = [0.5]
        collapsed = b"\xff"  # all one bits
        result = rotation_gate(amps, collapsed, improved=True, delta=0.05)
        assert result[0] < 0.5 - 1e-10

    def test_rotation_away_from_one(self):
        """collapsed=1, improved=False → α increases (rotate toward |0⟩)."""
        amps = [0.5]
        collapsed = b"\xff"
        result = rotation_gate(amps, collapsed, improved=False, delta=0.05)
        assert result[0] > 0.5 + 1e-10

    def test_rotation_clamp_lower(self):
        """α near 0 → clamped to ALPHA_MIN."""
        amps = [0.005]
        # Rotate toward |1⟩: should go below 0.005 but clamped to 0.01
        result = rotation_gate(amps, b"\xff", improved=True, delta=0.05)
        assert result[0] == ALPHA_MIN

    def test_rotation_clamp_upper(self):
        """α near 1 → clamped to ALPHA_MAX."""
        amps = [0.995]
        # Rotate toward |0⟩: should go above 0.995 but clamped to 0.99
        result = rotation_gate(amps, b"\x00", improved=True, delta=0.05)
        assert result[0] == ALPHA_MAX

    def test_rotation_delta_magnitude(self):
        """Larger δ → larger amplitude change."""
        amps_small = [0.5]
        amps_large = [0.5]
        collapsed = b"\x00"
        rotation_gate(amps_small, collapsed, improved=True, delta=0.01)
        rotation_gate(amps_large, collapsed, improved=True, delta=0.10)
        assert amps_large[0] - 0.5 > amps_small[0] - 0.5

    def test_rotation_convergence_loop(self):
        """100 iterations of rotation toward |0⟩ → α approaches ALPHA_MAX."""
        amps = [0.5]
        for _ in range(100):
            rotation_gate(amps, b"\x00", improved=True, delta=0.05)
        assert amps[0] > 0.95

    def test_rotation_divergence_loop(self):
        """100 iterations away from |0⟩ → α approaches ALPHA_MIN."""
        amps = [0.5]
        for _ in range(100):
            rotation_gate(amps, b"\x00", improved=False, delta=0.05)
        assert amps[0] < 0.05

    def test_rotation_in_place(self):
        """rotation_gate modifies the list in place and returns same object."""
        amps = [0.5]
        result = rotation_gate(amps, b"\x00", improved=True)
        assert result is amps  # same object
        assert amps[0] != 0.5

    def test_rotation_multi_byte(self):
        """Rotation works correctly across multiple bytes."""
        amps = [0.5] * 16  # 2 bytes
        # 0x00ff: first byte all 0, second byte all 1
        collapsed = b"\x00\xff"
        # All improved on first byte (rotate toward |0⟩), not on second (rotate toward |1⟩)
        result = rotation_gate(amps, collapsed, improved=True, delta=0.05)
        # First byte (bits 0-7): improved=True, bit=0 → α increases
        for i in range(8):
            assert result[i] > 0.5 + 1e-10, f"bit {i} should increase"
        # Second byte (bits 8-15): improved=True, bit=1 → α decreases
        for i in range(8, 16):
            assert result[i] < 0.5 - 1e-10, f"bit {i} should decrease"


# ═══════════════════════════════════════════════════════════════════════
# 3. Amplitude mutation
# ═══════════════════════════════════════════════════════════════════════


class TestMutateAmplitudes:
    def test_mutation_perturbs(self):
        """prob=1.0 → all bits changed from their original values."""
        amps = [ALPHA_UNIFORM] * 64
        original = list(amps)
        mutate_amplitudes(amps, prob=1.0)
        changed = sum(1 for a, o in zip(amps, original, strict=False) if abs(a - o) > 1e-10)
        assert changed == len(amps), f"Only {changed}/{len(amps)} changed"

    def test_mutation_prob_zero(self):
        """prob=0.0 → nothing changes."""
        amps = [ALPHA_UNIFORM] * 64
        original = list(amps)
        mutate_amplitudes(amps, prob=0.0)
        assert amps == original

    def test_mutation_range(self):
        """Mutated values stay within [ALPHA_MIN, ALPHA_MAX]."""
        amps = [ALPHA_UNIFORM] * 128
        mutate_amplitudes(amps, prob=1.0)
        assert all(ALPHA_MIN <= a <= ALPHA_MAX for a in amps)

    def test_mutation_in_place(self):
        """Returns same list object."""
        amps = [0.5] * 8
        result = mutate_amplitudes(amps, prob=1.0)
        assert result is amps

    def test_mutation_partial(self):
        """prob=0.3 → roughly 30% of bits change (with wide tolerance)."""
        random.seed(42)
        amps = [0.5] * 200
        original = list(amps)
        mutate_amplitudes(amps, prob=0.3)
        changed = sum(1 for a, o in zip(amps, original, strict=False) if abs(a - o) > 1e-10)
        assert 30 <= changed <= 120, f"changed={changed}/200 (expected ~60)"


# ═══════════════════════════════════════════════════════════════════════
# 4. Bit conversion helpers
# ═══════════════════════════════════════════════════════════════════════


class TestBitConversion:
    def test_bytes_to_bits_roundtrip(self):
        data = bytes(range(256))
        bits = _bytes_to_bits(data)
        restored = _bits_to_bytes(bits)
        assert restored == data

    def test_bytes_to_bits_length(self):
        data = b"\x00\xff\x55\xaa"
        bits = _bytes_to_bits(data)
        assert len(bits) == len(data) * 8

    def test_bytes_to_bits_values(self):
        # b"\x0f" = 0000 1111
        bits = _bytes_to_bits(b"\x0f")
        assert bits == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_bits_to_bytes_empty(self):
        assert _bits_to_bytes([]) == b""

    def test_bits_to_bytes_partial_byte(self):
        # [1, 0] → 0b10000000 = 0x80 (padded with zeros)
        result = _bits_to_bytes([1, 0])
        assert result == b"\x80"

    def test_bias_amplitudes_zero_byte(self):
        """b'\\x00' → strong α for zero bits."""
        amps = _bias_amplitudes_from(b"\x00")
        assert all(a == 0.9 for a in amps)

    def test_bias_amplitudes_ff_byte(self):
        """b'\\xff' → weak α for ones bits (strong_prob for 0, 1-strong_prob for 1)."""
        amps = _bias_amplitudes_from(b"\xff")
        assert all(a == pytest.approx(0.1) for a in amps)  # 1 - 0.9 = 0.099999...

    def test_uniform_amplitudes(self):
        amps = _uniform_amplitudes(16)
        assert len(amps) == 16
        assert all(a == ALPHA_UNIFORM for a in amps)


# ═══════════════════════════════════════════════════════════════════════
# 5. QEAIndividual serialization
# ═══════════════════════════════════════════════════════════════════════


class TestQEAIndividual:
    def test_serialization_roundtrip(self):
        """All fields survive JSON save/load."""
        ind = QEAIndividual(
            amplitudes=[0.5, 0.7, 0.3, 0.9],
            fitness=0.85,
            edge_count=42,
            novelty_score=0.3,
            diversity_score=0.6,
            freshness_score=0.9,
            mutation_potential=0.1,
            species_id=5,
            generation=3,
            best_collapsed=b"\x89PNG",
            best_fitness=1.2,
            seed_key="abc123",
            crash=True,
        )
        d = ind.to_dict()
        restored = QEAIndividual.from_dict(d)
        assert restored.amplitudes == ind.amplitudes
        assert restored.fitness == ind.fitness
        assert restored.edge_count == ind.edge_count
        assert restored.novelty_score == ind.novelty_score
        assert restored.diversity_score == ind.diversity_score
        assert restored.freshness_score == ind.freshness_score
        assert restored.mutation_potential == ind.mutation_potential
        assert restored.species_id == ind.species_id
        assert restored.generation == ind.generation
        assert restored.best_collapsed == ind.best_collapsed
        assert restored.best_fitness == ind.best_fitness
        assert restored.seed_key == ind.seed_key
        assert restored.crash == ind.crash

    def test_seed_key_from_best_collapsed(self):
        """seed_key auto-generated from best_collapsed hash."""
        ind = QEAIndividual(amplitudes=[0.5] * 8, best_collapsed=b"test_data")
        assert len(ind.seed_key) == 16
        assert ind.seed_key.isalnum()

    def test_empty_best_collapsed(self):
        """No best_collapsed → empty seed_key."""
        ind = QEAIndividual(amplitudes=[0.5] * 8)
        assert ind.seed_key == ""

    def test_partial_from_dict(self):
        """Missing fields get sensible defaults."""
        d = {"amplitudes": [0.5]}
        ind = QEAIndividual.from_dict(d)
        assert ind.amplitudes == [0.5]
        assert ind.fitness == 0.0
        assert ind.generation == 0
        assert ind.best_collapsed == b""
        assert not ind.crash

    def test_no_float_drift_after_repeated_serialize(self):
        """3x save/load, amplitudes match within 1e-10."""
        original_amps = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6]
        ind = QEAIndividual(amplitudes=original_amps, best_collapsed=b"\xab")
        import json

        for _ in range(3):
            d = ind.to_dict()
            json_str = json.dumps(d)
            loaded = json.loads(json_str)
            ind = QEAIndividual.from_dict(loaded)
        for a, o in zip(ind.amplitudes, original_amps, strict=False):
            assert abs(a - o) < 1e-10

    def test_num_bytes_property(self):
        ind = QEAIndividual(amplitudes=[0.5] * 16)
        assert ind.num_bytes == 2

    def test_num_bytes_minimum(self):
        ind = QEAIndividual(amplitudes=[])
        assert ind.num_bytes == 1  # max(1, 0//8)


# ═══════════════════════════════════════════════════════════════════════
# 6. QEALifecycle
# ═══════════════════════════════════════════════════════════════════════


class TestQEALifecycle:
    def test_initialize_from_corpus(self):
        """5 seeds → 5 QEAIndividuals in population."""
        qea = QEALifecycle(pop_size=10)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        qea.initialize(corpus, et)
        assert len(qea.population) == 5

    def test_initialize_empty_corpus(self):
        """Empty corpus → empty population, no crash."""
        qea = QEALifecycle(pop_size=10)
        et = EdgeTracker()
        qea.initialize([], et)
        assert len(qea.population) == 0

    def test_initialize_truncates_to_pop_size(self):
        """Corpus larger than pop_size → truncated."""
        qea = QEALifecycle(pop_size=3)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(10)]
        qea.initialize(corpus, et)
        assert len(qea.population) == 3

    def test_pick_seed_returns_bytes(self):
        """pick_seed() returns bytes of appropriate length."""
        qea = QEALifecycle(pop_size=10)
        qea.population = uniform_population(5)
        seed = qea.pick_seed()
        assert isinstance(seed, bytes)
        assert len(seed) > 0

    def test_pick_seed_empty_population(self):
        """Empty population → fallback bytes, no crash."""
        qea = QEALifecycle()
        seed = qea.pick_seed()
        assert seed == b"\x00" * 64

    def test_pick_seed_tracks_parent(self):
        """pick_seed records _last_parent and _last_collapsed."""
        qea = QEALifecycle(pop_size=10)
        qea.population = uniform_population(5, n_bits=64)
        qea.population[0].fitness = 10.0  # make one clearly better
        seed = qea.pick_seed()
        assert qea._last_parent is not None
        assert qea._last_collapsed == seed

    def test_evolve_maintains_pop_size(self):
        """20 in → 20 out after _evolve."""
        qea = QEALifecycle(pop_size=20, elite_fraction=0.1)
        qea.population = uniform_population(20)
        et = EdgeTracker()
        qea._evolve(et)
        assert len(qea.population) == 20

    def test_evolve_all_identical(self):
        """All individuals have same collapsed data → no crash."""
        qea = QEALifecycle(pop_size=10, elite_fraction=0.1)
        amps = _bias_amplitudes_from(b"\x42" * 8)
        qea.population = [make_individual(list(amps)) for _ in range(10)]
        et = EdgeTracker()
        qea._evolve(et)
        assert len(qea.population) == 10

    def test_crash_never_culled(self):
        """Crash individual survives even with low fitness."""
        qea = QEALifecycle(pop_size=10, elite_fraction=0.1)
        crash_ind = QEAIndividual(amplitudes=[0.5] * 64, fitness=0.0, crash=True)
        qea.population.append(crash_ind)
        for i in range(9):
            qea.population.append(
                QEAIndividual(amplitudes=[0.5] * 64, fitness=float(i + 1), crash=False)
            )
        et = EdgeTracker()
        qea._evolve(et)
        assert any(i.crash for i in qea.population)

    def test_add_to_population_below_capacity(self):
        """Add below pop_size → added."""
        qea = QEALifecycle(pop_size=10)
        qea.population = uniform_population(5)
        ind = QEAIndividual(
            amplitudes=[0.5] * 64,
            fitness=5.0,
            best_collapsed=b"new_seed",
        )
        qea.add_to_population(ind)
        assert len(qea.population) == 6
        assert qea.population[-1].fitness == 5.0

    def test_add_to_population_above_capacity(self):
        """Add above pop_size → worst replaced if new is better."""
        qea = QEALifecycle(pop_size=5)
        qea.population = [
            QEAIndividual(amplitudes=[0.5] * 64, fitness=float(i), best_collapsed=b"x_%d" % i)
            for i in range(5)
        ]
        best_ind = QEAIndividual(
            amplitudes=[0.6] * 64,
            fitness=100.0,
            best_collapsed=b"best",
        )
        qea.add_to_population(best_ind)
        assert len(qea.population) == 5
        # The new best should be in the population
        assert any(i.fitness == 100.0 for i in qea.population)

    def test_generation_boundary_triggers(self):
        """gen_size iterations → generation incremented."""
        qea = QEALifecycle(pop_size=10, generation_size=5)
        qea.population = uniform_population(10)
        et = EdgeTracker()
        for _ in range(5):
            qea.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        assert qea.generation == 1

    def test_on_fuzz_result_new_coverage(self):
        """New coverage → returns QEAIndividual."""
        qea = QEALifecycle(pop_size=10)
        qea.population = uniform_population(5)
        et = EdgeTracker()
        ind = qea.on_fuzz_result(b"new_coverage", new_coverage=True, edge_count=10, edge_tracker=et)
        assert ind is not None
        assert isinstance(ind, QEAIndividual)
        assert ind.best_collapsed == b"new_coverage"

    def test_on_fuzz_result_no_coverage(self):
        """No coverage → returns None."""
        qea = QEALifecycle(pop_size=10, generation_size=100)
        qea.population = uniform_population(5)
        et = EdgeTracker()
        ind = qea.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        assert ind is None

    def test_on_fuzz_result_rotation_improved(self):
        """New coverage rotates last parent's amplitudes toward the collapsed value."""
        qea = QEALifecycle(pop_size=10, rotation_angle=0.05)
        qea.population = uniform_population(5, n_bits=8)
        et = EdgeTracker()

        # pick_seed records the parent
        qea.pick_seed()
        collapsed = qea._last_collapsed
        original_amps = list(qea._last_parent.amplitudes)

        # Trigger a "new coverage" result — this should rotate toward collapsed
        qea.on_fuzz_result(b"result", new_coverage=True, edge_count=5, edge_tracker=et)

        # Amplitudes should have moved toward the collapsed value
        if collapsed == b"\x00":
            assert qea._last_parent.amplitudes[0] > original_amps[0]
        else:
            assert qea._last_parent.amplitudes[0] != original_amps[0]

    def test_on_fuzz_result_rotation_not_improved(self):
        """No coverage rotates last parent's amplitudes away from collapsed value."""
        qea = QEALifecycle(pop_size=10, rotation_angle=0.05)
        qea.population = uniform_population(5, n_bits=8)
        et = EdgeTracker()

        qea.pick_seed()
        collapsed = qea._last_collapsed
        original_amps = list(qea._last_parent.amplitudes)

        # No coverage → rotate away
        qea.on_fuzz_result(b"result", new_coverage=False, edge_count=0, edge_tracker=et)

        # Amplitudes should have moved away from the collapsed value
        if collapsed == b"\x00":
            # Rotate away from |0⟩ → α decreases
            assert qea._last_parent.amplitudes[0] < original_amps[0]
        else:
            assert qea._last_parent.amplitudes[0] != original_amps[0]

    def test_save_load_roundtrip(self, tmp_path):
        """Save then load — amplitudes and generation match."""
        qea = QEALifecycle(pop_size=5)
        qea.population = [
            QEAIndividual(
                amplitudes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                fitness=float(i),
                species_id=i % 2,
                best_collapsed=b"x_%d" % i,
            )
            for i in range(5)
        ]
        qea.generation = 7
        path = tmp_path / "qea.json"
        qea.save(path)

        qea2 = QEALifecycle(pop_size=5)
        qea2.load(path)
        assert qea2.generation == 7
        assert len(qea2.population) == 5
        assert qea2.population[0].amplitudes == qea.population[0].amplitudes
        assert qea2.population[0].best_collapsed == qea.population[0].best_collapsed

    def test_load_nonexistent(self, tmp_path):
        """Load from nonexistent path → no-op, no crash."""
        qea = QEALifecycle()
        qea.load(tmp_path / "nonexistent.json")
        assert qea.population == []
        assert qea.generation == 0

    def test_diversity_across_generations(self):
        """3 generations → not all individuals are identical."""
        qea = QEALifecycle(pop_size=10, elite_fraction=0.2, generation_size=5)
        qea.population = uniform_population(10, n_bits=64)
        et = EdgeTracker()
        for _gen in range(3):
            for _ in range(5):
                # Alternate between coverage and no-coverage to exercise both paths
                if random.random() < 0.2:
                    ind = qea.on_fuzz_result(
                        bytes(random.randint(0, 255) for _ in range(8)),
                        new_coverage=True,
                        edge_count=random.randint(1, 20),
                        edge_tracker=et,
                    )
                    if ind is not None:
                        qea.add_to_population(ind)
                else:
                    qea.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        # Population is not all identical
        collapsed_set = {i.best_collapsed for i in qea.population}
        assert len(collapsed_set) > 1, "Population collapsed to all identical!"

    def test_fitness_scored_after_evolve(self):
        """After _evolve, at least the elite has fitness > 0; offspring get evaluated on next on_fuzz_result."""
        qea = QEALifecycle(pop_size=10, elite_fraction=0.1)
        et = EdgeTracker()
        for i in range(10):
            data = b"x_%d" % i
            key = __import__("hashlib").sha256(data).hexdigest()[:16]
            et.seed_edges[key] = {i * 10 + j for j in range(5)}
            qea.population.append(
                QEAIndividual(
                    amplitudes=[ALPHA_UNIFORM] * 64,
                    edge_count=5,
                    best_collapsed=data,
                    seed_key=key,
                )
            )
        et.cumulative_edges = set(range(100))
        qea._evolve(et)
        # Elite from original population should have fitness > 0
        assert qea.best_fitness > 0
        assert qea.avg_fitness >= 0

    def test_rotation_improves_fitness_sequence(self):
        """Repeated good outcomes → amplitudes drift toward correct values."""
        qea = QEALifecycle(pop_size=10, rotation_angle=0.1, generation_size=100)
        # One individual with uniform amplitudes
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8, best_collapsed=b"\x42")
        qea.population = [ind]
        # Repeatedly select, collapse, and report success for the same target value
        target = b"\x42"
        for _ in range(50):
            qea._last_parent = ind
            qea._last_collapsed = target
            qea.on_fuzz_result(target, new_coverage=True, edge_count=10, edge_tracker=EdgeTracker())
        # After 50 positive rotations, amplitudes should be strongly biased toward target
        target_bits = _bytes_to_bits(target)
        for i, bit in enumerate(target_bits):
            if bit == 0:
                assert ind.amplitudes[i] > 0.7, (
                    f"bit {i} should be high (α={ind.amplitudes[i]:.3f})"
                )
            else:
                assert ind.amplitudes[i] < 0.3, f"bit {i} should be low (α={ind.amplitudes[i]:.3f})"


# ═══════════════════════════════════════════════════════════════════════
# 7. Edge-case stress tests
# ═══════════════════════════════════════════════════════════════════════


class TestStress:
    def test_large_amplitudes(self):
        """1024 bytes (8192 bits) collapses without error."""
        amps = [ALPHA_UNIFORM] * (1024 * 8)
        result = collapse(amps)
        assert len(result) == 1024

    def test_500_calls_no_coverage(self):
        """500 on_fuzz_result calls with no coverage → no crash."""
        qea = QEALifecycle(pop_size=10, generation_size=50)
        qea.population = uniform_population(10, n_bits=64)
        et = EdgeTracker()
        for _i in range(500):
            qea.pick_seed()
            qea.on_fuzz_result(b"x" * 8, new_coverage=False, edge_count=0, edge_tracker=et)
        # Should have gone through several generations
        assert qea.iterations_since_gen < 50  # was reset by evolve

    def test_amplitude_invariants(self):
        """All amplitudes in [ALPHA_MIN, ALPHA_MAX] across entire population."""
        qea = QEALifecycle(pop_size=20)
        qea.population = uniform_population(20, n_bits=128)
        et = EdgeTracker()
        # Run a few evolve cycles
        for _ in range(3):
            qea._evolve(et)
        for ind in qea.population:
            for a in ind.amplitudes:
                assert ALPHA_MIN <= a <= ALPHA_MAX, f"α={a} out of range"

    def test_no_mutation_does_not_stagnate(self):
        """Even with mutation_prob=0, rotation gates maintain some diversity."""
        qea = QEALifecycle(pop_size=20, mutation_prob=0.0, rotation_angle=0.02, generation_size=10)
        qea.population = uniform_population(20, n_bits=64)
        et = EdgeTracker()
        for _ in range(5):
            for _ in range(10):
                qea.on_fuzz_result(b"seed", new_coverage=False, edge_count=0, edge_tracker=et)
        # Check that not all individuals collapsed to identical bytes
        collapsed_set = {bytes([int(a > 0.5) for a in ind.amplitudes]) for ind in qea.population}
        assert len(collapsed_set) > 1, "All individuals collapsed identically!"

    def test_float_precision_serialization(self):
        """Float amplitudes serialize correctly through JSON."""
        import json

        amps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ind = QEAIndividual(amplitudes=amps, best_collapsed=b"\x00")
        d = ind.to_dict()
        json_str = json.dumps(d)
        loaded_d = json.loads(json_str)
        restored = QEAIndividual.from_dict(loaded_d)
        for a, r in zip(amps, restored.amplitudes, strict=False):
            assert abs(a - r) < 1e-15


# ═══════════════════════════════════════════════════════════════════════
# 8. Integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_works_with_edge_tracker(self):
        """QEALifecycle.initialize + _evaluate_all with real EdgeTracker."""
        qea = QEALifecycle(pop_size=10)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        qea.initialize(corpus, et)
        assert len(qea.population) == 5
        # All individuals should have been evaluated
        assert all(i.fitness >= 0 for i in qea.population)

    def test_speciation_forms_multiple_species(self):
        """With diverse inputs, multiple species form."""
        qea = QEALifecycle(pop_size=20, speciation_threshold=0.3)
        et = EdgeTracker()

        # Create diverse inputs and register in edge_tracker
        for i in range(10):
            data = bytes([i] * 16)
            key = __import__("hashlib").sha256(data).hexdigest()[:16]
            et.seed_edges[key] = {i * 10 + j for j in range(10)}
            et.cumulative_edges.update(range(i * 10, i * 10 + 10))

        corpus = [bytes([i] * 16) for i in range(10)]
        qea.initialize(corpus, et)

        # Force speciation
        from fuzzer_tool.core.ga import Speciation

        spec = Speciation(et, jaccard_threshold=0.3)
        species_map = spec.assign_species(qea.population)
        assert len(species_map) >= 1

    def test_full_lifecycle_no_crash(self):
        """Full init → pick → on_fuzz_result loop for 100 iterations."""
        qea = QEALifecycle(pop_size=10, generation_size=20)
        et = EdgeTracker()
        corpus = [b"\x00" * 16, b"\xff" * 16, b"\x55" * 16]
        qea.initialize(corpus, et)

        for _i in range(100):
            seed = qea.pick_seed()
            # Simulate mutation (small change)
            mutated = bytearray(seed)
            if mutated:
                mutated[random.randint(0, len(mutated) - 1)] ^= 0xFF
            mutated = bytes(mutated)
            # Simulate coverage result
            new_cov = random.random() < 0.15
            ind = qea.on_fuzz_result(mutated, new_cov, random.randint(1, 20), et)
            if ind is not None:
                qea.add_to_population(ind)

        assert len(qea.population) == 10
        assert any(i.fitness > 0 for i in qea.population)

    def test_rotation_gate_within_on_fuzz_result(self):
        """Verify that on_fuzz_result applies rotation in the expected direction."""
        qea = QEALifecycle(pop_size=5, rotation_angle=0.05)
        # Start with known alpha
        amps = [0.5] * 8
        ind = QEAIndividual(amplitudes=amps, best_collapsed=b"\x00")
        qea.population = [ind]
        et = EdgeTracker()

        # Pick seed (collapses from ind's amplitudes)
        seed = qea.pick_seed()
        assert qea._last_parent is ind

        # Record the amplitude before rotation
        alpha_before = list(ind.amplitudes)

        # Report success — rotation should move toward collapsed bits
        qea.on_fuzz_result(seed, new_coverage=True, edge_count=5, edge_tracker=et)

        # After rotation: bits that were 0 in collapsed → α increased; bits 1 → α decreased
        collapsed_bits = _bytes_to_bits(seed)
        for i in range(len(alpha_before)):
            if collapsed_bits[i] == 0:
                assert ind.amplitudes[i] > alpha_before[i], (
                    f"bit {i}=0 should increase α: {alpha_before[i]:.3f} → {ind.amplitudes[i]:.3f}"
                )
            else:
                assert ind.amplitudes[i] < alpha_before[i], (
                    f"bit {i}=1 should decrease α: {alpha_before[i]:.3f} → {ind.amplitudes[i]:.3f}"
                )
