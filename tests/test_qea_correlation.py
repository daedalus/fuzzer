"""Tests for the intra-byte correlation (partial entanglement) extension to QEA.

See docs/handover/handover_qea_hilbert_space_analysis_2026-08-31.md for the
analysis this implements: QEA's per-bit independent amplitudes are a
product-state approximation that cannot represent that two bits within a
byte tend to be right or wrong together. This adds an opt-in (default off)
8x8 pairwise coupling matrix per byte, Hebbian-updated alongside the
rotation gate.

Test categories:
1. _zero_coupling / _alpha_to_field helpers
2. collapse_correlated statistics
3. update_couplings direction and clamping
4. QEAIndividual coupling field + serialization
5. QEALifecycle wiring (opt-in, off-by-default)
6. Learning / integration
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.qea import (
    ALPHA_UNIFORM,
    QEAIndividual,
    QEALifecycle,
    _alpha_to_field,
    _zero_coupling,
    collapse_correlated,
    update_couplings,
)

# ═══════════════════════════════════════════════════════════════════════
# 1. Helpers
# ═══════════════════════════════════════════════════════════════════════


class TestZeroCoupling:
    def test_shape(self):
        c = _zero_coupling(4)
        assert c.shape == (4, 8, 8)

    def test_all_zero(self):
        c = _zero_coupling(3)
        assert np.all(c == 0.0)

    def test_minimum_one_byte(self):
        """num_bytes=0 still produces a valid (1, 8, 8) tensor, matching
        QEAIndividual.num_bytes' own max(1, ...) floor."""
        c = _zero_coupling(0)
        assert c.shape == (1, 8, 8)


class TestAlphaToField:
    def test_uniform_alpha_gives_zero_field(self):
        """α = ALPHA_UNIFORM (P(0)=P(1)=0.5) → log-odds = 0."""
        field = _alpha_to_field(np.array([ALPHA_UNIFORM]))
        assert abs(field[0]) < 1e-6

    def test_high_alpha_gives_positive_field(self):
        """α near 1 (P(bit=0) high) → positive log-odds for bit=0."""
        field = _alpha_to_field(np.array([0.9]))
        assert field[0] > 0

    def test_low_alpha_gives_negative_field(self):
        """α near 0 (P(bit=1) high) → negative log-odds for bit=0."""
        field = _alpha_to_field(np.array([0.1]))
        assert field[0] < 0

    def test_no_overflow_at_extremes(self):
        """Clipping keeps the field finite even at the α∈{0,1} boundary."""
        field = _alpha_to_field(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(field))


# ═══════════════════════════════════════════════════════════════════════
# 2. collapse_correlated statistics
# ═══════════════════════════════════════════════════════════════════════


class TestCollapseCorrelated:
    def test_output_shape_matches_collapse(self):
        amps = np.array([ALPHA_UNIFORM] * 16, dtype=np.float64)  # 2 bytes
        coupling = _zero_coupling(2)
        result = collapse_correlated(amps, coupling)
        assert isinstance(result, bytes)
        assert len(result) == 2

    def test_empty_amplitudes(self):
        result = collapse_correlated(np.array([], dtype=np.float64), _zero_coupling(0))
        assert result == b""

    def test_zero_coupling_matches_marginals(self):
        """With all-zero coupling, per-bit marginals should match collapse()'s
        α² law (statistically) -- zero coupling should not introduce bias."""
        n_bits = 8
        amps = np.array([0.9] * n_bits, dtype=np.float64)  # P(bit=0) = 0.81
        coupling = _zero_coupling(1)
        trials = 400
        zero_count = 0
        for _ in range(trials):
            result = collapse_correlated(amps, coupling, n_sweeps=3)
            byte = result[0]
            for shift in range(8):
                bit = (byte >> (7 - shift)) & 1
                if bit == 0:
                    zero_count += 1
        p_zero = zero_count / (trials * n_bits)
        assert 0.70 <= p_zero <= 0.90, f"p_zero={p_zero} should be near 0.81"

    def test_positive_coupling_correlates_bits(self):
        """Strong positive J[0,1] with neutral fields → bit 0 and bit 1
        agree far more than the 50% chance rate."""
        amps = np.array([ALPHA_UNIFORM] * 8, dtype=np.float64)
        coupling = _zero_coupling(1)
        coupling[0, 0, 1] = 5.0
        coupling[0, 1, 0] = 5.0
        trials = 500
        agree = 0
        for _ in range(trials):
            result = collapse_correlated(amps, coupling, n_sweeps=5)
            byte = result[0]
            bit0 = (byte >> 7) & 1
            bit1 = (byte >> 6) & 1
            if bit0 == bit1:
                agree += 1
        p_agree = agree / trials
        assert p_agree > 0.85, f"p_agree={p_agree} should be strongly correlated"

    def test_negative_coupling_anticorrelates_bits(self):
        """Strong negative J[0,1] with neutral fields → bit 0 and bit 1
        disagree far more than the 50% chance rate."""
        amps = np.array([ALPHA_UNIFORM] * 8, dtype=np.float64)
        coupling = _zero_coupling(1)
        coupling[0, 0, 1] = -5.0
        coupling[0, 1, 0] = -5.0
        trials = 500
        disagree = 0
        for _ in range(trials):
            result = collapse_correlated(amps, coupling, n_sweeps=5)
            byte = result[0]
            bit0 = (byte >> 7) & 1
            bit1 = (byte >> 6) & 1
            if bit0 != bit1:
                disagree += 1
        p_disagree = disagree / trials
        assert p_disagree > 0.85, f"p_disagree={p_disagree} should be strongly anti-correlated"

    def test_uncoupled_bits_unaffected_by_distant_coupling(self):
        """Coupling between bits 0/1 shouldn't bias the marginal of bit 4,
        which has no nonzero coupling entries at all."""
        amps = np.array([ALPHA_UNIFORM] * 8, dtype=np.float64)
        coupling = _zero_coupling(1)
        coupling[0, 0, 1] = 5.0
        coupling[0, 1, 0] = 5.0
        trials = 400
        zero_count = 0
        for _ in range(trials):
            result = collapse_correlated(amps, coupling, n_sweeps=5)
            bit4 = (result[0] >> 3) & 1
            if bit4 == 0:
                zero_count += 1
        p_zero = zero_count / trials
        assert 0.35 <= p_zero <= 0.65, f"p_zero={p_zero} should stay near 0.5"

    def test_multi_byte_independent(self):
        """Coupling in byte 0 has no effect on byte 1's bits."""
        amps = np.array([ALPHA_UNIFORM] * 16, dtype=np.float64)  # 2 bytes
        coupling = _zero_coupling(2)
        coupling[0, 0, 1] = 8.0
        coupling[0, 1, 0] = 8.0
        results = [collapse_correlated(amps, coupling, n_sweeps=5) for _ in range(200)]
        # Second byte should show no strong bit0/bit1 agreement bias
        agree = sum(1 for r in results if ((r[1] >> 7) & 1) == ((r[1] >> 6) & 1))
        p_agree = agree / len(results)
        assert p_agree < 0.75, f"p_agree={p_agree} -- byte 1 should be uncoupled"


# ═══════════════════════════════════════════════════════════════════════
# 3. update_couplings
# ═══════════════════════════════════════════════════════════════════════


class TestUpdateCouplings:
    def test_matching_bits_improved_increases_coupling(self):
        """Both bits 0, same value (both 0 → same sign s=+1), improved=True
        → J[0,1] moves positive (reinforce the agreement)."""
        coupling = _zero_coupling(1)
        collapsed = b"\x00"  # all bits 0 -> all s=+1
        update_couplings(coupling, collapsed, improved=True, delta=0.1)
        assert coupling[0, 0, 1] > 0
        assert coupling[0, 1, 0] > 0  # symmetric

    def test_mismatched_bits_improved_decreases_coupling(self):
        """bit0=0 (s=+1), bit1=1 (s=-1), improved=True → J[0,1] moves
        negative (reinforce the disagreement)."""
        coupling = _zero_coupling(1)
        collapsed = bytes([0b01000000])  # bit0=0, bit1=1, rest 0
        update_couplings(coupling, collapsed, improved=True, delta=0.1)
        assert coupling[0, 0, 1] < 0

    def test_improved_false_negates_direction(self):
        coupling_pos = _zero_coupling(1)
        coupling_neg = _zero_coupling(1)
        collapsed = b"\x00"
        update_couplings(coupling_pos, collapsed, improved=True, delta=0.1)
        update_couplings(coupling_neg, collapsed, improved=False, delta=0.1)
        assert coupling_pos[0, 0, 1] > 0
        assert coupling_neg[0, 0, 1] < 0
        assert math.isclose(coupling_pos[0, 0, 1], -coupling_neg[0, 0, 1], abs_tol=1e-12)

    def test_diagonal_stays_zero(self):
        coupling = _zero_coupling(1)
        collapsed = b"\xff"
        update_couplings(coupling, collapsed, improved=True, delta=0.5)
        assert np.all(np.diagonal(coupling[0]) == 0.0)

    def test_symmetric_after_update(self):
        coupling = _zero_coupling(2)
        collapsed = b"\xa5\x3c"
        update_couplings(coupling, collapsed, improved=True, delta=0.05)
        for b in range(2):
            np.testing.assert_allclose(coupling[b], coupling[b].T)

    def test_clips_to_coupling_max(self):
        coupling = _zero_coupling(1)
        collapsed = b"\x00"
        for _ in range(1000):
            update_couplings(coupling, collapsed, improved=True, delta=0.5, coupling_max=1.0)
        assert np.all(coupling <= 1.0)
        assert np.all(coupling >= -1.0)

    def test_in_place_and_returns_same_object(self):
        coupling = _zero_coupling(1)
        result = update_couplings(coupling, b"\x00", improved=True, delta=0.1)
        assert result is coupling

    def test_short_collapsed_padded(self):
        """collapsed shorter than the coupling's byte count is padded with
        zero bits, mirroring rotation_gate's pad/truncate handling."""
        coupling = _zero_coupling(2)  # expects 2 bytes / 16 bits
        # Should not raise even though collapsed is only 1 byte.
        update_couplings(coupling, b"\xff", improved=True, delta=0.1)
        assert coupling.shape == (2, 8, 8)


# ═══════════════════════════════════════════════════════════════════════
# 4. QEAIndividual coupling field
# ═══════════════════════════════════════════════════════════════════════


class TestQEAIndividualCoupling:
    def test_default_coupling_is_none(self):
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8)
        assert ind.coupling is None

    def test_list_coupling_converted_to_ndarray(self):
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8, coupling=np.zeros((1, 8, 8)).tolist())
        assert isinstance(ind.coupling, np.ndarray)
        assert ind.coupling.shape == (1, 8, 8)

    def test_to_dict_none_coupling(self):
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8)
        d = ind.to_dict()
        assert d["coupling"] is None

    def test_to_dict_with_coupling(self):
        coupling = _zero_coupling(1)
        coupling[0, 0, 1] = 0.5
        coupling[0, 1, 0] = 0.5
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8, coupling=coupling)
        d = ind.to_dict()
        assert d["coupling"] is not None
        assert d["coupling"][0][0][1] == 0.5

    def test_round_trip_serialization(self):
        coupling = _zero_coupling(1)
        coupling[0, 2, 5] = -0.75
        coupling[0, 5, 2] = -0.75
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8, coupling=coupling)
        restored = QEAIndividual.from_dict(ind.to_dict())
        np.testing.assert_allclose(restored.coupling, coupling, atol=1e-6)

    def test_round_trip_none_coupling(self):
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8)
        restored = QEAIndividual.from_dict(ind.to_dict())
        assert restored.coupling is None

    def test_json_round_trip(self):
        """Same JSON-through-serialization check as
        test_float_precision_serialization in test_qea.py, but for coupling."""
        import json

        coupling = _zero_coupling(1)
        coupling[0, 0, 3] = 0.123456
        coupling[0, 3, 0] = 0.123456
        ind = QEAIndividual(amplitudes=[ALPHA_UNIFORM] * 8, coupling=coupling)
        json_str = json.dumps(ind.to_dict())
        restored = QEAIndividual.from_dict(json.loads(json_str))
        np.testing.assert_allclose(restored.coupling, coupling, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 5. QEALifecycle wiring
# ═══════════════════════════════════════════════════════════════════════


class TestQEALifecycleCorrelation:
    def test_default_disabled(self):
        qea = QEALifecycle(pop_size=5)
        assert qea.use_correlation is False

    def test_disabled_by_default_no_coupling_after_initialize(self):
        qea = QEALifecycle(pop_size=5)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        qea.initialize(corpus, et)
        assert all(ind.coupling is None for ind in qea.population)

    def test_enabled_gives_coupling_after_initialize(self):
        qea = QEALifecycle(pop_size=5, use_correlation=True)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        qea.initialize(corpus, et)
        assert all(ind.coupling is not None for ind in qea.population)
        for ind in qea.population:
            assert ind.coupling.shape == (ind.num_bytes, 8, 8)

    def test_disabled_behavior_unaffected(self):
        """With use_correlation=False, pick_seed()/on_fuzz_result() run
        identically to the pre-existing (uncorrelated) path -- no coupling
        ever gets created or touched."""
        qea = QEALifecycle(pop_size=5, use_correlation=False)
        et = EdgeTracker()
        corpus = [b"seed_%d" % i for i in range(5)]
        qea.initialize(corpus, et)
        for _ in range(20):
            seed = qea.pick_seed()
            qea.on_fuzz_result(seed, new_coverage=False, edge_count=0, edge_tracker=et)
        assert all(ind.coupling is None for ind in qea.population)

    def test_on_fuzz_result_updates_parent_coupling(self):
        qea = QEALifecycle(pop_size=1, use_correlation=True, correlation_delta=0.1)
        ind = QEAIndividual(
            amplitudes=[ALPHA_UNIFORM] * 8,
            best_collapsed=b"\x00",
            coupling=_zero_coupling(1),
        )
        qea.population = [ind]
        et = EdgeTracker()
        seed = qea.pick_seed()
        assert qea._last_parent is ind
        before = ind.coupling.copy()
        qea.on_fuzz_result(seed, new_coverage=True, edge_count=1, edge_tracker=et)
        assert not np.allclose(before, ind.coupling), "coupling should change after feedback"

    def test_offspring_get_fresh_zero_coupling(self):
        qea = QEALifecycle(pop_size=6, elite_fraction=0.2, use_correlation=True)
        et = EdgeTracker()
        corpus = [b"x_%d" % i for i in range(6)]
        qea.initialize(corpus, et)
        # Bias one individual's coupling away from zero so we can tell
        # elites (carried over) apart from offspring (freshly bred).
        qea.population[0].fitness = 1000.0  # guaranteed elite
        qea.population[0].coupling[0, 0, 1] = 0.9
        qea._evolve(et)
        # Offspring (non-elite slots) should all start at zero coupling.
        n_elite = max(1, int(6 * 0.2))
        for ind in qea.population[n_elite:]:
            assert np.all(ind.coupling == 0.0)

    def test_new_coverage_individual_gets_coupling(self):
        qea = QEALifecycle(pop_size=5, use_correlation=True)
        et = EdgeTracker()
        et.cumulative_edges = set(range(10))
        new_ind = qea.on_fuzz_result(b"\x01\x02", new_coverage=True, edge_count=3, edge_tracker=et)
        assert new_ind is not None
        assert new_ind.coupling is not None
        assert new_ind.coupling.shape == (new_ind.num_bytes, 8, 8)

    def test_full_lifecycle_no_crash_with_correlation(self):
        """Mirrors TestIntegration.test_full_lifecycle_no_crash in
        test_qea.py, with use_correlation=True."""
        import random as _random

        qea = QEALifecycle(pop_size=10, generation_size=20, use_correlation=True)
        et = EdgeTracker()
        corpus = [b"\x00" * 16, b"\xff" * 16, b"\x55" * 16]
        qea.initialize(corpus, et)

        for _i in range(100):
            seed = qea.pick_seed()
            mutated = bytearray(seed)
            if mutated:
                mutated[_random.randint(0, len(mutated) - 1)] ^= 0xFF
            mutated = bytes(mutated)
            new_cov = _random.random() < 0.15
            ind = qea.on_fuzz_result(mutated, new_cov, _random.randint(1, 20), et)
            if ind is not None:
                qea.add_to_population(ind)

        assert len(qea.population) == 10
        assert any(i.fitness > 0 for i in qea.population)


# ═══════════════════════════════════════════════════════════════════════
# 6. Learning / integration
# ═══════════════════════════════════════════════════════════════════════


class TestCorrelationLearning:
    def test_repeated_reward_of_matching_pair_learns_positive_coupling(self):
        """Repeatedly rewarding a target where byte0's bit0==bit1 should
        drive that pair's coupling positive over many iterations, the
        correlation analogue of test_rotation_improves_fitness_sequence
        in test_qea.py."""
        qea = QEALifecycle(pop_size=1, rotation_angle=0.1, use_correlation=True, correlation_delta=0.05)
        ind = QEAIndividual(
            amplitudes=[ALPHA_UNIFORM] * 8,
            best_collapsed=b"\x00",
            coupling=_zero_coupling(1),
        )
        qea.population = [ind]
        target = b"\xc0"  # 11000000: bit0=1, bit1=1 (agree)
        et = EdgeTracker()
        for _ in range(80):
            qea._last_parent = ind
            qea._last_collapsed = target
            qea.on_fuzz_result(target, new_coverage=True, edge_count=10, edge_tracker=et)
        assert ind.coupling[0, 0, 1] > 0.3, (
            f"expected learned positive coupling, got {ind.coupling[0, 0, 1]:.3f}"
        )

    def test_repeated_reward_of_mismatched_pair_learns_negative_coupling(self):
        qea = QEALifecycle(pop_size=1, rotation_angle=0.1, use_correlation=True, correlation_delta=0.05)
        ind = QEAIndividual(
            amplitudes=[ALPHA_UNIFORM] * 8,
            best_collapsed=b"\x00",
            coupling=_zero_coupling(1),
        )
        qea.population = [ind]
        target = b"\x80"  # 10000000: bit0=1, bit1=0 (disagree)
        et = EdgeTracker()
        for _ in range(80):
            qea._last_parent = ind
            qea._last_collapsed = target
            qea.on_fuzz_result(target, new_coverage=True, edge_count=10, edge_tracker=et)
        assert ind.coupling[0, 0, 1] < -0.3, (
            f"expected learned negative coupling, got {ind.coupling[0, 0, 1]:.3f}"
        )

    def test_learned_coupling_biases_future_collapses(self):
        """After learning a positive bit0/bit1 coupling, collapse_correlated
        should reproduce that agreement more often than collapse() would."""
        coupling = _zero_coupling(1)
        # Simulate many rounds of Hebbian reinforcement toward agreement.
        for _ in range(50):
            update_couplings(coupling, b"\xc0", improved=True, delta=0.05)
        amps = np.array([ALPHA_UNIFORM] * 8, dtype=np.float64)
        agree = sum(
            1
            for _ in range(300)
            if ((r := collapse_correlated(amps, coupling, n_sweeps=5))[0] >> 7 & 1)
            == (r[0] >> 6 & 1)
        )
        assert agree / 300 > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
