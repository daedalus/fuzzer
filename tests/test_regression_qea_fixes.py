"""Regression tests for four QEA defects.

Each class pins behaviour that was previously wrong. The comment at the top
of each class records what the broken behaviour was, so a future refactor
that reintroduces it fails here with an obvious explanation rather than a
bare assertion error.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.core.qea import (
    QEAIndividual,
    QEALifecycle,
    _bias_amplitudes_from,
    collapse,
)


def _edge_tracker(total_edges=8, weight=1.0):
    """Minimal EdgeTracker stub with the three attributes QEA touches."""
    et = MagicMock()
    et.cumulative_edges = set(range(total_edges))
    et.seed_edges = {}
    et.compute_wasserstein_weight.return_value = weight
    return et


def _full_population(qea, n, base_fitness=0.05):
    """Fill a population with already-evaluated, non-zero-fitness members.

    Fitness values must stay inside the range FitnessFunction can actually
    produce — its weights sum to 0.9, so fabricating values like 5.0 would
    make even a maximally-fit newcomer lose and would test nothing.
    """
    for i in range(n):
        ind = QEAIndividual(
            amplitudes=_bias_amplitudes_from(bytes([i]) * 8),
            best_collapsed=bytes([i]) * 8,
        )
        ind.fitness = base_fitness + i * 0.05
        qea.population.append(ind)


# ── Bug 1: new-coverage individuals were never scored ──────────────────


class TestNewCoverageIndividualIsScored:
    """Previously ``on_fuzz_result`` built the individual without scoring it,
    leaving fitness at 0.0. ``add_to_population`` admits on fitness, so once
    the population filled every coverage-finding seed lost the comparison and
    was silently discarded — QEA never learned from any discovery."""

    def test_returned_individual_has_nonzero_fitness(self):
        qea = QEALifecycle(pop_size=5, generation_size=10_000)
        et = _edge_tracker()
        ind = qea.on_fuzz_result(b"discovery", True, 4, et)
        assert ind is not None
        assert ind.fitness > 0.0, "coverage individual must be scored before admission"

    def test_individual_is_admitted_into_a_full_population(self):
        qea = QEALifecycle(pop_size=5, generation_size=10_000)
        _full_population(qea, 5)
        et = _edge_tracker()
        ind = qea.on_fuzz_result(b"discovery", True, 8, et)
        qea.add_to_population(ind)
        assert any(i.best_collapsed == b"discovery" for i in qea.population)

    def test_unscored_individual_would_lose_to_every_member(self):
        """Pins the exact pre-fix mechanism, independent of the fix's shape.

        An individual entering admission with the default fitness of 0.0 is
        rejected by a full population of evaluated members. This is why
        scoring before returning is load-bearing rather than cosmetic.
        """
        qea = QEALifecycle(pop_size=5, generation_size=10_000)
        _full_population(qea, 5)
        unscored = QEAIndividual(
            amplitudes=_bias_amplitudes_from(b"discovery"),
            best_collapsed=b"discovery",
            edge_count=999,
        )
        assert unscored.fitness == 0.0
        qea.add_to_population(unscored)
        assert not any(i.best_collapsed == b"discovery" for i in qea.population)

    def test_scoring_matches_evaluate_all(self):
        """The new individual must be scored the same way members are."""
        qea = QEALifecycle(pop_size=5, generation_size=10_000)
        et = _edge_tracker(total_edges=8, weight=1.4)
        ind = qea.on_fuzz_result(b"discovery", True, 4, et)

        twin = QEAIndividual(
            amplitudes=_bias_amplitudes_from(b"discovery"),
            edge_count=4,
            generation=qea.generation,
            best_collapsed=b"discovery",
            seed_key=ind.seed_key,
        )
        qea.population = [twin]
        qea._evaluate_all(et)
        assert twin.fitness == pytest.approx(ind.fitness)
        assert twin.diversity_score == pytest.approx(ind.diversity_score)

    def test_no_individual_returned_without_coverage(self):
        qea = QEALifecycle(pop_size=5, generation_size=10_000)
        assert qea.on_fuzz_result(b"nothing", False, 0, _edge_tracker()) is None


# ── Bug 2: rotation gate credited a stale parent ───────────────────────


class TestRotationGateClearsParent:
    """``pick_seed`` runs only when QEA wins seed arbitration, but
    ``on_fuzz_result`` runs every iteration. Previously ``_last_parent`` was
    never cleared, so one individual absorbed rotations driven by unrelated
    seeds — overwhelmingly ``improved=False`` — pinning its amplitudes to the
    clamps and destroying the uncertainty QEA exists to maintain."""

    def test_parent_is_cleared_after_one_rotation(self):
        qea = QEALifecycle(pop_size=4, generation_size=10_000)
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], _edge_tracker())
        qea.pick_seed()
        assert qea._last_parent is not None
        qea.on_fuzz_result(b"x", False, 0, _edge_tracker())
        assert qea._last_parent is None
        assert qea._last_collapsed == b""

    def test_unrelated_iterations_do_not_touch_amplitudes(self):
        """Only the iteration immediately following pick_seed may rotate."""
        qea = QEALifecycle(pop_size=4, generation_size=10_000)
        et = _edge_tracker()
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
        qea.pick_seed()
        parent = qea._last_parent
        qea.on_fuzz_result(b"x", False, 0, et)  # the one legitimate rotation
        after_first = parent.amplitudes.copy()

        for _ in range(64):  # unrelated iterations, no pick_seed
            qea.on_fuzz_result(b"unrelated", False, 0, et)

        np.testing.assert_array_equal(parent.amplitudes, after_first)

    def test_amplitudes_do_not_saturate_from_unrelated_traffic(self):
        """The pre-fix failure mode: 100% of amplitudes pinned to the clamps."""
        qea = QEALifecycle(pop_size=4, generation_size=10_000)
        et = _edge_tracker()
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
        qea.pick_seed()
        parent = qea._last_parent
        for _ in range(64):
            qea.on_fuzz_result(b"unrelated", False, 0, et)

        pinned = np.mean((parent.amplitudes <= 0.011) | (parent.amplitudes >= 0.989))
        assert pinned < 0.5, f"{pinned:.0%} of amplitudes saturated from stale rotations"

    def test_rotation_still_applies_on_the_matching_iteration(self):
        """Clearing must not disable legitimate feedback."""
        qea = QEALifecycle(pop_size=4, generation_size=10_000)
        et = _edge_tracker()
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
        qea.pick_seed()
        parent = qea._last_parent
        before = parent.amplitudes.copy()
        qea.on_fuzz_result(b"x", True, 1, et)
        assert not np.array_equal(parent.amplitudes, before)


# ── Bug 3: generation boundary starved during productive runs ──────────


class TestGenerationBoundaryAlwaysChecked:
    """The ``new_coverage`` branch returned before the boundary check, so
    generations only advanced during unproductive stretches. A run that kept
    finding coverage never evolved, and ``iterations_since_gen`` grew without
    bound."""

    def test_generations_advance_while_finding_coverage(self):
        qea = QEALifecycle(pop_size=4, generation_size=5)
        et = _edge_tracker()
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
        for _ in range(60):
            qea.on_fuzz_result(b"cov", True, 1, et)
        assert qea.generation > 0, "evolution starved during a productive run"
        assert qea.iterations_since_gen < qea.generation_size

    def test_coverage_and_barren_runs_advance_alike(self):
        """Generation count must not depend on whether coverage was found."""
        et = _edge_tracker()
        counts = []
        for found in (True, False):
            qea = QEALifecycle(pop_size=4, generation_size=5)
            qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
            for _ in range(60):
                qea.on_fuzz_result(b"x", found, 1 if found else 0, et)
            counts.append(qea.generation)
        assert counts[0] == counts[1] == 12

    def test_individual_still_returned_when_boundary_coincides(self):
        """Hitting the boundary must not swallow the discovered individual."""
        qea = QEALifecycle(pop_size=4, generation_size=1)
        et = _edge_tracker()
        qea.initialize([b"aaaa", b"bbbb", b"cccc", b"dddd"], et)
        ind = qea.on_fuzz_result(b"cov", True, 1, et)
        assert ind is not None
        assert qea.generation == 1


# ── Bug 4: asymmetric bias drifted inputs toward set bits ──────────────


class TestBiasIsSymmetric:
    """``_bias_amplitudes_from`` complemented the amplitude (α = 1 - p)
    instead of the probability. At the 0.9 default a one bit was retained
    with probability 0.99 against 0.81 for a zero bit, drifting every collapse
    toward 0xFF and compounding through each breed/re-bias cycle."""

    def test_retention_probability_is_equal_for_both_bit_values(self):
        zeros = _bias_amplitudes_from(b"\x00", strong_prob=0.9)
        ones = _bias_amplitudes_from(b"\xff", strong_prob=0.9)
        p_keep_zero = zeros[0] ** 2
        p_keep_one = 1 - ones[0] ** 2
        assert p_keep_zero == pytest.approx(p_keep_one)
        assert p_keep_zero == pytest.approx(0.81)

    @pytest.mark.parametrize("strong_prob", [0.6, 0.7, 0.8, 0.9, 0.99])
    def test_symmetry_holds_across_strong_prob(self, strong_prob):
        zeros = _bias_amplitudes_from(b"\x00", strong_prob=strong_prob)
        ones = _bias_amplitudes_from(b"\xff", strong_prob=strong_prob)
        assert zeros[0] ** 2 == pytest.approx(1 - ones[0] ** 2)
        assert ones[0] == pytest.approx(math.sqrt(1 - strong_prob**2))

    def test_collapse_does_not_drift_bit_density(self):
        """The observable symptom: 1-bit density rose ~9% per collapse."""
        rng = np.random.RandomState(0)
        data = rng.randint(0, 256, 512, dtype=np.uint8).tobytes()
        source = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).mean()

        amps = _bias_amplitudes_from(data, strong_prob=0.9)
        np.random.seed(0)
        densities = [
            np.unpackbits(np.frombuffer(collapse(amps), dtype=np.uint8)).mean() for _ in range(200)
        ]
        assert abs(np.mean(densities) - source) < 0.02

    def test_mixed_byte_biases_each_bit_toward_its_own_value(self):
        amps = _bias_amplitudes_from(b"\xaa", strong_prob=0.9)  # 10101010
        for i, bit in enumerate((1, 0, 1, 0, 1, 0, 1, 0)):
            p_keep = (1 - amps[i] ** 2) if bit else (amps[i] ** 2)
            assert p_keep == pytest.approx(0.81), f"bit {i} (value {bit}) not retained at 0.81"

    def test_amplitudes_stay_in_valid_range(self):
        """α must remain a valid amplitude for every strong_prob."""
        for sp in (0.5, 0.7071, 0.9, 1.0):
            amps = _bias_amplitudes_from(b"\x00\xff\xa5", strong_prob=sp)
            assert np.all(amps >= 0.0)
            assert np.all(amps <= 1.0)
            assert not np.any(np.isnan(amps))
