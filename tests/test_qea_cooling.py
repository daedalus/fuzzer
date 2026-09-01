"""Tests for QEA's opt-in algorithmic cooling (Δθ decay) schedule.

rotation_gate()'s cos(Δθ)/sin(Δθ) rotation already decelerates *per bit*
as that bit's own amplitude approaches certainty -- a local effect with
no memory of generation count. Algorithmic cooling is a separate, global
schedule that shrinks the base angle Δθ itself as generations pass,
anchored to elite_reset_every's cycle boundary so a reset also restores
full step size (see the "Algorithmic cooling" section in qea.py above
collapse_correlated()).

Default is off, so these tests also pin down that the feature is fully
inert unless explicitly enabled -- existing callers, existing saved
populations, and every pre-existing test see zero behavior change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from fuzzer_tool.core.qea import QEAIndividual, QEALifecycle


def _edge_tracker(total_edges=8, weight=1.0):
    et = MagicMock()
    et.cumulative_edges = set(range(total_edges))
    et.seed_edges = {}
    et.compute_wasserstein_weight.return_value = weight
    return et


class TestCoolingDisabledByDefault:
    def test_use_cooling_defaults_false(self):
        qea = QEALifecycle()
        assert qea.use_cooling is False

    def test_effective_angle_is_constant_when_disabled(self):
        qea = QEALifecycle(rotation_angle=0.05, use_cooling=False)
        for gen in (0, 1, 50, 10_000):
            qea.generation = gen
            assert qea._effective_rotation_angle() == 0.05

    def test_disabled_ignores_elite_reset_every(self):
        qea = QEALifecycle(rotation_angle=0.07, use_cooling=False, elite_reset_every=10)
        qea.generation = 7
        assert qea._effective_rotation_angle() == 0.07


class TestCoolingEnabledNoResetCycle:
    """elite_reset_every=0: no cycle boundary, decay runs against raw generation."""

    def test_generation_zero_is_base_angle(self):
        qea = QEALifecycle(
            rotation_angle=0.05, use_cooling=True, elite_reset_every=0, cooling_decay=0.9
        )
        qea.generation = 0
        assert qea._effective_rotation_angle() == 0.05

    def test_decays_monotonically_with_generation(self):
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=0,
            cooling_decay=0.9,
            cooling_min_angle=0.0,
        )
        angles = []
        for gen in range(6):
            qea.generation = gen
            angles.append(qea._effective_rotation_angle())
        assert all(a2 <= a1 for a1, a2 in zip(angles, angles[1:]))
        assert angles[-1] < angles[0]

    def test_matches_closed_form_decay(self):
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=0,
            cooling_decay=0.9,
            cooling_min_angle=0.0,
        )
        qea.generation = 4
        assert qea._effective_rotation_angle() == 0.05 * (0.9**4)

    def test_never_decays_below_floor(self):
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=0,
            cooling_decay=0.5,
            cooling_min_angle=0.01,
        )
        qea.generation = 10_000  # 0.05 * 0.5**10000 underflows far past the floor
        assert qea._effective_rotation_angle() == 0.01


class TestCoolingEnabledWithResetCycle:
    """elite_reset_every>0: Δθ decays within a cycle, snaps back at each boundary."""

    def test_snaps_back_at_cycle_boundary(self):
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=5,
            cooling_decay=0.8,
            cooling_min_angle=0.0,
        )
        for gen in (0, 5, 10, 25):
            qea.generation = gen
            assert qea._effective_rotation_angle() == 0.05

    def test_decays_within_a_cycle(self):
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=5,
            cooling_decay=0.8,
            cooling_min_angle=0.0,
        )
        qea.generation = 3  # 3rd step within the [0,5) cycle
        assert qea._effective_rotation_angle() == 0.05 * (0.8**3)

    def test_cycle_position_matches_across_cycles(self):
        # generation=2 and generation=2+5=7 are both position 2 in their
        # respective cycles and must decay identically.
        qea = QEALifecycle(
            rotation_angle=0.05,
            use_cooling=True,
            elite_reset_every=5,
            cooling_decay=0.8,
            cooling_min_angle=0.0,
        )
        qea.generation = 2
        early = qea._effective_rotation_angle()
        qea.generation = 7
        later = qea._effective_rotation_angle()
        assert early == later


class TestCoolingIntegratesWithOnFuzzResult:
    """The rotation gate applied inside on_fuzz_result() must use the
    cooled angle, not the constant self.rotation_angle, once enabled.
    """

    def _lifecycle_with_pending_parent(self, **kwargs) -> QEALifecycle:
        qea = QEALifecycle(pop_size=1, rotation_angle=0.05, **kwargs)
        parent = QEAIndividual(amplitudes=np.full(8, 0.5, dtype=np.float64))
        qea.population = [parent]
        qea._last_parent = parent
        qea._last_collapsed = b"\xff"
        return qea

    def test_cooled_rotation_moves_amplitude_less_than_uncooled(self):
        edge_tracker = _edge_tracker()

        cold = self._lifecycle_with_pending_parent(
            use_cooling=True, elite_reset_every=0, cooling_decay=0.5, cooling_min_angle=0.0
        )
        cold.generation = 20  # deep into decay: 0.05 * 0.5**20 ~ 0
        cold_parent = cold._last_parent
        cold.on_fuzz_result(b"\xff", new_coverage=True, edge_count=1, edge_tracker=edge_tracker)

        hot = self._lifecycle_with_pending_parent(use_cooling=False)
        hot_parent = hot._last_parent
        hot.on_fuzz_result(b"\xff", new_coverage=True, edge_count=1, edge_tracker=edge_tracker)

        cold_move = np.abs(cold_parent.amplitudes - 0.5).sum()
        hot_move = np.abs(hot_parent.amplitudes - 0.5).sum()
        assert cold_move < hot_move

    def test_clears_pending_parent_same_as_before(self):
        qea = self._lifecycle_with_pending_parent(use_cooling=True, elite_reset_every=5)
        qea.on_fuzz_result(b"\xff", new_coverage=True, edge_count=1, edge_tracker=_edge_tracker())
        assert qea._last_parent is None
        assert qea._last_collapsed == b""
