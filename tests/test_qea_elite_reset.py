"""QEA incumbent discard and the zero-coupling endpoint.

Two ports from the p-bit lattice study:

* elites are carried forward verbatim and are what tournament selection
  keeps drawing, so a slot can be held indefinitely by an individual whose
  region is exhausted. ``elite_reset_every`` breaks that anchor.
* the study's largest effect came from turning the coupling term *off*,
  which was only findable because coupling strength was a parameter. QEA's
  couplings are the rotation gate and the amplitude bias; both must be
  settable to their no-op values.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.qea import QEAIndividual, QEALifecycle, rotation_gate


class _StubEdgeTracker:
    cumulative_edges: set = set()

    def compute_wasserstein_weight(self, _seed_key):
        return 1.0


def _populate(qea: QEALifecycle, n: int) -> None:
    for i in range(n):
        ind = QEAIndividual(
            amplitudes=np.full(64, 0.5),
            edge_count=i,
            best_collapsed=bytes([i % 256]) * 8,
        )
        ind.fitness = float(i)
        qea.population.append(ind)


class TestEliteReset:
    def test_elites_are_carried_by_default(self):
        # _evolve re-scores fitness, so identify survivors by identity
        # against the pre-evolution population rather than by rank.
        qea = QEALifecycle(pop_size=20, elite_fraction=0.25)
        _populate(qea, 20)
        before = set(map(id, qea.population))
        qea._evolve(_StubEdgeTracker())
        assert before & set(map(id, qea.population))

    def test_reset_generation_drops_every_elite(self):
        # elite_reset_every=2 fires when (generation + 1) % 2 == 0.
        qea = QEALifecycle(pop_size=20, elite_fraction=0.25, elite_reset_every=2)
        _populate(qea, 20)
        qea.generation = 1
        carried = set(map(id, qea.population))
        qea._evolve(_StubEdgeTracker())
        assert not (carried & set(map(id, qea.population)))
        assert len(qea.population) == 20

    def test_reset_is_periodic_not_permanent(self):
        qea = QEALifecycle(pop_size=20, elite_fraction=0.25, elite_reset_every=2)
        _populate(qea, 20)
        qea.generation = 2  # (2 + 1) % 2 == 1 -> not a reset generation
        before = set(map(id, qea.population))
        qea._evolve(_StubEdgeTracker())
        assert before & set(map(id, qea.population))

    def test_zero_disables(self):
        qea = QEALifecycle(pop_size=20, elite_fraction=0.25, elite_reset_every=0)
        _populate(qea, 20)
        for gen in range(1, 5):
            qea.generation = gen
            before = set(map(id, qea.population))
            qea._evolve(_StubEdgeTracker())
            assert before & set(map(id, qea.population))


class TestZeroCoupling:
    def test_rotation_angle_zero_is_a_no_op(self):
        amps = np.array([0.2, 0.5, 0.8, 0.3])
        before = amps.copy()
        rotation_gate(amps, b"\xa5", improved=True, delta=0.0)
        assert np.array_equal(amps, before)
        rotation_gate(amps, b"\xa5", improved=False, delta=0.0)
        assert np.array_equal(amps, before)

    def test_lifecycle_accepts_the_zero_endpoint(self):
        qea = QEALifecycle(rotation_angle=0.0, strong_bias=0.5)
        assert qea.rotation_angle == 0.0
        assert qea.strong_bias == 0.5
