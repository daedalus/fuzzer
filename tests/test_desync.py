"""Tests for pulse-coupled desynchronization of the parallel sync cadence."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.circular_stats import order_parameter
from fuzzer_tool.core.desync import (
    DEFAULT_DAMPING,
    initial_offset,
    next_delay,
    phase_shift,
)

PERIOD = 30.0


def _spread(phases, period=PERIOD):
    """Kuramoto order parameter of a set of firing phases: 0 is uniform."""
    r, _ = order_parameter([2 * math.pi * p / period for p in phases])
    return r


# ── initial stagger ────────────────────────────────────────────────────


def test_initial_offset_spreads_workers_over_one_period():
    n = 8
    offsets = [initial_offset(i, n, PERIOD) for i in range(n)]
    assert offsets[0] == 0.0
    assert offsets == sorted(offsets)
    assert max(offsets) < PERIOD
    gaps = [b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    assert all(g == pytest.approx(PERIOD / n) for g in gaps)


def test_initial_offset_is_fully_incoherent():
    # FALSIFICATION: the current code gives every worker the same offset.
    # Evenly spaced phases cancel; a constant offset does not.
    n = 12
    assert _spread([initial_offset(i, n, PERIOD) for i in range(n)]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert _spread([0.0] * n) == pytest.approx(1.0)


def test_single_worker_gets_no_offset():
    assert initial_offset(0, 1, PERIOD) == 0.0


def test_zero_workers_is_treated_as_one():
    # ADVERSARIAL: n_workers is plumbed from a CLI flag.
    assert initial_offset(0, 0, PERIOD) == 0.0


# ── phase shift ────────────────────────────────────────────────────────


def test_uniform_neighbours_are_a_fixed_point():
    # Own phase 10, neighbours at 0 and 20: already evenly spaced.
    assert phase_shift(10.0, [0.0, 20.0], PERIOD) == pytest.approx(0.0)


def test_moves_away_from_a_crowded_neighbour():
    # Neighbour just behind, nothing ahead -> move forward.
    shift = phase_shift(10.0, [9.0], PERIOD)
    assert shift > 0


def test_moves_backward_when_crowded_from_ahead():
    shift = phase_shift(10.0, [11.0], PERIOD)
    assert shift < 0


def test_two_workers_converge_to_antiphase():
    own, other = 10.0, 11.0
    for _ in range(60):
        own = (own + phase_shift(own, [other], PERIOD)) % PERIOD
        other = (other + phase_shift(other, [own], PERIOD)) % PERIOD
    gap = abs(own - other) % PERIOD
    assert min(gap, PERIOD - gap) == pytest.approx(PERIOD / 2, abs=0.2)


def test_only_the_phase_neighbours_matter():
    # DESYNC is a local rule: a node beyond the immediate neighbours on
    # either side must not affect the midpoint.
    near = phase_shift(10.0, [5.0, 15.0], PERIOD)
    far = phase_shift(10.0, [5.0, 15.0, 1.0, 25.0], PERIOD)
    assert near == pytest.approx(far)


def test_wraps_across_the_period_boundary():
    # Own phase just after 0, neighbours straddling the wrap.
    shift = phase_shift(0.5, [29.5, 2.5], PERIOD)
    assert shift == pytest.approx(DEFAULT_DAMPING * (2.0 - 1.0) / 2.0)


def test_no_neighbours_means_no_coupling():
    assert phase_shift(10.0, [], PERIOD) == 0.0


def test_coincident_neighbour_is_a_known_fixed_point():
    # LIMITATION, pinned rather than papered over: two nodes at exactly the
    # same phase are a symmetric unstable equilibrium and no deterministic
    # local rule breaks it. This is why `initial_offset` staggers the fleet
    # up front instead of relying on convergence from a cold collision.
    assert phase_shift(10.0, [10.0], PERIOD) == 0.0


def test_shift_is_bounded_by_the_damping():
    # ADVERSARIAL: an unbounded shift could drive the next delay negative.
    bound = DEFAULT_DAMPING * PERIOD / 2
    for own in (0.0, 7.3, 15.0, 29.9):
        for nb in ([1.0], [0.1, 29.9], [14.0, 16.0], [2.0, 3.0, 4.0]):
            assert abs(phase_shift(own, nb, PERIOD)) <= bound + 1e-9


def test_damping_scales_the_correction():
    half = phase_shift(10.0, [9.0], PERIOD, damping=0.5)
    full = phase_shift(10.0, [9.0], PERIOD, damping=1.0)
    assert full == pytest.approx(2 * half)


@pytest.mark.parametrize("bad", [0.0, -0.5, 2.1])
def test_damping_outside_the_stable_range_is_rejected(bad):
    with pytest.raises(ValueError):
        phase_shift(10.0, [9.0], PERIOD, damping=bad)


@pytest.mark.parametrize("bad", [0.0, -30.0])
def test_non_positive_period_is_rejected(bad):
    with pytest.raises(ValueError):
        phase_shift(1.0, [2.0], bad)


# ── delay ──────────────────────────────────────────────────────────────


def test_delay_is_one_period_when_uncoupled():
    assert next_delay(10.0, [], PERIOD) == PERIOD


def test_delay_stays_positive_under_maximum_correction():
    # The sync loop subtracts this from a wall clock; a non-positive delay
    # would spin.
    for own in (0.0, 5.0, 17.5, 29.999):
        for nb in ([own - 0.001], [own + 0.001], [own + PERIOD / 2]):
            assert next_delay(own, nb, PERIOD, damping=1.0) > 0.0


def test_delay_tracks_the_shift():
    shift = phase_shift(10.0, [9.0], PERIOD)
    assert next_delay(10.0, [9.0], PERIOD) == pytest.approx(PERIOD + shift)


# ── convergence ────────────────────────────────────────────────────────


def test_a_clustered_fleet_spreads_out():
    # FALSIFICATION: with the coupling removed this stays clustered forever,
    # which is exactly the current behaviour of the sync loop.
    n = 8
    phases = [10.0 + 0.4 * i for i in range(n)]  # all inside a 3s window
    before = _spread(phases)
    assert before > 0.9

    for _ in range(200):
        for i in range(n):
            others = phases[:i] + phases[i + 1 :]
            phases[i] = (phases[i] + phase_shift(phases[i], others, PERIOD)) % PERIOD

    after = _spread(phases)
    assert after < 0.1, f"order parameter {before:.3f} -> {after:.3f}"


def test_convergence_reaches_even_spacing():
    n = 6
    phases = [1.0 * i for i in range(n)]
    for _ in range(400):
        for i in range(n):
            others = phases[:i] + phases[i + 1 :]
            phases[i] = (phases[i] + phase_shift(phases[i], others, PERIOD)) % PERIOD

    ordered = sorted(phases)
    gaps = [b - a for a, b in zip(ordered[:-1], ordered[1:], strict=True)]
    gaps.append(PERIOD - ordered[-1] + ordered[0])
    assert max(gaps) - min(gaps) < 0.5, gaps
