"""Tests for generic Floyd cycle detection (core/cycle_detect.py).

Uses small hand-verifiable modular-arithmetic sequences (rho-shaped:
a tail then a cycle) so `mu` and `period` have known exact values,
plus the two real call sites' shapes (float-vector state with
tolerance-based equality) to make sure the tolerance path works too.
"""

import numpy as np
import pytest

from fuzzer_tool.core.cycle_detect import cesaro_average, floyd_detect


def test_pure_cycle_from_start():
    """x -> (x + 1) % 5 starting at 0: mu=0, period=5 (no tail)."""
    result = floyd_detect(0, lambda x: (x + 1) % 5, is_close=lambda a, b: a == b)
    assert result is not None
    assert result.mu == 0
    assert result.period == 5


def test_rho_shaped_sequence_with_tail():
    """Classic rho shape: a tail of known length, then a cycle of known length.

    f(x) = x+1 for x < 3 (tail 0,1,2), then f(x) = 3 + (x-3+1) % 4 for
    x >= 3 (a 4-cycle: 3,4,5,6,3,4,...). So mu=3, period=4.
    """

    def f(x):
        if x < 3:
            return x + 1
        return 3 + (x - 3 + 1) % 4

    result = floyd_detect(0, f, is_close=lambda a, b: a == b)
    assert result is not None
    assert result.mu == 3
    assert result.period == 4
    assert 3 <= result.state < 7  # representative state is inside the cycle


def test_no_cycle_within_budget_returns_none():
    """A strictly increasing (non-repeating) sequence never collides."""
    result = floyd_detect(0, lambda x: x + 1, is_close=lambda a, b: a == b, max_steps=500)
    assert result is None


def test_fixed_point_is_a_period_1_cycle():
    """A step function that immediately settles is a cycle of length 1."""
    result = floyd_detect(5, lambda x: 5, is_close=lambda a, b: a == b)
    assert result is not None
    assert result.period == 1


def test_float_vector_state_with_tolerance():
    """The real call sites compare float vectors within a tolerance, not
    exact equality — a 2-cycle that swaps two numpy array components."""
    P = np.array([[0.0, 1.0], [1.0, 0.0]])

    def step(v):
        return v @ P

    def is_close(a, b):
        return bool(np.abs(a - b).sum() < 1e-9)

    x0 = np.array([0.9, 0.1])
    result = floyd_detect(x0, step, is_close, max_steps=100)
    assert result is not None
    assert result.period == 2


@pytest.mark.parametrize("period", [2, 3, 5])
def test_period_matches_modulus(period):
    result = floyd_detect(0, lambda x, m=period: (x + 1) % m, is_close=lambda a, b: a == b)
    assert result.mu == 0
    assert result.period == period


def test_cesaro_average_of_alternating_scalars():
    """Averaging a 2-cycle that alternates 10 and 20 should give 15."""
    state = 10
    step = lambda x: 30 - x  # noqa: E731 - toggles 10 <-> 20
    avg = cesaro_average(state, step, period=2)
    assert avg == 15.0


def test_cesaro_average_of_vector_cycle():
    """Same, but for a list-valued state using explicit add/scale, matching
    the pure-Python power-iteration call site's usage."""
    state = [1.0, 0.0]

    def step(v):
        return [v[1], v[0]]  # swap

    avg = cesaro_average(
        state,
        step,
        period=2,
        add=lambda a, b: [x + y for x, y in zip(a, b, strict=False)],
        scale=lambda a, k: [x * k for x in a],
    )
    assert avg == [0.5, 0.5]


def test_cesaro_average_period_one_is_the_state_itself():
    state = 42
    avg = cesaro_average(state, lambda x: x, period=1)
    assert avg == 42.0
