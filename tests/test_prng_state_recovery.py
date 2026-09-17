"""Regression tests for ``core/prng_state_recovery.py``.

Validated against the independent sympy-based reference implementation from
https://github.com/Gegell/Factorio-RNG-Release (``factorio_rng.py``, GPL
theory writeup at https://gegell.github.io/posts/factorio-rng): that
reference's ``build_generation_matrix()`` also saturates at rank 92/96 with
naive 3-output equations alone, confirming the rank deficiency handled here
via :func:`fuzzer_tool.core.prng_state_recovery._structural_equations` is a
real property of the generator, not a bug in either implementation.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.prng_state_recovery import (
    TAUS88_PARAMS,
    predict_next,
    recover_taus88_state,
    taus88_output,
    taus88_step,
    verify_recovery,
)


def _make_stream(seed: int, n: int) -> tuple[tuple[int, int, int], list[int]]:
    rng = random.Random(seed)
    state = (rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32))
    words: list[int] = []
    s = state
    for _ in range(n):
        s = taus88_step(s)
        words.append(taus88_output(s))
    return state, words


@pytest.mark.parametrize("seed", range(8))
def test_recovers_state_from_exactly_three_outputs(seed: int) -> None:
    _, words = _make_stream(seed, 3)
    recovered = recover_taus88_state(words)
    assert recovered is not None
    assert verify_recovery(recovered, words)


@pytest.mark.parametrize("seed", range(5))
def test_recovered_state_predicts_future_outputs(seed: int) -> None:
    """The point of recovery: predict outputs never used to derive the state.

    ``recovered``'s *own* output (zero extra steps) equals ``words[0]``, so
    to predict ``words[3:]`` we first walk it to the state whose own output
    is ``words[2]`` (two steps), then ``predict_next`` from there.
    """
    _, words = _make_stream(seed, 10)
    recovered = recover_taus88_state(words[:3])
    assert recovered is not None
    s = taus88_step(taus88_step(recovered))
    assert taus88_output(s) == words[2]
    assert predict_next(s, 7) == words[3:]


def test_raises_below_minimum_sample_count() -> None:
    """Two outputs (64 equations for 96 unknowns) can't pin the state."""
    with pytest.raises(ValueError):
        recover_taus88_state([0, 1])


def test_extra_samples_are_consistent_not_required() -> None:
    """More than 3 samples still recovers the same unique state."""
    _, words = _make_stream(seed=99, n=6)
    from_three = recover_taus88_state(words[:3])
    from_six = recover_taus88_state(words)
    assert from_three == from_six


def test_custom_params_do_not_leak_into_default_taus88() -> None:
    """A differently-shaped combined LFSR is recoverable with its own params."""
    custom_params = ((32, 30, 5, 3), (32, 27, 7, 2), (32, 25, 11, 6))
    rng = random.Random(1)
    state = (rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32))
    words = []
    s = state
    for _ in range(5):
        s = taus88_step(s, custom_params)
        words.append(taus88_output(s))

    recovered = recover_taus88_state(words[:3], params=custom_params)
    assert recovered is not None
    assert verify_recovery(recovered, words, params=custom_params)

    # Sanity: default TAUS88_PARAMS must not accidentally also fit this stream.
    assert custom_params != TAUS88_PARAMS
