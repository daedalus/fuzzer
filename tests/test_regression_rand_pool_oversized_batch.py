"""Batch draws must return what they were asked for.

``random_list`` and ``randint_list`` refilled the pool and then sliced
``self._pool[idx : idx + count]``. numpy slicing past the end of an array
truncates instead of raising, so any count above ``_POOL_ENTRIES`` came back
short while ``_idx`` advanced by the full count.

Reachable through ``operators.py`` prefilling ``_dict_scratch`` with
``max(n_mutations * 8, 64)``: defaults stay under the pool, but
``-M/--mutations`` has no upper bound. Past the 4096th entry the dict
operators guard on ``idx < len(scratch)`` and become silent no-ops, and one
site falls back to dictionary token 0, so the bandit scores them as failures
for a reason unrelated to the target.
"""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool

OVER = _POOL_ENTRIES * 2 + 37  # spans three refills, ends mid-pool


@pytest.mark.parametrize(
    "count", [1, 64, _POOL_ENTRIES - 1, _POOL_ENTRIES, _POOL_ENTRIES + 1, OVER]
)
def test_random_list_returns_requested_count(count):
    assert len(RandPool(seed=1).random_list(count)) == count


@pytest.mark.parametrize(
    "count", [1, 64, _POOL_ENTRIES - 1, _POOL_ENTRIES, _POOL_ENTRIES + 1, OVER]
)
def test_randint_list_returns_requested_count(count):
    assert len(RandPool(seed=1).randint_list(0, 255, count)) == count


@pytest.mark.parametrize("width", [2, 10, 256, 1000])
def test_randint_list_respects_bounds_across_refills(width):
    values = RandPool(seed=3).randint_list(7, 7 + width - 1, OVER)

    assert len(values) == OVER
    assert min(values) >= 7
    assert max(values) <= 7 + width - 1


def test_randint_list_accepts_a_negative_lower_bound():
    """uint32 pool words rejected a negative offset outright."""
    values = RandPool(seed=4).randint_list(-5, 5, 100)

    assert len(values) == 100
    assert min(values) >= -5
    assert max(values) <= 5


def test_random_list_values_stay_in_unit_interval():
    values = RandPool(seed=5).random_list(OVER)

    assert len(values) == OVER
    assert min(values) >= 0.0
    assert max(values) < 1.0


def test_oversized_draw_is_not_a_repeated_pool():
    """Adversarial: a naive chunk loop that forgets to refill would repeat."""
    values = RandPool(seed=6).random_list(OVER)
    first = values[:_POOL_ENTRIES]
    second = values[_POOL_ENTRIES : 2 * _POOL_ENTRIES]

    assert first != second


def test_index_is_consistent_after_an_oversized_draw():
    """_idx advanced past the pool, so later scalar draws read stale words."""
    pool = RandPool(seed=7)
    pool.random_list(OVER)

    assert 0 <= pool._idx <= _POOL_ENTRIES
    assert 0.0 <= pool.random() < 1.0


def test_draws_continue_correctly_after_an_oversized_draw():
    pool = RandPool(seed=8)
    pool.randint_list(0, 255, OVER)
    tail = pool.randint_list(0, 9, 5000)

    assert len(tail) == 5000
    assert set(tail) <= set(range(10))


def test_oversized_draw_is_seed_reproducible():
    assert RandPool(seed=9).random_list(OVER) == RandPool(seed=9).random_list(OVER)


def test_small_draws_are_unchanged_by_the_fix():
    """A draw that fits in the pool must keep its existing sequence."""
    pool = RandPool(seed=11)
    if pool._idx + 100 > _POOL_ENTRIES:
        pool._refill()
    expected = (pool._pool[pool._idx : pool._idx + 100].astype(np.float64) / 2**32).tolist()

    assert RandPool(seed=11).random_list(100) == expected
