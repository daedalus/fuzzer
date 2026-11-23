"""poisson_array runs on its own stream, independent of pool draws."""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool

_LAMS = np.array([0.5, 3.0, 40.0, 700.0])


def _draw(pool):
    return pool.poisson_array(np.tile(_LAMS, 50)).tolist()


def test_independent_of_pool_draws():
    a = RandPool(seed=11)
    b = RandPool(seed=11)
    for _ in range(5000):
        b.randrange(97)
    assert _draw(a) == _draw(b)


def test_independent_of_generator_backed_helpers():
    a = RandPool(seed=12)
    b = RandPool(seed=12)
    b.shuffle(list(range(64)))
    b.gauss_list(0.0, 1.0, 100)
    b.sample(list(range(50)), 7)
    assert _draw(a) == _draw(b)


def test_independent_of_gate_retry():
    # An unreachable gate forces retries (then starvation) on the main
    # stream; the aux stream must be untouched by the reseeds.
    a = RandPool(seed=13)
    b = RandPool(seed=13, min_hinf=99.0)
    with pytest.raises(RuntimeError):
        b.randrange(10)
    assert _draw(a) == _draw(b)


def test_deterministic_and_seed_sensitive():
    assert _draw(RandPool(seed=1)) == _draw(RandPool(seed=1))
    assert _draw(RandPool(seed=1)) != _draw(RandPool(seed=2))
    assert _draw(RandPool(seed=None)) == _draw(RandPool(seed=None))


def test_reseed_restarts_stream():
    p = RandPool(seed=5)
    first = _draw(p)
    _draw(p)
    p.reseed(5)
    assert _draw(p) == first


def test_inject_entropy_changes_stream_deterministically():
    a, b, c = RandPool(seed=6), RandPool(seed=6), RandPool(seed=6)
    a.inject_entropy(b"abc")
    b.inject_entropy(b"abc")
    assert _draw(a) == _draw(b)
    assert _draw(a) != _draw(c)


def test_main_stream_unchanged_by_aux_draws():
    a = RandPool(seed=21)
    b = RandPool(seed=21)
    _draw(b)
    assert [a.randrange(1000) for _ in range(5000)] == [b.randrange(1000) for _ in range(5000)]


def test_bad_rates_raise_valueerror():
    p = RandPool(seed=0)
    for bad in (-1.0, float("nan"), float("inf"), 1e19):
        with pytest.raises(ValueError):
            p.poisson_array(np.array([bad]))
    assert p.poisson_array(np.array([0.0, 0.0])).tolist() == [0, 0]
