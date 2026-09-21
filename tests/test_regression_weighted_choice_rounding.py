"""``weighted_choice`` could raise IndexError on float64 rounding.

``r = self.random() * total`` is mathematically guaranteed ``< total``
(``self.random()`` draws from ``[0.0, 1.0)`` exactly, since it divides a
uint32 by a power of two), but float64 multiplication rounding can still
push ``r`` up to exactly ``total`` -- more likely as ``total`` grows, e.g.
via large or many weights accumulating rounding error. When that happens,
``bisect.bisect_right(cum, r)`` returns ``n`` (one past the last valid
index), and ``seq[n]`` raises ``IndexError: list index out of range``.

Reached in production through ``seed_entropy_gradient.py``'s
``select()``, which calls ``weighted_choice`` on the corpus with
entropy-gradient weights.
"""

import bisect
import itertools

import pytest

from fuzzer_tool.core.rand_pool import RandPool


def test_weighted_choice_never_indexerrors_on_boundary_r(monkeypatch):
    """r landing exactly on total (post-rounding) must not raise."""
    seq = ["a", "b", "c", "d"]
    weights = [1.0, 1.0, 1.0, 1.0]
    total = sum(weights)

    pool = RandPool(seed=1)
    # Force the float-rounding edge case deterministically: random() * total
    # rounds up to exactly total. RandPool uses __slots__, so patch the class.
    monkeypatch.setattr(RandPool, "random", lambda self: 1.0)

    # Sanity check this really is the boundary condition being guarded against.
    cum = list(itertools.accumulate(weights))
    assert bisect.bisect_right(cum, 1.0 * total) == len(seq)

    result = pool.weighted_choice(seq, weights)
    assert result in seq


def test_weighted_choice_boundary_returns_last_element(monkeypatch):
    """r == total should resolve to the last element, not wrap or skip."""
    seq = ["a", "b", "c"]
    weights = [2.0, 3.0, 5.0]

    pool = RandPool(seed=2)
    monkeypatch.setattr(RandPool, "random", lambda self: 1.0)

    assert pool.weighted_choice(seq, weights) == "c"


def test_weighted_choice_zero_weights_still_raises_indexerror():
    """All-zero weights (total <= 0) remains a caller-bug IndexError.

    This is unchanged, deliberate behavior (see
    TestWeightedChoice.test_all_weights_zero_raises in test_rand_pool.py) --
    only the float-rounding boundary case above is being fixed here, not
    this one.
    """
    seq = ["a", "b", "c"]
    weights = [0.0, 0.0, 0.0]

    pool = RandPool(seed=3)
    with pytest.raises(IndexError):
        pool.weighted_choice(seq, weights)


@pytest.mark.parametrize("seed", range(20))
def test_weighted_choice_many_draws_stay_in_range(seed):
    """Broad sweep: normal draws must always return a valid element."""
    seq = list(range(50))
    weights = [float(i + 1) for i in range(50)]
    pool = RandPool(seed=seed)
    for _ in range(2000):
        result = pool.weighted_choice(seq, weights)
        assert result in seq
