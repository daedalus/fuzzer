"""Modulo-bias guard and min-entropy gate for RandPool."""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool, _min_entropy


def test_min_entropy_detects_heavy_hitter_shannon_misses():
    rng = np.random.default_rng(1)
    heavy = np.concatenate([np.zeros(205, np.uint8), rng.integers(0, 256, 3891, dtype=np.uint8)])
    assert _min_entropy(heavy.tobytes()) < 5.0
    assert _min_entropy(rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()) > 6.5


def test_hinf_gate_passes_on_healthy_stream():
    p = RandPool(seed=3, min_hinf=6.5)
    for _ in range(20000):
        p.randint(0, 9)
    assert p.pool_min_entropy_now() > 6.5


def test_hinf_gate_raises_when_unsatisfiable():
    p = RandPool(seed=3, min_hinf=8.0)
    with pytest.raises(RuntimeError, match="min-entropy starvation"):
        p.randint(0, 9)


@pytest.mark.parametrize("n", [(3 << 30), int(2**32 * 2 / 3)])
def test_large_range_is_unbiased_by_rejection(n):
    # draw % n would put P(x < 2**32 - n) at 2/2**32-per-point ~double; check
    # the lower third is not over-represented beyond sampling noise.
    p = RandPool(seed=11)
    xs = np.array([p.randrange(n) for _ in range(40000)])
    cut = (1 << 32) - n
    frac = float((xs < cut).mean())
    assert abs(frac - cut / n) < 0.02
    assert 0 <= xs.min() and xs.max() < n


def test_list_apis_agree_on_range():
    p = RandPool(seed=5)
    assert all(0 <= v < (3 << 30) for v in p.randrange_list(3 << 30, 50))
    assert all(5 <= v <= 5 + (3 << 30) for v in p.randint_list(5, 5 + (3 << 30), 50))
    assert len(p.randrange_list(1 << 32, 10)) == 10
