"""Unseeded-run reproducibility: RandPool(seed=None) is deterministic.

Background (LWN "Python's two modules for random numbers" + internal audit)
---------------------------------------------------------------------------
``random.seed(None)`` and ``np.random.seed(None)`` reseed from OS entropy, so
two unseeded runs diverge.  ``RandPool`` deliberately does *not*:
``seed=None`` and ``reseed(None)`` both fall through to the fixed constant
``_NO_SEED_FALLBACK``, documented in the module docstring.  That keeps every
pool — seeded or not — byte-for-byte reproducible.

The legacy global ``np.random`` stream is a different story: Fuzzer still
calls ``np.random.seed(None)`` for an unseeded run (see
``test_regression_numpy_global_seed.py``), so QEA / Monte-Carlo draws remain
non-deterministic when ``--seed`` is omitted.  That split is intentional;
these tests pin the *pool* side of it so a future change that accidentally
routes ``seed=None`` through OS entropy is caught.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import (
    _NO_SEED_FALLBACK,
    RandPool,
    get_default_rand_pool,
    reset_default_rand_pool,
)


def _draw_seq(pool: RandPool, n: int = 64) -> list[int]:
    return [pool.randint(0, 255) for _ in range(n)] + [
        pool.randrange(1000) for _ in range(n // 2)
    ]


class TestUnseededRandPoolIsDeterministic:
    def test_two_none_pools_match(self):
        a = RandPool(seed=None)
        b = RandPool(seed=None)
        assert _draw_seq(a) == _draw_seq(b)

    def test_none_equals_explicit_fallback(self):
        a = RandPool(seed=None)
        b = RandPool(seed=_NO_SEED_FALLBACK)
        assert _draw_seq(a) == _draw_seq(b)

    def test_reseed_none_resets_to_same_stream(self):
        p = RandPool(seed=99)
        _draw_seq(p, 20)  # advance
        p.reseed(None)
        q = RandPool(seed=None)
        assert _draw_seq(p) == _draw_seq(q)

    def test_default_pool_none_is_deterministic(self):
        reset_default_rand_pool(seed=None)
        a = get_default_rand_pool()
        seq_a = _draw_seq(a)
        reset_default_rand_pool(seed=None)
        b = get_default_rand_pool()
        assert _draw_seq(b) == seq_a


class TestSeededStillDiffers:
    def test_different_seeds_diverge(self):
        a = RandPool(seed=1)
        b = RandPool(seed=2)
        assert _draw_seq(a) != _draw_seq(b)

    def test_none_differs_from_other_seed(self):
        a = RandPool(seed=None)
        b = RandPool(seed=12345)
        assert _draw_seq(a) != _draw_seq(b)
