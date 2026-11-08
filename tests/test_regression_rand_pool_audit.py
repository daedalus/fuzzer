"""Regressions from the RandPool audit (2026-11-08).

Each test pins one defect: stream reuse, correlated default/hotpath
streams, Floyd order bias, numpy type leaks, inconsistent zero-weight
handling, negative-seed crash, slow ``choice`` on long sequences, and
per-refill entropy work nobody reads.
"""

import itertools

import numpy as np
import pytest

from fuzzer_tool.core import rand_pool as rp
from fuzzer_tool.core.rand_pool import (
    _POOL_ENTRIES,
    RandPool,
    get_default_rand_pool,
    reseed_default_rand_pool,
    reset_default_rand_pool,
)
from tests.test_regression_no_global_random import _make_fuzzer

U32 = 2**32


@pytest.fixture(autouse=True)
def _restore_shared_state():
    """Leave the default singleton and Floyd flag as other tests expect."""
    floyd = rp.rand_floyd_enabled()
    yield
    reset_default_rand_pool()
    rp.configure_rand_floyd(floyd)


def _raw_pools(seed: int, count: int) -> list[np.ndarray]:
    """Pools a seeded RandPool refills with, derived straight from numpy."""
    gen = np.random.default_rng(seed)
    return [gen.integers(0, U32, size=_POOL_ENTRIES, dtype=np.uint32) for _ in range(count)]


class _ScriptedFloyd(RandPool):
    """RandPool whose ``randrange`` replays a fixed script."""

    __slots__ = ("_script",)

    def __init__(self, script):
        super().__init__(seed=0)
        self._script = iter(script)

    def randrange(self, n):
        t = next(self._script)
        assert 0 <= t < n
        return t


# ── randbytes: exact-multiple drain must not rewind the cursor ───────────


def test_regression_randbytes_exact_multiple_no_reuse():
    pools = _raw_pools(1, 3)
    p = RandPool(seed=1)
    p.randbytes(2 * _POOL_ENTRIES)

    # Next bytes come from the third pool, not a replay of the second.
    assert p.randbytes(16) == (pools[2][:16] % 256).astype(np.uint8).tobytes()


def test_regression_randbytes_single_pool_then_draw():
    """Adversarial: one exact pool from a fresh pool, then a scalar draw."""
    pools = _raw_pools(3, 2)
    p = RandPool(seed=3)
    p.randbytes(_POOL_ENTRIES)
    assert p._draw() == int(pools[1][0])


# ── Default pool must not mirror the hotpath pool ────────────────────────


@pytest.mark.parametrize("seed", [42, 0, None])
def test_regression_default_pool_decorrelated(seed):
    reseed_default_rand_pool(seed)
    default = [get_default_rand_pool().randint(0, U32 - 1) for _ in range(64)]
    hot = RandPool(seed=seed)
    assert default != [hot.randint(0, U32 - 1) for _ in range(64)]

    # Still a pure function of the seed.
    reseed_default_rand_pool(seed)
    assert [get_default_rand_pool().randint(0, U32 - 1) for _ in range(64)] == default


def test_regression_fuzzer_default_pool_decorrelated():
    """Wiring: Fuzzer init and stall reseed use the derived stream."""
    f = _make_fuzzer(seed=42)
    hot = [f._rng.randint(0, U32 - 1) for _ in range(64)]
    assert [get_default_rand_pool().randint(0, U32 - 1) for _ in range(64)] != hot

    f._reseed_after_stall()
    hot = [f._rng.randint(0, U32 - 1) for _ in range(64)]
    assert [get_default_rand_pool().randint(0, U32 - 1) for _ in range(64)] != hot


# ── Floyd: output order must be a uniform permutation ────────────────────


def test_regression_floyd_order_uniform():
    """n=4, k=3: the 24 draw scripts map 1:1 onto the 24 ordered samples."""
    n, k = 4, 3
    scripts = itertools.product(*(range(j + 1) for j in range(n - k, n)))
    outs = [tuple(_ScriptedFloyd(s)._floyd_indices(n, k)) for s in scripts]
    assert sorted(outs) == sorted(itertools.permutations(range(n), k))


def test_regression_floyd_collision_path():
    """Adversarial: every draw collides, so j itself is inserted each time."""
    # j=2: t=0 → [0]; j=3: t=0 (hit) → 3 after 0 → [0, 3]; j=4: t=3 (hit) → [0, 3, 4]
    assert _ScriptedFloyd([0, 0, 3])._floyd_indices(5, 3) == [0, 3, 4]


# ── Python types, not numpy scalars ──────────────────────────────────────


def test_regression_sample_int_returns_python_ints():
    out = RandPool(seed=6).sample(100, 5)
    assert all(type(v) is int for v in out)


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.gauss_list(0.0, 1.0, 4),
        lambda p: p.expovariate_list(1.0, 4),
        lambda p: p.betavariate_list(2.0, 3.0, 4),
        lambda p: p.gammavariate_list(2.0, 1.0, 4),
        lambda p: p.gammavariate_list(2.0, 0.0, 4),
        lambda p: p.lognormvariate_list(0.0, 1.0, 4),
    ],
)
def test_regression_float_lists_return_python_floats(call):
    assert all(type(v) is float for v in call(RandPool(seed=6)))


# ── Zero weights raise the same way everywhere ───────────────────────────


@pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0, -1.0]])
def test_regression_weighted_choice_list_zero_weights(weights):
    with pytest.raises(IndexError):
        RandPool().weighted_choice_list([1, 2], weights, 3)


@pytest.mark.parametrize("weights", [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
def test_regression_categorical_zero_weights(weights):
    with pytest.raises(IndexError):
        RandPool().categorical(np.array(weights), 3)


# ── Negative seeds ───────────────────────────────────────────────────────


def test_regression_negative_seed_accepted():
    a = RandPool(seed=-1).randbytes(32)
    assert RandPool(seed=-1).randbytes(32) == a
    assert RandPool(seed=1).randbytes(32) != a

    p = RandPool(seed=7)
    p.reseed(-1)
    assert p.randbytes(32) == a


def test_regression_nonneg_seed_stream_unchanged():
    """Falsification: the negative-seed fix must not move existing streams."""
    pools = _raw_pools(12345, 1)
    assert RandPool(seed=12345)._draw() == int(pools[0][0])


# ── choice on long sequences uses the pool ───────────────────────────────


def test_regression_choice_long_seq_uses_pool():
    seq = list(range(1000))
    p = RandPool(seed=8)
    p._refill()
    start = p._idx
    expected = seq[p._pool_l[start] % len(seq)]
    assert p.choice(seq) == expected
    assert p._idx == start + 1


def test_regression_choice_huge_range():
    """Adversarial: a span wider than one pool word composes draws."""
    p = RandPool(seed=8)
    p._refill()
    start = p._idx
    v = p.choice(range(2**40))
    assert 0 <= v < 2**40
    assert p._idx >= start + 2


# ── Entropy work only when gated ─────────────────────────────────────────


def test_regression_refill_skips_entropy_ungated(monkeypatch):
    def boom(_data):
        raise AssertionError("entropy computed on an ungated refill")

    monkeypatch.setattr(rp, "_shannon_entropy", boom)
    RandPool(seed=1).randbytes(3 * _POOL_ENTRIES)


def test_regression_pool_entropy_lazy_matches():
    p = RandPool(seed=2)
    assert p.pool_entropy() is None
    p.randbytes(10)
    assert p.pool_entropy() == rp._shannon_entropy(p._m256.tobytes())
