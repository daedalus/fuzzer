"""Regressions from the RandPool audit (2026-11-12).

Each test pins one defect: sequence ``sample`` skipping the wide-range
path, ``gammavariate_array`` diverging from the scalar on rate <= 0,
uninitialized reads in ``pool_min_entropy_now``, unvalidated
``weighted_choice`` weights, untagged default-pool seeding, entropy gates
not rechecked after a retry, stale ``pool_entropy`` after a reseed, and an
opaque ``categorical([])`` error.
"""

import bisect
import itertools
import warnings

import numpy as np
import pytest

from fuzzer_tool.core import rand_pool as rp
from fuzzer_tool.core.rand_pool import (
    _POOL_ENTRIES,
    RandPool,
    configure_default_rand_pool_seed,
    get_default_rand_pool,
    reseed_default_rand_pool,
    reset_default_rand_pool,
)

U32 = 2**32
WIDE = 2**40
DRAWS = 64


@pytest.fixture(autouse=True)
def _restore_default_pool():
    """Leave the default singleton as other tests expect."""
    yield
    reset_default_rand_pool()


def _raw_pool(seed: int) -> list[int]:
    """First pool a seeded RandPool refills with, straight from numpy."""
    gen = np.random.default_rng(seed)
    return gen.integers(0, U32, size=_POOL_ENTRIES, dtype=np.uint32).tolist()


def _words(pool: RandPool, n: int = DRAWS) -> list[int]:
    return [pool.randint(0, U32 - 1) for _ in range(n)]


# ── sample(sequence) must take the wide-range path like sample(int) ──────


@pytest.mark.parametrize("n", [WIDE, 3 << 30])
@pytest.mark.parametrize("k", [1, 2])
def test_regression_sample_seq_wide_range(n, k):
    """Oracle: ``randrange`` on a twin pool, distinct-pair rule for k=2."""
    got = RandPool(seed=5).sample(range(n), k)

    twin = RandPool(seed=5)
    a = twin.randrange(n)
    expected = [a]
    if k == 2:
        b = twin.randrange(n - 1)
        expected.append(b + 1 if b >= a else b)
    assert got == expected


def test_regression_sample_seq_wide_reaches_high_words():
    """Adversarial: indices above 2**32 must be reachable."""
    p = RandPool(seed=1)
    assert max(p.sample(range(WIDE), 1)[0] for _ in range(DRAWS)) >= U32


@pytest.mark.parametrize("k", [1, 2])
def test_regression_sample_seq_small_stream_unchanged(k):
    """Falsification: small populations keep the ``_draw() % n`` stream."""
    n = 100
    raw = _raw_pool(9)
    expected = [raw[0] % n]
    if k == 2:
        b = raw[1] % (n - 1)
        expected.append(b + 1 if b >= expected[0] else b)
    assert RandPool(seed=9).sample(list(range(n)), k) == expected


# ── gammavariate_array: rate <= 0 -> 0.0, matching the scalar ────────────


def test_regression_gamma_array_nonpositive_rate():
    alphas = [2.0, 3.0, 4.0]
    betas = [0.5, 0.0, -2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = RandPool(seed=4).gammavariate_array(alphas, betas)

    draws = np.random.default_rng(4).gamma(alphas)
    assert got.tolist() == [draws[0] / betas[0], 0.0, 0.0]
    assert RandPool(seed=4).gammavariate(2.0, 0.0) == got[1]


def test_regression_gamma_array_broadcast_scalar_rate():
    """Adversarial: a scalar zero rate broadcasts over every arm."""
    got = RandPool(seed=4).gammavariate_array([1.0, 2.0], 0.0)
    assert got.tolist() == [0.0, 0.0]


def test_regression_gamma_array_positive_unchanged():
    """Falsification: positive rates keep the numpy stream exactly."""
    alphas, betas = [1.5, 2.5], [2.0, 4.0]
    expected = np.random.default_rng(6).gamma(alphas) / np.asarray(betas)
    assert RandPool(seed=6).gammavariate_array(alphas, betas).tolist() == expected.tolist()


# ── pool_min_entropy_now: no read before a refill ────────────────────────


def test_regression_min_entropy_before_refill_raises():
    with pytest.raises(RuntimeError, match="not been refilled"):
        RandPool(seed=1).pool_min_entropy_now()


def test_regression_min_entropy_after_reseed_raises():
    """Adversarial: the batch left over from the old stream is stale."""
    p = RandPool(seed=1)
    p.random()
    p.reseed(2)
    with pytest.raises(RuntimeError, match="not been refilled"):
        p.pool_min_entropy_now()


def test_regression_min_entropy_after_refill_matches():
    """Falsification: a refilled pool still reports its batch."""
    p = RandPool(seed=1)
    p.random()
    raw = np.asarray(_raw_pool(1), dtype=np.uint32).astype(np.uint8)
    assert p.pool_min_entropy_now() == rp._min_entropy(raw.tobytes())


# ── pool_entropy: None after a reseed, not the old batch ─────────────────


@pytest.mark.parametrize("reseed", [lambda p: p.reseed(3), lambda p: p.inject_entropy(b"x")])
def test_regression_pool_entropy_after_reseed(reseed):
    p = RandPool(seed=2)
    p.random()
    reseed(p)
    assert p.pool_entropy() is None
    p.random()
    assert p.pool_entropy() == rp._shannon_entropy(p._m256.tobytes())


# ── weighted_choice: reject negative or mismatched weights ───────────────


@pytest.mark.parametrize("weights", [[5.0, -4.0, 1.0], [-1.0, 3.0, 0.0]])
def test_regression_weighted_choice_negative(weights):
    with pytest.raises(ValueError, match="negative"):
        RandPool(seed=1).weighted_choice("abc", weights)


@pytest.mark.parametrize("weights", [[1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
def test_regression_weighted_choice_length(weights):
    """Adversarial: shorter and longer weight vectors both raise."""
    with pytest.raises(ValueError, match="weights"):
        RandPool(seed=1).weighted_choice("abcd", weights)


def test_regression_weighted_choice_valid_unchanged():
    """Falsification: valid weights pick by ``random() * total`` + bisect."""
    weights = [1.0, 0.0, 3.0, 2.0]
    cum = list(itertools.accumulate(weights))
    r = _raw_pool(7)[0] / U32 * cum[-1]
    expected = "abcd"[bisect.bisect_right(cum, r, 0, len(weights) - 1)]
    assert RandPool(seed=7).weighted_choice("abcd", weights) == expected


# ── Default pool seeding never mirrors RandPool(seed) ────────────────────


def _derived(seed) -> list[int]:
    reseed_default_rand_pool(seed)
    return _words(get_default_rand_pool())


@pytest.mark.parametrize("seed", [None, 0, 42, -3])
@pytest.mark.parametrize("make", ["get", "configure", "reset"])
def test_regression_default_pool_seed_tagged(monkeypatch, seed, make):
    expected = _derived(seed)
    monkeypatch.setattr(rp, "_default_instance", None)

    if make == "get":
        pool = get_default_rand_pool(seed)
    elif make == "configure":
        configure_default_rand_pool_seed(seed)
        pool = get_default_rand_pool()
    else:
        reset_default_rand_pool(seed)
        pool = get_default_rand_pool()

    got = _words(pool)
    assert got == expected
    assert got != _words(RandPool(seed=seed))


def test_regression_default_pool_seed_control():
    """Control (rule 46): the derived stream equals a second run of itself."""
    assert _derived(42) == _derived(42)
    assert _derived(42) != _derived(43)


# ── Entropy gates: every gate rechecked after any retry ──────────────────


class _Script:
    """Replays fixed entropy readings and counts calls."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, _data):
        v = self.values[self.calls]
        self.calls += 1
        return v


def test_regression_gates_recheck_after_retry(monkeypatch):
    """Shannon retry lands a batch that fails H_inf: it must retry again.

    batch0: H_inf ok, Shannon low -> retry
    batch1: H_inf low              -> retry (old code accepted it)
    batch2: H_inf ok, Shannon ok   -> accept
    """
    hinf = _Script([8.0, 0.0, 8.0])
    shannon = _Script([0.0, 8.0])
    monkeypatch.setattr(rp, "_min_entropy", hinf)
    monkeypatch.setattr(rp, "_shannon_entropy", shannon)

    p = RandPool(seed=1, min_entropy=7.0, min_hinf=7.0)
    p.random()
    assert (hinf.calls, shannon.calls) == (3, 2)


def test_regression_gates_raise_after_retries(monkeypatch):
    """Adversarial: a gate that never passes still raises its own message."""
    monkeypatch.setattr(rp, "_min_entropy", lambda _d: 9.0)
    monkeypatch.setattr(rp, "_shannon_entropy", lambda _d: 0.0)
    with pytest.raises(RuntimeError, match="entropy starvation"):
        RandPool(seed=1, min_entropy=7.0, min_hinf=7.0).random()


def test_regression_single_gate_stream_unchanged(monkeypatch):
    """Falsification: one gate, one retry: reseed from the live stream."""
    hinf = _Script([0.0, 8.0])
    monkeypatch.setattr(rp, "_min_entropy", hinf)
    p = RandPool(seed=1, min_hinf=7.0)

    gen = np.random.default_rng(1)
    gen.integers(0, U32, size=_POOL_ENTRIES, dtype=np.uint32)
    retry = np.random.default_rng(int(gen.integers(0, 2**63)))
    expected = int(retry.integers(0, U32, size=_POOL_ENTRIES, dtype=np.uint32)[0])
    assert p._draw() == expected


# ── categorical: empty weights name the problem ──────────────────────────


def test_regression_categorical_empty():
    with pytest.raises(IndexError, match="empty"):
        RandPool(seed=1).categorical(np.array([]), 1)


def test_regression_gamma_array_empty():
    """Adversarial: zero arms return an empty array, as before the guard."""
    assert RandPool(seed=4).gammavariate_array([], []).shape == (0,)


# ── Empty ranges raise, as in CPython's random ───────────────────────────


@pytest.mark.parametrize("n", [0, -5])
def test_regression_randrange_empty_raises(n):
    with pytest.raises(ValueError, match="empty range"):
        RandPool(seed=1).randrange(n)


@pytest.mark.parametrize(("a", "b"), [(5, 4), (0, -1), (10, -(2**40))])
def test_regression_randint_empty_raises(a, b):
    """Adversarial: inverted ranges, including a wide one, raise."""
    with pytest.raises(ValueError, match="empty range"):
        RandPool(seed=1).randint(a, b)


@pytest.mark.parametrize(("a", "b"), [(5, 4), (0, -1)])
def test_regression_exhaustive_mirrors_empty_raise(a, b):
    """Wiring: ExhaustivePool keeps RandPool's contract under enumeration."""
    from fuzzer_tool.core.exhaustive_pool import ExhaustivePool

    pool = ExhaustivePool()
    with pytest.raises(ValueError, match="empty range"):
        pool.randint(a, b)
    with pytest.raises(ValueError, match="empty range"):
        pool.randrange(b - a + 1)


def test_regression_randrange_single_unchanged():
    """Falsification: valid ranges keep the ``_draw() % n`` stream."""
    raw = _raw_pool(2)
    p = RandPool(seed=2)
    assert [p.randrange(10), p.randint(3, 8)] == [raw[0] % 10, 3 + raw[1] % 6]


# ── geometric: 53-bit uniform, no 2**-32 tail cap ────────────────────────


def _u53(hi: int, lo: int) -> float:
    """CPython's ``random()``: 27 + 26 bits of two 32-bit words."""
    return ((hi >> 5) * 67108864 + (lo >> 6)) / 2**53


def test_regression_geometric_53_bit():
    p_ok = 0.25
    raw = _raw_pool(3)
    u = _u53(raw[0], raw[1])
    expected = int(np.floor(np.log(1.0 - u) / np.log1p(-p_ok)))
    pool = RandPool(seed=3)
    assert pool.geometric(p_ok) == expected
    assert pool._idx == 2


def test_regression_geometric_tail_past_32_bits():
    """Adversarial: the all-ones script reaches past the old 2**-32 cap."""
    p = 1e-6
    old_cap = int(np.floor(np.log(2.0**-32) / np.log1p(-p)))
    pool = RandPool(seed=0)
    pool._refill()
    pool._pool_l[pool._idx] = U32 - 1
    pool._pool_l[pool._idx + 1] = U32 - 1
    assert pool.geometric(p) > old_cap
