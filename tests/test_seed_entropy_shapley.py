"""Entropy §5, permutation-Shapley step: ``core/schedulers/seed_entropy_shapley.py``."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fuzzer_tool.core.byte_entropy import CumulativeByteEntropy, byte_histogram
from fuzzer_tool.core.schedulers.seed_entropy_shapley import loo_entropy, shapley_entropy

DIVERSE = bytes(range(256))
FLAT = b"\x00" * 256
HALF = bytes(range(128)) * 2
TOP = bytes(range(128, 256)) * 2


def _slab(corpus):
    rows = [byte_histogram(s, 1 << 20) for s in corpus]
    return (
        np.stack([np.asarray(c, dtype=np.int64) for c, _ in rows]),
        np.array([t for _, t in rows], dtype=np.int64),
    )


def _bits(seeds):
    pool = CumulativeByteEntropy()
    for s in seeds:
        pool.add(s)
    return pool.bits()


def _exact_shapley(corpus):
    n = len(corpus)
    phi = [0.0] * n
    perms = list(itertools.permutations(range(n)))
    for order in perms:
        seen = []
        for i in order:
            before = _bits([corpus[j] for j in seen]) if seen else 0.0
            seen.append(i)
            phi[i] += _bits([corpus[j] for j in seen]) - before
    return [p / len(perms) for p in phi]


def test_efficiency_sums_to_pooled_entropy():
    corpus = [DIVERSE, FLAT, HALF, TOP, b"hello world"]
    counts, sizes = _slab(corpus)
    phi, _ = shapley_entropy(counts, sizes, n_perm=16, rng=np.random.default_rng(0))
    assert phi.sum() == pytest.approx(_bits(corpus), abs=1e-9)


def test_converges_to_exact_shapley():
    corpus = [DIVERSE, FLAT, HALF, TOP]
    counts, sizes = _slab(corpus)
    phi, se = shapley_entropy(counts, sizes, n_perm=400, rng=np.random.default_rng(1))
    for got, want, s in zip(phi, _exact_shapley(corpus), se, strict=True):
        assert abs(got - want) < max(4 * s, 1e-6)


def test_near_duplicates_priced_by_shapley_not_loo():
    """Same-histogram pair: LOO scores 0/0 while Shapley splits the credit."""
    corpus = [HALF, HALF[::-1], TOP]
    counts, sizes = _slab(corpus)
    loo = loo_entropy(counts, sizes)
    phi, se = shapley_entropy(counts, sizes, n_perm=400, rng=np.random.default_rng(2))
    assert abs(phi[0] - phi[1]) < 4 * float(np.hypot(se[0], se[1])) + 1e-9
    assert phi[0] > loo[0] + 0.5


def test_loo_matches_fresh_pool_oracle():
    corpus = [DIVERSE, FLAT, HALF, TOP, b"hello world"]
    counts, sizes = _slab(corpus)
    for i, g in enumerate(loo_entropy(counts, sizes)):
        assert g == pytest.approx(_bits(corpus) - _bits(corpus[:i] + corpus[i + 1 :]), abs=1e-9)


def test_empty_and_single():
    assert shapley_entropy(np.zeros((0, 256)), np.zeros(0))[0].size == 0
    counts, sizes = _slab([HALF])
    phi, _ = shapley_entropy(counts, sizes, n_perm=2, rng=np.random.default_rng(0))
    assert phi[0] == pytest.approx(7.0)
