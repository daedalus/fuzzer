"""Sort-based set ops vs numpy's.

numpy 2.4's ``np.unique`` hashes: 6.1 s on 4.9M int64 keys against 0.09 s
for sort + adjacent-diff. ICFG build and the Katz recompute call it on
millions of packed edge keys.
"""

import time

import numpy as np
import pytest

from fuzzer_tool.core.np_sets import sorted_isin, sorted_unique


def _lcg(n: int, seed: int, mod: int, dtype) -> np.ndarray:
    out = np.empty(n, dtype=np.uint64)
    x = seed
    for i in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        out[i] = (x >> 11) % mod
    return out.astype(dtype)


_CASES = {
    "empty": np.array([], dtype=np.int64),
    "single": np.array([7], dtype=np.int64),
    "all_equal": np.full(50, 3, dtype=np.int64),
    "negative": np.array([5, -1, -1, 0, 5, -9], dtype=np.int64),
    "dupes_int64": _lcg(2000, 1, 300, np.int64),
    "wide_uint64": _lcg(500, 2, 2**63, np.uint64),
    "packed_keys": (_lcg(800, 3, 1000, np.int64) << 32) | _lcg(800, 4, 1000, np.int64),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_sorted_unique_matches_numpy(name):
    a = _CASES[name]
    got = sorted_unique(a)
    want = np.unique(a)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_control_numpy_against_itself(name):
    """Hard Rule 46."""
    a = _CASES[name]
    assert np.array_equal(np.unique(a), np.unique(a.copy()))


def test_input_not_mutated():
    a = np.array([3, 1, 2, 1], dtype=np.int64)
    sorted_unique(a)
    assert a.tolist() == [3, 1, 2, 1]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_sorted_isin_matches_numpy(name):
    keys = sorted_unique(
        np.concatenate([_CASES[name], _CASES["dupes_int64"].astype(_CASES[name].dtype)])
    )
    ref = sorted_unique(_CASES[name])
    assert np.array_equal(sorted_isin(keys, ref), np.isin(keys, ref))


def test_isin_empty_reference():
    """Adversarial: searchsorted into nothing must not index out of range."""
    keys = np.array([1, 2, 3], dtype=np.int64)
    assert sorted_isin(keys, np.array([], dtype=np.int64)).tolist() == [False] * 3


def test_isin_key_past_last():
    keys = np.array([1, 9], dtype=np.int64)
    assert sorted_isin(keys, np.array([1, 5], dtype=np.int64)).tolist() == [True, False]


def test_large_is_fast():
    a = np.arange(3_000_000, dtype=np.int64)[::-1] % 1_500_000
    t = time.perf_counter()
    sorted_unique(a)
    assert time.perf_counter() - t < 1.0
