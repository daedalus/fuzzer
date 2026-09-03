"""Two rewrites inside the dieharder-family tests, both pinned to oracles.

``permutation_test`` and ``serial_test`` were rewritten onto identities
that leave the returned p-value bit-identical. The oracles below are the
previous implementations carried verbatim, so any drift shows up as a
numeric difference rather than as a silently different statistic.

The rewrites are:

* ``permutation_test`` sorted its blocks twice in one expression, took
  ranks via ``argsort(argsort(...))``, and mapped each rank vector to a
  permutation index through a per-row tuple construction and dict lookup.
  A rank vector is a permutation, so reading it as a base-k numeral is
  injective and the whole (n, k) block converts in one dot product.
* ``serial_test`` built its m-bit and (m-1)-bit sliding windows
  independently, but the second is the first shifted right by one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core import randomness as R
from fuzzer_tool.core.randomness import _perm_index, _perm_radix_table, _rank_rows


def permutation_test_oracle(draws, k: int = 5) -> float:
    """The pre-rewrite body, kept verbatim."""
    d = np.asarray(draws)
    if k < 3 or k > 6:
        raise ValueError("k must be in 3..6")
    ncell = math.factorial(k)
    n = d.size // k
    if n < 10 * ncell:
        return 1.0
    blocks = d[: n * k].reshape(n, k)
    blocks = blocks[(np.sort(blocks, axis=1)[:, 1:] != np.sort(blocks, axis=1)[:, :-1]).all(axis=1)]
    if blocks.shape[0] < 10 * ncell:
        return 1.0
    index = _perm_index(k)
    ranks = np.argsort(np.argsort(blocks, axis=1), axis=1)
    idx = np.fromiter((index[tuple(r)] for r in ranks.tolist()), dtype=np.int64, count=len(ranks))
    counts = np.bincount(idx, minlength=ncell)
    exp = blocks.shape[0] / ncell
    x2 = float(np.sum((counts - exp) ** 2) / exp)
    return R.chisq_sf(x2, ncell - 1)


def serial_test_oracle(data: bytes, m: int = 8) -> float:
    """The pre-rewrite body, kept verbatim."""
    b = R._bits(data).astype(np.int64)
    n = b.size
    if n < (1 << m) * 5 or m < 2:
        return 1.0

    def psi2(mm: int) -> float:
        if mm <= 0:
            return 0.0
        ext = np.concatenate([b, b[: mm - 1]])
        idx = np.zeros(n, dtype=np.int64)
        for k in range(mm):
            idx = (idx << 1) | ext[k : k + n]
        counts = np.bincount(idx, minlength=1 << mm)
        return float(np.sum(counts.astype(np.float64) ** 2)) * (1 << mm) / n - n

    return R.chisq_sf(psi2(m) - psi2(m - 1), 1 << (m - 1))


class TestPermutationRadixTable:
    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_base_k_encoding_is_injective_over_permutations(self, k):
        powers, lookup = _perm_radix_table(k)
        keys = {int(np.dot(np.asarray(p, dtype=np.int64), powers)) for p in _perm_index(k)}
        assert len(keys) == math.factorial(k)

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_lookup_reproduces_perm_index(self, k):
        powers, lookup = _perm_radix_table(k)
        for perm, idx in _perm_index(k).items():
            key = int(np.dot(np.asarray(perm, dtype=np.int64), powers))
            assert lookup[key] == idx

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_non_permutation_keys_stay_unmapped(self, k):
        """-1 rather than 0, so a misuse fails loudly instead of scoring row 0."""
        _powers, lookup = _perm_radix_table(k)
        assert int((lookup >= 0).sum()) == math.factorial(k)
        assert int(lookup.min()) == -1

    def test_table_is_cached(self):
        assert _perm_radix_table(5) is _perm_radix_table(5)


class TestRankScatterIsTheInverseArgsort:
    """The p-value cannot see this, so it is asserted structurally.

    Replacing ``argsort(argsort(b))`` with a single ``argsort`` looks like
    a valid simplification and is not: ``argsort`` gives the positions in
    sorted order, whose inverse permutation is the rank vector. But the
    chi-square is a histogram over permutation *labels*, and the inverse
    map is a bijection on those labels, so swapping ranks for their
    inverse only relabels the histogram and leaves the statistic
    identical. An oracle on the returned p-value therefore cannot
    distinguish the two, and this test exists because injecting that
    exact mistake left every p-value assertion in this module green.
    """

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_rank_rows_equals_double_argsort(self, k):
        rng = np.random.default_rng(41 + k)
        blocks = rng.integers(0, 137, size=(500, k))
        assert np.array_equal(_rank_rows(blocks), np.argsort(np.argsort(blocks, axis=1), axis=1))

    def test_rank_rows_is_not_a_bare_argsort(self):
        """Guards the simplification that looks right and is not."""
        rng = np.random.default_rng(7)
        blocks = rng.integers(0, 137, size=(500, 5))
        assert not np.array_equal(_rank_rows(blocks), np.argsort(blocks, axis=1))

    def test_rank_rows_inverts_the_order(self):
        rng = np.random.default_rng(19)
        blocks = rng.integers(0, 137, size=(200, 5))
        order = np.argsort(blocks, axis=1)
        ranks = _rank_rows(blocks)
        assert np.array_equal(
            np.take_along_axis(ranks, order, axis=1), np.tile(np.arange(5), (200, 1))
        )

    def test_chi_square_is_blind_to_the_relabelling(self):
        """Why the structural test above is needed — documented, not assumed."""
        rng = np.random.default_rng(13)
        blocks = rng.integers(0, 137, size=(4000, 5))
        index = _perm_index(5)
        order = np.argsort(blocks, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(5), axis=1)

        def histogram(rows):
            counts = np.bincount([index[tuple(r)] for r in rows.tolist()], minlength=120)
            return np.sort(counts)

        assert np.array_equal(histogram(ranks), histogram(order))


class TestPermutationTestMatchesOracle:
    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_random_streams(self, k):
        rng = np.random.default_rng(20260903 + k)
        for _ in range(8):
            size = int(rng.integers(10 * 720 * k, 12 * 720 * k))
            draws = rng.integers(0, 137, size=size)
            assert R.permutation_test(draws, k) == pytest.approx(
                permutation_test_oracle(draws, k), abs=1e-12
            )

    def test_structured_stream(self):
        """A stream the test should reject — agreement must hold there too."""
        draws = np.tile(np.arange(5), 40000)
        assert R.permutation_test(draws, 5) == pytest.approx(
            permutation_test_oracle(draws, 5), abs=1e-12
        )

    def test_heavily_tied_stream(self):
        """Small alphabet — most blocks are discarded by the tie filter."""
        rng = np.random.default_rng(3)
        draws = rng.integers(0, 4, size=200000)
        assert R.permutation_test(draws, 5) == pytest.approx(
            permutation_test_oracle(draws, 5), abs=1e-12
        )

    def test_short_stream_returns_one(self):
        assert R.permutation_test(np.arange(50), 5) == 1.0

    @pytest.mark.parametrize("k", [2, 7])
    def test_out_of_range_k_still_raises(self, k):
        with pytest.raises(ValueError, match="k must be in 3..6"):
            R.permutation_test(np.arange(100000), k)


class TestSerialTestWindowShift:
    @pytest.mark.parametrize("m", [2, 3, 5, 8, 10])
    def test_shorter_window_is_the_longer_one_shifted(self, m):
        """The identity the rewrite rests on, asserted directly."""
        rng = np.random.default_rng(5)
        b = R._bits(bytes(rng.integers(0, 256, 4000, dtype=np.uint8))).astype(np.int64)
        n = b.size

        def window(mm):
            ext = np.concatenate([b, b[: mm - 1]]) if mm > 1 else b
            idx = np.zeros(n, dtype=np.int64)
            for k in range(mm):
                idx = (idx << 1) | ext[k : k + n]
            return idx

        assert np.array_equal(window(m) >> 1, window(m - 1))

    @pytest.mark.parametrize("m", [2, 4, 8, 10])
    def test_matches_oracle(self, m):
        rng = np.random.default_rng(11 + m)
        for _ in range(5):
            data = bytes(rng.integers(0, 256, int(rng.integers(3000, 20000)), dtype=np.uint8))
            assert R.serial_test(data, m) == pytest.approx(serial_test_oracle(data, m), abs=1e-12)

    def test_structured_data_matches_oracle(self):
        data = bytes(range(256)) * 80
        assert R.serial_test(data, 8) == pytest.approx(serial_test_oracle(data, 8), abs=1e-12)

    def test_constant_data_matches_oracle(self):
        data = b"\x00" * 20000
        assert R.serial_test(data, 8) == pytest.approx(serial_test_oracle(data, 8), abs=1e-12)

    def test_short_input_returns_one(self):
        assert R.serial_test(b"\x00" * 4, 8) == 1.0
