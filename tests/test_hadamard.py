"""Tests for core/hadamard: Sylvester + Paley I Hadamard matrices."""

import itertools

import pytest

from fuzzer_tool.core.hadamard import hadamard, hadamard_rows

ORDERS = (1, 2, 4, 8, 12, 16, 20, 24, 32)
WORD_ORDERS = (2, 4, 8, 12, 16, 20, 24, 32)
UNBUILT = (3, 5, 6, 28, 36)


def _gram(h):
    """H H^T, computed with plain sums (independent of the module)."""
    return [[sum(a * b for a, b in zip(r, s, strict=True)) for s in h] for r in h]


def _is_hadamard(h) -> bool:
    n = len(h)
    want = [[n if i == j else 0 for j in range(n)] for i in range(n)]
    return all(len(r) == n and set(r) <= {-1, 1} for r in h) and _gram(h) == want


def _unpack(rows, n):
    return [[-1 if w >> j & 1 else 1 for j in range(n)] for w in rows]


def _strength2(cols) -> bool:
    """Every column pair takes each of the four sign pairs n/4 times."""
    n = len(cols[0])
    want = {(a, b): n // 4 for a in (-1, 1) for b in (-1, 1)}
    for ci, cj in itertools.combinations(cols, 2):
        pairs = list(zip(ci, cj, strict=True))
        if {c: pairs.count(c) for c in want} != want:
            return False
    return True


class TestHadamard:
    @pytest.mark.parametrize("n", ORDERS)
    def test_gram_is_n_identity(self, n):
        assert _is_hadamard(hadamard(n))

    def test_gram_oracle_rejects_non_orthogonal(self):
        # Control: one sign flipped breaks orthogonality of its row.
        h = hadamard(8)
        h[3][5] = -h[3][5]
        assert not _is_hadamard(h)

    @pytest.mark.parametrize("n", UNBUILT)
    def test_unbuilt_orders_are_none(self, n):
        assert hadamard(n) is None

    @pytest.mark.parametrize("n", (0, -4))
    def test_adversarial_non_positive_order_is_none(self, n):
        assert hadamard(n) is None


class TestHadamardRows:
    @pytest.mark.parametrize("n", WORD_ORDERS)
    def test_rows_form_hadamard(self, n):
        assert _is_hadamard(_unpack(hadamard_rows(n), n))

    @pytest.mark.parametrize("n", WORD_ORDERS)
    def test_normalized(self, n):
        rows = hadamard_rows(n)
        assert rows[0] == 0
        assert all(w & 1 == 0 for w in rows)

    @pytest.mark.parametrize("n", (4, 8, 12, 16, 20, 24, 32))
    def test_columns_are_strength_2_oa(self, n):
        h = _unpack(hadamard_rows(n), n)
        cols = [[r[j] for r in h] for j in range(1, n)]
        assert _strength2(cols)

    def test_oa_oracle_rejects_repeated_row(self):
        h = _unpack(hadamard_rows(8), 8)
        h[1] = h[2]
        assert not _strength2([[r[j] for r in h] for j in range(1, 8)])

    def test_rows_fit_order_bits(self):
        assert all(w < 1 << 32 for w in hadamard_rows(32))

    @pytest.mark.parametrize("n", (28, 3, 0))
    def test_adversarial_unbuilt_order_raises(self, n):
        with pytest.raises(ValueError):
            hadamard_rows(n)
