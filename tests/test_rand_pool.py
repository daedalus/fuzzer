"""Tests for RandPool — numpy-accelerated batched random number pool."""

import pytest

from fuzzer_tool.core.rand_pool import RandPool, _POOL_ENTRIES


class TestRandbytes:
    def test_no_repeat(self):
        """Consecutive randbytes(16) calls must return different bytes."""
        p = RandPool()
        results = [p.randbytes(16) for _ in range(20)]
        assert len(set(results)) == 20, "randbytes returned identical outputs"

    def test_advances_pool(self):
        """After randbytes(n), subsequent randint calls return different values."""
        p1 = RandPool()
        p2 = RandPool()
        # Advance p1 by consuming bytes
        p1.randbytes(100)
        # p1 and p2 should now be at different positions
        vals1 = [p1.randint(0, 255) for _ in range(10)]
        vals2 = [p2.randint(0, 255) for _ in range(10)]
        assert vals1 != vals2, "randbytes did not advance the pool"

    def test_exact_pool_boundary(self):
        """randbytes(POOL_ENTRIES) consumes exactly one pool, idx resets to 0."""
        p = RandPool()
        p.randbytes(_POOL_ENTRIES)
        assert p._idx == 0

    def test_across_boundary(self):
        """randbytes(POOL_ENTRIES + 10) drains current + refill, idx = 10."""
        p = RandPool()
        p.randbytes(_POOL_ENTRIES + 10)
        assert p._idx == 10

    def test_across_multiple_refills(self):
        """randbytes with large n spans multiple refills correctly."""
        p = RandPool()
        n = _POOL_ENTRIES * 3 + 42
        result = p.randbytes(n)
        assert len(result) == n
        assert p._idx == 42

    def test_large(self):
        """randbytes(10000) works correctly across multiple refills."""
        p = RandPool()
        result = p.randbytes(10000)
        assert len(result) == 10000
        assert len(set(result)) > 100, "Large randbytes output has no diversity"

    def test_zero(self):
        """randbytes(0) returns empty bytes."""
        assert RandPool().randbytes(0) == b""

    def test_negative(self):
        """randbytes(-1) returns empty bytes."""
        assert RandPool().randbytes(-1) == b""

    def test_single(self):
        """randbytes(1) returns exactly 1 byte."""
        result = RandPool().randbytes(1)
        assert len(result) == 1
        assert isinstance(result, bytes)

    def test_does_not_corrupt_pool(self):
        """After randbytes, other methods (randint, choice) still work correctly."""
        p = RandPool()
        p.randbytes(200)
        # These should not raise or return garbage
        v1 = p.randint(0, 100)
        v2 = p.choice([1, 2, 3, 4, 5])
        assert 0 <= v1 <= 100
        assert v2 in [1, 2, 3, 4, 5]

    def test_returns_bytes_type(self):
        """randbytes always returns bytes, not bytearray or numpy array."""
        p = RandPool()
        for n in [1, 10, 100, 4096]:
            result = p.randbytes(n)
            assert isinstance(result, bytes)

    def test_content_not_all_zeros(self):
        """randbytes output is not trivially all zeros."""
        p = RandPool()
        result = p.randbytes(256)
        assert result != b"\x00" * 256


class TestRandint:
    def test_basic(self):
        p = RandPool()
        for _ in range(100):
            v = p.randint(0, 10)
            assert 0 <= v <= 10

    def test_range(self):
        p = RandPool()
        vals = {p.randint(0, 1) for _ in range(200)}
        assert vals == {0, 1}

    def test_advances_pool(self):
        p1 = RandPool()
        p2 = RandPool()
        p1.randint(0, 255)  # advance p1
        v1 = p1.randint(0, 255)
        v2 = p2.randint(0, 255)
        # Not guaranteed different, but pool positions differ
        assert p1._idx != p2._idx


class TestRandrange:
    def test_basic(self):
        p = RandPool()
        for _ in range(100):
            v = p.randrange(10)
            assert 0 <= v < 10


class TestChoice:
    def test_basic(self):
        p = RandPool()
        seq = [10, 20, 30, 40, 50]
        for _ in range(100):
            v = p.choice(seq)
            assert v in seq

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            RandPool().choice([])


class TestRandintList:
    def test_basic(self):
        p = RandPool()
        result = p.randint_list(0, 10, 50)
        assert len(result) == 50
        assert all(0 <= v <= 10 for v in result)

    def test_empty(self):
        assert RandPool().randint_list(0, 10, 0) == []
