"""Tests for RandPool — numpy-accelerated batched random number pool."""

import pytest

from fuzzer_tool.core.rand_pool import RandPool, _POOL_ENTRIES


# ── Pool internals ──────────────────────────────────────────────────────


class TestPoolInternals:
    def test_refill_fills_all_entries(self):
        """_refill populates all pool entries."""
        p = RandPool()
        p._refill()
        assert p._idx == 0
        assert p._pool.shape == (_POOL_ENTRIES,)
        assert p._m256.shape == (_POOL_ENTRIES,)

    def test_refill_resets_idx(self):
        """_refill always sets idx to 0."""
        p = RandPool()
        p._idx = 100
        p._refill()
        assert p._idx == 0

    def test_draw_triggers_refill(self):
        """_draw refills when pool is exhausted."""
        p = RandPool()
        p._idx = _POOL_ENTRIES  # force refill on next draw
        p._draw()
        assert p._idx == 1

    def test_m256_is_mod256_of_pool(self):
        """_m256 contains pool values modulo 256."""
        p = RandPool()
        p._refill()
        for i in range(_POOL_ENTRIES):
            assert p._m256[i] == p._pool[i] % 256

    def test_initial_state(self):
        """Fresh pool has idx at POOL_ENTRIES (triggers refill on first use)."""
        p = RandPool()
        assert p._idx == _POOL_ENTRIES


# ── Randbytes edge cases ───────────────────────────────────────────────


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
        p1.randbytes(100)
        vals1 = [p1.randint(0, 255) for _ in range(10)]
        vals2 = [p2.randint(0, 255) for _ in range(10)]
        assert vals1 != vals2, "randbytes did not advance the pool"

    def test_exact_pool_boundary(self):
        """randbytes(POOL_ENTRIES) consumes exactly one pool, idx resets to 0."""
        p = RandPool()
        p.randbytes(_POOL_ENTRIES)
        assert p._idx == 0

    def test_just_under_pool_boundary(self):
        """randbytes(POOL_ENTRIES - 1) stays in fast path, idx = POOL_ENTRIES - 1."""
        p = RandPool()
        p.randbytes(_POOL_ENTRIES - 1)
        assert p._idx == _POOL_ENTRIES - 1

    def test_just_over_pool_boundary(self):
        """randbytes(POOL_ENTRIES + 1) drains + refill, idx = 1."""
        p = RandPool()
        p.randbytes(_POOL_ENTRIES + 1)
        assert p._idx == 1

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

    def test_exact_two_pools(self):
        """randbytes(POOL_ENTRIES * 2) consumes exactly 2 pools, idx = 0."""
        p = RandPool()
        result = p.randbytes(_POOL_ENTRIES * 2)
        assert len(result) == _POOL_ENTRIES * 2
        assert p._idx == 0

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

    def test_interleaved_with_randint(self):
        """randbytes interleaved with randint maintains correct pool state."""
        p = RandPool()
        # Consume some pool first to be in the middle
        for _ in range(100):
            p.randint(0, 255)
        before_idx = p._idx
        p.randbytes(50)
        mid_idx = p._idx
        p.randint(0, 255)
        after_idx = p._idx
        assert mid_idx == before_idx + 50
        assert after_idx == mid_idx + 1

    def test_pool_partially_consumed(self):
        """randbytes on partially consumed pool works correctly."""
        p = RandPool()
        p.randint(0, 255)  # consume 1
        remaining = _POOL_ENTRIES - 1
        result = p.randbytes(remaining)
        assert len(result) == remaining
        assert p._idx == _POOL_ENTRIES  # triggers refill on next use


# ── Randint edge cases ─────────────────────────────────────────────────


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
        p1.randint(0, 255)
        v1 = p1.randint(0, 255)
        v2 = p2.randint(0, 255)
        assert p1._idx != p2._idx

    def test_single_value(self):
        """randint(a, a) always returns a."""
        p = RandPool()
        for _ in range(50):
            assert p.randint(5, 5) == 5

    def test_zero_range(self):
        """randint(0, 0) always returns 0."""
        p = RandPool()
        for _ in range(50):
            assert p.randint(0, 0) == 0

    def test_width_256_fast_path(self):
        """randint(0, 255) uses pre-computed %256 array."""
        p = RandPool()
        vals = [p.randint(0, 255) for _ in range(100)]
        assert all(0 <= v <= 255 for v in vals)

    def test_negative_width(self):
        """randint(a, b) where a > b returns a."""
        p = RandPool()
        assert p.randint(10, 5) == 10

    def test_triggers_refill(self):
        """randint triggers refill when pool exhausted."""
        p = RandPool()
        p._idx = _POOL_ENTRIES
        p.randint(0, 255)
        assert p._idx == 1


# ── Randrange edge cases ───────────────────────────────────────────────


class TestRandrange:
    def test_basic(self):
        p = RandPool()
        for _ in range(100):
            v = p.randrange(10)
            assert 0 <= v < 10

    def test_single_value(self):
        """randrange(1) always returns 0."""
        p = RandPool()
        for _ in range(50):
            assert p.randrange(1) == 0

    def test_zero_returns_zero(self):
        """randrange(0) returns 0 (edge case)."""
        assert RandPool().randrange(0) == 0

    def test_negative_returns_zero(self):
        """randrange(-5) returns 0 (edge case)."""
        assert RandPool().randrange(-5) == 0


# ── Choice edge cases ──────────────────────────────────────────────────


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

    def test_single_element(self):
        """choice with single-element sequence always returns that element."""
        p = RandPool()
        for _ in range(50):
            assert p.choice([42]) == 42

    def test_bytes_input(self):
        """choice works with bytes input."""
        p = RandPool()
        seq = b"hello"
        for _ in range(50):
            v = p.choice(seq)
            assert v in seq

    def test_tuple_input(self):
        """choice works with tuple input."""
        p = RandPool()
        seq = (1, 2, 3)
        for _ in range(50):
            assert p.choice(seq) in seq

    def test_large_sequence(self):
        """choice with sequence >256 elements uses numpy fallback."""
        p = RandPool()
        seq = list(range(300))
        for _ in range(50):
            v = p.choice(seq)
            assert v in seq

    def test_advances_pool(self):
        """choice advances the pool position."""
        p = RandPool()
        p.choice([1, 2, 3])
        assert p._idx > 0


# ── Choice list edge cases ─────────────────────────────────────────────


class TestChoiceList:
    def test_basic(self):
        p = RandPool()
        result = p.choice_list([10, 20, 30], 5)
        assert len(result) == 5
        assert all(v in [10, 20, 30] for v in result)

    def test_empty_sequence_raises(self):
        with pytest.raises(IndexError):
            RandPool().choice_list([], 5)

    def test_zero_count(self):
        """choice_list with count=0 returns empty list."""
        assert RandPool().choice_list([1, 2, 3], 0) == []

    def test_negative_count(self):
        """choice_list with negative count returns empty list."""
        assert RandPool().choice_list([1, 2, 3], -1) == []

    def test_count_exceeds_sequence(self):
        """choice_list with count > len(seq) still works (with replacement)."""
        p = RandPool()
        result = p.choice_list([1, 2], 10)
        assert len(result) == 10
        assert all(v in [1, 2] for v in result)


# ── Weighted choice edge cases ─────────────────────────────────────────


class TestWeightedChoice:
    def test_basic(self):
        p = RandPool()
        seq = ["a", "b", "c"]
        weights = [1.0, 0.0, 0.0]
        for _ in range(50):
            assert p.weighted_choice(seq, weights) == "a"

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            RandPool().weighted_choice([], [1.0])

    def test_all_weights_zero_raises(self):
        """All zero weights — degenerate case raises IndexError."""
        p = RandPool()
        with pytest.raises(IndexError):
            p.weighted_choice([1, 2], [0.0, 0.0])


class TestWeightedChoiceList:
    def test_basic(self):
        p = RandPool()
        result = p.weighted_choice_list(["a", "b"], [1.0, 0.0], 5)
        assert len(result) == 5
        assert all(v == "a" for v in result)

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            RandPool().weighted_choice_list([], [1.0], 5)

    def test_zero_k(self):
        """weighted_choice_list with k=0 returns empty list."""
        assert RandPool().weighted_choice_list([1, 2], [1.0, 1.0], 0) == []


# ── Sample edge cases ──────────────────────────────────────────────────


class TestSample:
    def test_int_population(self):
        """sample(10, 3) returns 3 unique ints in [0, 10)."""
        p = RandPool()
        result = p.sample(10, 3)
        assert len(result) == 3
        assert all(0 <= v < 10 for v in result)
        assert len(set(result)) == 3

    def test_list_population(self):
        """sample([a,b,c], 2) returns 2 unique elements."""
        p = RandPool()
        result = p.sample([10, 20, 30], 2)
        assert len(result) == 2
        assert len(set(result)) == 2
        assert all(v in [10, 20, 30] for v in result)

    def test_bytes_population(self):
        """sample works with bytes input."""
        p = RandPool()
        result = p.sample(b"abcde", 3)
        assert len(result) == 3
        assert len(set(result)) == 3

    def test_k_exceeds_population(self):
        """sample clamps k to population size."""
        p = RandPool()
        result = p.sample(5, 10)
        assert len(result) == 5

    def test_k_zero(self):
        """sample with k=0 returns empty list."""
        assert RandPool().sample(10, 0) == []

    def test_k_one(self):
        """sample with k=1 returns single-element list."""
        p = RandPool()
        result = p.sample(10, 1)
        assert len(result) == 1
        assert 0 <= result[0] < 10

    def test_k_two(self):
        """sample with k=2 uses optimized path."""
        p = RandPool()
        result = p.sample(10, 2)
        assert len(result) == 2
        assert result[0] != result[1]


# ── Shuffle edge cases ─────────────────────────────────────────────────


class TestShuffle:
    def test_small_list(self):
        """shuffle on list < 8 uses Fisher-Yates path."""
        p = RandPool()
        seq = [1, 2, 3, 4, 5]
        original = seq[:]
        p.shuffle(seq)
        assert sorted(seq) == sorted(original)

    def test_large_list(self):
        """shuffle on list >= 8 uses numpy path."""
        p = RandPool()
        seq = list(range(100))
        original = seq[:]
        p.shuffle(seq)
        assert sorted(seq) == sorted(original)

    def test_empty_list(self):
        """shuffle on empty list is a no-op."""
        p = RandPool()
        seq = []
        p.shuffle(seq)
        assert seq == []

    def test_single_element(self):
        """shuffle on single-element list is a no-op."""
        p = RandPool()
        seq = [42]
        p.shuffle(seq)
        assert seq == [42]


# ── Continuous distributions ───────────────────────────────────────────


class TestContinuousDistributions:
    def test_gauss(self):
        p = RandPool()
        vals = [p.gauss(0, 1) for _ in range(100)]
        assert all(isinstance(v, float) for v in vals)
        # Not all the same
        assert len(set(vals)) > 1

    def test_gauss_list(self):
        p = RandPool()
        result = p.gauss_list(0, 1, 50)
        assert len(result) == 50
        assert all(isinstance(v, float) for v in result)

    def test_gauss_list_zero_count(self):
        assert RandPool().gauss_list(0, 1, 0) == []

    def test_expovariate(self):
        p = RandPool()
        vals = [p.expovariate(1.0) for _ in range(100)]
        assert all(v >= 0 for v in vals)
        assert len(set(vals)) > 1

    def test_expovariate_zero_rate(self):
        """expovariate(0) returns inf."""
        assert RandPool().expovariate(0) == float("inf")

    def test_expovariate_negative_rate(self):
        """expovariate with negative rate returns inf."""
        assert RandPool().expovariate(-1) == float("inf")

    def test_expovariate_list(self):
        p = RandPool()
        result = p.expovariate_list(1.0, 50)
        assert len(result) == 50
        assert all(v >= 0 for v in result)

    def test_expovariate_list_zero_rate(self):
        """expovariate_list(0) returns list of inf."""
        result = RandPool().expovariate_list(0, 5)
        assert result == [float("inf")] * 5

    def test_expovariate_list_zero_count(self):
        assert RandPool().expovariate_list(1.0, 0) == []

    def test_betavariate(self):
        p = RandPool()
        vals = [p.betavariate(2, 5) for _ in range(100)]
        assert all(0 <= v <= 1 for v in vals)
        assert len(set(vals)) > 1

    def test_betavariate_list(self):
        p = RandPool()
        result = p.betavariate_list(2, 5, 50)
        assert len(result) == 50
        assert all(0 <= v <= 1 for v in result)

    def test_betavariate_list_zero_count(self):
        assert RandPool().betavariate_list(2, 5, 0) == []

    def test_gammavariate(self):
        p = RandPool()
        vals = [p.gammavariate(2, 1) for _ in range(100)]
        assert all(v >= 0 for v in vals)
        assert len(set(vals)) > 1

    def test_gammavariate_zero_beta(self):
        """gammavariate with beta=0 returns 0."""
        assert RandPool().gammavariate(2, 0) == 0.0

    def test_gammavariate_list(self):
        p = RandPool()
        result = p.gammavariate_list(2, 1, 50)
        assert len(result) == 50
        assert all(v >= 0 for v in result)

    def test_gammavariate_list_zero_beta(self):
        """gammavariate_list with beta=0 returns zeros."""
        result = RandPool().gammavariate_list(2, 0, 5)
        assert result == [0.0] * 5

    def test_gammavariate_list_zero_count(self):
        assert RandPool().gammavariate_list(2, 1, 0) == []

    def test_lognormvariate(self):
        p = RandPool()
        vals = [p.lognormvariate(0, 1) for _ in range(100)]
        assert all(v > 0 for v in vals)
        assert len(set(vals)) > 1

    def test_lognormvariate_list(self):
        p = RandPool()
        result = p.lognormvariate_list(0, 1, 50)
        assert len(result) == 50
        assert all(v > 0 for v in result)

    def test_lognormvariate_list_zero_count(self):
        assert RandPool().lognormvariate_list(0, 1, 0) == []


# ── Random list edge cases ─────────────────────────────────────────────


class TestRandomList:
    def test_basic(self):
        p = RandPool()
        result = p.random_list(10)
        assert len(result) == 10
        assert all(0.0 <= v < 1.0 for v in result)

    def test_zero_count(self):
        assert RandPool().random_list(0) == []

    def test_negative_count(self):
        assert RandPool().random_list(-1) == []


# ── Randrange list edge cases ──────────────────────────────────────────


class TestRandrangeList:
    def test_basic(self):
        p = RandPool()
        result = p.randrange_list(10, 50)
        assert len(result) == 50
        assert all(0 <= v < 10 for v in result)

    def test_zero_n(self):
        assert RandPool().randrange_list(0, 10) == []

    def test_zero_count(self):
        assert RandPool().randrange_list(10, 0) == []

    def test_negative_n(self):
        assert RandPool().randrange_list(-1, 10) == []


# ── Randint list edge cases ────────────────────────────────────────────


class TestRandintList:
    def test_basic(self):
        p = RandPool()
        result = p.randint_list(0, 10, 50)
        assert len(result) == 50
        assert all(0 <= v <= 10 for v in result)

    def test_empty(self):
        assert RandPool().randint_list(0, 10, 0) == []

    def test_single_value(self):
        """randint_list(a, a, n) returns n copies of a."""
        p = RandPool()
        result = p.randint_list(5, 5, 10)
        assert result == [5] * 10

    def test_width_256_fast_path(self):
        """randint_list(0, 255, n) uses pre-computed %256 array."""
        p = RandPool()
        result = p.randint_list(0, 255, 100)
        assert all(0 <= v <= 255 for v in result)

    def test_width_256_with_offset(self):
        """randint_list(100, 355, n) uses %256 fast path + offset."""
        p = RandPool()
        result = p.randint_list(100, 355, 100)
        assert all(100 <= v <= 355 for v in result)

    def test_invalid_width(self):
        """randint_list(a, b) where a > b returns empty."""
        assert RandPool().randint_list(10, 5, 10) == []

    def test_triggers_refill(self):
        """randint_list triggers refill when pool exhausted."""
        p = RandPool()
        p._idx = _POOL_ENTRIES
        p.randint_list(0, 10, 5)
        assert p._idx == 5
