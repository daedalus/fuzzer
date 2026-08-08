"""Tests for the digest-budget clamp, power-of-two sizing, and bytes fast path."""

import os

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.services.fuzzer import EXEC_DEDUP_RETRIES, Fuzzer


class TestDigestBudget:
    def test_k_never_exceeds_digest_width(self):
        # Tight error rates push the ideal k past what a single SHA-256 can
        # supply. Every configuration must still fit inside 256 bits.
        for capacity, error_rate in [
            (1_000, 1e-3),
            (10_000, 1e-4),
            (100_000, 1e-2),
            (1_000_000, 1e-6),
            (5_000_000, 1e-9),
        ]:
            bf = BloomFilter(capacity=capacity, error_rate=error_rate)
            assert bf._k * bf._bits_per_slice <= BloomFilter.DIGEST_BITS
            assert bf._k >= 1

    def test_digest_limited_flag_reports_clamping(self):
        loose = BloomFilter(capacity=100_000, error_rate=0.01)
        assert not loose.digest_limited
        assert loose._k == loose._k_ideal

        tight = BloomFilter(capacity=1_000_000, error_rate=1e-6)
        assert tight.digest_limited
        assert tight._k < tight._k_ideal

    def test_clamped_filter_still_honours_error_rate(self):
        bf = BloomFilter(capacity=10_000, error_rate=1e-4)
        for i in range(10_000):
            bf.add(f"present_{i}")
        fp = sum(1 for i in range(20_000) if bf.query(f"absent_{i}"))
        assert fp / 20_000 < 0.01

    def test_no_false_negatives(self):
        bf = BloomFilter(capacity=5_000, error_rate=1e-3)
        keys = [f"k{i}" for i in range(5_000)]
        for k in keys:
            bf.add(k)
        assert all(bf.query(k) for k in keys)


class TestSizing:
    def test_exact_power_of_two_is_not_doubled(self):
        # m_ideal for this configuration lands just above 2^13; the filter
        # must not round up a further full doubling.
        bf = BloomFilter(capacity=1024, error_rate=0.01)
        assert bf.m & (bf.m - 1) == 0
        assert bf.m < 32768

    def test_mask_and_slice_width_agree(self):
        bf = BloomFilter(capacity=50_000, error_rate=1e-3)
        assert bf._mask == bf.m - 1
        assert 1 << bf._bits_per_slice == bf.m

    def test_backing_array_is_bytearray(self):
        bf = BloomFilter(capacity=100)
        assert isinstance(bf._bits, bytearray)
        assert len(bf._bits) == (bf.m + 7) // 8


class TestUpdateBytes:
    def test_first_insert_reports_unseen(self):
        bf = BloomFilter(capacity=1000)
        assert bf.update_bytes(b"payload") is False
        assert bf.update_bytes(b"payload") is True

    def test_distinct_keys_are_independent(self):
        bf = BloomFilter(capacity=1000, error_rate=1e-3)
        assert bf.update_bytes(b"\x00\x01") is False
        assert bf.update_bytes(b"\x00\x02") is False
        assert bf.update_bytes(b"\x00\x01") is True

    def test_empty_key_is_tracked(self):
        bf = BloomFilter(capacity=1000)
        assert bf.update_bytes(b"") is False
        assert bf.update_bytes(b"") is True

    def test_counts_insertions(self):
        bf = BloomFilter(capacity=1000)
        for i in range(10):
            bf.update_bytes(bytes([i]))
        assert bf.n_added == 10
        bf.update_bytes(b"\x00")
        assert bf.n_added == 10

    def test_reset_on_full_wipes_at_capacity(self):
        bf = BloomFilter(capacity=64, error_rate=1e-3)
        keys = [os.urandom(8) for _ in range(64)]
        for k in keys:
            bf.update_bytes(k, reset_on_full=True)
        assert bf.n_added == 64
        # The next insert crosses capacity and starts a fresh generation.
        bf.update_bytes(os.urandom(8), reset_on_full=True)
        assert bf.n_added == 1
        assert bf.load_factor < 0.05

    def test_reset_off_by_default(self):
        # Without reset_on_full the filter keeps absorbing past capacity.
        # An overloaded filter false-positives, so n_added may lag the insert
        # count; what matters is that no generational wipe occurred.
        bf = BloomFilter(capacity=8, error_rate=1e-3)
        for i in range(32):
            bf.update_bytes(bytes([i]))
        assert bf.n_added > bf.capacity

    def test_clear_resets_counter(self):
        bf = BloomFilter(capacity=100)
        bf.update_bytes(b"x")
        bf.clear()
        assert bf.n_added == 0
        assert bf.update_bytes(b"x") is False


class _StubFuzzer:
    """Minimal stand-in exposing only what _dedup_mutate touches."""

    def __init__(self, mutants, dedup_execs=True, capacity=1000):
        self._mutants = list(mutants)
        self._dedup_execs = dedup_execs
        self._exec_bloom = BloomFilter(capacity=capacity, error_rate=1e-3)
        self._dedup_hits = 0
        self._dedup_gaveup = 0
        self.mutate_calls = 0

    def mutate(self, data):
        self.mutate_calls += 1
        return self._mutants.pop(0)

    _dedup_mutate = Fuzzer._dedup_mutate


class TestDedupMutate:
    def test_novel_mutant_passes_through_with_one_mutate(self):
        f = _StubFuzzer([b"novel"])
        assert f._dedup_mutate(b"seed") == b"novel"
        assert f.mutate_calls == 1
        assert f._dedup_hits == 0

    def test_repeat_is_rerolled(self):
        f = _StubFuzzer([b"dup", b"dup", b"fresh"])
        f._exec_bloom.update_bytes(b"dup")
        assert f._dedup_mutate(b"seed") == b"fresh"
        assert f._dedup_hits == 2
        assert f._dedup_gaveup == 0

    def test_gives_up_after_retry_budget(self):
        f = _StubFuzzer([b"dup"] * 8)
        f._exec_bloom.update_bytes(b"dup")
        assert f._dedup_mutate(b"seed") == b"dup"
        assert f._dedup_hits == EXEC_DEDUP_RETRIES
        assert f._dedup_gaveup == 1
        # Budget is bounded: 1 initial mutate + one per retry.
        assert f.mutate_calls == EXEC_DEDUP_RETRIES + 1

    def test_disabled_skips_the_filter(self):
        f = _StubFuzzer([b"dup"], dedup_execs=False)
        f._exec_bloom.update_bytes(b"dup")
        assert f._dedup_mutate(b"seed") == b"dup"
        assert f.mutate_calls == 1
        assert f._dedup_hits == 0

    def test_accepts_bytearray_mutants(self):
        f = _StubFuzzer([bytearray(b"buf"), bytearray(b"buf"), bytearray(b"other")])
        assert bytes(f._dedup_mutate(b"seed")) == b"buf"
        assert bytes(f._dedup_mutate(b"seed")) == b"other"
        assert f._dedup_hits == 1

    def test_filter_is_populated_as_a_side_effect(self):
        f = _StubFuzzer([b"a", b"b"])
        f._dedup_mutate(b"seed")
        assert f._exec_bloom.update_bytes(b"a") is True
