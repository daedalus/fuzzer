"""Tests for BloomFilter."""

from fuzzer_tool.core.bloom import BloomFilter


class TestBloomFilter:
    def test_add_and_query(self):
        bf = BloomFilter(capacity=1000)
        bf.add("hello")
        assert bf.query("hello")
        assert not bf.query("world")

    def test_query_false_positive_rate(self):
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(500):
            bf.add(f"key_{i}")
        false_positives = sum(1 for i in range(500, 1000) if bf.query(f"key_{i}"))
        assert false_positives < 50

    def test_update_new_key(self):
        bf = BloomFilter(capacity=1000)
        assert bf.update("foo") is False
        assert bf.query("foo")

    def test_update_existing_key(self):
        bf = BloomFilter(capacity=1000)
        bf.add("bar")
        assert bf.update("bar") is True

    def test_update_idempotent(self):
        bf = BloomFilter(capacity=1000)
        bf.add("x")
        assert bf.update("x") is True
        assert bf.update("x") is True

    def test_load_factor_empty(self):
        bf = BloomFilter(capacity=1000)
        assert bf.load_factor == 0.0

    def test_load_factor_increases(self):
        bf = BloomFilter(capacity=1000)
        bf.add("a")
        lf1 = bf.load_factor
        assert lf1 > 0.0
        for i in range(100):
            bf.add(f"item_{i}")
        assert bf.load_factor > lf1

    def test_load_factor_bounded(self):
        bf = BloomFilter(capacity=100)
        for i in range(100):
            bf.add(f"item_{i}")
        assert bf.load_factor <= 1.0

    def test_clear(self):
        bf = BloomFilter(capacity=1000)
        bf.add("test")
        assert bf.query("test")
        bf.clear()
        assert not bf.query("test")
        assert bf.load_factor == 0.0

    def test_clear_resets_state(self):
        bf = BloomFilter(capacity=1000)
        for i in range(50):
            bf.add(f"item_{i}")
        bf.clear()
        for i in range(50):
            assert not bf.query(f"item_{i}")

    def test_capacity_1(self):
        bf = BloomFilter(capacity=1)
        bf.add("only")
        assert bf.query("only")

    def test_empty_key(self):
        bf = BloomFilter(capacity=100)
        bf.add("")
        assert bf.query("")
        assert not bf.query("x")

    def test_unicode_key(self):
        bf = BloomFilter(capacity=100)
        bf.add("café")
        assert bf.query("café")
        assert not bf.query("cafe")
