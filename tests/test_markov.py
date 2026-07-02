"""Tests for MarkovChain."""

from fuzzer_tool.core.markov import MarkovChain


class TestMarkovChain:
    def test_init_defaults(self):
        mc = MarkovChain()
        assert mc.order == 1
        assert not mc.is_trained()

    def test_train_makes_trained(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABCD")
        assert mc.is_trained()
        assert len(mc.transitions) > 0

    def test_train_corpus(self):
        mc = MarkovChain(order=1)
        mc.train_corpus([b"ABC", b"DEF"])
        assert mc.is_trained()

    def test_generate_length(self):
        mc = MarkovChain(order=1)
        mc.train(b"AAAA")
        result = mc.generate(8)
        assert isinstance(result, bytes)
        assert len(result) == 8

    def test_generate_untrained_returns_bytes(self):
        mc = MarkovChain(order=1)
        result = mc.generate(4)
        assert len(result) == 4

    def test_sample_byte_range(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABCD")
        for _ in range(100):
            b = mc.sample_byte(b"A")
            assert 0 <= b <= 255

    def test_order_zero(self):
        mc = MarkovChain(order=0)
        mc.train(b"XYZ")
        assert mc.is_trained()
        result = mc.generate(4)
        assert len(result) == 4

    def test_codelength_trained_input(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABABABAB")
        # Known pattern should have low codelength
        cl = mc.codelength(b"ABABABAB")
        assert cl < 8 * 8  # less than random (64 bits)

    def test_codelength_random_input(self):
        mc = MarkovChain(order=1)
        mc.train(b"AAAA")
        # Random bytes should have high codelength
        cl = mc.codelength(bytes(range(256)))
        assert cl > 0

    def test_codelength_empty(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABCD")
        assert mc.codelength(b"") == 0.0

    def test_codelength_untrained(self):
        mc = MarkovChain(order=1)
        # No training — falls back to 8 bits/byte
        cl = mc.codelength(b"ABC")
        assert cl == 3 * 8.0

    def test_codelength_ratio(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABABABAB")
        ratio = mc.codelength_ratio(b"ABABABAB")
        assert 0.0 <= ratio <= 8.0

    def test_codelength_ratio_empty(self):
        mc = MarkovChain(order=1)
        assert mc.codelength_ratio(b"") == 0.0

    def test_snapshot_and_check_plateau_not_trained(self):
        mc = MarkovChain()
        assert not mc.snapshot_and_check_plateau()

    def test_snapshot_and_check_plateau_too_few(self):
        mc = MarkovChain()
        mc._contexts_seen = 10
        mc._trains_since_snapshot = 1  # below interval
        assert not mc.snapshot_and_check_plateau()
