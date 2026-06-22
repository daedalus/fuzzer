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
