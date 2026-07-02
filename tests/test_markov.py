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

    def test_snapshot_and_check_plateau_full_path(self):
        mc = MarkovChain()
        mc._snapshot_interval = 1
        mc._contexts_seen = 200
        # First call: sets prev_snapshot, no comparison yet
        mc.train(b"ABABABAB" * 50)
        mc.snapshot_and_check_plateau()
        # Second call: same data → low JS → plateau
        mc.train(b"ABABABAB" * 50)
        result2 = mc.snapshot_and_check_plateau()
        assert isinstance(result2, bool)

    def test_build_snapshot(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABCD")
        snap = mc._build_snapshot()
        assert b"" in snap or any(k for k in snap)
        for _ctx, dist in snap.items():
            assert isinstance(dist, dict)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-10  # normalized

    def test_js_between_snapshots_identical(self):
        snap = {b"\x00": {65: 0.5, 66: 0.5}}
        js = MarkovChain._js_between_snapshots(snap, snap)
        assert js == 0.0

    def test_js_between_snapshots_different(self):
        snap_a = {b"\x00": {65: 1.0}}
        snap_b = {b"\x00": {66: 1.0}}
        js = MarkovChain._js_between_snapshots(snap_a, snap_b)
        assert js > 0.0

    def test_js_between_snapshots_disjoint_contexts(self):
        snap_a = {b"A": {65: 1.0}}
        snap_b = {b"B": {66: 1.0}}
        js = MarkovChain._js_between_snapshots(snap_a, snap_b)
        assert js > 0.0

    def test_js_between_snapshots_empty(self):
        assert MarkovChain._js_between_snapshots({}, {}) == 0.0

    def test_save_and_load_roundtrip(self, tmp_path):
        mc = MarkovChain(order=2)
        mc.train(b"HELLO WORLD")
        mc.train(b"HELLO THERE")
        path = str(tmp_path / "markov.json")
        assert mc.save(path)

        mc2 = MarkovChain(order=2)
        assert mc2.load(path)
        assert mc2.order == 2
        assert mc2.is_trained()
        assert len(mc2.transitions) == len(mc.transitions)
        assert mc2._contexts_seen == mc._contexts_seen

    def test_save_failure(self, tmp_path):
        mc = MarkovChain()
        assert not mc.save("/nonexistent/dir/file.json")

    def test_load_failure(self, tmp_path):
        mc = MarkovChain()
        assert not mc.load("/nonexistent/file.json")

    def test_load_corrupt_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        mc = MarkovChain()
        assert not mc.load(str(p))

    def test_load_preserves_transitions(self, tmp_path):
        mc = MarkovChain(order=1)
        mc.train(b"AAAA")
        path = str(tmp_path / "m.json")
        mc.save(path)

        mc2 = MarkovChain()
        mc2.load(path)
        assert len(mc2.transitions) > 0

    def test_generate_untrained_fallback(self):
        mc = MarkovChain(order=1)
        # Untrained: generate picks random bytes (line 97)
        result = mc.generate(10)
        assert len(result) == 10

    def test_sample_byte_untrained(self):
        mc = MarkovChain(order=1)
        # Untrained: sample_byte returns random byte (line 120)
        for _ in range(100):
            b = mc.sample_byte(b"X")
            assert 0 <= b <= 255

    def test_order_two(self):
        mc = MarkovChain(order=2)
        mc.train(b"ABCDABCD")
        assert len(mc.transitions) > 0
        result = mc.generate(8)
        assert len(result) == 8

    def test_generate_length_one(self):
        mc = MarkovChain(order=1)
        mc.train(b"AB")
        result = mc.generate(1)
        assert len(result) == 1


class TestPerplexity:
    def test_perplexity_trained_input(self):
        """Known pattern → low perplexity (model explains it)."""
        mc = MarkovChain(order=1)
        mc.train(b"ABABABAB")
        pp = mc.perplexity(b"ABABABAB")
        assert 1.0 <= pp <= 256.0
        # Model has strong prediction → PP should be low
        assert pp < 20

    def test_perplexity_random_input(self):
        """Random bytes → high perplexity (model can't explain)."""
        mc = MarkovChain(order=1)
        mc.train(b"AAAA")
        pp = mc.perplexity(bytes(range(256)))
        assert pp > 100

    def test_perplexity_empty(self):
        mc = MarkovChain()
        assert mc.perplexity(b"") == 1.0

    def test_perplexity_untrained(self):
        mc = MarkovChain()
        pp = mc.perplexity(b"ABC")
        assert pp == 256.0  # uniform: 2^8 = 256

    def test_perplexity_range(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABCD" * 100)
        pp = mc.perplexity(b"ABCD")
        assert pp >= 1.0
        assert pp <= 256.0

    def test_corpus_perplexity(self):
        mc = MarkovChain(order=1)
        mc.train(b"ABABABAB")
        stats = mc.corpus_perplexity([b"ABABABAB", b"ABABABAB"])
        assert stats["mean"] > 0
        assert stats["median"] > 0
        assert stats["low_surprise_count"] >= 1  # well-predicted inputs

    def test_corpus_perplexity_empty(self):
        mc = MarkovChain()
        stats = mc.corpus_perplexity([])
        assert stats["mean"] == 0
