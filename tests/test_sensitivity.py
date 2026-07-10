"""Tests for per-byte sensitivity tracker."""

from fuzzer_tool.core.sensitivity import ByteSensitivityTracker


class TestByteSensitivityTracker:
    def test_init(self):
        t = ByteSensitivityTracker()
        assert t.max_seeds == 100
        assert t.sample_rate == 0.1

    def test_analyze_seed(self):
        t = ByteSensitivityTracker(sample_rate=1.0)
        seed = b"\x00" * 10
        original_edges = {1, 2, 3}

        def exec_fn(data):
            if data[0] != 0:
                return {4, 5, 6}
            return original_edges

        scores = t.analyze_seed(seed, original_edges, exec_fn)
        assert len(scores) == 10
        assert scores[0] > 0.0
        assert t.has_data(seed)

    def test_analyze_seed_idempotent(self):
        t = ByteSensitivityTracker(sample_rate=1.0)
        seed = b"\x00" * 5
        call_count = [0]

        def exec_fn(data):
            call_count[0] += 1
            return {1, 2}

        t.analyze_seed(seed, {1, 2}, exec_fn)
        t.analyze_seed(seed, {1, 2}, exec_fn)
        assert call_count[0] == 5

    def test_get_weighted_position(self):
        t = ByteSensitivityTracker(sample_rate=1.0)
        seed = b"\x00" * 10

        def exec_fn(data):
            if data[0] != 0:
                return {10, 20, 30}
            return {1, 2, 3}

        t.analyze_seed(seed, {1, 2, 3}, exec_fn)
        pos = t.get_weighted_position(seed, 10)
        assert pos is not None
        assert 0 <= pos < 10

    def test_get_weighted_position_no_data(self):
        t = ByteSensitivityTracker()
        assert t.get_weighted_position(b"\x00" * 10, 10) is None

    def test_lru_eviction(self):
        t = ByteSensitivityTracker(max_seeds=2, sample_rate=1.0)
        for i in range(4):
            seed = bytes([i]) * 5
            t.analyze_seed(seed, {1}, lambda d: {1})
        assert len(t._analyzed) == 2

    def test_save_load(self):
        t = ByteSensitivityTracker(sample_rate=0.5)
        seed = b"\x00" * 10
        t.analyze_seed(seed, {1, 2}, lambda d: {1, 2})
        data = t.save()
        t2 = ByteSensitivityTracker()
        t2.load(data)
        assert t2.has_data(seed)
        assert t2.sample_rate == 0.5
