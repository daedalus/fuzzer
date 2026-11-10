"""Tests for the window influence tracker."""

from fuzzer_tool.core.analyzers.analyzer_influence import FILLS, WINDOW, WindowInfluenceTracker

SEED = bytes(4 * WINDOW)
BASE = {1, 2, 3}


def _gate(data):
    # Only window 1 steers the trace.
    if data[WINDOW : 2 * WINDOW] != SEED[WINDOW : 2 * WINDOW]:
        return {1, 2, 9}
    return BASE


class TestWindowInfluence:
    def test_hot_window_scores_highest(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        scores = t.analyze_seed(SEED, BASE, _gate)
        assert len(scores) == len(SEED)
        assert min(scores[WINDOW : 2 * WINDOW]) > 0.0
        assert max(scores[:WINDOW] + scores[2 * WINDOW :]) == 0.0

    def test_cost_matches_execs(self):
        t = WindowInfluenceTracker(sample_rate=0.5)
        calls = []
        t.analyze_seed(SEED, BASE, lambda d: calls.append(d) or BASE)
        assert len(calls) == t.cost(SEED)
        assert t.cost(SEED) == 2 * FILLS

    def test_idempotent(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        calls = []
        t.analyze_seed(SEED, BASE, lambda d: calls.append(d) or BASE)
        t.analyze_seed(SEED, BASE, lambda d: calls.append(d) or BASE)
        assert len(calls) == 4 * FILLS

    def test_falsification_constant_target_zero(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        scores = t.analyze_seed(SEED, BASE, lambda d: BASE)
        assert set(scores) == {0.0}
        assert t.get_weighted_position(SEED, len(SEED)) is None

    def test_adversarial_empty_and_short(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        assert t.analyze_seed(b"", BASE, lambda d: BASE) == []
        scores = t.analyze_seed(b"\x00\x01\x02", BASE, lambda d: {7})
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_adversarial_empty_traces(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        assert set(t.analyze_seed(SEED, set(), lambda d: set())) == {0.0}

    def test_weighted_position_in_hot_window(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        t.analyze_seed(SEED, BASE, _gate)
        picks = {t.get_weighted_position(SEED, len(SEED)) for _ in range(50)}
        assert picks <= set(range(WINDOW, 2 * WINDOW))

    def test_eviction(self):
        t = WindowInfluenceTracker(max_seeds=1, sample_rate=1.0)
        a, b = bytes([1]) * 16, bytes([2]) * 16
        t.analyze_seed(a, BASE, lambda d: BASE)
        t.analyze_seed(b, BASE, lambda d: BASE)
        assert not t.has_data(a)
        assert t.has_data(b)

    def test_save_load_roundtrip(self):
        t = WindowInfluenceTracker(sample_rate=1.0)
        t.analyze_seed(SEED, BASE, _gate)
        u = WindowInfluenceTracker()
        u.load(t.save())
        assert u.scores(SEED) == t.scores(SEED)


def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path):
    from tests.test_shim_touched_atomic_env import _fuzz_kwargs

    assert _fuzz_kwargs(monkeypatch, tmp_path, ["--influence"])["influence"] is True
    assert _fuzz_kwargs(monkeypatch, tmp_path, [])["influence"] is False
    assert _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])["influence"] is False
