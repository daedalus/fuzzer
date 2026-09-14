"""Unit tests for GradientBanditScheduler."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.gradient import GradientBanditScheduler


def test_select_single_arm():
    s = GradientBanditScheduler(rng=RandPool(seed=1))
    s.init_arm("a")
    assert s.select_op(["a"]) == "a"


def test_select_uniform_when_prefs_equal():
    s = GradientBanditScheduler(temperature=1.0, temp_decay=1.0, rng=RandPool(seed=42))
    ops = ["a", "b", "c"]
    for o in ops:
        s.init_arm(o)
    counts = {o: 0 for o in ops}
    for _ in range(3000):
        counts[s.select_op(ops)] += 1
    # Roughly uniform at equal prefs
    for o in ops:
        assert 700 < counts[o] < 1300, counts


def test_gradient_moves_preference():
    s = GradientBanditScheduler(alpha=0.5, temperature=1.0, temp_decay=1.0, rng=RandPool(seed=7))
    ops = ["good", "bad"]
    for o in ops:
        s.init_arm(o)
    # Force many successful updates on "good"
    for _ in range(50):
        op = s.select_op(ops)
        s.record(op, success=(op == "good"), weight=1.0)
    # "good" should now have higher preference
    assert s.preferences["good"] > s.preferences["bad"]


def test_shadow_record_ignored():
    s = GradientBanditScheduler(alpha=0.5, rng=RandPool(seed=3))
    s.init_arm("a")
    s.init_arm("b")
    s.select_op(["a", "b"])
    h_before = dict(s.preferences)
    # Record a different arm (shadow) — should not change preferences
    s.record("x", success=True, weight=1.0)
    assert s.preferences == h_before


def test_bandit_stats():
    s = GradientBanditScheduler(rng=RandPool(seed=0))
    s.init_arm("z")
    s.select_op(["z"])
    s.record("z", success=True)
    stats = s.bandit_stats()
    assert stats["gradient_pulls"] == 1
    assert "temperature" in stats
    assert stats["best_op"] == "z"
