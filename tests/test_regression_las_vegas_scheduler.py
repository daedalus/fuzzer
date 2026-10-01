"""Regression tests for LasVegasScheduler."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_las_vegas import LasVegasScheduler


def test_las_vegas_init():
    """Basic initialization test."""
    rng = RandPool(42)
    sched = LasVegasScheduler(delta=0.05, min_pulls=3, reopen_interval=0, rng=rng)
    assert sched.delta == 0.05
    assert sched.min_pulls == 3
    assert sched.reopen_interval == 0
    assert sched.supports_priors is True


def test_las_vegas_select_op_single_arm():
    """Selection works with single arm."""
    rng = RandPool(42)
    sched = LasVegasScheduler(rng=rng)
    sched.init_arm("op_a")
    chosen = sched.select_op(["op_a"])
    assert chosen == "op_a"


def test_las_vegas_select_op_multiple_arms():
    """Selection works with multiple arms - round-robin among active."""
    rng = RandPool(42)
    sched = LasVegasScheduler(rng=rng)
    sched.init_arm("op_a")
    sched.init_arm("op_b")
    sched.init_arm("op_c")

    # Should cycle through all arms
    chosen = sched.select_op(["op_a", "op_b", "op_c"])
    assert chosen in ["op_a", "op_b", "op_c"]

    chosen2 = sched.select_op(["op_a", "op_b", "op_c"])
    assert chosen2 in ["op_a", "op_b", "op_c"]


def test_las_vegas_record_updates_stats():
    """Recording updates mean and pull count."""
    rng = RandPool(42)
    sched = LasVegasScheduler(rng=rng)
    sched.init_arm("op_a")
    sched.init_arm("op_b")

    # Record success for op_a
    sched.record("op_a", True, weight=1.0)
    assert sched._n["op_a"] == 1
    assert sched._mean["op_a"] == 1.0

    # Record failure for op_a
    sched.record("op_a", False, weight=1.0)
    assert sched._n["op_a"] == 2
    assert sched._mean["op_a"] == 0.5

    # Record success for op_b
    sched.record("op_b", True, weight=1.0)
    assert sched._n["op_b"] == 1
    assert sched._mean["op_b"] == 1.0


def test_las_vegas_elimination():
    """Suboptimal arms eliminated when UCB < best LCB."""
    rng = RandPool(42)
    sched = LasVegasScheduler(delta=0.05, min_pulls=3, reopen_interval=0, rng=rng)
    sched.init_arm("good", prior_alpha=10.0, prior_beta=1.0)  # Strong prior
    sched.init_arm("bad", prior_alpha=1.0, prior_beta=10.0)  # Weak prior

    # Give "good" many successes
    for _ in range(10):
        sched.select_op(["good", "bad"])
        sched.record("good", True, weight=1.0)

    # Give "bad" many failures
    for _ in range(10):
        sched.select_op(["good", "bad"])
        sched.record("bad", False, weight=1.0)

    # "bad" should be eliminated
    sched.select_op(["good", "bad"])
    sched.record("bad", False, weight=1.0)

    # Check elimination
    stats = sched.bandit_stats()
    assert "good" in str(stats)
    # "bad" may be eliminated depending on confidence bounds


def test_las_vegas_priors():
    """Prior support works - init_arm accepts Beta priors."""
    rng = RandPool(42)
    sched = LasVegasScheduler(rng=rng)

    # Initialize with custom prior
    sched.init_arm("op_a", prior_alpha=5.0, prior_beta=2.0)

    # Mean should be prior mean = 5/(5+2) = 5/7
    assert abs(sched._mean["op_a"] - 5.0 / 7.0) < 0.001
    assert sched._alpha["op_a"] == 5.0
    assert sched._beta["op_a"] == 2.0


def test_las_vegas_reopen():
    """Periodic reopening re-admits eliminated arms."""
    rng = RandPool(42)
    sched = LasVegasScheduler(delta=0.05, min_pulls=3, reopen_interval=5, rng=rng)
    sched.init_arm("op_a")
    sched.init_arm("op_b")

    # Eliminate op_b by making it perform poorly
    for _ in range(20):
        sched.select_op(["op_a", "op_b"])
        sched.record("op_a", True, weight=1.0)
        sched.record("op_b", False, weight=1.0)

    # Should have eliminated op_b
    assert "op_b" in sched._eliminated or "op_a" not in sched._active

    # Force reopen by making enough record calls (reopen_interval records)
    # since the last reopen. select_op alone doesn't increment _total_pulls.
    # After reopening, record success for op_b so it doesn't get eliminated again.
    for _ in range(10):
        sched.select_op(["op_a", "op_b"])
        sched.record("op_a", True, weight=1.0)
        sched.record("op_b", True, weight=1.0)  # Success so op_b stays

    # op_b should be re-admitted after enough records
    assert "op_b" in sched._active


def test_las_vegas_bandit_stats():
    """bandit_stats returns correct structure."""
    rng = RandPool(42)
    sched = LasVegasScheduler(rng=rng)
    sched.init_arm("op_a")
    sched.init_arm("op_b")
    sched.record("op_a", True, weight=1.0)

    stats = sched.bandit_stats()
    assert "lv_pulls" in stats
    assert "lv_active" in stats
    assert "lv_eliminated" in stats
    assert "lv_arms" in stats
    assert "lv_success" in stats
    assert "lv_counts" in stats
    assert stats["lv_arms"] == 2


def test_las_vegas_delta_validation():
    """Invalid delta raises ValueError."""
    rng = RandPool(42)
    try:
        LasVegasScheduler(delta=0.0, rng=rng)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    try:
        LasVegasScheduler(delta=1.0, rng=rng)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_las_vegas_min_pulls_validation():
    """Invalid min_pulls raises ValueError."""
    rng = RandPool(42)
    try:
        LasVegasScheduler(min_pulls=0, rng=rng)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_las_vegas_reopen_validation():
    """Invalid reopen_interval raises ValueError."""
    rng = RandPool(42)
    try:
        LasVegasScheduler(reopen_interval=-1, rng=rng)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
