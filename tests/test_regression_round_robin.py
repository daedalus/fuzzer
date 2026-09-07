"""Regression test for RoundRobinScheduler operator selection."""

from fuzzer_tool.core.schedulers.round_robin import RoundRobinScheduler


def test_round_robin_basic_cycle():
    """Round-robin cycles through operators in registration order."""
    sched = RoundRobinScheduler()
    ops = ["op_a", "op_b", "op_c"]

    # Register arms
    for op in ops:
        sched.init_arm(op)

    # First cycle: should return ops in order
    assert sched.select_op(ops) == "op_a"
    assert sched.select_op(ops) == "op_b"
    assert sched.select_op(ops) == "op_c"

    # Second cycle: wraps around
    assert sched.select_op(ops) == "op_a"
    assert sched.select_op(ops) == "op_b"
    assert sched.select_op(ops) == "op_c"


def test_round_robin_full_cycle_then_repeat():
    """Cycles through every operator once before any operator is repeated."""
    sched = RoundRobinScheduler()
    ops = [f"op_{i}" for i in range(10)]

    for op in ops:
        sched.init_arm(op)

    # First full cycle: every op appears exactly once in registration order
    first_cycle = [sched.select_op(ops) for _ in range(len(ops))]
    assert first_cycle == ops
    assert len(set(first_cycle)) == len(ops)

    # Second full cycle: same order, no operator repeated before the cycle completes
    second_cycle = [sched.select_op(ops) for _ in range(len(ops))]
    assert second_cycle == ops
    assert len(set(second_cycle)) == len(ops)

    # Third full cycle: still the same order
    third_cycle = [sched.select_op(ops) for _ in range(len(ops))]
    assert third_cycle == ops

    # Core round-robin invariant: in any window of k consecutive selections
    # (where k = number of registered ops), each op appears at most once.
    # An op can only appear twice within k calls if the cycle has completed
    # and started a new one.
    n = len(ops)
    selections = [sched.select_op(ops) for _ in range(n * 5)]
    for window_start in range(len(selections) - n):
        window = selections[window_start : window_start + n]
        assert len(set(window)) == n, (
            f"Window [{window_start}:{window_start + n}] has duplicate op; window={window}"
        )


def test_round_robin_filters_to_available_ops():
    """Selects only from ops that are in the available list."""
    sched = RoundRobinScheduler()
    all_ops = ["op_a", "op_b", "op_c", "op_d", "op_e"]

    for op in all_ops:
        sched.init_arm(op)

    # Only op_b and op_d are available this round
    available = ["op_b", "op_d"]

    # Should cycle through available ops in registration order
    assert sched.select_op(available) == "op_b"
    assert sched.select_op(available) == "op_d"
    assert sched.select_op(available) == "op_b"
    assert sched.select_op(available) == "op_d"


def test_round_robin_skips_unavailable_ops():
    """Advances index even when current arm is not in available ops."""
    sched = RoundRobinScheduler()
    all_ops = ["op_a", "op_b", "op_c", "op_d"]

    for op in all_ops:
        sched.init_arm(op)

    # First select: op_a is available, index advances to 1
    assert sched.select_op(["op_a"]) == "op_a"

    # Next select: available is op_c, index is at 1 (op_b)
    # Should skip op_b (not in available) and return op_c
    assert sched.select_op(["op_c"]) == "op_c"


def test_round_robin_single_op():
    """Returns the single op when only one is available."""
    sched = RoundRobinScheduler()
    sched.init_arm("only_op")

    assert sched.select_op(["only_op"]) == "only_op"
    assert sched.select_op(["only_op"]) == "only_op"


def test_round_robin_empty_ops():
    """Returns empty string when no ops available."""
    sched = RoundRobinScheduler()
    sched.init_arm("op_a")

    assert sched.select_op([]) == ""


def test_round_robin_record_is_noop():
    """record() doesn't affect selection (no learning)."""
    sched = RoundRobinScheduler()
    ops = ["op_a", "op_b", "op_c"]

    for op in ops:
        sched.init_arm(op)

    # Record many successes for op_a
    for _ in range(100):
        sched.record("op_a", True, weight=1.0)

    # Selection should still cycle: op_a, op_b, op_c
    assert sched.select_op(ops) == "op_a"
    assert sched.select_op(ops) == "op_b"
    assert sched.select_op(ops) == "op_c"


def test_round_robin_bandit_stats():
    """bandit_stats returns success/failure counts."""
    sched = RoundRobinScheduler()
    ops = ["op_a", "op_b"]

    for op in ops:
        sched.init_arm(op)

    sched.record("op_a", True, weight=1.0)
    sched.record("op_a", True, weight=2.0)
    sched.record("op_b", False, weight=1.0)

    stats = sched.bandit_stats()
    assert stats["op_a"] == (3.0, 0.0)
    assert stats["op_b"] == (0.0, 1.0)


def test_round_robin_supports_priors_false():
    """Round-robin declares no prior support."""
    sched = RoundRobinScheduler()
    assert getattr(sched, "supports_priors", True) is False
