"""Integration tests for op_uniform scheduler wiring."""

from fuzzer_tool.cli import commands
from fuzzer_tool.core.schedulers.op_uniform import UniformScheduler
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer


def test_op_uniform_in_available_list():
    """op_uniform should be added to available list when enabled."""
    f = make_minimal_fuzzer(seed=42)
    f._use_uniform = True
    f._uniform = UniformScheduler()
    ops = ["op_a", "op_b", "op_c"]
    for op in ops:
        f._uniform.init_arm(op)

    engine = OperatorEngine(f)
    f._meta_strategy_cached = None

    selected = engine.select_op(ops)
    assert selected in ops, f"Selected {selected} not in {ops}"


def test_op_uniform_picks_randomly():
    """op_uniform should pick operators uniformly at random."""
    f = make_minimal_fuzzer(seed=42)
    f._use_uniform = True
    f._uniform = UniformScheduler()
    ops = ["op_a", "op_b", "op_c"]
    for op in ops:
        f._uniform.init_arm(op)

    engine = OperatorEngine(f)
    f._meta_strategy_cached = None

    results = [engine.select_op(ops) for _ in range(10)]
    unique_ops = set(results)
    assert len(unique_ops) >= 2, f"Expected at least 2 unique ops, got {unique_ops}"


def test_op_uniform_in_hail_mary():
    """op_uniform should be in _HAIL_MARY_FLAGS."""
    assert "op_uniform" in commands._HAIL_MARY_FLAGS
