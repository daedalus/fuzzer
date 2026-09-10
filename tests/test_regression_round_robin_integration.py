"""Simplified test for round_robin scheduler integration."""

from fuzzer_tool.core import schedulers as S
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import install_scheduler_surface


class MockFuzzer:
    """Minimal fuzzer for testing select_op() without full Fuzzer setup."""

    def __init__(self):
        self.mc = None
        self.mc_bandit = False
        self.mc_cem = False
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self._invasion = False
        self._stall_recovery_active = False
        self.corpus = []
        self.seed_meta = {}
        self.dictionary = []
        self.markov_trained = False
        self.grammar = None

        # Scheduler flags
        self._use_round_robin = True
        self._round_robin = S.RoundRobinScheduler()
        install_scheduler_surface(self)

        self._last_mopt_particles = []
        self._prev_bandit_op = None

        self._rng = RandPool()


def test_round_robin_in_available_list():
    """round_robin should be added to available list when enabled."""
    f = MockFuzzer()
    ops = ["op_a", "op_b", "op_c"]
    for op in ops:
        f._round_robin.init_arm(op)

    f._rng = RandPool(seed=42)
    engine = OperatorEngine(f)

    # Reset cached strategy to force re-evaluation
    f._meta_strategy_cached = None

    # This should select round_robin from the available list
    selected = engine.select_op(ops)
    assert selected in ops, f"Selected {selected} not in {ops}"


def test_round_robin_cycles_deterministically():
    """round_robin should cycle through ops in fixed order."""
    f = MockFuzzer()
    ops = ["op_a", "op_b", "op_c"]
    for op in ops:
        f._round_robin.init_arm(op)

    f._rng = RandPool(seed=42)
    engine = OperatorEngine(f)

    # First call should return op_a
    assert engine.select_op(ops) == "op_a"
    assert engine.select_op(ops) == "op_b"
    assert engine.select_op(ops) == "op_c"
    assert engine.select_op(ops) == "op_a"  # Wraps around
