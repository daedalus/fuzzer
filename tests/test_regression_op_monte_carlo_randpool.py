"""op_monte_carlo draws its Gaussian noise from RandPool, not global np.random.

``spectral_gap`` (λ₂ power-iteration start vector) and ``correlated_select``
(correlated Thompson noise) called ``np.random.randn``. Global numpy state is
outside every RandPool snapshot (Hard Rule 16), so a resumed run diverged and
``Fuzzer`` had to keep seeding the legacy global stream.
"""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import op_monte_carlo
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

OPS = ["a", "b", "c", "d"]
PICKS = 64


def _cov() -> dict[str, dict[str, float]]:
    # Diagonally dominant, so positive definite: the Cholesky path runs.
    return {x: {y: (1.0 if x == y else 0.3) for y in OPS} for x in OPS}


def _sched(seed: int) -> MonteCarloScheduler:
    mc = MonteCarloScheduler(rng=RandPool(seed=seed))
    for op in OPS:
        mc.init_arm(op)
    mc.operator_covariance = lambda **_: _cov()
    return mc


def _ring(mc: MonteCarloScheduler) -> MonteCarloScheduler:
    # A -> B -> C -> D -> A with a leak, so λ₂ is non-trivial.
    for i, op in enumerate(OPS):
        nxt = OPS[(i + 1) % len(OPS)]
        mc.transition_counts[op][nxt] = 9
        mc.transition_counts[op][OPS[(i + 2) % len(OPS)]] = 1
        mc.transition_total[op] = 10
    return mc


def _picks(seed: int, global_seed: int) -> list[str]:
    np.random.seed(global_seed)
    mc = _sched(seed)
    out = []
    for _ in range(PICKS):
        np.random.random(3)  # foreign global draws between picks
        out.append(mc.correlated_select(OPS))
    return out


def _global_untouched(fn) -> bool:
    np.random.seed(7)
    before = np.random.get_state()[1].copy(), np.random.get_state()[2]
    fn()
    after = np.random.get_state()[1], np.random.get_state()[2]
    return bool(np.array_equal(before[0], after[0])) and before[1] == after[1]


def test_regression_op_monte_carlo_randpool():
    """Global numpy seed and foreign draws must not change selections."""
    assert _picks(5, global_seed=1) == _picks(5, global_seed=2)


def test_control_same_seed_identical():
    """Hard Rule 46: the reference matches a second run of itself."""
    assert _picks(5, global_seed=1) == _picks(5, global_seed=1)


def test_falsify_pool_seed_drives_selection():
    """A different RandPool seed must change the selection sequence."""
    assert _picks(5, global_seed=1) != _picks(6, global_seed=1)


def test_correlated_select_leaves_global_state():
    mc = _sched(3)
    assert _global_untouched(lambda: mc.correlated_select(OPS))


def test_spectral_gap_leaves_global_state():
    mc = _ring(_sched(3))
    assert _global_untouched(mc.spectral_gap)


def test_spectral_gap_reproducible_under_global_noise():
    np.random.seed(1)
    gap_a = _ring(_sched(9)).spectral_gap()
    np.random.seed(2)
    np.random.random(100)
    gap_b = _ring(_sched(9)).spectral_gap()
    assert gap_a == gap_b
    assert 0.0 <= gap_a <= 1.0


@pytest.mark.parametrize("ops", [OPS[:3], OPS])
def test_adversarial_no_numpy_fallback(monkeypatch, ops):
    """Pure-Python path draws from the same pool: same picks as numpy path."""
    expect = _sched(4).correlated_select(ops)
    monkeypatch.setattr(op_monte_carlo, "_HAS_NUMPY", False)
    got = _sched(4).correlated_select(ops)
    assert got == expect
    assert got in ops


def test_regression_chol_readonly_diag():
    """np.diag returns a read-only view: `_chol` raised on every call.

    `_chol_py` applies the same regularisation in pure Python, so it is an
    independent oracle; a zero and a negative diagonal exercise the clamp.
    """
    matrix = [[0.0, 0.1, 0.0], [0.1, -2.0, 0.2], [0.0, 0.2, 3.0]]
    got = MonteCarloScheduler._chol(matrix)
    want = MonteCarloScheduler._chol_py(matrix)
    assert got is not None and want is not None
    assert np.allclose(got, np.asarray(want))
