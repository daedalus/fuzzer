"""operator_kernel rewrites must stay bit-identical to the set-union original."""

import pytest

from fuzzer_tool.core import shapley
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.shapley import KERNEL_NUMPY_MIN_OPS, ShapleyAttribution


def _reference(sa, operators):
    """The original nested-loop kernel, verbatim semantics."""
    kernel = {op: {} for op in operators}
    for i, a in enumerate(operators):
        ea = sa._operator_edges.get(a, set())
        for j, b in enumerate(operators):
            if i == j:
                kernel[a][b] = 1.0
            elif i < j:
                eb = sa._operator_edges.get(b, set())
                union = len(ea | eb)
                sim = len(ea & eb) / union if union else 0.0
                kernel[a][b] = kernel[b][a] = sim
    return kernel


def _attribution(n, size, seed):
    rng = RandPool(seed)
    sa = ShapleyAttribution()
    for i in range(n):
        sa.record({f"op{i:03d}"}, 1, set(rng.randint_list(0, 4 * size, size)))
    return sa


# Both sides of the numpy cutoff, plus a tiny and an empty-heavy case.
@pytest.mark.parametrize("n", [2, 5, KERNEL_NUMPY_MIN_OPS - 1, KERNEL_NUMPY_MIN_OPS, 70])
def test_bit_identical_to_reference(n):
    sa = _attribution(n, 40, n)
    ops = sorted(sa._operator_edges) + ["never_recorded", "also_empty"]

    # Control (Hard Rule 46): the oracle against a second run of itself.
    assert _reference(sa, ops) == _reference(sa, ops)
    assert sa.operator_kernel(ops) == _reference(sa, ops)


def test_pure_python_path_above_cutoff(monkeypatch):
    """No numpy: the pure loop must serve large n with the same values."""
    sa = _attribution(KERNEL_NUMPY_MIN_OPS + 8, 30, 9)
    ops = sorted(sa._operator_edges)
    monkeypatch.setattr(shapley, "_HAS_NUMPY", False)
    assert sa.operator_kernel(ops) == _reference(sa, ops)


def test_falsify_oracle_detects_wrong_kernel():
    """Falsification: a perturbed kernel must not compare equal."""
    sa = _attribution(6, 20, 3)
    ops = sorted(sa._operator_edges)
    got = sa.operator_kernel(ops)
    got[ops[0]][ops[1]] += 1e-15
    assert got != _reference(sa, ops)
