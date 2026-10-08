"""Regression: --seed did not cover the global ``np.random`` state.

``RandPool`` owns an independent ``np.random.default_rng(seed)`` Generator.
That Generator shares NO state with the legacy module-level ``np.random.*``
functions, so every global draw (QEA collapse/mutate, op_monte_carlo's
spectral probe and correlated-Thompson noise) ran off OS entropy regardless
of ``--seed``. A first fix seeded the global stream from ``Fuzzer``; the
final one moved every consumer onto ``RandPool`` (Hard Rule 16) and deleted
the global seeding. The scan below keeps the legacy stream out of ``src/``.
"""

import ast
from pathlib import Path

import numpy as np

import fuzzer_tool

# Module-level np.random names that do NOT touch the legacy global state.
_GENERATOR_API = {
    "default_rng",
    "Generator",
    "SeedSequence",
    "BitGenerator",
    "PCG64",
    "PCG64DXSM",
    "Philox",
    "SFC64",
    "MT19937",
}


def _global_draws(path: Path) -> list[str]:
    """``np.random.<legacy>`` / ``numpy.random.<legacy>`` uses in *path*."""
    hits = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr in _GENERATOR_API:
            continue
        mod = node.value
        if not (isinstance(mod, ast.Attribute) and mod.attr == "random"):
            continue
        if isinstance(mod.value, ast.Name) and mod.value.id in ("np", "numpy"):
            hits.append(f"{path.name}:{node.lineno} np.random.{node.attr}")
    return hits


def test_regression_no_global_numpy_draws_in_src():
    src = Path(fuzzer_tool.__file__).parent
    hits = [h for f in sorted(src.rglob("*.py")) for h in _global_draws(f)]
    assert hits == []


def test_scan_detects_global_draw(tmp_path):
    """Falsification: the scan must flag a legacy draw, or it proves nothing."""
    f = tmp_path / "bad.py"
    f.write_text("import numpy as np\nx = np.random.randn(3)\ny = np.random.default_rng(1)\n")
    assert _global_draws(f) == ["bad.py:2 np.random.randn"]


def test_randpool_is_not_backed_by_global_numpy():
    # Reseeding the global stream between two same-seed pools must not move
    # RandPool's output: the streams are separate.
    from fuzzer_tool.core.rand_pool import RandPool

    np.random.seed(11111)
    after_a = [RandPool(seed=99).randint(0, 1_000_000) for _ in range(20)]
    np.random.seed(22222)
    after_b = [RandPool(seed=99).randint(0, 1_000_000) for _ in range(20)]
    assert after_a == after_b


def test_qea_draws_are_reproducible_under_seed():
    # QEA draws from its injected RandPool, not the global stream
    # (see test_regression_qea_randpool.py); the pool seed governs it.
    from fuzzer_tool.core import qea
    from fuzzer_tool.core.rand_pool import RandPool

    amplitudes = np.full(64, 0.5, dtype=np.float64)

    first = qea.collapse(amplitudes.copy(), RandPool(seed=4242))
    second = qea.collapse(amplitudes.copy(), RandPool(seed=4242))
    assert first == second

    # A differing seed must actually move it, or the assert above would pass
    # on a constant.
    third = qea.collapse(amplitudes.copy(), RandPool(seed=9999))
    assert third != first


def test_qea_mutate_amplitudes_reproducible_under_seed():
    from fuzzer_tool.core import qea
    from fuzzer_tool.core.rand_pool import RandPool

    base = np.full(64, 0.5, dtype=np.float64)

    a = qea.mutate_amplitudes(base.copy(), rng=RandPool(seed=31337))
    b = qea.mutate_amplitudes(base.copy(), rng=RandPool(seed=31337))
    assert np.array_equal(np.asarray(a), np.asarray(b))
