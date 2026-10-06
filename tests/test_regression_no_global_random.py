"""Hard Rule 16: the fuzzer never seeds or draws from the global ``random``.

``random.seed`` in ``Fuzzer.__init__`` / ``_reseed_after_stall`` reseeded a
process-wide stream shared with every other library. The draws that relied on
it now come from ``RandPool``; the seed feeds the default pool instead.
"""

import ast
import pathlib
import random
import tempfile
from unittest.mock import patch

import pytest

from fuzzer_tool.core.rand_pool import get_default_rand_pool

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool"
RANDOM_NAMES = {"random", "_random", "_rand", "std_random"}


def _make_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


def _global_draws_untouched(fn):
    """Run *fn*; the global ``random`` state must be identical afterwards."""
    random.seed(0)
    before = random.getstate()
    fn()
    assert random.getstate() == before


# ── Falsification: the source never calls random.seed ───────────────────


def _seed_calls(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "seed"):
            continue
        if isinstance(func.value, ast.Name) and func.value.id in RANDOM_NAMES:
            yield f"{path.relative_to(SRC)}:{node.lineno}"


def test_no_random_seed_call_in_src():
    found = [hit for p in SRC.rglob("*.py") for hit in _seed_calls(p)]
    assert found == []


def test_scanner_detects_a_seed_call(tmp_path):
    """Control: the scanner flags a planted violation."""
    bad = tmp_path / "bad.py"
    bad.write_text("import random\nrandom.seed(1)\n")
    with patch.object(pathlib.Path, "relative_to", lambda self, _: self.name):
        assert list(_seed_calls(bad)) == ["bad.py:2"]


# ── Fuzzer seeding ──────────────────────────────────────────────────────


def test_init_leaves_global_random_untouched():
    _global_draws_untouched(lambda: _make_fuzzer(seed=42))


def test_stall_reseed_leaves_global_random_untouched():
    f = _make_fuzzer(seed=42)
    _global_draws_untouched(f._reseed_after_stall)


def test_seed_drives_default_pool():
    _make_fuzzer(seed=7)
    a = [get_default_rand_pool().randint(0, 255) for _ in range(16)]
    _make_fuzzer(seed=7)
    b = [get_default_rand_pool().randint(0, 255) for _ in range(16)]
    _make_fuzzer(seed=8)
    c = [get_default_rand_pool().randint(0, 255) for _ in range(16)]
    assert a == b
    assert a != c


def test_stall_reseed_moves_default_pool_deterministically():
    f = _make_fuzzer(seed=7)
    f._reseed_after_stall()
    a = [get_default_rand_pool().randint(0, 255) for _ in range(16)]
    g = _make_fuzzer(seed=7)
    g._reseed_after_stall()
    b = [get_default_rand_pool().randint(0, 255) for _ in range(16)]
    assert a == b


# ── Former global-random draw sites ─────────────────────────────────────


def test_shapley_inverse_iteration_no_global():
    from fuzzer_tool.core.shapley import ShapleyAttribution as Cls

    lap = [[1.0, -0.5], [-0.5, 1.0]]
    _global_draws_untouched(lambda: Cls._inverse_iteration_py(lap, 2))


def test_tree_mutator_fallbacks_no_global():
    from fuzzer_tool.core.tree_mutator import lightweight_tree_mutate

    data = bytes(range(64))
    _global_draws_untouched(lambda: [lightweight_tree_mutate(data) for _ in range(20)])


def test_gradient_descent_fallback_no_global():
    from fuzzer_tool.core.gradient_descent import _candidate_positions

    buf = bytes(256)
    _global_draws_untouched(lambda: _candidate_positions(buf, b"\x01\x02"))


def test_edge_tracker_pair_sample_no_global():
    from fuzzer_tool.core.edge_tracker import EdgeTracker

    t = EdgeTracker()
    _global_draws_untouched(lambda: t.update_correlation(set(range(40))))


def test_qea_tournament_no_global():
    from fuzzer_tool.core.qea import QEALifecycle

    qea = QEALifecycle()
    pool = [type("I", (), {"fitness": float(i)})() for i in range(8)]
    _global_draws_untouched(lambda: qea._tournament_select(pool))


def test_frameshift_discover_no_global():
    from fuzzer_tool.core.analyzers.analyzer_frameshift import FrameShift

    fs = FrameShift()
    _global_draws_untouched(lambda: fs.discover_relations(bytes(64), lambda b: len(b) & 3))


@pytest.mark.parametrize("seed", [1, 2**40])
def test_seed_accepts_wide_values(seed):
    """Adversarial: seeds wider than 32 bits still construct and reseed."""
    f = _make_fuzzer(seed=seed)
    assert 0 <= f._reseed_after_stall() < 2**32
