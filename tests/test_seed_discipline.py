"""Guard for seed discipline (docs/port-backlog.md, item F6).

The seed plugin in ``conftest.py`` is worth nothing if new tests keep
constructing their own unseeded RNGs, so the rule is enforced here rather than
left to review. Two things are checked:

1. **No bare ``random.Random()`` in the test suite.** An unseeded RNG in a test
   is the worst of both worlds -- the failure is real, and it is
   unreproducible. This is not hypothetical: the sweep that landed P0-1 found
   ``test_new_operators.py::TestMagicValues::test_inserts_magic_value``
   failing on 0.66% of runs with nothing to re-run, and
   two more of the same shape were confirmed and fixed alongside it
   (``test_bloom_exec_dedup`` at 0.33%, ``test_mb_cbh_reanchor`` at 1.0%),
   all unseeded RNGs in tests asserting statistical properties.

2. **The plugin itself still works.** ``pytest_report_header`` must emit the
   seed, and ``--fuzz-seed`` must round-trip, because the whole design rests
   on the seed reaching stdout before collection -- i.e. surviving a suite
   that segfaults or hangs during a test, both of which this suite has done.

The bare-``Random()`` check is deliberately a whole-suite scan rather than a
per-file one: the point is that the count is zero and stays zero.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent


def _conftest():
    """Load ``tests/conftest.py`` as a module.

    pytest imports conftest under an internal name that is not importable by
    ``import conftest`` under the default import mode, and the plugin hooks
    are what is under test here, so load it by path.
    """
    spec = importlib.util.spec_from_file_location("_conftest_uut", TESTS_DIR / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS_DIR.rglob("test_*.py"))


def _bare_random_calls(tree: ast.AST) -> list[int]:
    """Line numbers of ``random.Random()`` / ``Random()`` with no arguments."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name == "Random":
            hits.append(node.lineno)
    return hits


def test_no_unseeded_random_in_tests():
    """No test may construct an RNG whose seed cannot be recovered.

    Seed it from the ``random_seed`` fixture (per-test, derived from the
    session seed) or from an explicit literal if the test is a genuinely
    deterministic example. Either is fine; neither is optional.
    """
    offenders = []
    for path in _test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders += [f"{path.relative_to(TESTS_DIR)}:{line}" for line in _bare_random_calls(tree)]
    assert not offenders, (
        "Unseeded random.Random() in tests -- a failure here is real but "
        "unreproducible. Use the `random_seed` fixture:\n  " + "\n  ".join(offenders)
    )


def test_report_header_publishes_seed(pytestconfig):
    """The seed must be recoverable from the header, not from a test body.

    A test that prints its own seed prints nothing when the process dies, and
    a seed that is not in the CI log is the same as no seed at all.
    """
    header = _conftest().pytest_report_header(pytestconfig)
    match = re.search(r"--fuzz-seed=(0x[0-9a-f]+)", header)
    assert match, f"seed not recoverable from report header: {header!r}"
    assert int(match.group(1), 0) == pytestconfig.fuzz_seed


def test_random_seed_fixture_is_derived_not_shared(random_seed, pytestconfig):
    """Per-test derivation, so deselecting tests does not move the others.

    A test that fails under ``-k foo`` must fail identically under a full run
    with the same ``--fuzz-seed``; handing every test the raw session seed
    would make reproduction depend on which tests ran alongside it.
    """
    assert isinstance(random_seed, int)
    assert 0 <= random_seed < 2**64
    assert random_seed != pytestconfig.fuzz_seed, "fixture handed out the session seed raw"


@pytest.mark.parametrize(
    ("raw", "expected"), [("0x5eed", 0x5EED), ("24301", 24301), ("0o777", 511)]
)
def test_fuzz_seed_option_round_trips(raw, expected):
    """``--fuzz-seed`` must accept the 0x notation the header prints.

    The header advertises ``--fuzz-seed=0x...``; if the option parsed as plain
    decimal, copy-pasting the reproduce line out of a CI log would either
    error or -- worse -- silently reproduce a different seed.
    """

    class _Config:
        def getoption(self, _name):
            return raw

    config = _Config()
    _conftest().pytest_configure(config)
    assert config.fuzz_seed == expected


def test_fuzz_seed_defaults_to_random():
    """Absent the flag the seed is fresh, or the suite explores one point forever."""
    pytest_configure = _conftest().pytest_configure

    class _Config:
        def getoption(self, _name):
            return None

    seeds = set()
    for _ in range(8):
        config = _Config()
        pytest_configure(config)
        seeds.add(config.fuzz_seed)
    assert len(seeds) == 8, "default seed is not varying between sessions"
