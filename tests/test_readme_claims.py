"""README headline counts must come from the sources of truth (2026-11-24 review).

The README claimed 271 operators and 41 scheduling strategies for months
while the registry held 364 and the strategy pool 60. The drift was in the
conservative direction (the project understated itself) but systematic:
nothing tied the published numbers to the tables that define them, so they
were hand-maintained while the code was machine-written at high frequency.

These tests regenerate the headline numbers from the operator registry and
the strategy sources, so an operator or scheduler PR either updates the
README in the same change or turns the suite red. The test-function claim
is a floor check (the count moves daily; a floor cannot go stale upward).
"""

import inspect
import re
from pathlib import Path

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services import operators as operators_module
from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def _strategy_count() -> int:
    """Every name the Elo ballot can hold.

    ``operator_strategy_pool`` appends names conditionally on the enabled
    schedulers; the *set* of names it can ever append is the strategy
    census. Parsed from source (not a hand-kept tuple) so a new arm is
    counted the moment its ``available.append`` lands.
    """
    src = inspect.getsource(operators_module.operator_strategy_pool)
    pool = set(re.findall(r'available\.append\("([a-z0-9_]+)"\)', src))
    return len(pool | set(_OPERATOR_STRATEGY_NAMES))


def _test_function_count() -> int:
    return sum(
        len(re.findall(r"^\s*def test_", p.read_text(), re.M))
        for p in (REPO / "tests").rglob("test_*.py")
    )


def _headline() -> str:
    return README.read_text().splitlines()[4]


def test_operator_count_matches_registry():
    n = len(REGISTRY.names())
    line = _headline()
    m = re.search(r"(\d+) mutation operators across (\d+) categories", line)
    assert m, f"README headline no longer states the operator count: {line!r}"
    assert int(m.group(1)) == n, (
        f"README says {m.group(1)} operators; the registry holds {n}. "
        "Regenerate the headline (see also REGISTRY.names())."
    )
    assert int(m.group(2)) == len(REGISTRY.categories()), (
        f"README says {m.group(2)} categories; the registry holds {len(REGISTRY.categories())}."
    )


def test_strategy_count_matches_pool():
    n = _strategy_count()
    line = _headline()
    m = re.search(r"(\d+) operator-scheduling strategies", line)
    assert m, f"README headline no longer states the strategy count: {line!r}"
    assert int(m.group(1)) == n, (
        f"README says {m.group(1)} strategies; the ballot can hold {n} "
        "(operator_strategy_pool + _OPERATOR_STRATEGY_NAMES)."
    )


def test_test_function_claim_is_a_true_floor():
    """The '21,000+ test functions' claim must not overstate the suite."""
    m = re.search(r"(\d+),000\+ test functions", README.read_text())
    assert m, "README no longer states a test-function count"
    claimed = int(m.group(1)) * 1000
    actual = _test_function_count()
    assert claimed <= actual, (
        f"README claims {claimed:,}+ test functions; the tree holds {actual:,}. Lower the claim."
    )
