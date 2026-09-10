"""The shared operator mock must satisfy everything ``select_op`` reads.

``OperatorEngine.select_op`` builds the Elo ballot from bare attribute
reads -- ``if f._use_c2ucb and f._c2ucb`` -- so a mock fuzzer missing one
pair raises AttributeError the moment the ballot is built, before any
assertion in the test runs.

That is not hypothetical. Adding the C2UCB scheduler broke 21 tests across
six files, none of them about C2UCB: five hand-rolled fake fuzzers and the
shared ``make_minimal_fuzzer`` all had to grow ``_use_c2ucb``/``_c2ucb``
and none did. The operator env docstring already names the underlying
cause -- the operator's declared contract is the whole ``Fuzzer`` object,
29 attributes of shadow, port item F1 -- and until that is paid down, the
contract needs a tripwire.

This is that tripwire. It reads the attribute set out of ``select_op``
itself, so the next scheduler added to the ballot fails one test that names
the missing attribute rather than twenty scattered ones that do not.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from fuzzer_tool.services import operators as operators_mod
from tests.support.operator_env import make_minimal_fuzzer


def _attributes_select_op_reads() -> set[str]:
    tree = ast.parse(inspect.getsource(operators_mod))
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "select_op"]
    assert fns, "select_op not found -- this test's premise is stale"

    # `f` is the local the method binds the fuzzer to; anything read off it
    # is part of the contract a mock has to satisfy.
    return {
        node.attr
        for fn in fns
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "f"
    }


_READS = sorted(_attributes_select_op_reads())


def test_discovery_found_the_reads():
    """Guard: an empty set would make the assertion below vacuous."""
    assert len(_READS) >= 30, f"only found {len(_READS)} attribute reads"


def test_minimal_fuzzer_has_everything_select_op_reads():
    mock = make_minimal_fuzzer(seed=1)
    missing = [name for name in _READS if not hasattr(mock, name)]

    assert not missing, (
        "select_op reads these off the fuzzer and make_minimal_fuzzer does not "
        f"provide them, so the ballot raises AttributeError: {missing}"
    )


def test_the_ballot_builds_and_is_empty_by_default():
    """Falsification for the above: presence is not enough, it has to run.

    Every scheduler defaults off, so a mock with the full surface must
    produce a ballot -- not an exception -- and that ballot must be empty.
    An attribute present but set to a truthy placeholder would pass the
    hasattr check and then dispatch into a scheduler that is not there.
    """
    from fuzzer_tool.services.operators import OperatorEngine

    engine = OperatorEngine(make_minimal_fuzzer(seed=1))
    op = engine.select_op(["bit_flip", "byte_flip"])

    assert op in ("bit_flip", "byte_flip"), (
        "with every scheduler off, select_op must still return one of the "
        f"offered operators, got {op!r}"
    )


@pytest.mark.parametrize("sched", ["c2ucb", "cmaes", "contextual", "cucb", "ducb"])
def test_each_scheduler_pair_is_present_and_falsy(sched):
    """The pair has to be both there and off; either half missing is a bug."""
    mock = make_minimal_fuzzer(seed=1)

    assert hasattr(mock, f"_use_{sched}"), f"_use_{sched} missing"
    assert hasattr(mock, f"_{sched}"), f"_{sched} missing"
    assert not getattr(mock, f"_use_{sched}")
    assert not getattr(mock, f"_{sched}")
