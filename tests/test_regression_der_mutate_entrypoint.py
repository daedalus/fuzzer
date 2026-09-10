"""``DerMutator`` has two entry points, and they must agree on the op list.

The class was written as one public method per operator. That is a fine
shape and the operator handlers use it directly -- `_op_der_len_mutate`
names `mutate_length`, and so on -- but it left DER outside every sweep
that discovers mutators and drives `mutate()`. The dispatch-arity check,
the parser-truncation sweep and the field-overflow sweep all skipped it, so
DER length and tag arithmetic was the one format family nothing exercised
in bulk.

`mutate()` closes that. The cost is a second place that knows which four
methods are operators, so the list lives in `DerMutator.OPERATORS` and this
file asserts it still matches what `services/operators.py` dispatches. A
fifth DER operator added to one and not the other is exactly the drift this
guards -- the same failure that put fourteen operators outside the no-op
sweep and left six copies of the scheduler ballot.
"""

from __future__ import annotations

import ast
import inspect

from fuzzer_tool.core.mutations.der import DerMutator, parse_der
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services import operators as operators_mod


def _methods_the_handlers_dispatch() -> set[str]:
    """The method names passed to `_der_mutate` from the `_op_der_*` handlers."""
    tree = ast.parse(inspect.getsource(operators_mod))
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "_der_mutate" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return found


def test_discovery_found_the_handlers():
    """Guard: an empty set would make the assertion below vacuous."""
    assert len(_methods_the_handlers_dispatch()) >= 4


def test_operators_tuple_matches_what_the_handlers_dispatch():
    dispatched = _methods_the_handlers_dispatch()

    assert set(DerMutator.OPERATORS) == dispatched, (
        "DerMutator.OPERATORS and the _op_der_* handlers disagree; one gained "
        f"an operator the other did not: {set(DerMutator.OPERATORS) ^ dispatched}"
    )


def test_every_listed_operator_exists_and_takes_the_call():
    for name in DerMutator.OPERATORS:
        method = getattr(DerMutator, name, None)
        assert method is not None, f"OPERATORS names {name}, which does not exist"
        params = inspect.signature(method).parameters
        assert "max_len" in params and "rng" in params, (
            f"{name} must take max_len and rng like its siblings: {list(params)}"
        )


def test_unparseable_input_falls_back_to_the_generator():
    m = DerMutator(seed=1)
    out = m.mutate(b"\xff" * 64, max_len=4096, rng=RandPool(seed=1))

    assert parse_der(out) is not None, "the fallback must produce parseable DER"


def test_max_len_is_honoured():
    """f5435af's lesson: a generator reached through a keyword-less call
    silently used its own default and ignored the cap."""
    m = DerMutator(seed=1)
    for cap in (8, 16, 64):
        for seed in range(6):
            out = m.mutate(b"\xff" * 64, max_len=cap, rng=RandPool(seed=seed))
            assert len(out) <= cap, f"max_len={cap} produced {len(out)} bytes"


def test_parseable_input_is_returned_or_mutated_never_none():
    """`mutate` must always return bytes; the per-method contract is None."""
    m = DerMutator(seed=1)
    seed = m._generate_random_der(max_len=4096, rng=RandPool(seed=4))
    assert parse_der(seed) is not None, "fixture guard: the seed must parse"

    rng = RandPool(seed=7)
    for _ in range(2000):
        out = m.mutate(bytes(seed), max_len=4096, rng=rng)
        assert isinstance(out, bytes)


def test_some_operator_actually_changes_a_parseable_input():
    """Falsification for the test above: returning the input unchanged every
    time would satisfy it. At least one draw has to do real work."""
    m = DerMutator(seed=1)
    seed = m._generate_random_der(max_len=4096, rng=RandPool(seed=4))

    rng = RandPool(seed=7)
    changed = sum(m.mutate(bytes(seed), max_len=4096, rng=rng) != seed for _ in range(500))

    assert changed > 0, "every draw returned the input unchanged"
