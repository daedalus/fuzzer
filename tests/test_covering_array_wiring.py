"""The covering-array mutators build their rows with covering_array.OPERATOR_STRATEGY."""

from __future__ import annotations

import ast
import random
from pathlib import Path

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core.mutations import covering_array_gzip, covering_array_mutate

_MUTATIONS = Path(covering_array_mutate.__file__).parent


def test_operator_strategy_is_density():
    assert ca.OPERATOR_STRATEGY == "density"


def test_every_operator_generate_call_passes_the_strategy():
    # Source-level guard: a new generate() call that forgets strategy= would
    # silently fall back to "aetg" for that operator.
    calls = 0
    for path in sorted(_MUTATIONS.glob("covering_array_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "covering_array"
            ):
                calls += 1
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                assert kw.get("strategy") == "covering_array.OPERATOR_STRATEGY", path.name
    assert calls == 4


def test_png_mutator_rows_come_from_density():
    from tests.test_covering_array_mutate import _make_png

    m = covering_array_mutate.PngCoveringArrayMutator()
    m.mutate(_make_png(), random.Random(7))
    expect = ca.generate(
        covering_array_mutate._VALUE_SETS, t=2, rng=random.Random(7), strategy="density"
    )
    assert m._rows == expect


def test_gzip_mutator_rows_come_from_density():
    m = covering_array_gzip.GzipCoveringArrayMutator()
    data = b"\x1f\x8b\x08\x00" + b"\x00" * 16
    m.mutate(data, random.Random(7))
    expect = ca.generate(
        covering_array_gzip._VALUE_SETS, t=2, rng=random.Random(7), strategy="density"
    )
    assert m._rows == expect
