"""A ``rng: RandPool`` parameter must only be sent methods RandPool has.

The Hard Rule 16 migration (8312b15) rewrote hundreds of stdlib ``random``
calls onto ``RandPool``, and RandPool's API is close to but not the same as
the stdlib module's. One rename was guessed:
``MonteCarloScheduler._null_js_samples_python`` called
``rng.shuffle_list(...)``, by analogy with the real batch helpers
(``random_list``, ``choice_list``, ``randint_list``). There is no
``shuffle_list``; there is ``shuffle``, in place.

That backend therefore raised AttributeError on every call, and nothing
noticed because it is the ``not _HAS_NUMPY`` fallback and numpy is always
installed in CI. A wrong method name on a duck-typed argument is invisible
until the line runs, which for a fallback can be never.

Checked by annotation: a function that says ``rng: RandPool`` has declared
what it expects, so every ``rng.<attr>(...)`` in its body is checkable
without running it. Parameters that are deliberately duck-typed -- the ones
annotated ``Any``, which take a scripted double in tests -- are out of
scope by construction, which is the right line: this test enforces a
declaration, not a guess about intent.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from fuzzer_tool.core.rand_pool import RandPool

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool"
_RANDPOOL_API = {name for name in dir(RandPool) if not name.startswith("__")}


def _annotated_rng_functions():
    """Yield (path, function, parameter name) for every ``x: RandPool`` arg."""
    for path in sorted(_SRC.rglob("*.py")):
        # rand_pool.py's own `self._rng` is the numpy Generator underneath,
        # not a RandPool; it is the implementation, not a consumer.
        if path.name == "rand_pool.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                if arg.annotation is None:
                    continue
                # `from __future__ import annotations` is on in most of the
                # tree, so annotations arrive as source text either way.
                text = ast.unparse(arg.annotation)
                if "RandPool" not in text:
                    continue
                yield path, node, arg.arg


_TARGETS = list(_annotated_rng_functions())


def test_discovery_found_the_annotated_parameters():
    """Guard: an empty list would make the assertion below vacuous."""
    assert len(_TARGETS) >= 10, f"only found {len(_TARGETS)} annotated rng parameters"


@pytest.mark.parametrize(
    "target",
    _TARGETS,
    ids=lambda t: f"{t[0].name}:{t[1].name}:{t[2]}" if isinstance(t, tuple) else str(t),
)
def test_only_real_randpool_methods_are_called(target):
    path, fn, param = target
    called = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == param
        ):
            called.add(node.func.attr)

    missing = sorted(called - _RANDPOOL_API)
    assert not missing, (
        f"{path.name}:{fn.lineno} {fn.name}() declares `{param}: RandPool` and calls "
        f"{missing} on it, which RandPool does not have -- AttributeError the first "
        "time that line runs"
    )
