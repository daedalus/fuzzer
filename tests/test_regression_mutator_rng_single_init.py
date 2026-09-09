"""Every format mutator owns exactly one pool, built once, never stdlib.

Before the consolidation these classes carried ``_rng = random`` as a
*class* attribute -- shared across every instance, pointing at the stdlib
module, and therefore unseeded by ``--seed`` (Hard Rule 16). Others carried
``_rng = None`` and only ever got a value part-way through ``mutate()``, so
any helper reached before that assignment saw None.

The discovery is by walking the package, not a hardcoded list: a new
mutator added later is covered without editing this file (Hard Rule: no
hardcoded counts).
"""

import ast
import importlib
import inspect
import pkgutil

import pytest

import fuzzer_tool.core.mutations as mutations_pkg
from fuzzer_tool.core.rand_pool import RandPool


def _mutator_classes():
    """Every ``*Mutator`` class in core.mutations that reads ``self._rng``."""
    found = []
    for mod in pkgutil.iter_modules(mutations_pkg.__path__):
        m = importlib.import_module(f"{mutations_pkg.__name__}.{mod.name}")
        for name, obj in vars(m).items():
            if not inspect.isclass(obj) or obj.__module__ != m.__name__:
                continue
            if not name.endswith("Mutator"):
                continue
            src = inspect.getsource(obj)
            if "self._rng" in src:
                found.append(pytest.param(obj, id=f"{mod.name}.{name}"))
    return found


MUTATORS = _mutator_classes()


def test_discovery_found_the_mutators():
    # Falsification guard: if the walk silently returns nothing, every
    # parametrized test below vacuously passes.
    assert len(MUTATORS) >= 20


@pytest.mark.parametrize("cls", MUTATORS)
class TestSinglePoolPerMutator:
    def test_constructs_with_no_arguments(self, cls):
        cls()

    def test_instance_owns_a_randpool(self, cls):
        assert isinstance(cls()._rng, RandPool)

    def test_pool_is_per_instance_not_shared(self, cls):
        # The old class attribute made every instance share one RNG.
        assert cls()._rng is not cls()._rng

    def test_no_class_level_rng_attribute(self, cls):
        # ``_rng`` must come from __init__, so it lives in the instance dict
        # (or a slot), never on the class itself.
        assert "_rng" not in vars(cls)

    def test_seed_is_honoured(self, cls):
        a, b = cls(seed=1234), cls(seed=1234)
        left = [a._rng.randint(0, 255) for _ in range(16)]
        right = [b._rng.randint(0, 255) for _ in range(16)]
        assert left == right

        other = cls(seed=4321)
        assert left != [other._rng.randint(0, 255) for _ in range(16)]

    def test_source_never_falls_back_to_stdlib_random(self, cls):
        """No ``self._rng or random`` and no ``self._rng = ... random``.

        Parsed rather than grepped so a mention inside a docstring or a
        comment cannot fail the test, and so the check survives reformatting.
        """
        tree = ast.parse(inspect.getsource(cls))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            reads_rng = any(
                isinstance(v, ast.Attribute) and v.attr == "_rng" for v in node.values
            )
            names = {v.id for v in node.values if isinstance(v, ast.Name)}
            assert not (reads_rng and "random" in names), (
                f"{cls.__name__} still falls back to the stdlib random module"
            )
