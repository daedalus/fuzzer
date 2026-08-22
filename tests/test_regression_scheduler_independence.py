"""Regression tests: schedulers are fully independent of each other.

User requirement: schedulers compete, so no scheduler may import or call
another scheduler; they may only share neutral dependencies. Elo arbitration
sits on top in the services layer. These tests statically check the
import graph of every module in core/schedulers/ and verify the shared
operator→category taxonomy used to be the only cross-scheduler coupling.
"""

import ast
import inspect
from pathlib import Path

import fuzzer_tool.core.schedulers as schedulers_pkg
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.schedulers import GPUCBScheduler, HierarchicalBanditScheduler

SCHEDULERS_DIR = Path(inspect.getfile(schedulers_pkg)).parent
SCHEDULER_MODULES = {
    "epsilon_greedy",
    "exp3",
    "gp_ucb",
    "hierarchical",
    "monte_carlo",
    "mopt",
    "replicator",
}


def _imported_module_names(source: str) -> list[str]:
    """Return dotted module names this source imports (import / from-import)."""
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestSchedulersIndependent:
    """No scheduler module may import another scheduler module."""

    def test_no_scheduler_imports_another_scheduler(self):
        for module_name in sorted(SCHEDULER_MODULES):
            source = (SCHEDULERS_DIR / f"{module_name}.py").read_text()
            for imported in _imported_module_names(source):
                # Only the package __init__ may reference the sibling modules.
                assert not imported.startswith("fuzzer_tool.core.schedulers."), (
                    f"{module_name}.py imports sibling scheduler module {imported!r} — "
                    "schedulers must be independent of each other"
                )

    def test_gp_ucb_has_no_hierarchical_reference(self):
        source = (SCHEDULERS_DIR / "gp_ucb.py").read_text()
        assert "HierarchicalBanditScheduler" not in source

    def test_shared_categories_alias(self):
        assert HierarchicalBanditScheduler.CATEGORIES is OPERATOR_CATEGORIES
        assert list(OPERATOR_CATEGORIES.keys()) == list(
            HierarchicalBanditScheduler.CATEGORIES.keys()
        )

    def test_gp_ucb_uses_shared_categories(self):
        scheduler = GPUCBScheduler()
        # Sorted, not dict order: OPERATOR_CATEGORIES maps to sets, so
        # unsorted iteration made an operator's assigned category depend on
        # PYTHONHASHSEED wherever it appears in more than one. The scheduler
        # still draws from exactly the shared taxonomy, which is what this
        # test exists to check.
        assert scheduler._cat_names == sorted(OPERATOR_CATEGORIES)
        assert len(scheduler._cat_names) >= 1
        assert set(scheduler._op_to_cat) == {
            op for ops in OPERATOR_CATEGORIES.values() for op in ops
        }
        for op, cat in scheduler._op_to_cat.items():
            assert op in OPERATOR_CATEGORIES[cat]
