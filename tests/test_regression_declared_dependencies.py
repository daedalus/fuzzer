"""Regression test: every unconditionally-imported package is declared.

corpus_manager.py added a module-level `import xxhash` for fast corpus
dedup hashing, but xxhash was never added to pyproject's dependencies.
Because corpus_manager sits in the CLI import chain, a fresh install
without xxhash broke the *entire* CLI — not merely an optional feature —
with a bare ModuleNotFoundError at startup.

The class of bug is easy to reintroduce: adding a top-level import is a
one-line change, and it works fine in any environment where the package
already happens to be installed. This test walks the package for
unconditional third-party imports and checks each against the declared
dependencies, so the omission fails in CI rather than on a user's fresh
checkout.

Imports guarded by try/except or placed inside functions are optional by
construction and are intentionally not required to be declared.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "fuzzer_tool"
PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# Import name -> distribution name, where they differ.
_IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "z3": "z3-solver",
}


def _declared_distributions() -> set[str]:
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text())
    project = data.get("project", {})
    names: set[str] = set()
    for spec in project.get("dependencies", []):
        names.add(_dist_name(spec))
    for extra in project.get("optional-dependencies", {}).values():
        for spec in extra:
            names.add(_dist_name(spec))
    return names


def _dist_name(spec: str) -> str:
    for sep in (">=", "==", "<=", "~=", ">", "<", "[", ";"):
        spec = spec.split(sep)[0]
    return spec.strip().lower().replace("_", "-")


def _stdlib_names() -> set[str]:
    return set(getattr(sys, "stdlib_module_names", set()))


def _unconditional_third_party_imports() -> dict[str, list[str]]:
    """module top-level import name -> files importing it unconditionally."""
    stdlib = _stdlib_names()
    found: dict[str, list[str]] = {}

    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in tree.body:  # module level only
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import
                    continue
                if node.module:
                    names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name == "fuzzer_tool" or name.startswith("_"):
                    continue
                found.setdefault(name, []).append(str(path.relative_to(SRC)))
    return found


def test_unconditional_imports_are_declared_dependencies():
    declared = _declared_distributions()
    missing = []
    for import_name, files in sorted(_unconditional_third_party_imports().items()):
        dist = _IMPORT_TO_DIST.get(import_name, import_name).lower().replace("_", "-")
        if dist not in declared:
            missing.append(f"{import_name} (dist {dist!r}) imported by {files[:3]}")
    assert not missing, (
        "third-party packages imported unconditionally but not declared in "
        "pyproject dependencies — a fresh install will fail at import time:\n  "
        + "\n  ".join(missing)
    )


def test_xxhash_specifically_is_declared():
    """Pins the original occurrence: corpus_manager imports it at module
    level and sits in the CLI import chain."""
    assert "xxhash" in _declared_distributions()


def test_helper_detects_the_imports_it_should():
    """Guard the guard: if the AST walk silently found nothing, the test
    above would vacuously pass."""
    found = _unconditional_third_party_imports()
    assert "numpy" in found, "AST walk found no numpy import — detector broken"
