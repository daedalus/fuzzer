#!/usr/bin/env python3
"""Find test functions with identical bodies.

Identity = normalized AST of (params, decorators, body); docstrings and
comments are ignored, ``pass``-only stubs skipped. Largest groups first.

Cross-file groups may bind different module-level names (a per-file
``SCHED`` class, helper or ``FLAGS``): same text, different test. Check
the imports before merging.

Usage: tools/find_dup_tests.py [PATH ...]   (default: tests/)
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "tests"
TEST_GLOB = "test_*.py"
TEST_PREFIX = "test"
SELF = "self"

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Hit:
    """One test function in a duplicate group."""

    path: Path
    owner: str  # enclosing class, "" at module level
    name: str
    line: int
    size: int  # body lines after the def


def _strip_doc(body: list[ast.stmt]) -> list[ast.stmt]:
    # Docstrings differ between copies; they are not behavior.
    first = body[0] if body else None
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if is_doc else body


def _params(fn: _FuncDef) -> str:
    # Drop a leading ``self`` so a method matches its module-level copy.
    args = fn.args
    pos = args.args
    if pos and pos[0].arg == SELF:
        args = ast.arguments(**{**vars(args), "args": pos[1:]})
    return ast.dump(args)


def _key(fn: _FuncDef) -> tuple | None:
    body = _strip_doc(fn.body)
    if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
        return None

    decos = tuple(ast.dump(d) for d in fn.decorator_list)
    return (_params(fn), decos, tuple(ast.dump(s) for s in body))


def _scan(path: Path, groups: dict) -> None:
    # Module-level tests and methods of top-level/nested classes.
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        print(f"skip {path}: {exc.msg}", file=sys.stderr)
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef):
            continue

        owner = node.name if isinstance(node, ast.ClassDef) else ""
        for fn in node.body:
            if not isinstance(fn, _FuncDef) or not fn.name.startswith(TEST_PREFIX):
                continue

            key = _key(fn)
            if key is None:
                continue

            size = (fn.end_lineno or fn.lineno) - fn.lineno
            groups[key].append(Hit(path, owner, fn.name, fn.lineno, size))


def find_dups(roots: list[Path]) -> list[list[Hit]]:
    """Return groups of 2+ identical tests under *roots*, largest first."""
    groups: dict[tuple, list[Hit]] = defaultdict(list)
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob(TEST_GLOB))
        for path in files:
            _scan(path, groups)

    dups = [g for g in groups.values() if len(g) > 1]
    dups.sort(key=lambda g: (-g[0].size, str(g[0].path), g[0].line))
    return dups


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path(DEFAULT_ROOT)]
    dups = find_dups(roots)

    for group in dups:
        print("---")
        for h in group:
            print(f"  {h.path}:{h.line} {h.owner}.{h.name} ({h.size} lines)")

    print(f"{len(dups)} groups", file=sys.stderr)
    return 1 if dups else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
