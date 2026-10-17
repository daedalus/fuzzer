"""Regression: tools still launched `fuzz -n` after bd2a5f4 removed it.

bd2a5f4 dropped `fuzz -n/--iterations` for `--max-execs`. Seven tools kept
building `[..., "fuzz", ..., "-n", N]`; argparse rejected it, so every
bench_paired cell exited in 0.5 s and recorded "NO COVERAGE" (measured:
40/40 cells on the fuzzgoat set).
"""

import ast
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
REMOVED = {"-n", "--iterations"}


def _fuzz_argv_literals(tree: ast.AST):
    """List/tuple literals that contain the string "fuzz"."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        consts = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if "fuzz" in consts:
            yield node.lineno, consts


def test_regression_tools_use_max_execs():
    offenders = []
    for path in sorted(TOOLS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, consts in _fuzz_argv_literals(tree):
            if REMOVED & set(consts):
                offenders.append(f"{path.relative_to(TOOLS.parent)}:{lineno}")
    assert offenders == []


def test_scanner_flags_a_removed_flag():
    # Control: the scanner must catch the exact shape it guards against.
    tree = ast.parse('cmd = [sys.executable, "-m", "fuzzer_tool", "fuzz", t, "-n", "5"]')
    assert any(REMOVED & set(c) for _, c in _fuzz_argv_literals(tree))
