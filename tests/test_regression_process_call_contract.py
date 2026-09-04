"""Regression: callers of run_target_stdin/run_target_file drifted to a stale arity.

Both helpers return ``(returncode, stderr, pid)``. Five call sites still unpacked
two names, which raises ``ValueError: too many values to unpack`` the moment the
real adapter runs:

- ``differential.diff_run`` (four sites) -- so ``--differential-target`` did not
  merely produce a dead signal, it crashed on the first seed. Every test for
  ``diff_run`` mocked ``run_target_stdin`` with a 2-tuple, so the suite
  validated the stale contract instead of the real one; one test even carried a
  comment naming the bug and mocked around it.
- ``stats_reporter._replay_crashes`` -- guarded by ``except Exception``, so every
  crash replay silently recorded -2 ("execution error") instead of the real
  returncode.
- ``commands`` ASAN verify, file_mode branch -- same broad guard, and it also
  omitted the required ``target_args`` positional. The stdin branch immediately
  below it unpacks three names correctly, which is what a half-finished
  migration looks like.

The mock-arity tests below are the ones that matter long term: a unit test that
mocks a collaborator can only be as correct as the contract it encodes, so the
mock's shape is asserted against the real signature rather than hardcoded.
"""

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.adapters import process

HELPERS = ("run_target_stdin", "run_target_file")
SRC_ROOT = Path(inspect.getfile(process)).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent


def _run_in_fresh_interpreter(script: str) -> str:
    """Execute `script` in a new Python process and return its stdout.

    These assertions need real fork+exec through run_target_stdin, with no
    mocks -- that is the whole point, since every mocked test missed the stale
    arity. But the interpreter running the suite cannot be trusted to fork
    safely: tests/test_abort_override.py dlopens the AFL shim .so into this
    process via InProcessRunner(direct_lite=True), and after that any
    run_target_stdin call here dies with SIGSEGV. Reproduce with

        pytest tests/test_abort_override.py::TestAbortOverride\
    ::test_shim_so_inprocess_abort_does_not_crash \
               tests/test_regression_differential_drift_wiring.py

    which exits 139, while either file alone exits 0. pytest-randomly is
    enabled here, so ordering cannot be relied on to keep them apart. Spawning
    a clean interpreter keeps the no-mocks property without depending on what
    else has run.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return proc.stdout.strip()


def _declared_arity(name: str) -> int:
    """Number of elements the helper's return annotation declares."""
    ann = inspect.signature(getattr(process, name)).return_annotation
    # tuple[int, str, int] -> 3
    return len(getattr(ann, "__args__", ()))


@pytest.mark.parametrize("name", HELPERS)
def test_helper_returns_three_values(name):
    assert _declared_arity(name) == 3


@pytest.mark.parametrize("name", HELPERS)
def test_annotation_matches_runtime_return(name):
    """The annotation is the contract callers are checked against, so pin it to
    what the function actually returns rather than trusting the annotation."""
    src = inspect.getsource(getattr(process, name))
    returns = [
        node
        for node in ast.walk(ast.parse(src.lstrip()))
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]
    assert returns, f"{name} has no tuple return to check"
    for node in returns:
        assert len(node.value.elts) == _declared_arity(name), (
            f"{name} returns {len(node.value.elts)} values at line {node.lineno} "
            f"but is annotated for {_declared_arity(name)}"
        )


def _unpack_sites(root: Path):
    """Every ``a, b = run_target_*(...)`` assignment under root."""
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in HELPERS and isinstance(node.targets[0], ast.Tuple):
                yield path, node.lineno, name, len(node.targets[0].elts)


def test_no_caller_unpacks_a_stale_arity():
    bad = [
        f"{path}:{lineno} unpacks {got} from {name} (returns {_declared_arity(name)})"
        for path, lineno, name, got in _unpack_sites(SRC_ROOT)
        if got != _declared_arity(name)
    ]
    assert not bad, "stale call contract:\n" + "\n".join(bad)


def test_run_target_file_callers_pass_target_args():
    """target_args is positional-required; omitting it is a TypeError, and the
    broad ``except Exception`` at the ASAN-verify call site swallowed it."""
    required = [
        p.name
        for p in inspect.signature(process.run_target_file).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    bad = []
    for path in SRC_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "run_target_file":
                continue
            supplied = {k.arg for k in node.keywords if k.arg}
            missing = [
                a
                for i, a in enumerate(required)
                if i >= len(node.args) and a not in supplied
            ]
            if missing:
                bad.append(f"{path}:{node.lineno} missing {missing}")
    assert not bad, "incomplete call:\n" + "\n".join(bad)


def _mock_payloads(tree):
    """Tuples configured as a mock's return_value / side_effect.

    Scoped deliberately to mock configuration rather than every tuple literal in
    the file: an assertion on diff_run's own ``(diverged, description)`` return
    is a legitimate 2-tuple and must not be flagged.
    """
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            if node.targets[0].attr in ("return_value", "side_effect"):
                value = node.value
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("return_value", "side_effect"):
                    value = kw.value
        if value is None:
            continue
        items = value.elts if isinstance(value, ast.List | ast.Tuple) else [value]
        for item in items:
            if isinstance(item, ast.Tuple):
                yield item


def test_diff_run_mocks_use_the_real_arity():
    """The suite's own mocks are the reason this drifted undetected. Assert that
    no test configures a process-helper mock narrower than the real return."""
    expected = _declared_arity("run_target_stdin")
    bad = []
    for path in TESTS_ROOT.glob("test_*differential*.py"):
        source = path.read_text()
        if "run_target_stdin" not in source and "run_target_file" not in source:
            continue
        for item in _mock_payloads(ast.parse(source)):
            if len(item.elts) != expected:
                bad.append(f"{path.name}:{item.lineno} {len(item.elts)}-tuple mock")
    assert not bad, (
        f"process helpers return {expected} values; these mocks disagree:\n"
        + "\n".join(bad)
    )


def test_diff_run_against_real_subprocesses(tmp_path):
    """End-to-end, no mocks: the bug was invisible to every mocked test."""
    a = tmp_path / "a.sh"
    a.write_text("#!/bin/sh\nexit 0\n")
    b = tmp_path / "b.sh"
    b.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    for p in (a, b):
        p.chmod(0o755)

    out = _run_in_fresh_interpreter(f"""
        from fuzzer_tool.services.differential import diff_run
        print(diff_run({str(a)!r}, {str(b)!r}, b"payload"))
        print(diff_run({str(a)!r}, {str(a)!r}, b"payload"))
    """)
    first, second = out.splitlines()
    assert first.startswith("(True,") and "returncode" in first
    assert second == "(False, 'identical')"
