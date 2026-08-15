"""Shared test fixtures and environment guards.

Several suites need a toolchain component that is not a declared project
dependency -- clang, or the optional ``smt`` extra (z3). When it is absent
those tests errored out rather than skipping, so a machine without clang
reported 28 failures that said nothing about the code under test. CI hits
this too: ``pip install -e ".[dev]"`` does not pull in z3.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest


def _has(tool: str) -> bool:
    return shutil.which(tool) is not None


requires_clang = pytest.mark.skipif(not _has("clang"), reason="clang not installed")
requires_gcc = pytest.mark.skipif(not _has("gcc"), reason="gcc not installed")
requires_z3 = pytest.mark.skipif(
    importlib.util.find_spec("z3") is None,
    reason="z3-solver not installed (optional 'smt' extra)",
)


@pytest.fixture(scope="session", autouse=True)
def _fuzz_loader_built():
    """Build ``fuzz_loader`` once, before any test runs.

    W3: in a fresh clone the binary does not exist (it is gitignored), and
    the suite *hangs* rather than fails -- a test that starts a
    ForkserverRunner blocks on a reply from a loader that was never built,
    and the join only unblocks after the runner's grace period, per test.
    Building it up front is a one-off ~0.3s and turns the failure mode from
    "CI wedged" into "these tests skip".

    Deliberately best-effort: no compiler is a legitimate environment (the
    tests that need the loader already skip on ``_ensure_compiled() is
    None``), so a failure here must not abort collection.
    """
    try:
        from fuzzer_tool.adapters.forkserver import _ensure_compiled

        _ensure_compiled()
    except Exception:  # pragma: no cover - environment-dependent
        pass


def pytest_make_parametrize_id(config, val):
    """Shorten long byte-string parametrize IDs to keep test output readable."""
    if isinstance(val, bytes) and len(val) > 32:
        return f"bytes[{len(val)}]"
    return None
