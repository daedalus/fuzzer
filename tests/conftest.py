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
