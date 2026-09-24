"""Regression: cond_stmt carried an unused, buggy duplicate wall solver.
Z3Solver.solve_comparison_wall is the single implementation."""

from __future__ import annotations

from fuzzer_tool.core import cond_stmt
from fuzzer_tool.core.smt_solver import Z3Solver


def test_regression_minimax_duplicate_removed():
    assert not hasattr(cond_stmt, "solve_comparison_wall_minimax")
    assert not hasattr(cond_stmt, "_check_comparison_satisfied")


def test_canonical_wall_solver_present():
    """Falsification: the surviving solver must still exist."""
    assert callable(getattr(Z3Solver, "solve_comparison_wall", None))
