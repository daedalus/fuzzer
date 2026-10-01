"""Regression: the operator chi-squared test ran on cells with expected < 5.

``_run_chi2_operator_test`` fed every operator with >= 1 execution into a
2 x K table. An operator run once that hit once has expected successes
~0.01 at a 1% base rate, contributing ~100 to chi2 alone, so two operators
with identical rates were logged as "significant". Rows whose expected cells
fall below Cochran's minimum are now dropped before the test.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.chi_squared import COCHRAN_MIN_EXPECTED, drop_sparse_rows
from fuzzer_tool.services.fuzzer import Fuzzer

SIGNIFICANT = "significant (p<0.05)"


def _expected(table, r, c):
    rows = [sum(row) for row in table]
    cols = [sum(row[j] for row in table) for j in range(len(table[0]))]
    return rows[r] * cols[c] / sum(rows)


def test_drops_rows_below_cochran_minimum():
    table = [[10.0, 990.0], [1.0, 0.0], [12.0, 988.0]]
    kept = drop_sparse_rows(table)
    assert kept == [table[0], table[2]]
    assert _expected(table, 1, 0) < COCHRAN_MIN_EXPECTED


def test_keeps_rows_at_the_minimum():
    """Edge: a cell exactly at the minimum is valid (>=, not >)."""
    table = [[5.0, 5.0], [5.0, 5.0]]
    assert _expected(table, 0, 0) == COCHRAN_MIN_EXPECTED
    assert drop_sparse_rows(table) == table


def test_empty_column_drops_everything():
    """Adversarial: no successes anywhere -> every expected success is 0;
    there is nothing to test, and nothing must divide by zero."""
    assert drop_sparse_rows([[0.0, 5.0], [0.0, 7.0]]) == []
    assert drop_sparse_rows([]) == []


def _fuzzer(counts: dict[str, int], success: dict[str, int]):
    return SimpleNamespace(op_counts=counts, op_success=success)


def test_regression_single_lucky_op_not_significant(caplog):
    """Falsification: identical 1% operators plus one 1-for-1 operator."""
    f = _fuzzer({"a": 1000, "b": 1000, "lucky": 1}, {"a": 10, "b": 10, "lucky": 1})
    with caplog.at_level(logging.DEBUG):
        Fuzzer._run_chi2_operator_test(f)
    assert SIGNIFICANT not in caplog.text


def test_real_difference_still_significant(caplog):
    """Control: the filter must not suppress a well-sampled difference."""
    f = _fuzzer({"a": 1000, "b": 1000}, {"a": 10, "b": 60})
    with caplog.at_level(logging.DEBUG):
        Fuzzer._run_chi2_operator_test(f)
    assert SIGNIFICANT in caplog.text


@pytest.mark.parametrize("n", [1, 3])
def test_all_sparse_runs_no_test(caplog, n):
    """Adversarial: when no row survives, no chi2 line is logged at all."""
    f = _fuzzer({"a": n, "b": n}, {"a": 1, "b": 0})
    with caplog.at_level(logging.DEBUG):
        Fuzzer._run_chi2_operator_test(f)
    assert "op heterogeneity" not in caplog.text


def test_regression_filter_recomputes_marginals():
    """Dropping rows shifts the marginals: rows valid against the full table
    can fall below the minimum against the survivors. Filter to a fixpoint."""
    table = [[0.0, 100.0], [1.0, 999.0], [9.0, 9991.0]] + [[1.0, 0.0]] * 600
    kept = drop_sparse_rows(table)
    for r in range(len(kept)):
        for c in range(2):
            assert _expected(kept, r, c) >= COCHRAN_MIN_EXPECTED
