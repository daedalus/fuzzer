"""Tests for core/mutations/sql_text.sql_mutate (sqlite_read.c SQL-text path)."""

import time

import pytest

from fuzzer_tool.core.mutations.sql_text import (
    CLAUSES,
    CTE_PREFIX,
    LITERAL_EDGES,
    MODES,
    SQL_KEYWORDS,
    sql_mutate,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 65536


def _mode(name):
    return [m.__name__ for m in MODES].index(name)


def _run(data, *choice_idxs, max_len=_MAX):
    return sql_mutate(data, ScriptedRng(choice_idxs=choice_idxs), max_len)


def test_keyword_swap_is_case_insensitive():
    # Candidates: select(0), from(1). Swap FROM -> WHERE.
    out = _run(b"select a from t", _mode("keyword_swap"), 1, SQL_KEYWORDS.index(b"WHERE"))
    assert out == b"select a WHERE t"


def test_literal_edge_on_string():
    edge = LITERAL_EDGES.index(b"x'00ff'")
    assert _run(b"SELECT 'a'", _mode("literal_edge"), 0, edge) == b"SELECT x'00ff'"


def test_literal_edge_doubled_quote_is_one_token():
    # 'it''s' is a single literal; the only other literal is 2.
    edge = LITERAL_EDGES.index(b"NULL")
    out = _run(b"SELECT 'it''s', 2", _mode("literal_edge"), 1, edge)
    assert out == b"SELECT 'it''s', NULL"


def test_subquery_wrap():
    assert _run(b"SELECT 1", _mode("subquery_wrap"), 0) == b"SELECT (SELECT 1)"


def test_clause_insert_before_semicolon():
    clause = CLAUSES.index(b" LIMIT -1")
    assert _run(b"SELECT 1;", _mode("clause_insert"), 0, clause) == b"SELECT 1 LIMIT -1;"


def test_clause_insert_at_end():
    clause = CLAUSES.index(b" ORDER BY 1")
    # Candidates: before ';' (0), end of buffer (1).
    out = _run(b"SELECT 1;SELECT 2", _mode("clause_insert"), 1, clause)
    assert out == b"SELECT 1;SELECT 2 ORDER BY 1"


def test_cte_prepend_second_statement():
    out = _run(b"SELECT 1;SELECT 2", _mode("cte_prepend"), 1)
    assert out == b"SELECT 1;" + CTE_PREFIX + b"SELECT 2"


def test_stmt_dup():
    assert _run(b"SELECT 1;SELECT 2", _mode("stmt_dup"), 0) == b"SELECT 1;SELECT 1;SELECT 2"


def test_stmt_dup_without_semicolon_adds_one():
    assert _run(b"SELECT 1", _mode("stmt_dup"), 0) == b"SELECT 1;SELECT 1"


# Falsification: nothing to edit must decline.
def test_regression_no_literal_declines():
    assert _run(b"SELECT a FROM t", _mode("literal_edge")) is None


def test_no_tokens_declines():
    assert sql_mutate(b"", ScriptedRng(), _MAX) is None
    assert sql_mutate(b" \x00 ", ScriptedRng(), _MAX) is None


def test_over_budget_declines():
    assert _run(b"SELECT 1", _mode("cte_prepend"), 0, max_len=16) is None


def test_adversarial_unterminated_quote_run_is_fast():
    data = b"SELECT '" + b"''" * 30000
    t0 = time.perf_counter()
    for seed in range(20):
        out = sql_mutate(data, RandPool(seed=seed), _MAX)
        assert out is None or len(out) <= _MAX
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("max_len", [1, 2, 8, 256])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [b";" * 300, b"SELECT " * 60, bytes(range(256)), b"x'" + b"f" * 200]
    for seed in range(40):
        rng = RandPool(seed=seed)
        for blob in blobs:
            out = sql_mutate(blob[:max_len], rng, max_len)
            assert out is None or len(out) <= max_len
