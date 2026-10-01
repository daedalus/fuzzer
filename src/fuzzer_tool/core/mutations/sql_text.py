"""Token-level SQL mutations for ``sqlite_read.c``'s SQL-text path.

``sqlite_chunk_mutate`` edits database images; any other input is fed to
the SQL parser and bytecode engine with only flat-byte mutation behind it.
This module lexes SQL (tolerant regex, like ``json_struct``) and applies one
edit:

    mode           example                          reaches
    -------------  -------------------------------  -------------------------
    keyword_swap   SELECT a FROM t -> ... WHERE t   parser error recovery
    literal_edge   SELECT 1 -> SELECT zeroblob(..)  affinity / overflow paths
    subquery_wrap  SELECT 1 -> SELECT (SELECT 1)    nested query planner
    clause_insert  SELECT 1 -> SELECT 1 LIMIT -1    clause code generation
    cte_prepend    SELECT 1 -> WITH RECURSIVE ...   CTE / recursion engine
    stmt_dup       SELECT 1 -> SELECT 1;SELECT 1    multi-statement prepare

Returns None to decline: no candidate token, or the result exceeds max_len.
"""

import itertools
import re
from enum import IntEnum

_TOKEN = re.compile(
    rb"(?P<s>'(?:[^']+|'')*'?)"
    rb"|(?P<n>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    rb"|(?P<w>[A-Za-z_][A-Za-z0-9_]*)"
    rb"|(?P<p>[;(),*=<>+\-/|.])",
)
_MAX_TOKENS = 4096
_SEMI = b";"[0]


class Kind(IntEnum):
    STR = 0
    NUM = 1
    WORD = 2
    PUNCT = 3


_GROUP_KIND = {"s": Kind.STR, "n": Kind.NUM, "w": Kind.WORD, "p": Kind.PUNCT}

SQL_KEYWORDS = (
    b"SELECT",
    b"FROM",
    b"WHERE",
    b"INSERT",
    b"INTO",
    b"VALUES",
    b"UPDATE",
    b"SET",
    b"DELETE",
    b"CREATE",
    b"TABLE",
    b"INDEX",
    b"VIEW",
    b"TRIGGER",
    b"DROP",
    b"ALTER",
    b"JOIN",
    b"LEFT",
    b"NATURAL",
    b"UNION",
    b"EXCEPT",
    b"INTERSECT",
    b"ORDER",
    b"GROUP",
    b"HAVING",
    b"LIMIT",
    b"OFFSET",
    b"DISTINCT",
    b"AS",
    b"ON",
    b"USING",
    b"NOT",
    b"NULL",
    b"IS",
    b"IN",
    b"LIKE",
    b"GLOB",
    b"MATCH",
    b"BETWEEN",
    b"CASE",
    b"WHEN",
    b"THEN",
    b"ELSE",
    b"END",
    b"CAST",
    b"COLLATE",
    b"PRAGMA",
    b"VIRTUAL",
    b"WITHOUT",
    b"ROWID",
    b"RECURSIVE",
    b"OVER",
    b"PARTITION",
    b"WINDOW",
    b"FILTER",
    b"RETURNING",
    b"CONFLICT",
    b"REPLACE",
    b"ESCAPE",
    b"EXISTS",
)
_KEYWORD_SET = frozenset(SQL_KEYWORDS)

# Integer limits, affinity traps, blob literals and allocator-sized calls.
LITERAL_EDGES = (
    b"9223372036854775807",
    b"-9223372036854775808",
    b"9223372036854775808",
    b"1e999",
    b"-0.0",
    b"x''",
    b"x'00ff'",
    b"X'0'",
    b"NULL",
    b"''",
    b"'\x00'",
    b"zeroblob(2147483647)",
    b"randomblob(-1)",
    b"char(0)",
    b"printf('%.*c',2147483647,'x')",
    b"CAST(1 AS BLOB)",
    b"(SELECT NULL)",
)

CLAUSES = (
    b" ORDER BY 1",
    b" GROUP BY 1 HAVING 1",
    b" LIMIT -1",
    b" LIMIT 1 OFFSET 9223372036854775807",
    b" COLLATE NOCASE",
    b" UNION ALL SELECT 1",
    b" WINDOW w AS (ORDER BY 1)",
    b" RETURNING *",
    b" ESCAPE '\\'",
)

CTE_PREFIX = b"WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c LIMIT 100) "


class _Tok:
    __slots__ = ("kind", "start", "end")

    def __init__(self, kind: Kind, start: int, end: int):
        self.kind = kind
        self.start = start
        self.end = end


def _lex(data: bytes) -> list[_Tok]:
    return [
        _Tok(_GROUP_KIND[m.lastgroup], m.start(), m.end())
        for m in itertools.islice(_TOKEN.finditer(data), _MAX_TOKENS)
    ]


def _is_keyword(data: bytes, t: _Tok) -> bool:
    return t.kind == Kind.WORD and data[t.start : t.end].upper() in _KEYWORD_SET


def _semis(data: bytes, toks: list[_Tok]) -> list[_Tok]:
    return [t for t in toks if t.kind == Kind.PUNCT and data[t.start] == _SEMI]


def _replace(data: bytes, t: _Tok, repl: bytes) -> bytes:
    return data[: t.start] + repl + data[t.end :]


def _insert(data: bytes, pos: int, repl: bytes) -> bytes:
    return data[:pos] + repl + data[pos:]


# ── modes: (data, toks, rng) -> bytes | None ───────────────────────────


def keyword_swap(data, toks, rng):
    cands = [t for t in toks if _is_keyword(data, t)]
    if not cands:
        return None
    return _replace(data, rng.choice(cands), rng.choice(SQL_KEYWORDS))


def literal_edge(data, toks, rng):
    cands = [t for t in toks if t.kind in (Kind.STR, Kind.NUM)]
    if not cands:
        return None
    return _replace(data, rng.choice(cands), rng.choice(LITERAL_EDGES))


def subquery_wrap(data, toks, rng):
    """Wrap a literal or identifier: ``1`` -> ``(SELECT 1)``."""
    cands = [
        t
        for t in toks
        if t.kind in (Kind.STR, Kind.NUM) or (t.kind == Kind.WORD and not _is_keyword(data, t))
    ]
    if not cands:
        return None

    t = rng.choice(cands)
    return _replace(data, t, b"(SELECT " + data[t.start : t.end] + b")")


def clause_insert(data, toks, rng):
    """Append a clause at a statement end: before a ';' or at EOF."""
    cands = [t.start for t in _semis(data, toks)] + [len(data)]
    return _insert(data, rng.choice(cands), rng.choice(CLAUSES))


def cte_prepend(data, toks, rng):
    """Prefix a recursive CTE at a statement start."""
    cands = [0] + [t.end for t in _semis(data, toks)]
    return _insert(data, rng.choice(cands), CTE_PREFIX)


def stmt_dup(data, toks, rng):
    """Duplicate one ';'-terminated statement in place."""
    bounds = [0] + [t.end for t in _semis(data, toks)]
    if bounds[-1] != len(data):
        bounds.append(len(data))

    k = rng.choice(range(len(bounds) - 1))
    stmt = data[bounds[k] : bounds[k + 1]]
    if not stmt.endswith(b";"):
        return data + b";" + stmt
    return _insert(data, bounds[k + 1], stmt)


MODES = (keyword_swap, literal_edge, subquery_wrap, clause_insert, cte_prepend, stmt_dup)


def sql_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one token-level SQL edit; None when there is nothing to do."""
    toks = _lex(data)
    if not toks:
        return None

    mode = rng.choice(MODES)
    out = mode(data, toks, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
