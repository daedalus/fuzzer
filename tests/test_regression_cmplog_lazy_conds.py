"""cmplog drains ingest plain tuples; CondStmt objects are built on demand.

``_parse_lines`` built a CondStmt (two dataclasses) for every unique CMP
line, ~10k per FFmpeg drain (6.8 s of 23 s over 151 drains), while the
ingest only reads five fields. Only the PRNG-state learner reads
``last_conds``, so it is materialised lazily from the tuples.
"""

import pytest

from fuzzer_tool.adapters.track_parser import cond_tuples_from_cmplog_text, conds_from_cmplog_text
from fuzzer_tool.core import cond_stmt
from fuzzer_tool.core.cmplog import CmplogCollector

_LINES = [
    "CMP 41424344 61626364 1 4 401000",
    "CMP 41424344 61626364 1 4 401000",  # exact repeat
    "CMP 0102 0304 0 2",  # no pc
    "CMP 0102 0304 0 2 402000",  # same operands, other pc
    "CMP 41424344 61626364 0 4 401000",  # same pair, other result
    "OP 4 1234abcd",  # non-CMP record
    "CMP zz 01 0 1",  # malformed hex
    "CMP 01",  # too short
    "CMP  ",  # empty operands
    "  CMP deadbeef cafebabe 1 4 403000  ",  # surrounding whitespace
]


def _ref_ingest(conds):
    """Oracle: the pre-change _ingest_conds, over CondStmt objects, into fresh maps."""
    pair_set, new_pairs, pc_map, cmp_map, tokens = set(), [], {}, {}, {}
    for c in conds:
        pair = (c.base.op_a, c.base.op_b)
        if pair not in pair_set:
            pair_set.add(pair)
            new_pairs.append(pair)
            if c.base.pc is not None:
                pc_map[pair] = c.base.pc
            if c.base.result is not None and c.base.width is not None:
                cmp_map[pair] = (c.base.result, c.base.width)
        tokens[c.base.op_a] = None
        tokens[c.base.op_b] = None
    return new_pairs, pc_map, cmp_map, list(tokens)


def test_control_parser_matches_itself():
    """Rule 46: the CondStmt oracle is deterministic on identical input."""
    a = [x.base for x in conds_from_cmplog_text(_LINES)]
    b = [x.base for x in conds_from_cmplog_text(_LINES)]
    assert [(x.op_a, x.op_b, x.width, x.result, x.pc) for x in a] == [
        (x.op_a, x.op_b, x.width, x.result, x.pc) for x in b
    ]


def test_tuples_match_condstmt_fields():
    """Falsification: tuples carry exactly the fields, order and dedup of the
    CondStmt parser (adversarial lines included)."""
    ref = [
        (c.base.op_a, c.base.op_b, c.base.width, c.base.result, c.base.pc)
        for c in conds_from_cmplog_text(_LINES)
    ]
    assert cond_tuples_from_cmplog_text(_LINES) == ref


def test_drain_builds_no_condstmt(monkeypatch):
    """Falsification: the hot path constructs zero CondStmt objects."""
    calls = {"n": 0}
    real = cond_stmt.CondStmt.from_cmplog_pair.__func__

    def counting(cls, *a, **k):
        calls["n"] += 1
        return real(cls, *a, **k)

    monkeypatch.setattr(cond_stmt.CondStmt, "from_cmplog_pair", classmethod(counting))
    c = CmplogCollector()
    c._parse_lines(_LINES)
    assert calls["n"] == 0

    conds = c.last_conds  # materialised on demand
    assert calls["n"] == len(conds) > 0
    assert c.last_conds is conds  # cached for the drain


def test_lazy_conds_equal_eager_list():
    """Falsification: the on-demand list equals the old eager one."""
    c = CmplogCollector()
    c._parse_lines(_LINES)
    want = conds_from_cmplog_text(_LINES)
    got = c.last_conds
    assert [
        (x.base.cmpid, x.base.op_a, x.base.op_b, x.base.width, x.base.result, x.base.pc)
        for x in got
    ] == [
        (x.base.cmpid, x.base.op_a, x.base.op_b, x.base.width, x.base.result, x.base.pc)
        for x in want
    ]


def test_collector_state_unchanged():
    """Falsification: pairs, pc/cmp maps and tokens equal the CondStmt-driven
    ingest (oracle replays the old loop over conds_from_cmplog_text)."""
    c = CmplogCollector()
    c._parse_lines(_LINES)
    new_pairs, pc_map, cmp_map, tokens = _ref_ingest(conds_from_cmplog_text(_LINES))
    assert c.pairs[: len(new_pairs)] == new_pairs
    assert {p: c._pair_pc[p] for p in pc_map} == pc_map
    assert {p: c._pair_cmp[p] for p in cmp_map} == cmp_map
    assert c.tokens[: len(tokens)] == tokens


def test_next_drain_invalidates():
    """Adversarial: a new drain must not serve the previous drain's conds."""
    c = CmplogCollector()
    c._parse_lines(_LINES)
    first = c.last_conds
    c._parse_lines(["CMP aabbccdd 11223344 1 4 404000"])
    assert c.last_conds is not first
    assert [x.base.op_a for x in c.last_conds] == [bytes.fromhex("aabbccdd")]


@pytest.mark.parametrize("value", [[], ["sentinel"]])
def test_assignment_still_works(value):
    """Adversarial: callers (and tests) that assign last_conds keep working."""
    c = CmplogCollector()
    c._parse_lines(_LINES)
    c.last_conds = value
    assert c.last_conds == value


_OPS = ["DIV 00010000 4 0x405000", "GEP 7f000000 4 0x406000"]


def _full_state(c):
    return (
        list(c.pairs),
        dict(c._pair_pc),
        dict(c._pair_cmp),
        list(c.tokens),
        list(c.last_pairs),
        dict(c._pair_occurrence),
        list(c._pending_redqueen),
    )


@pytest.mark.parametrize("repeat", [2, 50])
def test_repeated_lines_same_state(repeat):
    """Adversarial: a drain of k copies of each line (the FFmpeg shape) must
    leave exactly the state of one copy -- including DIV/GEP operand pairs."""
    base = _LINES + _OPS
    once, many = CmplogCollector(), CmplogCollector()
    once._parse_lines(list(base))
    many._parse_lines([ln for ln in base for _ in range(repeat)])
    assert _full_state(many) == _full_state(once)


def test_parsers_see_each_line_once(monkeypatch):
    """Falsification: both line parsers get the deduplicated drain."""
    from fuzzer_tool.core import cmplog as cmplog_mod

    seen = []
    real_c, real_p = cmplog_mod.cond_tuples_from_cmplog_text, cmplog_mod.pairs_from_operand_records
    monkeypatch.setattr(
        cmplog_mod, "cond_tuples_from_cmplog_text", lambda ls: (seen.append(len(ls)), real_c(ls))[1]
    )
    monkeypatch.setattr(
        cmplog_mod, "pairs_from_operand_records", lambda ls: (seen.append(len(ls)), real_p(ls))[1]
    )
    base = _LINES + _OPS
    CmplogCollector()._parse_lines(base * 20)
    assert seen == [len(dict.fromkeys(base))] * 2
