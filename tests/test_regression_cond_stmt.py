"""Regression tests for ``fuzzer_tool.core.cond_stmt`` and the track parser."""

from __future__ import annotations

from fuzzer_tool.adapters.track_parser import (
    conds_from_cmplog_text,
    iter_track_lines,
    parse_track_file,
    parse_track_json,
    parse_track_line,
)
from fuzzer_tool.core.cond_stmt import (
    CondState,
    CondStmt,
    CondStmtBase,
    conds_from_cmplog_pairs,
    filter_cond_list,
)

# ---------------------------------------------------------------------------
# CondStmtBase
# ---------------------------------------------------------------------------


class TestCondStmtBase:
    def test_key_identity(self):
        base = CondStmtBase(cmpid=1, op_a=b"ab", op_b=b"cd", width=2, result=1, pc=0x1000)
        assert base.key == (1, b"ab", b"cd", 2)

    def test_str_contains_pc(self):
        base = CondStmtBase(cmpid=1, op_a=b"ab", op_b=b"cd", width=2, result=1, pc=0x1000)
        s = str(base)
        assert "id=1" in s
        assert "pc=0x1000" in s

    def test_str_without_pc(self):
        base = CondStmtBase(cmpid=2, op_a=b"a", op_b=b"b", width=1, result=0, pc=None)
        s = str(base)
        assert "pc=" not in s


# ---------------------------------------------------------------------------
# CondStmt constructors
# ---------------------------------------------------------------------------


class TestCondStmt:
    def test_from_cmplog_pair_defaults(self):
        c = CondStmt.from_cmplog_pair(1, b"AB", b"CD", width=2)
        assert c.base.cmpid == 1
        assert c.base.width == 2
        assert c.state is CondState.UNSOLVED
        assert c.offsets == ()

    def test_from_cmplog_pair_with_meta(self):
        c = CondStmt.from_cmplog_pair(5, b"\x00", b"\xff", width=1, result=-1, pc=0x400)
        assert c.base.result == -1
        assert c.base.pc == 0x400

    def test_from_track_record_valid(self):
        c = CondStmt.from_track_record(
            {
                "cmpid": 10,
                "arg1": [65, 66],
                "arg2": [67, 68],
                "size": 2,
                "condition": 1,
                "pc": 0x500,
                "offsets": [3, 4],
                "speed": 7,
                "is_desirable": True,
                "is_consistent": False,
                "state": "solved",
                "linear": True,
                "fuzz_times": 2,
                "num_minimal_optima": 1,
            }
        )
        assert c is not None
        assert c.base.cmpid == 10
        assert c.offsets == (3, 4)
        assert c.speed == 7
        assert c.state is CondState.SOLVED
        assert c.linear is True

    def test_from_track_record_malformed_returns_none(self):
        assert CondStmt.from_track_record({}) is None
        assert CondStmt.from_track_record({"cmpid": "bad"}) is None

    def test_update_from_input_finds_offsets(self):
        c = CondStmt.from_cmplog_pair(1, b"ab", b"cd", width=2)
        c.update_from_input(b"xxabxxabyy")
        assert c.offsets == (2, 3, 6, 7)

    def test_state_transitions(self):
        c = CondStmt.from_cmplog_pair(1, b"a", b"b", width=1)
        assert c.state is CondState.UNSOLVED
        c.mark_solved()
        assert c.state is CondState.SOLVED
        assert c.fuzz_times == 1
        c.mark_unsolvable()
        assert c.state is CondState.UNSOLVABLE
        assert c.fuzz_times == 2
        c.mark_timeout()
        assert c.state is CondState.TIMEOUT
        assert c.fuzz_times == 3

    def test_bump_minima(self):
        c = CondStmt.from_cmplog_pair(1, b"a", b"b", width=1)
        c.bump_minima()
        assert c.num_minimal_optima == 1


# ---------------------------------------------------------------------------
# filter_cond_list
# ---------------------------------------------------------------------------


def _make_cond(cmpid: int, state: CondState = CondState.UNSOLVED, speed: int = 0) -> CondStmt:
    c = CondStmt.from_cmplog_pair(cmpid, b"a", b"b", width=1)
    c.state = state
    c.speed = speed
    return c


class TestFilterCondList:
    def test_dedup_by_key(self):
        conds = [_make_cond(1), _make_cond(1), _make_cond(2)]
        out = filter_cond_list(conds)
        assert len(out) == 2
        assert {c.base.cmpid for c in out} == {1, 2}

    def test_drop_unsolvable(self):
        conds = [_make_cond(1, CondState.UNSOLVABLE), _make_cond(2, CondState.UNSOLVED)]
        out = filter_cond_list(conds, drop_unsolvable=True)
        assert len(out) == 1
        assert out[0].base.cmpid == 2

    def test_keep_unsolvable_when_disabled(self):
        conds = [_make_cond(1, CondState.UNSOLVABLE)]
        out = filter_cond_list(conds, drop_unsolvable=False)
        assert len(out) == 1

    def test_drop_timeout(self):
        conds = [_make_cond(1, CondState.TIMEOUT), _make_cond(2, CondState.UNSOLVED)]
        out = filter_cond_list(conds, drop_timeout=True)
        assert len(out) == 1
        assert out[0].base.cmpid == 2

    def test_drop_one_byte(self):
        conds = [_make_cond(1, CondState.ONE_BYTE), _make_cond(2, CondState.UNSOLVED)]
        out = filter_cond_list(conds, drop_one_byte=True)
        assert len(out) == 1
        assert out[0].base.cmpid == 2

    def test_max_speed_filter(self):
        conds = [_make_cond(1, speed=5), _make_cond(2, speed=15)]
        out = filter_cond_list(conds, max_speed=10)
        assert len(out) == 1
        assert out[0].base.cmpid == 2


# ---------------------------------------------------------------------------
# conds_from_cmplog_pairs
# ---------------------------------------------------------------------------


class TestCondsFromCmplogPairs:
    def test_basic_wrapping(self):
        pairs = [(b"AB", b"CD"), (b"EF", b"GH")]
        conds = conds_from_cmplog_pairs(pairs)
        assert len(conds) == 2
        assert conds[0].base.op_a == b"AB"
        assert conds[1].base.op_b == b"GH"

    def test_dedup(self):
        pairs = [(b"AB", b"CD"), (b"AB", b"CD")]
        conds = conds_from_cmplog_pairs(pairs)
        assert len(conds) == 1

    def test_with_meta(self):
        pairs = [(b"AB", b"CD")]
        pair_meta = {(b"AB", b"CD"): (-1, 2)}
        pair_pc = {(b"AB", b"CD"): 0x1000}
        conds = conds_from_cmplog_pairs(pairs, pair_meta=pair_meta, pair_pc=pair_pc)
        assert conds[0].base.result == -1
        assert conds[0].base.width == 2
        assert conds[0].base.pc == 0x1000

    def test_empty_input(self):
        assert conds_from_cmplog_pairs([]) == []


# ---------------------------------------------------------------------------
# Track parser
# ---------------------------------------------------------------------------


class TestTrackParser:
    def test_parse_track_line_minimal(self):
        c = parse_track_line("1 0 0 414243 444546 2 0")
        assert c is not None
        assert c.base.cmpid == 1
        assert c.base.op_a == b"ABC"
        assert c.base.op_b == b"DEF"
        assert c.base.width == 2
        assert c.base.result == 0

    def test_parse_track_line_with_pc(self):
        c = parse_track_line("2 0 0 00 01 1 -1 1000")
        assert c is not None
        assert c.base.result == -1
        assert c.base.pc == 0x1000

    def test_parse_track_line_malformed(self):
        assert parse_track_line("") is None
        assert parse_track_line("1 0 0") is None

    def test_parse_track_file_text(self, tmp_path):
        p = tmp_path / "track.txt"
        p.write_text("1 0 0 414243 444546 2 0\n2 0 0 00 01 1 -1 1000\n1 0 0 414243 444546 2 0\n")
        conds = parse_track_file(str(p))
        assert len(conds) == 2
        assert conds[0].base.cmpid == 1
        assert conds[1].base.pc == 0x1000

    def test_parse_track_file_text_missing(self, tmp_path):
        p = tmp_path / "missing.txt"
        assert parse_track_file(str(p)) == []

    def test_parse_track_json(self, tmp_path):
        p = tmp_path / "track.json"
        p.write_text('[{"cmpid":1,"arg1":[65,66],"arg2":[67,68],"size":2,"condition":1,"pc":4096}]')
        conds = parse_track_json(str(p))
        assert len(conds) == 1
        assert conds[0].base.pc == 4096
        assert conds[0].offsets == ()

    def test_parse_track_json_missing(self, tmp_path):
        assert parse_track_json(str(tmp_path / "nope.json")) == []

    def test_conds_from_cmplog_text(self):
        lines = [
            "CMP 414243 444546 0 2",
            "CMP 0001 ffff -1 1 1000",
        ]
        conds = conds_from_cmplog_text(lines)
        assert len(conds) == 2
        assert conds[0].base.op_a == b"ABC"
        assert conds[1].base.result == -1
        assert conds[1].base.pc == 0x1000

    def test_iter_track_lines(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("a\nb\n\nc\n")
        assert list(iter_track_lines(str(p))) == ["a", "b", "c"]
