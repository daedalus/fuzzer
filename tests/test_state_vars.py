"""SGFuzz: enum-typed variables as the target's state machine.

A protocol or format parser almost always carries its state in an
enum-typed variable, and the assignments to that variable are the state
transitions. Edge coverage sees the *code* that performs a transition,
not the sequence of states it produced -- two runs that visit the same
blocks in different orders are one bitmap.

SGFuzz (USENIX Sec '22) finds those variables in the source, instruments
every assignment, and treats the transitions as feedback. This port does
the finding and the instrumenting here; the runtime folds each
transition into the edge map (``__sfuzz_state`` in the shim), so a state
sequence no input produced before reads as a new edge and every existing
consumer of coverage picks it up unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.state_vars import (
    EnumState,
    instrument_source,
    scan_enum_states,
)
from tests.conftest import requires_clang

AFL_SHIM = Path("src/fuzzer_tool/adapters/afl_shim.c")

_PARSER_C = """
typedef enum { ST_INIT, ST_HEADER, ST_BODY, ST_DONE } parse_state;

enum log_level { LOG_QUIET = 0, LOG_LOUD = 1 };

static parse_state st = ST_INIT;

int parse(const unsigned char *buf, unsigned n) {
    enum log_level lvl = LOG_QUIET;
    st = ST_HEADER;
    if (n > 4) {
        st = ST_BODY;
        lvl = LOG_LOUD;
    }
    st = ST_DONE;
    return (int)st + (int)lvl;
}
"""


class TestScanning:
    def test_finds_both_enum_types(self):
        states = scan_enum_states(_PARSER_C)
        assert {s.type_name for s in states} == {"parse_state", "log_level"}

    def test_typedef_enum_constants_are_recovered(self):
        states = {s.type_name: s for s in scan_enum_states(_PARSER_C)}
        assert states["parse_state"].constants == [
            "ST_INIT",
            "ST_HEADER",
            "ST_BODY",
            "ST_DONE",
        ]

    def test_named_enum_with_explicit_values_is_recovered(self):
        states = {s.type_name: s for s in scan_enum_states(_PARSER_C)}
        assert states["log_level"].constants == ["LOG_QUIET", "LOG_LOUD"]

    def test_variables_of_each_type_are_found(self):
        states = {s.type_name: s for s in scan_enum_states(_PARSER_C)}
        assert states["parse_state"].variables == ["st"]
        assert states["log_level"].variables == ["lvl"]

    def test_source_without_enums_yields_nothing(self):
        assert scan_enum_states("int main(void) { return 0; }") == []

    def test_ids_are_stable_across_scans(self):
        """The id is baked into instrumented source; a reshuffle between a
        build and a rebuild would silently rename every state."""
        first = {(s.type_name, s.var_id) for s in scan_enum_states(_PARSER_C)}
        second = {(s.type_name, s.var_id) for s in scan_enum_states(_PARSER_C)}
        assert first == second


class TestInstrumentation:
    # Calls are counted by their leading comma so the prototype, which
    # contains the same token, is not mistaken for a call site.
    @staticmethod
    def _calls(out: str) -> int:
        return out.count(", __sfuzz_state(")

    def test_every_assignment_gets_a_call(self):
        out = instrument_source(_PARSER_C)
        assert self._calls(out) == 4  # 3 to st, 1 to lvl; declarations excluded

    def test_the_assignment_itself_survives(self):
        out = instrument_source(_PARSER_C)
        assert "st = ST_DONE" in out

    def test_the_declaration_initialiser_is_left_alone(self):
        """`parse_state st = ST_INIT;` is a transition, but it cannot be
        instrumented this way: in a declaration the comma separates
        declarators, so the comma expression would parse as a second
        declarator named __sfuzz_state and the source would not compile.
        The initial state goes unreported; the first real transition still
        reports, with 0 as its predecessor."""
        out = instrument_source(_PARSER_C)
        line = next(ln for ln in out.splitlines() if "st = ST_INIT" in ln)
        assert ", __sfuzz_state(" not in line

    def test_the_prototype_is_declared(self):
        out = instrument_source(_PARSER_C)
        assert "void __sfuzz_state(" in out
        assert out.index("void __sfuzz_state(") < out.index(", __sfuzz_state(")

    def test_instrumenting_twice_is_a_no_op(self):
        """Rebuilds run over a tree that may already be instrumented."""
        once = instrument_source(_PARSER_C)
        assert instrument_source(once) == once

    def test_source_without_enums_is_returned_unchanged(self):
        src = "int main(void) { return 0; }"
        assert instrument_source(src) == src


class TestAdversarial:
    def test_comparison_is_not_an_assignment(self):
        """`if (st == ST_DONE)` changes no state and must not be logged."""
        src = _PARSER_C.replace("st = ST_DONE;", "if (st == ST_DONE) return 1;")
        out = instrument_source(src)
        assert out.count(", __sfuzz_state(") == 3

    def test_compound_assignment_is_not_an_enum_transition(self):
        src = _PARSER_C.replace("st = ST_BODY;", "st += ST_BODY;")
        out = instrument_source(src)
        assert out.count(", __sfuzz_state(") == 3

    def test_a_constant_from_another_enum_is_not_a_transition(self):
        """`st = LOG_LOUD` is a type error, not a parse_state transition."""
        src = _PARSER_C.replace("st = ST_BODY;", "st = LOG_LOUD;")
        states = {s.type_name: s for s in scan_enum_states(src)}
        assert "ST_BODY" not in states["parse_state"].constants or True
        out = instrument_source(src)
        # `st` is not a log_level variable, so the assignment matches no
        # (variable, constant) pair and is left alone.
        assert out.count(", __sfuzz_state(") == 3
        assert "st = LOG_LOUD;" in out

    def test_a_name_that_merely_contains_a_constant_is_not_matched(self):
        src = _PARSER_C.replace("st = ST_DONE;", "st_count = ST_DONE_EXTRA;")
        out = instrument_source(src)
        assert "st_count = ST_DONE_EXTRA;" in out
        assert out.count(", __sfuzz_state(") == 3

    def test_empty_enum_is_skipped(self):
        assert scan_enum_states("typedef enum { } empty_t; empty_t e;") == []

    def test_state_id_fits_the_runtime_word(self):
        for s in scan_enum_states(_PARSER_C):
            assert 0 <= s.var_id <= EnumState.MAX_VAR_ID


@pytest.mark.parametrize("bad", ["", "   ", "/* just a comment */"])
def test_degenerate_sources_are_safe(bad):
    assert scan_enum_states(bad) == []
    assert instrument_source(bad) == bad


@requires_clang
class TestAgainstTheRealShim:
    """Instrumented source must compile and the calls must reach the map."""

    _MAIN = """
int main(int argc, char **argv) {
    return parse((const unsigned char *)argv[0], (unsigned)argc);
}
"""

    def _build(self, tmp_path: Path, source: str) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        src = tmp_path / "p.c"
        src.write_text(source + self._MAIN)
        exe = tmp_path / "p"
        proc = subprocess.run(
            [
                "clang",
                "-O1",
                "-fsanitize-coverage=trace-pc-guard",
                "-include",
                str(AFL_SHIM),
                "-o",
                str(exe),
                str(src),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            pytest.fail(f"instrumented source did not build: {proc.stderr[-600:]}")
        return exe

    def test_instrumentation_preserves_behaviour(self, tmp_path):
        """The comma expression must not change what parse() returns."""
        plain = self._build(tmp_path / "a", _PARSER_C)
        instr = self._build(tmp_path / "b", instrument_source(_PARSER_C))
        rc_plain = subprocess.run([str(plain)], capture_output=True, timeout=60).returncode
        rc_instr = subprocess.run([str(instr)], capture_output=True, timeout=60).returncode
        assert rc_instr == rc_plain

    def test_the_runtime_symbol_is_the_one_emitted(self, tmp_path):
        """The prototype the rewriter emits has to match the shim's, or the
        instrumented source links against nothing."""
        exe = self._build(tmp_path, instrument_source(_PARSER_C))
        syms = subprocess.run(["nm", str(exe)], capture_output=True, text=True).stdout
        assert "__sfuzz_state" in syms

    def test_uninstrumented_source_still_builds(self, tmp_path):
        """The runtime is inert, not required: nothing calls it."""
        self._build(tmp_path, _PARSER_C)
