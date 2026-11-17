"""Compiler-fuzzer source mutations (core/mutations/source_rules.py).

Port of AFL++ custom_mutators/compiler_fuzzer (afl-compiler-fuzzer, Groce et
al. CC'22): 64 mutation-testing rules over C-like text, applied at the first
match at/after a random position; 16 attempts.

Expected outputs are written out by hand from the rule definitions.
"""

import pytest

from fuzzer_tool.core.mutations.source_rules import (
    MAX_MUTANT_CHANGE,
    RULES,
    Rule,
    apply_rule,
    source_mutate,
)
from tests.support.scripted_rng import ScriptedRng

PROG = b"int f(int a, int b) {\n  if (a == b) return a + 1;\n  return 0;\n}\n"


def _rule(kind, *args):
    return next(r for r in RULES if r == Rule(kind, *args))


def test_rule_count_matches_upstream():
    assert len(RULES) == 64


def test_simple_replaces_first_match_at_or_after_pos():
    r = _rule("simple", b"==", b"!=")
    assert apply_rule(r, PROG, 0, len(PROG) + 8) == PROG.replace(b"==", b"!=", 1)


def test_simple_respects_position():
    """A match before pos is skipped: '+' at offset > pos only."""
    r = _rule("simple", b"+", b"-")
    data = b"a + b + c"
    assert apply_rule(r, data, 3, 64) == b"a + b - c"


def test_delim_rep_removes_statement_after_semicolon():
    r = _rule("rep", b";", b";", b";")
    data = b"x = 1; y = 2; z;"
    assert apply_rule(r, data, 0, 64) == b"x = 1; z;"


def test_delim_rep_empties_brace_block():
    r = _rule("rep", b"{", b"}", b"{}")
    assert apply_rule(r, b"if (a) { b(); c(); } d();", 0, 64) == b"if (a) {} d();"


def test_delim_swap_swaps_arguments():
    r = _rule("swap", b"(", b",", b")")
    assert apply_rule(r, b"f(a,b);", 0, 64) == b"f(b,a);"


def test_delim_dup_duplicates_line():
    r = _rule("dup", b"\n", b"\n")
    data = b"a;\nb;\nc;\n"
    assert apply_rule(r, data, 0, 64) == b"a;\nb;\n\nb;\nc;\n"


def test_no_match_returns_none():
    """Falsification: a rule whose anchor is absent changes nothing."""
    assert apply_rule(_rule("simple", b"&&", b"||"), b"a || b", 0, 64) is None


def test_span_bounded_to_max_change():
    """Adversarial: a closing delimiter past MAX_MUTANT_CHANGE is not used."""
    r = _rule("rep", b"{", b"}", b"")
    data = b"{" + b"x" * (MAX_MUTANT_CHANGE + 10) + b"}"
    assert apply_rule(r, data, 0, 1 << 20) is None


def test_max_len_refuses_growth():
    """Adversarial: a mutant longer than max_len is refused, not truncated."""
    r = _rule("dup", b"\n", b"\n")
    data = b"a;\nb;\n"
    assert apply_rule(r, data, 0, len(data)) is None


def test_empty_result_refused():
    r = _rule("rep", b"{", b"}", b"")
    assert apply_rule(r, b"{}", 0, 64) is None


def test_source_mutate_drives_rule_and_pos_from_rng():
    """Scripted draws: position then rule index; the first applicable try wins."""
    idx = RULES.index(_rule("simple", b"==", b"!="))
    rng = ScriptedRng(randints=[0, idx])
    assert source_mutate(PROG, rng, len(PROG) + 8) == PROG.replace(b"==", b"!=", 1)


def test_source_mutate_gives_up_after_tries():
    rng = ScriptedRng(randints=[0, RULES.index(_rule("simple", b"&&", b"||"))] * 16)
    assert source_mutate(b"no operators here", rng, 64) is None


@pytest.mark.parametrize("data", [b"", b"x"])
def test_tiny_input(data):
    assert source_mutate(data, ScriptedRng(randints=[0, 0] * 16), 64) is None


class TestOperator:
    def test_registered_and_text_gated(self):
        from types import SimpleNamespace

        from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY

        assert "c_source_mutate" in REGISTRY.categories()["structural"]
        gate = _AVAILABLE["c_source_mutate"]
        assert gate(SimpleNamespace(), PROG) is True
        assert gate(SimpleNamespace(), bytes(range(256))) is False

    def test_handler_mutates_program(self):
        from types import SimpleNamespace

        from fuzzer_tool.core.rand_pool import RandPool
        from fuzzer_tool.services.operators import OperatorEngine

        class _Engine(OperatorEngine):
            ctx = None

        eng = _Engine.__new__(_Engine)
        eng.ctx = SimpleNamespace(_rng=RandPool(seed=2), max_len=4096)
        outs = {bytes(eng._op_c_source_mutate(bytearray(PROG), 0, PROG)) for _ in range(20)}
        assert outs and all(o != PROG for o in outs)
