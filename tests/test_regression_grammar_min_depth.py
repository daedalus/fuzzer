"""Regression: depth exhaustion wrote ``b"?"`` into generated inputs.

A recursive grammar (expressions, JSON) ran out of ``max_depth`` and
``_expand_rule`` substituted ``b"?"``: ~15 per output on the arithmetic
grammar of Gopinath & Zeller, *Building Fast Fuzzers* (arXiv 1911.07707).
Fix (paper §4): precompute each rule's minimum depth and pick only
alternatives that can still terminate within the depth left.
"""

from __future__ import annotations

import logging

import pytest

from fuzzer_tool.core.grammar import Grammar
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

EXPR = """start = expr
expr = term "+" expr | term "-" expr | term
term = factor "*" term | factor "/" term | factor
factor = "+" factor | "-" factor | "(" expr ")" | integer "." integer | integer
integer = digit integer | digit
digit = "0"|"1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
"""
EXPR_ALPHABET = set(b"0123456789+-*/().")
SEEDS = range(200)
GRAMMAR_FILES = ["dictionaries/png.gram", "dictionaries/der.gram"]


def _grammar(spec: str, seed: int = 0) -> Grammar:
    g = Grammar(seed=seed)
    g.parse(spec)
    return g


def _balanced(data: bytes) -> bool:
    level = 0
    for ch in data:
        level += (ch == ord("(")) - (ch == ord(")"))
        if level < 0:
            return False
    return level == 0


def _naive(g: Grammar, rng: RandPool, name: str, depth: int) -> bytes:
    """Pre-fix expansion, written independently: uniform alt, uniform count."""
    if depth <= 0 or name not in g.rules:
        return b"?"
    out = b""
    for token in rng.choice(g.rules[name]):
        if token[0] == "lit":
            out += token[1]
        elif token[0] == "ref":
            out += _naive(g, rng, token[1], depth - 1)
        else:
            _, sub, lo, hi = token
            for _ in range(rng.randint(lo, hi)):
                out += _naive(g, rng, sub, depth - 1)
    return out


@pytest.fixture
def grammar_log(caplog):
    caplog.set_level(logging.WARNING, logger="fuzzer_tool.core.grammar")
    return caplog


class TestNoPlaceholder:
    @pytest.mark.parametrize("boltzmann", [False, True])
    def test_regression_grammar_min_depth_expr(self, boltzmann, grammar_log):
        """Every seed yields a well-formed expression, no warning."""
        g = _grammar(EXPR)
        for seed in SEEDS:
            g._rng = RandPool(seed=seed)
            out = g.generate(boltzmann=boltzmann)
            assert set(out) <= EXPR_ALPHABET, (seed, out)
            assert _balanced(out), (seed, out)
        assert not grammar_log.records

    @pytest.mark.parametrize("depth", [3, 4, 6])
    def test_exact_min_depth_fits(self, depth, grammar_log):
        """A chain needing exactly ``depth`` levels fits ``max_depth=depth``."""
        rules = [f"r{i} = r{i + 1}" for i in range(depth - 1)]
        g = _grammar("\n".join([*rules, f'r{depth - 1} = "x"']))
        assert g.generate(max_depth=depth) == b"x"
        assert not grammar_log.records

    def test_recursion_forced_to_exit(self):
        """At depth 1 only the terminal alternative is affordable."""
        g = _grammar('a = "(" a ")" | "x"')
        assert g.generate(max_depth=1, rng=ScriptedRng(choice_idxs=[0])) == b"x"
        assert g.generate(max_depth=2, rng=ScriptedRng(choice_idxs=[0, 0])) == b"(x)"

    def test_optional_repeat_dropped(self):
        """``deep*`` that cannot fit repeats zero times."""
        g = _grammar('start = "<" deep* ">"\ndeep = d1\nd1 = "y"')
        assert g.generate(max_depth=2) == b"<>"

    def test_mandatory_repeat_alt_skipped(self):
        """``deep+`` that cannot fit rules out its alternative."""
        g = _grammar('start = deep+ | "z"\ndeep = d1\nd1 = "y"')
        assert g.generate(max_depth=2, rng=ScriptedRng(choice_idxs=[0])) == b"z"

    def test_tree_regen_clean(self, grammar_log):
        """TreeMutator subtree regeneration inherits the fix."""
        g = _grammar(EXPR)
        for seed in SEEDS:
            out = g.generate("factor", rng=RandPool(seed=seed))
            assert set(out) <= EXPR_ALPHABET, (seed, out)
        assert not grammar_log.records


class TestFalsification:
    @pytest.mark.parametrize("path", GRAMMAR_FILES)
    def test_unchanged_when_depth_plentiful(self, path):
        """Grammars that never hit the limit keep the same output and draws."""
        g = Grammar()
        g.parse_file(path)
        for seed in range(50):
            ref = _naive(g, RandPool(seed=seed), next(iter(g.rules)), g.max_depth)
            ref_again = _naive(g, RandPool(seed=seed), next(iter(g.rules)), g.max_depth)
            assert ref == ref_again  # control: the oracle agrees with itself
            assert b"?" not in ref, (path, seed)
            assert g.generate(rng=RandPool(seed=seed)) == ref, (path, seed)


class TestAdversarial:
    def test_cycle_without_exit_still_bounded(self, grammar_log):
        """``a = "x" a`` cannot terminate: falls back to ``?`` and warns."""
        g = _grammar('a = "x" a')
        out = g.generate(max_depth=5)
        assert out == b"xxxxx?"
        assert any("depth exhausted" in r.getMessage() for r in grammar_log.records)

    def test_chain_deeper_than_limit(self, grammar_log):
        """Min depth 12 under ``max_depth=10``: no viable alt, old fallback."""
        rules = [f"r{i} = r{i + 1}" for i in range(11)]
        g = _grammar("\n".join([*rules, 'r11 = "x"']))
        assert g.generate(max_depth=10) == b"?"
        assert grammar_log.records

    def test_unknown_rule_still_warns(self, grammar_log):
        """Undefined refs are grammar bugs: not silently routed around."""
        g = _grammar('start = missing | "ok"')
        assert g.generate(rng=ScriptedRng(choice_idxs=[0])) == b"?"
        assert any("unknown rule" in r.getMessage() for r in grammar_log.records)

    def test_merge_recomputes_depths(self):
        """A rule gaining a terminal alt via merge() becomes affordable."""
        g = _grammar('start = "<" a ">" | "-"\na = "(" a ")"')
        assert g.generate(max_depth=3, rng=ScriptedRng(choice_idxs=[0])) == b"-"
        g.merge(_grammar('a = "x"'))
        assert g.generate(max_depth=2, rng=ScriptedRng(choice_idxs=[0, 0])) == b"<x>"
