"""Regression: an escaped quote inside a quoted grammar literal.

The tokenizer ended a quoted literal at the first ``"``, so the shipped json
grammar's ``string = "\\"" text "\\""`` split into ``b'\\\\'``, ``b' text '``
and ``b''``: every generated JSON string was the literal bytes ``\\ text ``.
"""

from fuzzer_tool.core.grammar import GRAMMARS, Grammar, load_grammar
from fuzzer_tool.core.rand_pool import RandPool

QUOTE = b'"'


def _tokens(spec: str) -> list:
    g = Grammar()
    g.parse(spec)
    return g.rules


def test_regression_json_string_rule_tokens():
    rules = _tokens(GRAMMARS["json"])
    assert rules["string"] == [[("lit", QUOTE), ("ref", "text"), ("lit", QUOTE)]]


def test_regression_json_string_generates_quoted_text():
    g = load_grammar("json")
    g._rng = RandPool(seed=7)
    for _ in range(50):
        out = g.generate(rule="string")
        assert out.startswith(QUOTE) and out.endswith(QUOTE) and len(out) >= 2
        assert b"text" not in out and b"\\" not in out


def test_escaped_backslash_still_closes_the_literal():
    # "a\\" is `a` + backslash; the following quote closes it, `b` is a ref.
    rules = _tokens('r = "a\\\\" b')
    assert rules["r"] == [[("lit", b"a\\"), ("ref", "b")]]


def test_escaped_quote_in_single_quotes_and_bare_escape_after():
    rules = _tokens("r = 'it\\'s' \\x20 \"q\\\"\"")
    assert rules["r"] == [[("lit", b"it's"), ("lit", b" "), ("lit", b'q"')]]
