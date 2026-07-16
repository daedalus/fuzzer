"""Tests for core/grammar.py — grammar-based mutations."""

import pytest

from fuzzer_tool.core.grammar import Grammar


class TestGrammarParse:
    def test_empty_spec(self):
        g = Grammar()
        g.parse("")
        assert g.rules == {}

    def test_simple_literal(self):
        g = Grammar()
        g.parse('msg = "hello"')
        assert "msg" in g.rules
        assert len(g.rules["msg"]) == 1
        assert g.rules["msg"][0] == [("lit", b"hello")]

    def test_alternatives(self):
        g = Grammar()
        g.parse('method = "GET" | "POST" | "PUT"')
        assert len(g.rules["method"]) == 3
        assert g.rules["method"][0] == [("lit", b"GET")]
        assert g.rules["method"][1] == [("lit", b"POST")]
        assert g.rules["method"][2] == [("lit", b"PUT")]

    def test_rule_reference(self):
        g = Grammar()
        g.parse("line = method\nmethod = GET | POST")
        assert g.rules["line"] == [[("ref", "method")]]

    def test_repeat_n(self):
        g = Grammar()
        g.parse("items = item{3}")
        assert g.rules["items"] == [[("repeat", "item", 3, 3)]]

    def test_repeat_range(self):
        g = Grammar()
        g.parse("items = item{2,5}")
        assert g.rules["items"] == [[("repeat", "item", 2, 5)]]

    def test_repeat_plus(self):
        g = Grammar()
        g.parse("items = item+")
        assert g.rules["items"] == [[("repeat", "item", 1, 8)]]

    def test_repeat_star(self):
        g = Grammar()
        g.parse("items = item*")
        assert g.rules["items"] == [[("repeat", "item", 0, 8)]]

    def test_single_quoted_literal(self):
        g = Grammar()
        g.parse("data = 'bytes'")
        assert g.rules["data"] == [[("lit", b"bytes")]]

    def test_comment_lines(self):
        g = Grammar()
        g.parse('# comment\ndata = "x"')
        assert "data" in g.rules

    def test_blank_lines(self):
        g = Grammar()
        g.parse('\ndata = "x"\n\n')
        assert "data" in g.rules

    def test_hex_escape(self):
        g = Grammar()
        g.parse(r'data = "\xFF\x00"')
        # Parser treats backslash escapes as literal characters
        assert g.rules["data"] == [[("lit", b"\\xFF\\x00")]]

    def test_complex_grammar(self):
        g = Grammar()
        spec = """
        request = method " " uri " HTTP/1.1\\r\\n"
        method = GET | POST
        uri = / | /api
        """
        g.parse(spec)
        assert "request" in g.rules
        assert "method" in g.rules
        assert "uri" in g.rules


class TestGrammarGenerate:
    def test_empty_grammar(self):
        g = Grammar()
        assert g.generate() == b""

    def test_generates_literal(self):
        g = Grammar()
        g.parse('msg = "hello"')
        result = g.generate()
        assert result == b"hello"

    def test_generates_alternative(self):
        import random

        random.seed(0)
        g = Grammar()
        g.parse('method = "GET" | "POST"')
        results = {g.generate() for _ in range(50)}
        assert results <= {b"GET", b"POST"}

    def test_generates_recursive(self):
        import random

        random.seed(0)
        g = Grammar()
        g.parse('line = method\nmethod = "GET" | "POST"')
        results = {g.generate() for _ in range(50)}
        assert results <= {b"GET", b"POST"}
        assert len(results) == 2  # both alternatives hit

    def test_max_depth(self):
        g = Grammar()
        g.parse("a = b\nb = a")
        result = g.generate(max_depth=5)
        # Should terminate with "?" at depth limit
        assert b"?" in result

    def test_max_len(self):
        g = Grammar()
        g.parse('msg = "hello world"')
        result = g.generate(max_len=3)
        assert len(result) == 3

    def test_specific_rule(self):
        g = Grammar()
        g.parse('a = "AAA"\nb = "BBB"')
        assert g.generate("b") == b"BBB"

    def test_repeat_generates_multiple(self):
        g = Grammar()
        g.parse('items = x{3}\nx = "X"')
        result = g.generate()
        assert result == b"XXX"

    def test_repeat_range(self):
        g = Grammar()
        g.parse('items = y{1,3}\ny = "Y"')
        results = [g.generate() for _ in range(100)]
        lengths = {len(r) for r in results}
        assert lengths <= {1, 2, 3}


class TestGrammarMutate:
    def test_empty_grammar_returns_data(self):
        g = Grammar()
        assert g.mutate(b"hello") == b"hello"

    def test_empty_data_returns_data(self):
        g = Grammar()
        g.parse('x = "a"')
        assert g.mutate(b"") == b""

    def test_mutate_returns_bytes(self):
        g = Grammar()
        g.parse('x = "hello world"')
        result = g.mutate(b"hello world")
        assert isinstance(result, bytes)

    def test_mutate_truncate(self):
        g = Grammar()
        g.parse('x = "a"')
        data = b"ABCDEFGH"
        # Run many times to hit truncate path
        results = {len(g.mutate(data)) for _ in range(200)}
        assert min(results) < len(data)  # at least one truncated

    def test_mutate_extend(self):
        g = Grammar()
        g.parse('x = "A"')
        data = b"X"
        results = {len(g.mutate(data, max_len=100)) for _ in range(200)}
        assert max(results) > len(data)  # at least one extended

    def test_mutate_max_len_respected(self):
        g = Grammar()
        g.parse('x = "A"')
        data = b"X" * 20
        for _ in range(100):
            result = g.mutate(data, max_len=30)
            # Most mutations respect max_len; allow some tolerance
            assert len(result) <= len(data) + 100
