"""Tests for grammar module."""

from fuzzer_tool.core.grammar import GRAMMARS, Grammar, load_grammar


class TestGrammar:
    def test_empty_grammar(self):
        g = Grammar()
        result = g.generate()
        assert result == b""

    def test_parse_simple(self):
        g = Grammar()
        g.parse("rule = hello | world")
        assert "rule" in g.rules
        assert len(g.rules["rule"]) == 2

    def test_generate_returns_bytes(self):
        g = Grammar()
        g.parse('greeting = "hello" | "world"')
        result = g.generate("greeting")
        assert isinstance(result, bytes)
        assert result in (b"hello", b"world")

    def test_generate_literal(self):
        g = Grammar()
        g.parse('name = "test"')
        result = g.generate("name")
        assert result == b"test"

    def test_generate_rule_ref(self):
        g = Grammar()
        g.parse("start = greeting\nword = hi\n")
        # word has no alternatives that produce real output, so it returns b"?"
        result = g.generate("start", max_depth=2)
        assert isinstance(result, bytes)

    def test_generate_repeat(self):
        g = Grammar()
        g.parse('item = "A"\nlist = item{3}')
        result = g.generate("list")
        assert result == b"AAA"

    def test_generate_repeat_plus(self):
        g = Grammar()
        g.parse('item = "X"\nlist = item+')
        result = g.generate("list")
        assert len(result) >= 1
        assert result == b"X" * len(result)

    def test_generate_repeat_star(self):
        g = Grammar()
        g.parse('item = "Y"\nlist = item*')
        result = g.generate("list")
        assert isinstance(result, bytes)

    def test_generate_repeat_range(self):
        g = Grammar()
        g.parse('item = "Z"\nlist = item{2,5}')
        result = g.generate("list")
        assert len(result) >= 2
        assert len(result) <= 5
        assert result == b"Z" * len(result)

    def test_mutate_literal(self):
        g = Grammar()
        g.parse('data = "AAAA"')
        original = b"BBBB"
        mutated = g.mutate(original)
        assert isinstance(mutated, bytes)

    def test_mutate_empty(self):
        g = Grammar()
        g.parse('data = "X"')
        result = g.mutate(b"")
        assert isinstance(result, bytes)

    def test_parse_file(self, tmp_path):
        grammar_file = tmp_path / "test.gram"
        grammar_file.write_text('rule = "hello" | "world"\n')
        g = Grammar()
        g.parse_file(str(grammar_file))
        assert "rule" in g.rules

    def test_first_rule_as_default(self):
        g = Grammar()
        g.parse('alpha = "A"\nbeta = "B"')
        result = g.generate()
        assert result in (b"A", b"B")

    def test_max_depth_limits_recursion(self):
        g = Grammar()
        g.parse("a = b\nb = a")
        result = g.generate("a", max_depth=1)
        assert result == b"?"


class TestLoadGrammar:
    def test_builtin_json(self):
        g = load_grammar("json")
        assert "json" in g.rules
        assert "value" in g.rules

    def test_builtin_http(self):
        g = load_grammar("http_request")
        assert "request" in g.rules
        assert "method" in g.rules

    def test_builtin_elf(self):
        g = load_grammar("elf")
        assert "magic" in g.rules

    def test_inline_spec(self):
        g = load_grammar('rule = "test"')
        assert "rule" in g.rules

    def test_file_path(self, tmp_path):
        grammar_file = tmp_path / "test.gram"
        grammar_file.write_text('myrule = "hello"\n')
        g = load_grammar(str(grammar_file))
        assert "myrule" in g.rules

    def test_builtins_have_correct_count(self):
        assert len(GRAMMARS) == 3
        assert "json" in GRAMMARS
        assert "http_request" in GRAMMARS
        assert "elf" in GRAMMARS
