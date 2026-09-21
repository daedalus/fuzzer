"""Tests for the ``tree_generate`` operator (P2-2 of
docs/handover/handover_generators_2026-09-20.md).

``cycle_lemma_dyck_bytes`` (uniform Catalan sampling of balanced-delimiter
byte strings via the cycle lemma) shipped in tree_mutator.py with its own
unit tests but no caller anywhere in the operator table. This wires it as
a generator (not a mutator of the existing tree): it synthesizes a fresh
balanced fragment and splices it into the buffer, gated on the seed already
containing a bracket delimiter.
"""

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.tree_mutator import has_bracket_delimiter, partial_parse
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer


class TestHasBracketDelimiter:
    def test_true_for_parens(self):
        assert has_bracket_delimiter(b"foo(bar)")

    def test_true_for_brackets(self):
        assert has_bracket_delimiter(b"[1,2,3]")

    def test_true_for_braces(self):
        assert has_bracket_delimiter(b'{"a":1}')

    def test_false_for_plain_bytes(self):
        assert not has_bracket_delimiter(b"plain ascii, no nesting")

    def test_false_for_quotes_alone(self):
        # Quotes are excluded from _GEN_PAIRS: a run of quotes toggles
        # in/out rather than nesting.
        assert not has_bracket_delimiter(b"'just quotes' \"here\"")

    def test_false_for_empty(self):
        assert not has_bracket_delimiter(b"")


class TestTreeGenerateAvailability:
    def test_gated_off_without_delimiters(self):
        f = make_minimal_fuzzer(0x5EED)
        assert "tree_generate" not in REGISTRY.available(f, b"plain ascii only")

    def test_available_with_a_delimiter(self):
        f = make_minimal_fuzzer(0x5EED)
        assert "tree_generate" in REGISTRY.available(f, b"(x)")

    def test_registered_in_radamsa_category(self):
        assert "tree_generate" in REGISTRY.categories()["radamsa"]


class TestTreeGenerateOperator:
    def setup_method(self):
        self.f = make_minimal_fuzzer(0x5EED)
        self.engine = OperatorEngine(self.f)

    def test_inserts_a_balanced_fragment(self):
        buf = bytearray(b"(seed)")
        before_len = len(buf)
        self.engine._op_tree_generate(buf, 0, b"")
        assert len(buf) > before_len
        # The inserted fragment must itself be a fully-closed balanced run
        # somewhere in the result -- confirm the whole buffer still parses
        # with no dangling opens beyond what the original seed had, i.e.
        # the net open/close delta contributed by the insertion is zero.
        root = partial_parse(bytes(buf))

        def _count_unclosed(node):
            n = 0 if node.closed or node.open is None else 1
            for c in node.children:
                if not isinstance(c, (bytes, bytearray)):
                    n += _count_unclosed(c)
            return n

        assert _count_unclosed(root) == 0

    def test_never_exceeds_max_len(self):
        self.f.max_len = 10
        buf = bytearray(b"(x)")
        for _ in range(30):
            self.engine._op_tree_generate(buf, 0, b"")
            assert len(buf) <= 10

    def test_declines_when_buffer_already_at_max_len(self):
        self.f.max_len = 4
        buf = bytearray(b"(x)")  # already at 3, room=1 < 2 -> decline path
        # room = max_len - len(buf) = 1, below the 2-byte minimum fragment
        before = bytes(buf)
        self.engine._op_tree_generate(buf, 0, b"")
        assert bytes(buf) == before

    def test_empty_buffer_does_not_crash(self):
        buf = bytearray(b"")
        # Should decline cleanly rather than raising (no position to insert
        # relative to, and nothing to blend a fragment into).
        self.engine._op_tree_generate(buf, 0, b"")

    def test_deterministic_given_seed(self):
        f1 = make_minimal_fuzzer(0x1234)
        f2 = make_minimal_fuzzer(0x1234)
        e1, e2 = OperatorEngine(f1), OperatorEngine(f2)
        b1, b2 = bytearray(b"(seed)"), bytearray(b"(seed)")
        for _ in range(10):
            e1._op_tree_generate(b1, 0, b"")
            e2._op_tree_generate(b2, 0, b"")
        assert bytes(b1) == bytes(b2)
