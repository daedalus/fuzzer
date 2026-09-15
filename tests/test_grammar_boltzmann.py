"""Tests for Boltzmann sampling (docs/handover/handover_trees.md §7.4).

These exercise ``Grammar.generate_boltzmann`` / ``Grammar.generate(boltzmann=True)``,
the corrected sampler for grammars with recursive rules. See the block
comment above ``Grammar.generate_boltzmann`` in ``core/grammar.py`` for the
bias in the plain recursive-descent path this replaces.
"""

import statistics

import pytest

from fuzzer_tool.core.grammar import GRAMMARS, Grammar, load_grammar

RECURSIVE_SPEC = 'expr = "(" expr ")" | "x"'


class TestBoltzmannBasics:
    def test_empty_grammar(self):
        g = Grammar()
        assert g.generate_boltzmann() == b""
        assert g.generate(boltzmann=True) == b""

    def test_returns_bytes(self):
        g = Grammar(seed=1)
        g.parse(RECURSIVE_SPEC)
        result = g.generate_boltzmann("expr", max_depth=10)
        assert isinstance(result, bytes)

    def test_non_recursive_grammar_still_produces_valid_alternatives(self):
        g = Grammar(seed=1)
        g.parse('greeting = "hello" | "world"')
        for _ in range(30):
            assert g.generate(boltzmann=True) in (b"hello", b"world")

    def test_literal_only_rule(self):
        g = Grammar(seed=1)
        g.parse('name = "test"')
        assert g.generate_boltzmann("name") == b"test"

    def test_fixed_repeat_exact_count(self):
        g = Grammar(seed=1)
        g.parse('item = "A"\nlist = item{3}')
        assert g.generate_boltzmann("list") == b"AAA"

    def test_bounded_repeat_within_range(self):
        g = Grammar(seed=1)
        g.parse('item = "Z"\nlist = item{2,5}')
        for _ in range(50):
            result = g.generate_boltzmann("list", target_size=10)
            assert result == b"Z" * len(result)
            assert 2 <= len(result) <= 5

    def test_unknown_starting_rule_is_atom(self):
        g = Grammar(seed=1)
        g.parse('start = "x"')
        assert g.generate_boltzmann("does_not_exist") == b"?"

    def test_undefined_reference_expands_to_atom(self):
        g = Grammar(seed=1)
        g.parse("start = missing_rule")
        assert g.generate_boltzmann("start", max_depth=5) == b"?"

    def test_depth_exhaustion_yields_atom(self):
        g = Grammar(seed=1)
        g.parse(RECURSIVE_SPEC)
        # depth 0 must behave exactly like the plain generator's depth-0 case
        assert g.generate_boltzmann("expr", max_depth=0) == b"?"

    def test_max_len_truncates_and_bounds_work(self):
        g = Grammar(seed=1)
        g.parse(RECURSIVE_SPEC)
        result = g.generate_boltzmann("expr", max_depth=40, target_size=5000, max_len=16)
        assert len(result) == 16

    def test_deterministic_given_seed(self):
        g1 = Grammar(seed=99)
        g1.parse(RECURSIVE_SPEC)
        g2 = Grammar(seed=99)
        g2.parse(RECURSIVE_SPEC)
        outs1 = [g1.generate_boltzmann("expr", max_depth=12, target_size=8) for _ in range(25)]
        outs2 = [g2.generate_boltzmann("expr", max_depth=12, target_size=8) for _ in range(25)]
        assert outs1 == outs2

    def test_generate_dispatches_to_boltzmann(self):
        g = Grammar(seed=5)
        g.parse(RECURSIVE_SPEC)
        via_generate = g.generate("expr", max_depth=10, boltzmann=True, target_size=6)
        assert isinstance(via_generate, bytes)

    def test_default_generate_unaffected(self):
        # boltzmann defaults to False: existing callers see identical behavior.
        g = Grammar(seed=1)
        g.parse('greeting = "hello" | "world"')
        assert g.generate("greeting") in (b"hello", b"world")


class TestBoltzmannSizeTuning:
    """Expected size should track target_size for a recursive grammar."""

    @pytest.mark.parametrize("target", [4, 12, 30])
    def test_mean_size_tracks_target(self, target):
        g = Grammar(seed=123)
        g.parse(RECURSIVE_SPEC)
        sizes = [
            len(g.generate_boltzmann("expr", max_depth=40, target_size=target))
            for _ in range(1500)
        ]
        mean = statistics.mean(sizes)
        # Loose tolerance: this targets the *expected* size, not an exact
        # value, and sizes are odd integers only (1, 3, 5, ...) here.
        assert target * 0.5 <= mean <= target * 1.8

    def test_larger_target_gives_larger_mean(self):
        g = Grammar(seed=123)
        g.parse(RECURSIVE_SPEC)
        small = [
            len(g.generate_boltzmann("expr", max_depth=40, target_size=4)) for _ in range(800)
        ]
        large = [
            len(g.generate_boltzmann("expr", max_depth=40, target_size=40)) for _ in range(800)
        ]
        assert statistics.mean(large) > statistics.mean(small)


class TestBoltzmannCorrectsRecursiveDescentBias:
    """Mirrors docs/handover/handover_trees.md §5's Dyck-path measurement:
    a naive per-branch-uniform recursive generator does not sample sizes
    uniformly/as-designed, and the Boltzmann sampler is tunable where the
    naive one is not.
    """

    def test_naive_generator_collapses_to_near_minimal_size(self):
        # With expr = "(" expr ")" | "x" chosen uniformly, the walk
        # terminates ("x") with probability 1/2 at every step, so the
        # naive generator's size distribution is geometric and independent
        # of max_depth for any reasonably large depth -- it can't be
        # steered toward a larger typical size at all.
        g = Grammar(seed=42)
        g.parse(RECURSIVE_SPEC)
        shallow = [len(g.generate("expr", max_depth=10)) for _ in range(2000)]
        deep = [len(g.generate("expr", max_depth=40)) for _ in range(2000)]
        # Both settle near the geometric distribution's mean (~3); raising
        # the depth cap by 4x barely moves it.
        assert statistics.mean(shallow) < 5
        assert statistics.mean(deep) < 5

    def test_boltzmann_reaches_sizes_naive_generation_effectively_never_does(self):
        g = Grammar(seed=42)
        g.parse(RECURSIVE_SPEC)
        naive = [len(g.generate("expr", max_depth=40)) for _ in range(3000)]
        boltz = [
            len(g.generate("expr", max_depth=40, boltzmann=True, target_size=25))
            for _ in range(3000)
        ]
        threshold = 15
        naive_frac_large = sum(1 for s in naive if s >= threshold) / len(naive)
        boltz_frac_large = sum(1 for s in boltz if s >= threshold) / len(boltz)
        assert naive_frac_large < 0.01
        assert boltz_frac_large > 0.15


class TestBoltzmannOnBuiltinGrammars:
    """Smoke tests against the shipped GRAMMARS, which have several
    mutually-referencing rules (not just one self-recursive rule)."""

    @pytest.mark.parametrize("name", list(GRAMMARS))
    def test_builtin_grammar_generates(self, name):
        g = load_grammar(name)
        for _ in range(10):
            result = g.generate(boltzmann=True, max_len=256)
            assert isinstance(result, bytes)
            assert len(result) <= 256

    def test_json_grammar_boltzmann_varies_string_length(self):
        g = load_grammar("json")
        lengths = set()
        for _ in range(60):
            # `text` (word* | "") is the recursive/repeat-bearing rule that
            # Boltzmann tuning acts on; sampled directly rather than via
            # `string` to avoid a preexisting, unrelated parsing quirk on
            # `string`'s escaped-quote literals (present in plain
            # `generate()` too -- not something this change touches).
            result = g.generate("text", boltzmann=True, target_size=6, max_len=64)
            lengths.add(len(result))
        # word* inside text should realize more than one length across samples
        assert len(lengths) > 1
