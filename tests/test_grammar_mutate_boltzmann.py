"""Tests for wiring generate(boltzmann=True) into Grammar.mutate()'s
replacement-generation paths -- P2-2 of
docs/handover/handover_generators_2026-09-20.md.

Before this, generate_boltzmann() and cycle_lemma_dyck_bytes() were shipped,
tested, and unreachable from any campaign path. This wires the grammar half
behind an opt-in flag (default off, pending a G0 A/B) so the bias-corrected
sampler documented in Grammar.generate()'s docstring is actually reachable
from _mutate_extend / _mutate_insert / _mutate_replace_section.
"""

from unittest.mock import patch

from fuzzer_tool.core.grammar import Grammar

RECURSIVE_SPEC = 'expr = "(" expr ")" | "x"'


class TestBoltzmannMutateFlagDefault:
    def test_default_is_off(self):
        g = Grammar(seed=1)
        assert g.boltzmann_mutate is False

    def test_constructor_can_enable_it(self):
        g = Grammar(seed=1, boltzmann_mutate=True)
        assert g.boltzmann_mutate is True

    def test_attribute_is_settable_after_construction(self):
        g = Grammar(seed=1)
        g.boltzmann_mutate = True
        assert g.boltzmann_mutate is True


class TestBoltzmannMutateWiring:
    """Each replacement-generation path must forward the flag to generate()."""

    def _grammar(self, boltzmann):
        g = Grammar(seed=1, boltzmann_mutate=boltzmann)
        g.parse(RECURSIVE_SPEC)
        return g

    def test_extend_forwards_flag_off(self):
        g = self._grammar(False)
        with patch.object(g, "generate", wraps=g.generate) as spy:
            g._mutate_extend(b"x", max_len=4096)
        _, kwargs = spy.call_args
        assert kwargs["boltzmann"] is False

    def test_extend_forwards_flag_on(self):
        g = self._grammar(True)
        with patch.object(g, "generate", wraps=g.generate) as spy:
            g._mutate_extend(b"x", max_len=4096)
        _, kwargs = spy.call_args
        assert kwargs["boltzmann"] is True
        assert kwargs["target_size"] == 64

    def test_insert_forwards_flag(self):
        g = self._grammar(True)
        with patch.object(g, "generate", wraps=g.generate) as spy:
            g._mutate_insert(b"x", max_len=4096)
        _, kwargs = spy.call_args
        assert kwargs["boltzmann"] is True
        assert kwargs["target_size"] == 32

    def test_replace_section_forwards_flag_and_span_as_target_size(self):
        g = self._grammar(True)
        data = b"xxxxxxxxxxxxxxxxxxxxxxxxxx"
        with patch.object(g, "generate", wraps=g.generate) as spy:
            g._mutate_replace_section(data, max_len=4096)
        _, kwargs = spy.call_args
        assert kwargs["boltzmann"] is True
        # target_size must equal the span being replaced (end - start), which
        # is also the max_len passed for this call -- not some other constant
        # like the 32/64 used by insert/extend.
        assert kwargs["target_size"] == kwargs["max_len"]

    def test_mutate_end_to_end_still_returns_bytes_with_flag_on(self):
        g = self._grammar(True)
        for _ in range(20):
            out = g.mutate(b"(((x)))", max_len=256)
            assert isinstance(out, bytes)

    def test_mutate_end_to_end_deterministic_given_seed(self):
        a = self._grammar(True)
        b = self._grammar(True)
        seq_a = [a.mutate(b"(((x)))", max_len=256) for _ in range(10)]
        seq_b = [b.mutate(b"(((x)))", max_len=256) for _ in range(10)]
        assert seq_a == seq_b


class TestBoltzmannMutateOffPathUnaffected:
    """Default behavior (flag off) must be unchanged from before this wiring."""

    def test_generate_call_defaults_to_plain_descent(self):
        g = Grammar(seed=1)
        g.parse(RECURSIVE_SPEC)
        with patch.object(g, "generate_boltzmann", wraps=g.generate_boltzmann) as spy:
            for _ in range(10):
                g._mutate_extend(b"x", max_len=4096)
                g._mutate_insert(b"x", max_len=4096)
                g._mutate_replace_section(b"xxxxxx", max_len=4096)
        spy.assert_not_called()
