"""Tests for subtree-population crossover (docs/web_research_port_candidates_2026-08.md #8).

Port of GRIIN (ASE '23) / Grammarinator x AFL++ (2026): grammar-aware tree
crossover should be able to splice in a subtree harvested from a *different*
corpus entry, not just regenerate one from scratch or clone within the same
tree. ``TreeMutator``'s docstring already promised this as op 4
("Subtree splice") but the implementation never existed until now.
"""

import random

from fuzzer_tool.core.grammar import Grammar, SubtreePopulation, TreeMutator
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer


def _json_grammar() -> Grammar:
    g = Grammar()
    g.parse('root = {"key":"value"}')
    return g


class TestSubtreePopulation:
    def test_sample_unseen_rule_returns_none(self):
        pop = SubtreePopulation()
        assert pop.sample("value") is None

    def test_add_harvests_interior_nodes(self):
        tm = TreeMutator(_json_grammar())
        tree = tm.parse(b'{"key":"value"}')
        pop = SubtreePopulation()
        pop.add(tree, rng=random.Random(1))
        assert len(pop) > 0
        assert pop.sample("root") is not None

    @staticmethod
    def _interior_node(rule: str, marker: bytes):
        """A one-node-deep interior node: ``collect_interior`` only reports
        non-leaf nodes, so a bare ``TreeNode(rule=..., data=...)`` (a leaf)
        would never be harvested."""
        from fuzzer_tool.core.grammar import TreeNode

        return TreeNode(rule=rule, children=[TreeNode(rule="leaf", data=marker)])

    def test_reservoir_bounded_by_max_per_rule(self):
        """Falsification: harvesting more nodes than the cap must never
        grow the pool past ``max_per_rule`` for a single rule."""
        pop = SubtreePopulation(max_per_rule=4)
        rng = random.Random(7)
        for i in range(200):
            pop.add(self._interior_node("value", str(i).encode()), rng=rng)
        assert len(pop._pools["value"]) == 4

    def test_reservoir_sampling_reaches_late_items(self):
        """Adversarial: an item harvested long after the pool filled up
        must still have a nonzero chance of surviving eviction — a buggy
        reservoir (e.g. only replacing index 0) would never let it in."""
        seen_late_item = False
        for trial in range(200):
            pop = SubtreePopulation(max_per_rule=4)
            rng = random.Random(trial)
            for i in range(50):
                pop.add(self._interior_node("value", str(i).encode()), rng=rng)
            if any(n.children[0].data == b"49" for n in pop._pools["value"]):
                seen_late_item = True
                break
        assert seen_late_item, "item #49 never survived reservoir sampling across 200 trials"


class TestTreeSplice:
    def test_splice_falls_back_without_population(self):
        """No population supplied -> behaves like a plain subtree swap,
        never raises and always returns bytes that respect max_len."""
        tm = TreeMutator(_json_grammar())
        tm._rng = random.Random(0)
        tree = tm.parse(b'{"key":"value"}')
        result = tm._tree_splice(tree, max_len=64, population=None)
        assert isinstance(result, bytes)
        assert len(result) <= 64

    def test_splice_falls_back_on_empty_population(self):
        tm = TreeMutator(_json_grammar())
        tm._rng = random.Random(0)
        tree = tm.parse(b'{"key":"value"}')
        result = tm._tree_splice(tree, max_len=64, population=SubtreePopulation())
        assert isinstance(result, bytes)

    def test_splice_grafts_donor_subtree(self):
        """Splicing from a population seeded with a donor tree must be able
        to pull in bytes that never appeared in the original tree."""
        grammar = _json_grammar()
        tm = TreeMutator(grammar)
        target_tree = tm.parse(b'{"key":"value"}')
        donor_tree = tm.parse(b'{"other":"DONOR_MARKER_XYZ"}')

        pop = SubtreePopulation()
        pop.add(donor_tree, rng=random.Random(3))

        found_donor_bytes = False
        for seed in range(64):
            tm._rng = random.Random(seed)
            clone = tm._clone_tree(target_tree)
            out = tm._tree_splice(clone, max_len=4096, population=pop)
            if b"DONOR_MARKER_XYZ" in out or b"other" in out:
                found_donor_bytes = True
                break
        assert found_donor_bytes, "splice never grafted in bytes from the donor tree"

    def test_mutate_tree_op3_is_splice(self):
        """The op-3 branch in mutate_tree must route to _tree_splice, not
        silently stay a no-op (regression for the missing implementation)."""
        grammar = _json_grammar()
        tm = TreeMutator(grammar)
        tree = tm.parse(b'{"key":"value"}')
        donor_tree = tm.parse(b'{"other":"DONOR_MARKER_XYZ"}')
        pop = SubtreePopulation()
        pop.add(donor_tree, rng=random.Random(3))

        class _FixedOpRng:
            """Forces mutate_tree's op selection to pick splice (op index 3)."""

            def randint(self, a, b):
                if b == 5:  # the op-selection roll in mutate_tree
                    return 3
                return random.Random(0).randint(a, b)

        found_donor_bytes = False
        for _ in range(32):
            clone = tm._clone_tree(tree)
            out = tm.mutate_tree(clone, max_len=4096, rng=_FixedOpRng(), population=pop)
            if b"DONOR_MARKER_XYZ" in out or b"other" in out:
                found_donor_bytes = True
                break
        assert found_donor_bytes


class TestGrammarTreeMutateOperatorWiring:
    def _engine_with_grammar(self, corpus):
        f = make_minimal_fuzzer(seed=0x5EED)
        f.grammar = _json_grammar()
        f.corpus = corpus
        return OperatorEngine(f)

    def test_builds_and_reuses_population_across_calls(self):
        corpus = [b'{"a":"seed_one"}', b'{"b":"seed_two_marker"}']
        engine = self._engine_with_grammar(corpus)
        buf = bytearray(b'{"key":"value"}')
        engine._op_grammar_tree_mutate(buf, 0, bytes(buf))
        f = engine.f
        assert hasattr(f, "_subtree_population")
        assert len(f._subtree_population) > 0
        assert f._subtree_pop_next_idx == len(corpus)

        # Corpus growth is picked up incrementally on the next call.
        corpus.append(b'{"c":"seed_three"}')
        engine._op_grammar_tree_mutate(buf, 0, bytes(buf))
        assert f._subtree_pop_next_idx == len(corpus)

    def test_grammar_tree_mutate_returns_bytes(self):
        corpus = [b'{"a":"seed_one"}']
        engine = self._engine_with_grammar(corpus)
        buf = bytearray(b'{"key":"value"}')
        result = engine._op_grammar_tree_mutate(buf, 0, bytes(buf))
        assert isinstance(result, bytearray)

    def test_no_grammar_returns_none(self):
        """Adversarial: without a grammar the op must be a clean no-op,
        never touching the (nonexistent) population machinery."""
        f = make_minimal_fuzzer(seed=1)
        f.grammar = None
        engine = OperatorEngine(f)
        buf = bytearray(b"whatever")
        assert engine._op_grammar_tree_mutate(buf, 0, bytes(buf)) is None
