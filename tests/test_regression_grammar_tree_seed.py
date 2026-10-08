"""Regression: grammar_tree_mutate subtree generation ignored the passed rng.

``TreeMutator`` drew node choices from the rng given to ``mutate_tree`` but
called ``grammar.generate()``, which drew from ``Grammar._rng`` -- a fixed
``RandPool(None)`` stream from ``load_grammar``. Generated subtrees were
identical for every ``--seed`` and ignored stall reseeds.
"""

from __future__ import annotations

from fuzzer_tool.core.grammar import Grammar, TreeMutator, TreeNode
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

SPEC = 'start = a b\na = "x" | "y"\nb = "p" | "q"'

# mutate_tree op indices (see TreeMutator.mutate_tree).
OP_SWAP, OP_SPLICE, OP_RULE_SUB, OP_LEAF = 0, 3, 4, 5

# Interior nodes of _tree(): start(size 5), a(2), b(2) -> cumulative 5, 7, 9.
PICK_A = 5
# Rule-sub candidates (multi-alt only): a(2), b(2) -> cumulative 2, 4.
PICK_B = 2
ALT_FIRST, ALT_SECOND = 0, 1

SEED = 1234
ROUNDS = 32


class _Tripwire:
    """Fallback stream that fails the test on any draw."""

    def __getattr__(self, name):
        raise AssertionError(f"grammar fallback rng consumed: {name}")


def _grammar() -> Grammar:
    g = Grammar()
    g.parse(SPEC)
    g._rng = _Tripwire()
    return g


def _tree() -> TreeNode:
    a = TreeNode(rule="a", children=[TreeNode(data=b"x")])
    b = TreeNode(rule="b", children=[TreeNode(data=b"p")])
    return TreeNode(rule="start", children=[a, b])


def test_regression_tree_swap_uses_passed_rng():
    rng = ScriptedRng(randints=[OP_SWAP, PICK_A], choice_idxs=[ALT_SECOND])
    out = TreeMutator(_grammar()).mutate_tree(_tree(), rng=rng)
    assert out == b"y" + b"p"


def test_regression_splice_fallback_uses_passed_rng():
    rng = ScriptedRng(randints=[OP_SPLICE, PICK_A], choice_idxs=[ALT_SECOND])
    out = TreeMutator(_grammar()).mutate_tree(_tree(), rng=rng)
    assert out == b"y" + b"p"


def test_regression_rule_sub_uses_passed_rng():
    rng = ScriptedRng(randints=[OP_RULE_SUB, PICK_B], choice_idxs=[ALT_SECOND])
    out = TreeMutator(_grammar()).mutate_tree(_tree(), rng=rng)
    assert out == b"x" + b"q"


def test_regression_empty_leaf_generate_uses_passed_rng():
    rng = ScriptedRng(randints=[OP_LEAF], choice_idxs=[ALT_FIRST, ALT_SECOND, ALT_SECOND])
    out = TreeMutator(_grammar()).mutate_tree(TreeNode(rule="start"), rng=rng)
    assert out == b"y" + b"q"


def test_falsify_generate_without_rng_keeps_own_stream():
    """No rng passed: generate() must still draw from the grammar's stream."""
    g = Grammar()
    g.parse(SPEC)
    g._rng = ScriptedRng(choice_idxs=[ALT_FIRST, ALT_SECOND, ALT_FIRST])
    assert g.generate() == b"y" + b"p"


def test_adversarial_inplace_reseed_replays_generation():
    """Fuzzer reseeds ctx._rng in place on stall; generation must follow it."""
    pool = RandPool(seed=SEED)
    mut = TreeMutator(_grammar())
    runs = []
    for _ in range(2):
        pool.reseed(SEED)
        runs.append([mut.mutate_tree(_tree(), rng=pool) for _ in range(ROUNDS)])
    assert runs[0] == runs[1]
