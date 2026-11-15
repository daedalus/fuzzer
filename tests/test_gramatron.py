"""Gramatron-style walk mutations (core/gramatron.py; HexHive Gramatron, ISSTA'21).

An input is a walk: the choice sequence of a leftmost derivation. Replay
treats each choice as a hint (mod the alternatives live there), so every
mutant is grammatical by construction.

    random   keep walk[:i], regenerate the rest
    splice   walk1[:i] + walk2[j:] where both automaton states match
    recurse  repeat a segment that returns to its own state
"""

import re

import pytest

from fuzzer_tool.core.gramatron import MAX_STEPS, Gramatron
from fuzzer_tool.core.grammar import Grammar
from fuzzer_tool.core.rand_pool import RandPool

_SPEC = """
expr = term "+" expr | term
term = "(" expr ")" | num
num = "1" | "2" | "3"
"""


def _valid(data: bytes) -> bool:
    """Independent recursive-descent check of _SPEC (not the module's)."""
    pos = 0

    def expr() -> bool:
        nonlocal pos
        if not term():
            return False
        while pos < len(data) and data[pos : pos + 1] == b"+":
            pos += 1
            if not term():
                return False
        return True

    def term() -> bool:
        nonlocal pos
        if data[pos : pos + 1] == b"(":
            pos += 1
            if not expr() or data[pos : pos + 1] != b")":
                return False
            pos += 1
            return True
        if data[pos : pos + 1] in (b"1", b"2", b"3"):
            pos += 1
            return True
        return False

    return expr() and pos == len(data)


@pytest.fixture
def gram():
    g = Grammar(seed=1)
    g.parse(_SPEC)
    g.max_depth = 64  # recursion tests must not hit the depth fallback
    return Gramatron(g)


def _walks(gram, n, seed=7):
    rng = RandPool(seed=seed)
    return [gram.random_walk(rng) for _ in range(n)]


def test_oracle_rejects_garbage():
    """Control: the validator can fail, so the grammaticality tests mean something."""
    assert _valid(b"(1+2)+3")
    assert not _valid(b"(1+")
    assert not _valid(b"1++2")


def test_replay_is_deterministic(gram):
    for walk, out in _walks(gram, 20):
        assert gram.expand(walk) == out
        assert _valid(out)


def test_regen_keeps_prefix_bytes(gram):
    """Cut at a fixed step: the bytes produced before it survive, the rest is regenerated."""
    rng = RandPool(seed=3)
    for walk, out in _walks(gram, 20):
        _, off = gram.trace(walk)
        i = len(walk) // 2
        new_walk, new_out = gram.regen_from(walk, i, rng)
        assert _valid(new_out)
        assert new_walk[:i] == walk[:i]
        assert new_out.startswith(out[: off[i]])


def test_mutate_random_is_grammatical(gram):
    rng = RandPool(seed=4)
    for walk, _ in _walks(gram, 20):
        assert _valid(gram.mutate_random(walk, rng)[1])


def test_splice_is_prefix_plus_donor_tail(gram):
    """At a matched state, output = w1's bytes before step i + w2's bytes after step j."""
    pool = _walks(gram, 30)
    checked = 0
    for (w1, out1), (w2, out2) in zip(pool, pool[1:]):
        _, off1 = gram.trace(w1)
        _, off2 = gram.trace(w2)
        for i, j in gram.splice_points(w1, w2)[:10]:
            new_walk, new_out = gram.splice_at(w1, w2, i, j)
            assert _valid(new_out)
            assert new_walk[:i] == w1[:i]
            assert new_out.startswith(out1[: off1[i]])
            assert new_out.endswith(out2[off2[j] :])
            checked += 1
    assert checked, "no state-matched splice point in 29 pairs"


def test_mutate_splice_uses_a_splice_point(gram):
    rng = RandPool(seed=5)
    pool = _walks(gram, 30)
    results = [gram.mutate_splice(w1, w2, rng) for (w1, _), (w2, _) in zip(pool, pool[1:])]
    done = [r for r in results if r is not None]
    assert done and all(_valid(out) for _, out in done)


def test_recurse_repeats_segment(gram):
    rng = RandPool(seed=9)
    deep = max(_walks(gram, 50), key=lambda wo: wo[1].count(b"+"))
    res = gram.mutate_recurse(deep[0], rng)
    assert res is not None
    new_walk, new_out = res
    assert _valid(new_out)
    assert len(new_walk) > len(deep[0])


def test_out_of_range_choices_still_grammatical(gram):
    """Adversarial: hints from another context (or garbage) never break the grammar."""
    for walk in ([99] * 40, [7, 0, 13, 2, 255, 1], []):
        assert _valid(gram.expand(walk))


def test_left_recursive_grammar_is_bounded():
    """Adversarial: unbounded recursion stops at the step cap with valid-prefix output."""
    g = Grammar(seed=1)
    g.parse('a = "x" a | "x"')
    gt = Gramatron(g)
    out = gt.expand([0] * (10 * MAX_STEPS))
    assert re.fullmatch(rb"x+", out)


def test_empty_grammar():
    gt = Gramatron(Grammar(seed=1))
    assert gt.random_walk(RandPool(seed=1)) == ([], b"")


def test_walk_cache_is_bounded(gram):
    rng = RandPool(seed=11)
    for _ in range(gram.cache_size + 50):
        walk, out = gram.random_walk(rng)
        gram.remember(out, walk)
    assert len(gram) <= gram.cache_size


class TestOperators:
    """Registered ops: gated on a grammar, grammatical output, walks remembered."""

    def _engine(self, gram_obj):
        from types import SimpleNamespace

        from fuzzer_tool.services.operators import OperatorEngine

        class _Engine(OperatorEngine):
            ctx = None  # the real property reads a full Fuzzer

        eng = _Engine.__new__(_Engine)
        eng.ctx = SimpleNamespace(grammar=gram_obj, _rng=RandPool(seed=13), max_len=4096)
        return eng

    @pytest.mark.parametrize("op", ["gramatron_random", "gramatron_splice", "gramatron_recurse"])
    def test_registered_and_gated(self, op):
        from types import SimpleNamespace

        from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY

        assert op in REGISTRY.categories()["adaptive"]
        assert _AVAILABLE[op](SimpleNamespace(grammar=None), b"") is False
        assert _AVAILABLE[op](SimpleNamespace(grammar=object()), b"") is True

    @pytest.mark.parametrize("op", ["random", "splice", "recurse"])
    def test_chain_stays_grammatical(self, gram, op):
        """A seed without a walk becomes a fresh walk; its output then chains as a walk."""
        eng = self._engine(gram._g)
        handler = getattr(eng, f"_op_gramatron_{op}")
        buf = bytearray(b"not grammatical")
        for _ in range(30):
            buf = handler(buf, 0, bytes(buf))
            assert _valid(bytes(buf))
            assert gram_walk_known(eng, buf)

    def test_no_grammar_is_noop(self):
        eng = self._engine(None)
        assert eng._op_gramatron_random(bytearray(b"x"), 0, b"x") is None


def gram_walk_known(eng, buf) -> bool:
    return eng._gramatron().walk_for(bytes(buf)) is not None
