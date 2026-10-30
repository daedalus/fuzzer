"""Tests for lsystem_stochastic and lsystem_ctx (core/mutations/lsystem_*.py)."""

from __future__ import annotations

from fuzzer_tool.core.mutations.lsystem_ctx import EDGE, derive_ctx, lsystem_ctx
from fuzzer_tool.core.mutations.lsystem_expand import MAX_EXPAND
from fuzzer_tool.core.mutations.lsystem_stochastic import (
    SCALE,
    _rule,
    derive_stoch,
    lsystem_stochastic,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

BIG = 1 << 20
OPS = ("lsystem_stochastic", "lsystem_ctx")


class _MockFuzzer:
    def __init__(self):
        self.dictionary = []
        self.markov_trained = False
        self.mc = None
        self.mc_cem = False
        self.grammar = None
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self.seed_meta = {}
        self.corpus = []
        self.max_len = 4096


#: Half and quarter of the draw range: 1:1 and 1:3 rule thresholds.
HALF, QUARTER = SCALE // 2, SCALE // 4


def _tree_rules():
    # Article: 0 (1:1) -> 1[0]0 | 0, and 1 -> 11. Draws are in [0, SCALE).
    return {ord("0"): ((HALF, SCALE), (b"1[0]0", b"0")), ord("1"): ((SCALE,), (b"11",))}


class TestDeriveStoch:
    def test_first_branch(self):
        rng = ScriptedRng(randrange_lists=[[0]])
        assert derive_stoch(b"0", _tree_rules(), 1, BIG, rng) == b"1[0]0"

    def test_second_branch(self):
        rng = ScriptedRng(randrange_lists=[[SCALE - 1]])
        assert derive_stoch(b"0", _tree_rules(), 1, BIG, rng) == b"0"

    def test_boundary_goes_to_next_branch(self):
        # r equal to a threshold belongs to the next branch.
        below = ScriptedRng(randrange_lists=[[HALF - 1]])
        at = ScriptedRng(randrange_lists=[[HALF]])
        assert derive_stoch(b"0", _tree_rules(), 1, BIG, below) == b"1[0]0"
        assert derive_stoch(b"0", _tree_rules(), 1, BIG, at) == b"0"

    def test_two_generations(self):
        # gen 1: 0 -> 1[0]0.  gen 2 draws one value per symbol (5):
        # 1 -> 11, [ copied, 0 -> 1[0]0 (r=0), ] copied, 0 -> 0 (r=HALF).
        rng = ScriptedRng(randrange_lists=[[0], [HALF, HALF, 0, HALF, HALF]])
        out = derive_stoch(b"0", _tree_rules(), 2, BIG, rng)
        assert out == b"11[1[0]0]0"

    def test_unruled_symbols_copied(self):
        rng = ScriptedRng(randrange_lists=[[0, 0, 0]])
        assert derive_stoch(b"x-x", {}, 1, BIG, rng) == b"x-x"

    def test_cap_stops(self):
        rules = {ord("A"): ((SCALE,), (b"AAAA",))}
        rng = ScriptedRng(randrange_lists=[[0], [0] * 4, [0] * 16])  # 1 + 4 + 16 draws
        assert derive_stoch(b"A", rules, 3, 20, rng) == b"A" * 16

    def test_branch_frequency_follows_weights(self):
        # Fixed seed: 4000 draws of a 1:3 rule land near 25% first branch.
        rules = {ord("A"): ((QUARTER, SCALE), (b"X", b"Y"))}
        out = derive_stoch(b"A" * 4000, rules, 1, BIG, RandPool(seed=9))
        share = out.count(b"X") / len(out)
        assert 0.22 < share < 0.28

    def test_regression_lsystem_stochastic_exact_weights(self):
        """Every draw in [0, SCALE) once, through _rule's own thresholds:
        branch counts equal weight * SCALE / total exactly, for every
        reachable weight total (no float rounding at any boundary)."""
        for weights in ((1, 3), (1, 1), (2, 1, 1), (4, 4, 4), (1, 4, 3)):
            # _rule draws: branch count, then (start, length, weight) per
            # successor; successor i is the one-byte slice axiom[i].
            k = len(weights)
            draws = [k] + [v for i, w in enumerate(weights) for v in (i, 1, w)]
            axiom = bytes(range(ord("a"), ord("a") + k))
            rule = _rule(axiom, ScriptedRng(randints=draws))
            rng = ScriptedRng(randrange_lists=[list(range(SCALE))])
            out = derive_stoch(b"Z" * SCALE, {ord("Z"): rule}, 1, BIG, rng)

            want = [w * SCALE // sum(weights) for w in weights]
            assert [out.count(axiom[i : i + 1]) for i in range(k)] == want

    def test_control_same_seed_same_output(self):
        a = derive_stoch(b"0" * 50, _tree_rules(), 2, BIG, RandPool(seed=4))
        b = derive_stoch(b"0" * 50, _tree_rules(), 2, BIG, RandPool(seed=4))
        assert a == b


class TestDeriveCtx:
    def test_article_rule(self):
        rules = {(ord("b"), ord("a"), ord("c")): b"aa"}
        assert derive_ctx(b"bac", rules, 1, BIG) == b"baac"
        assert derive_ctx(b"xbacx", rules, 1, BIG) == b"xbaacx"

    def test_context_mismatch_is_identity(self):
        rules = {(ord("b"), ord("a"), ord("c")): b"aa"}
        for s in (b"bad", b"dac", b"a", b"ba", b"ac"):
            assert derive_ctx(s, rules, 3, BIG) == s

    def test_edge_sentinel(self):
        rules = {(EDGE, ord("a"), EDGE): b"aa"}
        assert derive_ctx(b"a", rules, 1, BIG) == b"aa"
        assert derive_ctx(b"a", rules, 2, BIG) == b"aa"  # contexts no longer match
        assert derive_ctx(b"ba", rules, 1, BIG) == b"ba"

    def test_parallel_contexts_read_old_string(self):
        rules = {(ord("a"), ord("b"), ord("a")): b"bb"}
        assert derive_ctx(b"ababa", rules, 1, BIG) == b"abbabba"

    def test_cap_stops(self):
        rules = {(EDGE, ord("a"), ord("a")): b"aaaa", (ord("a"), ord("a"), EDGE): b"aaaa"}
        assert len(derive_ctx(b"aa", rules, 6, 30)) <= 30

    def test_no_rules_noop(self):
        assert derive_ctx(b"abc", {}, 5, BIG) == b"abc"


class TestMutate:
    def test_scripted_stochastic(self):
        data = b"..AB......"
        # unit 2, offset 2, gens 1; A: 2 successors (0,2,w1)=AB (1,1,w1)=B;
        # B: 2 successors (0,1,w1)=A (1,2,w3)=B -> cumulative 1/2 and 1/4,
        # A picks B at r >= HALF, B picks A at r < QUARTER.
        rng = ScriptedRng(
            randints=[2, 2, 1, 2, 0, 2, 1, 1, 1, 1, 2, 0, 1, 1, 1, 2, 3],
            randrange_lists=[[HALF, 0]],
        )
        assert lsystem_stochastic(data, rng) == b"..BA......"

    def test_scripted_ctx(self):
        data = b"xxabcxxxxxx"
        # unit 3 (abc), offset 2, gens 1, 1 rule: pos 1 -> (a,b,c), slice (0,2)="ab".
        rng = ScriptedRng(randints=[3, 2, 1, 1, 1, 0, 2])
        assert lsystem_ctx(data, rng) == b"xx" + b"a" + b"ab" + b"c" + b"xxxxx"

    def test_empty_input(self):
        for fn in (lsystem_stochastic, lsystem_ctx):
            assert fn(b"", RandPool(seed=1)) == b""

    def test_length_preserved_adversarial(self):
        rng = RandPool(seed=11)
        for fn in (lsystem_stochastic, lsystem_ctx):
            for n in range(0, 40):
                for fill in (b"\x00", b"[", b"AB", bytes(range(256))):
                    data = (fill * n)[:n]
                    assert len(fn(data, rng)) == n

    def test_expansion_capped(self):
        rng = RandPool(seed=2)
        data = bytes(range(8)) * 1024
        for fn in (lsystem_stochastic, lsystem_ctx):
            for _ in range(50):
                out = fn(data, rng)
                assert len(out) == len(data)
                assert sum(x != y for x, y in zip(out, data, strict=True)) <= MAX_EXPAND

    def test_changes_input(self):
        data = bytes(range(32))
        for fn in (lsystem_stochastic, lsystem_ctx):
            rng = RandPool(seed=7)
            assert any(fn(data, rng) != data for _ in range(100))

    def test_deterministic_for_seed(self):
        data = bytes(range(48))
        for fn in (lsystem_stochastic, lsystem_ctx):
            assert fn(data, RandPool(seed=5)) == fn(data, RandPool(seed=5))


class TestWiring:
    def test_registered_regularity(self):
        for name in OPS:
            assert name in REGISTRY.names()
            assert REGISTRY.category_of(name) == "regularity"

    def test_dispatch_roundtrip(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(256))
        for name in OPS:
            out = dispatch[name](bytearray(data), 0, data)
            assert out is not None and len(out) == len(data)
