"""Tests for the lsystem_expand operator (core/mutations/lsystem_expand.py)."""

from __future__ import annotations

from fuzzer_tool.core.mutations.lsystem_expand import (
    MAX_EXPAND,
    MAX_GENS,
    SUCC_MAX,
    UNIT_MAX,
    derive,
    lsystem_expand,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

BIG = 1 << 20


def _fib_len(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return b


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


class TestDerive:
    # Oracles are the worked examples of the L-system article.
    def test_algae_generations(self):
        rules = {ord("A"): b"AB", ord("B"): b"A"}
        assert derive(b"A", rules, 0, BIG) == b"A"
        assert derive(b"A", rules, 1, BIG) == b"AB"
        assert derive(b"A", rules, 4, BIG) == b"ABAABABA"
        assert derive(b"A", rules, 5, BIG) == b"ABAABABAABAAB"

    def test_algae_lengths_are_fibonacci(self):
        rules = {ord("A"): b"AB", ord("B"): b"A"}
        for n in range(1, 12):
            assert len(derive(b"A", rules, n, BIG)) == _fib_len(n)

    def test_fractal_tree(self):
        rules = {ord("1"): b"11", ord("0"): b"1[0]0"}
        assert derive(b"0", rules, 2, BIG) == b"11[1[0]0]1[0]0"

    def test_cantor(self):
        rules = {ord("A"): b"ABA", ord("B"): b"BBB"}
        assert derive(b"A", rules, 2, BIG) == b"ABABBBABA"

    def test_unruled_symbols_are_identity(self):
        rules = {ord("F"): b"F+F"}
        assert derive(b"F-F", rules, 1, BIG) == b"F+F-F+F"

    def test_parallel_not_sequential(self):
        # S->SS applied one rule at a time reaches SSS; in parallel it cannot.
        out = {len(derive(b"S", {ord("S"): b"SS"}, n, BIG)) for n in range(1, 5)}
        assert out == {2, 4, 8, 16}
        assert 3 not in out

    def test_cap_stops_before_overflow(self):
        out = derive(b"A", {ord("A"): b"AAAA"}, 6, 100)
        assert out == b"A" * 64

    def test_cap_on_first_generation_returns_axiom(self):
        assert derive(b"A", {ord("A"): b"A" * 50}, 3, 10) == b"A"

    def test_empty_rules_noop(self):
        assert derive(b"abc", {}, 5, BIG) == b"abc"


class TestMutate:
    def test_scripted_d0l(self):
        data = b"xxABCyyyyyy"
        # mode 0 (D0L), unit 3, offset 2, gens 2, then per distinct symbol A,B,C:
        # (start, length) -> A=(0,2) "AB", B=(2,1) "C", C=(1,1) "B"
        rng = ScriptedRng(randints=[0, 3, 2, 2, 0, 2, 2, 1, 1, 1])
        out = lsystem_expand(data, rng)
        # Hand-derived: ABC -> AB,C,B = ABCB -> AB,C,B,C = ABCBC
        rules = {ord("A"): b"AB", ord("B"): b"C", ord("C"): b"B"}
        assert derive(b"ABC", rules, 1, BIG) == b"ABCB"
        gen2 = b"ABCBC"
        assert out == b"xx" + gen2 + b"yyyy"

    def test_scripted_fibonacci(self):
        data = b"..ABCD.."  # 6 bytes after the window offset
        # mode 1, unit 4, offset 2, gens 3 -> A="AB", B="CD", s_{k+1}=s_k+s_{k-1}
        rng = ScriptedRng(randints=[1, 4, 2, 3])
        a, b = b"AB", b"CD"
        s = [b, a]
        for _ in range(3):
            s = [s[1], s[1] + s[0]]
        assert lsystem_expand(data, rng) == data[:2] + s[1][: len(data) - 2]

    def test_empty_input(self):
        assert lsystem_expand(b"", RandPool(seed=1)) == b""

    def test_one_byte_input_uses_d0l(self):
        out = lsystem_expand(b"A", ScriptedRng(randints=[1, 1, 0, 1, 0, 1]))
        assert isinstance(out, bytes) and len(out) >= 1

    def test_length_preserved(self):
        rng = RandPool(seed=3)
        data = bytes(range(64))
        for _ in range(200):
            assert len(lsystem_expand(data, rng)) == len(data)

    def test_adversarial_sizes_and_alphabets(self):
        rng = RandPool(seed=11)
        for n in range(0, 40):
            for fill in (b"\x00", b"[", b"AB"):
                data = (fill * n)[:n]
                out = lsystem_expand(data, rng)
                assert isinstance(out, bytes)
                assert len(out) == n

    def test_changes_input(self):
        rng = RandPool(seed=7)
        data = bytes(range(32))
        outs = [lsystem_expand(data, rng) for _ in range(100)]
        assert any(o != data for o in outs)

    def test_big_buffer_gets_expansion_up_to_cap(self):
        # 4 KiB buffer: the expansion overwrites it from offset 0, length kept.
        rng = ScriptedRng(randints=[1, 2, 0, 6])
        data = bytes(range(256)) * 16
        out = lsystem_expand(data, rng)
        assert len(out) == len(data) and out[:MAX_EXPAND] != data[:MAX_EXPAND]

    def test_deterministic_for_seed(self):
        # Rule 46 control: same seed, same output.
        data = bytes(range(48))
        a = lsystem_expand(data, RandPool(seed=5))
        b = lsystem_expand(data, RandPool(seed=5))
        assert a == b

    def test_falsification_constants_sane(self):
        assert 1 <= SUCC_MAX <= UNIT_MAX and MAX_GENS >= 1 and MAX_EXPAND >= 64


class TestWiring:
    def test_registered_regularity(self):
        assert "lsystem_expand" in REGISTRY.names()
        assert REGISTRY.category_of("lsystem_expand") == "regularity"

    def test_dispatch_roundtrip(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(256))
        out = dispatch["lsystem_expand"](bytearray(data), 0, data)
        assert out is not None and len(out) <= fuzzer.max_len
