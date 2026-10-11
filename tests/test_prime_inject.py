"""Tests for the prime_inject operator (core/mutations/prime_inject.py)."""

from __future__ import annotations

from fuzzer_tool.core import numtheory as nt
from fuzzer_tool.core.mutations.prime_inject import prime_inject, values_for
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

WIDTHS = (1, 2, 4, 8)


class ScriptedRng:
    """randint pops a scripted value and asserts it lies in [a, b]."""

    def __init__(self, ints):
        self._ints = list(ints)

    def randint(self, a, b):
        v = self._ints.pop(0)
        assert a <= v <= b, f"scripted {v} outside [{a}, {b}]"
        return v


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


class TestTable:
    def test_values_fit_width(self):
        for w in WIDTHS:
            assert all(0 < v < 1 << (8 * w) for v in values_for(w))

    def test_boundary_primes_present(self):
        assert 251 in values_for(1)
        assert 65521 in values_for(2)
        assert 4294967291 in values_for(4)
        assert 18446744073709551557 in values_for(8)

    def test_pseudoprimes_are_composite_but_pass_base_2(self):
        assert 3825123056546413051 in values_for(8)
        fooling = [v for v in values_for(8) if not nt.is_prime(v) and pow(2, v - 1, v) == 1]
        assert 561 in fooling and 3825123056546413051 in fooling

    def test_semiprime_of_boundary_primes(self):
        p = nt.prev_prime(1 << 16)
        q = nt.prev_prime(p)
        assert p * q in values_for(4)

    def test_cached(self):
        assert values_for(4) is values_for(4)

    def test_control_against_itself(self):
        # Rule 46: two builds of the table agree.
        assert values_for.__wrapped__(8) == values_for.__wrapped__(8)


class TestMutate:
    def test_scripted_big_endian_slot(self):
        data = bytes(range(16))
        want = values_for(4)[0].to_bytes(4, "big")
        # width idx 2 (4 B), BE=1, value idx 0, region length 4, offset 8
        rng = ScriptedRng([2, 1, 0, 4, 8])
        out = prime_inject(data, rng)
        assert out == data[:8] + want + data[12:]

    def test_scripted_little_endian_slot(self):
        data = bytes(16)
        v = values_for(2)[1]
        rng = ScriptedRng([1, 0, 1, 2, 6])
        assert prime_inject(data, rng) == bytes(6) + v.to_bytes(2, "little") + bytes(8)

    def test_empty_input(self):
        out = prime_inject(b"", RandPool(seed=1))
        assert int.from_bytes(out, "big") in values_for(len(out))

    def test_length_preserved_and_changed(self):
        rng = RandPool(seed=7)
        data = bytes(64)
        outs = {prime_inject(data, rng) for _ in range(20)}
        assert all(len(o) == 64 for o in outs)
        assert all(o != data for o in outs)

    def test_one_byte_buffer(self):
        out = prime_inject(b"\x00", RandPool(seed=3))
        assert len(out) == 1 and out[0] in values_for(1)

    def test_adversarial_sizes(self):
        rng = RandPool(seed=11)
        for n in range(1, 40):
            assert len(prime_inject(bytes(n), rng)) == n

    def test_falsification_wrong_oracle(self):
        # A zero-filled buffer is never a table value: the check can fail.
        assert int.from_bytes(bytes(4), "big") not in values_for(4)


class TestWiring:
    def test_registered_regularity(self):
        assert "prime_inject" in REGISTRY.names()
        assert REGISTRY.category_of("prime_inject") == "regularity"

    def test_dispatch_roundtrip(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(256))
        out = dispatch["prime_inject"](bytearray(data), 0, data)
        assert out is not None and len(out) == len(data) and out != data
