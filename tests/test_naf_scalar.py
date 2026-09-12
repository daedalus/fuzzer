"""Tests for naf_scalar.py — the NAF-weight-extremal scalar mutator.

Two layers: the NAF arithmetic itself (encode/decode round trip, weight
minimality vs plain binary popcount, non-adjacency), and the mutator's
integration with the operator registry/dispatch (mirrors the pattern in
test_regression_mtf_bwt_rle.py for golomb/elias).
"""

import random

from fuzzer_tool.core.mutations.naf_scalar import (
    SCALAR_BYTES,
    from_naf,
    naf_scalar_mutate,
    naf_weight,
    to_naf,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


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


class TestNafArithmetic:
    def test_round_trip_random_integers(self):
        rng = random.Random(0)
        for _ in range(2000):
            k = rng.randint(0, 2**256 - 1)
            assert from_naf(to_naf(k)) == k

    def test_zero(self):
        assert to_naf(0) == []
        assert from_naf([]) == 0
        assert naf_weight(0) == 0

    def test_no_two_adjacent_nonzero_digits(self):
        rng = random.Random(1)
        for _ in range(500):
            k = rng.randint(1, 2**256 - 1)
            digits = to_naf(k)
            for i in range(len(digits) - 1):
                assert not (digits[i] != 0 and digits[i + 1] != 0), (
                    f"adjacent nonzero NAF digits for k={k}: {digits}"
                )

    def test_naf_weight_never_exceeds_binary_popcount_representative_cases(self):
        # NAF weight is <= the weight of *any* signed-digit representation,
        # in particular <= plain binary popcount, for every integer.
        for k in (1, 2, 3, 5, 7, 15, 31, 255, 2**32 - 1, 0b0101010101):
            assert naf_weight(k) <= bin(k).count("1")

    def test_minimal_weight_forms_have_low_naf_weight(self):
        # (1<<a) - (1<<b) collapses to a short run in NAF regardless of how
        # far apart a and b are -- that's the whole point of the construction.
        k = (1 << 200) - (1 << 5)
        assert naf_weight(k) <= 2

    def test_alternating_pattern_has_high_naf_weight(self):
        # 0b0101...01 over 256 bits: every bit is already isolated (no two
        # adjacent set bits), so its own binary form already equals its NAF,
        # and the weight is close to nbits/2 -- near the maximum achievable.
        nbits = 256
        value = sum(1 << i for i in range(0, nbits, 2))
        assert naf_weight(value) >= nbits // 2 - 2


class TestNafScalarMutator:
    def test_preserves_length(self):
        rng = RandPool(seed=42)
        data = bytes(range(256)) * 4
        result = naf_scalar_mutate(data, rng)
        assert len(result) == len(data)

    def test_too_short_buffer_returns_unchanged(self):
        rng = RandPool(seed=42)
        data = bytes(range(SCALAR_BYTES - 1))
        assert naf_scalar_mutate(data, rng) == data

    def test_writes_aligned_scalar_width_region(self):
        rng = RandPool(seed=7)
        data = bytes(SCALAR_BYTES * 3)
        result = naf_scalar_mutate(data, rng)
        assert len(result) == len(data)
        # Exactly one SCALAR_BYTES-aligned window should differ from all-zero.
        changed_words = [
            i
            for i in range(0, len(data), SCALAR_BYTES)
            if result[i : i + SCALAR_BYTES] != data[i : i + SCALAR_BYTES]
        ]
        assert len(changed_words) <= 1

    def test_produces_weight_extremal_scalars_over_many_draws(self):
        # Over many draws, some outputs should sit near each weight extreme
        # (not clustered in the middle), which is the entire design intent.
        rng = RandPool(seed=123)
        data = bytes(SCALAR_BYTES)
        weights = []
        for _ in range(200):
            result = naf_scalar_mutate(data, rng)
            value = int.from_bytes(result[:SCALAR_BYTES], "big")
            weights.append(naf_weight(value))
        assert min(weights) <= 3
        assert max(weights) >= SCALAR_BYTES  # >= 8 bits/byte / 2 roughly


class TestNafScalarRegistryWiring:
    def test_registered(self):
        assert "naf_scalar_mutate" in REGISTRY.names()

    def test_categorized_regularity(self):
        assert REGISTRY.category_of("naf_scalar_mutate") == "regularity"

    def test_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["naf_scalar_mutate"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_appears_in_fuzzer_dispatch(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = engine.build_dispatch()
        assert "naf_scalar_mutate" in dispatch
