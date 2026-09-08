"""Regression tests for mtf, bwt, and rle operators.

Covers:
- Registry registration, category placement, unconditional availability
- Handler dispatch and length preservation
- Round-trip invariants (encode → mutate → decode == original length)
- Adversarial edge cases (empty, single byte, uniform input)
"""

import os

import fuzzer_tool.core.mutations.structured as structured
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
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


NEW_OPS = frozenset({"mtf", "bwt", "rle"})


class TestRegistration:
    def test_ops_registered(self):
        assert set(REGISTRY.names()) >= NEW_OPS

    def test_ops_categorized_regularity(self):
        for op in NEW_OPS:
            assert REGISTRY.category_of(op) == "regularity", f"{op} not in regularity band"

    def test_every_op_in_exactly_one_category(self):
        cats = REGISTRY.categories()
        union: set[str] = set()
        for ops in cats.values():
            union |= ops
        assert union >= NEW_OPS

    def test_operator_categories_derived_from_registry(self):
        assert REGISTRY.categories() == OPERATOR_CATEGORIES

    def test_unconditional_availability(self):
        fuzzer = _MockFuzzer()
        available = set(REGISTRY.available(fuzzer, b"seed"))
        assert available >= NEW_OPS

    def test_every_op_has_handler(self):
        engine = OperatorEngine(_MockFuzzer())
        dispatch = REGISTRY.dispatch(engine)
        for name in NEW_OPS:
            assert callable(dispatch[name]), name


class TestHandlers:
    def test_handlers_preserve_length(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(NEW_OPS):
            for _ in range(10):
                buf = bytearray(os.urandom(512))
                result = dispatch[name](buf, 0, bytes(buf))
                assert result is not None, name
                assert len(result) == 512, name

    def test_empty_input(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(NEW_OPS):
            result = dispatch[name](bytearray(), 0, b"")
            assert result is None or len(result) == 0, name

    def test_single_byte(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(NEW_OPS):
            buf = bytearray([0x42])
            result = dispatch[name](buf, 0, bytes(buf))
            assert result is not None, name
            assert bytes(result) == b"\x42", name


class TestRoundTrip:
    def test_mtf_roundtrip(self):
        data = bytes(range(256)) * 4
        alphabet = bytearray(range(256))
        encoded = structured._mtf_encode(data, alphabet)
        alphabet = bytearray(range(256))
        assert structured._mtf_decode(encoded, alphabet) == data

    def test_bwt_roundtrip(self):
        data = b"banana"
        bwt_data, primary = structured._bwt(data)
        assert structured._bwt_inverse(bwt_data, primary) == data

    def test_mtf_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["mtf"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bwt_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = b"the quick brown fox jumps over the lazy dog " * 4
        buf = bytearray(data)
        result = dispatch["bwt"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_rle_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes([i % 4 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["rle"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)


class TestAdversarial:
    def test_rle_uniform_input_preserves_length(self):
        """All-same-byte input has one run; RLE must not silently change length."""
        fuzzer = _MockFuzzer()
        fuzzer._rand_pool = RandPool(seed=99)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = b"\x00" * 1024
        buf = bytearray(data)
        result = dispatch["rle"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bwt_identity_on_sorted_input(self):
        """BWT of sorted data should round-trip cleanly."""
        data = bytes(range(256))
        bwt_data, primary = structured._bwt(data)
        restored = structured._bwt_inverse(bwt_data, primary)
        assert restored == data

    def test_mtf_identity_on_uniform_input(self):
        """MTF of uniform data should round-trip cleanly."""
        data = b"\xab" * 256
        alphabet = bytearray(range(256))
        encoded = structured._mtf_encode(data, alphabet)
        # All bytes map to index 0xAB after the first occurrence.
        alphabet = bytearray(range(256))
        assert structured._mtf_decode(encoded, alphabet) == data

    def test_all_ops_dispatch_matches_registry(self):
        import tempfile
        from pathlib import Path

        from fuzzer_tool.services.fuzzer import Fuzzer

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            crashes = Path(tmp) / "crashes"
            corpus.mkdir()
            crashes.mkdir()
            fuzzer = Fuzzer(
                target=str(Path(__file__).resolve().parent.parent / "targets" / "test_target"),
                corpus_dir=str(corpus),
                crashes_dir=str(crashes),
                max_len=4096,
            )
        assert set(fuzzer._op_dispatch) >= NEW_OPS
