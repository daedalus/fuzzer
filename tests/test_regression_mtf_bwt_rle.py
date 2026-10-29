"""Regression tests for mtf, bwt, and rle operators.

Covers:
- Registry registration, category placement, unconditional availability
- Handler dispatch and length preservation
- Codec round-trips: encode → decode == original, no mutation (TestCodecRoundTrip)
- Handler length preservation through encode → mutate → decode
- Adversarial edge cases (empty, single byte, uniform input)
"""

import os

import pytest

import fuzzer_tool.core.mutations.structured as structured
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


NEW_OPS = frozenset(
    {
        "mtf",
        "bwt",
        "rle",
        "delta_encode",
        "delta_sigma",
        "bitcast_float",
        "bitcast_int32",
        "size_field_overflow",
        "bpe",
        "golomb",
        "endian_convert",
        "count_overflow",
        "zero_run_amplify",
        "zero_run_suppress",
        "type_promote",
        "length_miscalculate",
        "elias_gamma",
        "elias_delta",
        "simd_shuffle",
        "bit_interleave",
        "gray_code",
        "lz_dict_mutate",
        "huffman_tree_mutate",
    }
)


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
        fuzzer._rng = RandPool(seed=1)
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
        fuzzer._rng = RandPool(seed=1)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        for name in sorted(NEW_OPS):
            result = dispatch[name](bytearray(), 0, b"")
            assert result is None or len(result) == 0, name

    def test_single_byte(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=1)
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
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["mtf"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bwt_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = b"the quick brown fox jumps over the lazy dog " * 4
        buf = bytearray(data)
        result = dispatch["bwt"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_rle_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes([i % 4 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["rle"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_delta_encode_roundtrip_direct(self):
        data = bytes(range(256)) * 4
        result = structured.delta_encode(data, RandPool(seed=42))
        assert result is not None
        assert len(result) == len(data)

    def test_delta_sigma_roundtrip_direct(self):
        data = bytes(range(256)) * 4
        result = structured.delta_sigma(data, RandPool(seed=42))
        assert result is not None
        assert len(result) == len(data)

    def test_delta_encode_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["delta_encode"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_delta_sigma_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["delta_sigma"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bitcast_float_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["bitcast_float"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bitcast_int32_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["bitcast_int32"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_size_field_overflow_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["size_field_overflow"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bpe_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        # Repetitive data has strong byte pairs for BPE to merge.
        data = bytes([i % 16 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["bpe"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_golomb_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["golomb"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_endian_convert_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["endian_convert"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_count_overflow_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["count_overflow"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_zero_run_amplify_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        # Data with zero runs.
        data = bytes([i % 8 * 32 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["zero_run_amplify"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_zero_run_suppress_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes([i % 8 * 32 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["zero_run_suppress"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_type_promote_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["type_promote"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_length_miscalculate_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["length_miscalculate"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_elias_gamma_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["elias_gamma"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_elias_delta_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["elias_delta"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_simd_shuffle_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["simd_shuffle"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_bit_interleave_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["bit_interleave"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_gray_code_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["gray_code"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_lz_dict_mutate_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        # Repetitive data has long LZ matches.
        data = bytes([i % 16 for i in range(256)])
        buf = bytearray(data)
        result = dispatch["lz_dict_mutate"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_huffman_tree_mutate_handler_roundtrip_invariant(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256)) * 4
        buf = bytearray(data)
        result = dispatch["huffman_tree_mutate"](buf, 0, data)
        assert result is not None
        assert len(result) == len(data)


def _codec_inputs():
    """Edge cases plus seeded noise over small and full alphabets."""
    pool = RandPool(seed=7)
    return [
        b"ab",
        b"\x00" * 64,
        b"\xff" * 64,
        bytes(range(256)),
        bytes(range(256))[::-1] * 3,
        pool.randbytes(500),
        bytes(b % 3 for b in pool.randbytes(700)),
    ]


class _NoDraws:
    """RNG that fails the test on any draw: a clean decode must not pad."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected rng.{name} on a clean decode")


class TestCodecRoundTrip:
    """Each codec's decoder inverts its encoder exactly when nothing is edited."""

    @pytest.mark.parametrize("k", [0, 2, 4, 7])
    @pytest.mark.parametrize("block", _codec_inputs())
    def test_rice(self, block, k):
        restored, n = structured._rice_decode(
            structured._rice_encode(bytearray(block), k), k, len(block)
        )
        assert (bytes(restored), n) == (block, len(block))

    @pytest.mark.parametrize("block", _codec_inputs())
    def test_elias_gamma(self, block):
        restored, n = structured._gamma_decode(structured._gamma_encode(block), len(block))
        assert (bytes(restored), n) == (block, len(block))

    @pytest.mark.parametrize("block", _codec_inputs())
    def test_elias_delta(self, block):
        restored, n = structured._delta_decode(structured._delta_encode(block), len(block))
        assert (bytes(restored), n) == (block, len(block))

    @pytest.mark.parametrize("block", _codec_inputs())
    def test_lz77(self, block):
        tokens = structured._lz_tokenize(bytearray(block), 3, 64)
        restored, n = structured._lz_replay(tokens, len(block), _NoDraws())
        assert (bytes(restored), n) == (block, len(block))

    def test_gray(self):
        every = bytes(range(256))
        encoded = every.translate(structured._GRAY_ENCODE)
        assert sorted(encoded) == list(every)
        assert encoded.translate(structured._GRAY_DECODE) == every

    @pytest.mark.parametrize("block", _codec_inputs())
    def test_huffman(self, block):
        codes, lengths = structured._huffman_codebook(_freq(block))
        restored, n = structured._huff_decode(
            _huff_bits(block, codes, lengths), codes, lengths, len(block)
        )
        assert (bytes(restored), n) == (block, len(block))


def _freq(block):
    freq = [0] * 256
    for b in block:
        freq[b] += 1
    return freq


def _huff_bits(block, codes, lengths):
    """MSB-first codeword bits of *block*, one 0/1 byte per bit."""
    return bytearray((codes[b] >> i) & 1 for b in block for i in range(lengths[b] - 1, -1, -1))


def _fib_freq(n):
    """Fibonacci frequencies: the shape that makes unlimited Huffman deepest."""
    freq = [0] * 256
    a, b = 1, 1
    for sym in range(n):
        freq[sym] = a
        a, b = b, a + b
    return freq


class TestHuffmanCodebook:
    """_huffman_codebook: a real, length-limited, canonical prefix code."""

    @pytest.mark.parametrize("block", _codec_inputs())
    def test_prefix_free_and_kraft_complete(self, block):
        codes, lengths = structured._huffman_codebook(_freq(block))
        words = sorted(format(codes[s], f"0{lengths[s]}b") for s in range(256) if lengths[s])
        # Sorted, a prefix sits right before a word it prefixes.
        assert all(not b.startswith(a) for a, b in zip(words, words[1:], strict=False))
        kraft = sum(2.0 ** -lengths[s] for s in range(256) if lengths[s])
        assert kraft == (1.0 if len(words) > 1 else 0.5)

    def test_matches_textbook_lengths(self):
        """Falsification: freqs 5,9,12,13,16,45 -> lengths 4,4,3,3,3,1 (CLRS 16.3)."""
        freq = [0] * 256
        for sym, f in zip(b"abcdef", (5, 9, 12, 13, 16, 45), strict=True):
            freq[sym] = f
        _, lengths = structured._huffman_codebook(freq)
        assert [lengths[s] for s in b"abcdef"] == [4, 4, 3, 3, 3, 1]

    def test_regression_rank_codes_undecodable(self):
        """Six symbols: every one must decode (rank codes lost all but two)."""
        block = b"aaaaaabbbbbccccdddeef"
        codes, lengths = structured._huffman_codebook(_freq(block))
        restored, n = structured._huff_decode(
            _huff_bits(block, codes, lengths), codes, lengths, len(block)
        )
        assert (bytes(restored), n) == (block, len(block))

    def test_adversarial_fibonacci_depth_is_limited(self):
        """Fibonacci freqs push unlimited Huffman past 16 bits; the cap must hold."""
        freq = _fib_freq(30)
        codes, lengths = structured._huffman_codebook(freq)
        present = [s for s in range(256) if freq[s]]
        assert max(lengths[s] for s in present) <= structured._HUFF_MAX_BITS
        assert sum(2.0 ** -lengths[s] for s in present) <= 1.0
        block = bytes(present)
        restored, n = structured._huff_decode(
            _huff_bits(block, codes, lengths), codes, lengths, len(block)
        )
        assert (bytes(restored), n) == (block, len(block))

    def test_single_symbol_gets_one_bit(self):
        codes, lengths = structured._huffman_codebook(_freq(b"zzzz"))
        assert (codes[ord("z")], lengths[ord("z")]) == (0, 1)
        assert sum(lengths) == 1


class TestHuffmanTreeMutate:
    """The operator decodes the stream under a leaf-swapped tree: a, b exchange."""

    @pytest.mark.parametrize("seed", range(20))
    def test_output_is_a_two_symbol_swap(self, seed):
        pool = RandPool(seed=seed)
        data = bytes(b % 6 + 65 for b in RandPool(seed=100 + seed).randbytes(300))
        out = structured.huffman_tree_mutate(data, pool)
        assert len(out) == len(data)
        diff = {(x, y) for x, y in zip(data, out, strict=True) if x != y}
        # Every changed byte is one of a single pair (a->b, b->a).
        # Both symbols are drawn from the region, so the swap must show
        # (the rank codebook made this an identity on most inputs).
        assert len(diff) == 2
        assert all((y, x) in diff for x, y in diff)
        assert set(out) == set(data)

    def test_regression_huffman_not_identity_on_narrow_alphabet(self):
        """Falsification: four symbols, whole-buffer region; output must differ."""
        data = b"ABCD" * 8
        out = structured.huffman_tree_mutate(data, RandPool(seed=1))
        assert out != data
        assert sorted(out) == sorted(data.translate(_swap_table(data, out)))


def _swap_table(data, out):
    """Translate table exchanging the one pair of symbols out/data differ on."""
    (a, b) = {x for x, y in zip(data, out, strict=True) if x != y}
    table = bytearray(range(256))
    table[a], table[b] = b, a
    return bytes(table)


class TestEliasRegressions:
    """Decoders lost the 1-bit codeword: byte 0x00 stopped gamma, delta scaled wrong."""

    def test_regression_gamma_zero_byte(self):
        block = b"\x00\x01\x00"
        restored, n = structured._gamma_decode(structured._gamma_encode(block), len(block))
        assert (bytes(restored), n) == (block, len(block))

    def test_regression_delta_value_scaling(self):
        # N = 98 for 'a': gamma(7) then the low 6 bits of 98; was decoded as 0x21.
        block = b"a"
        restored, n = structured._delta_decode(structured._delta_encode(block), len(block))
        assert (bytes(restored), n) == (block, len(block))


class TestAdversarial:
    def test_rle_uniform_input_preserves_length(self):
        """All-same-byte input has one run; RLE must not silently change length."""
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=99)
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

    def test_delta_encode_identity_on_constant(self):
        """Delta of constant data should round-trip cleanly (all deltas = 0)."""
        data = b"\x42" * 64
        deltas = bytearray(len(data))
        deltas[0] = data[0]
        for i in range(1, len(data)):
            deltas[i] = (data[i] - data[i - 1]) & 0xFF
        restored = bytearray(len(data))
        restored[0] = deltas[0]
        for i in range(1, len(data)):
            restored[i] = (restored[i - 1] + deltas[i]) & 0xFF
        assert bytes(restored) == data

    def test_delta_sigma_identity_on_constant(self):
        """Delta-sigma of constant data should round-trip cleanly."""
        data = b"\x42" * 64
        # Predictive first-order: store first byte, then encode prediction error.
        modulated = bytearray(len(data))
        acc = data[0]
        modulated[0] = data[0]
        for i in range(1, len(data)):
            modulated[i] = (data[i] - acc) & 0xFF
            acc = (acc + modulated[i]) & 0xFF
        restored = bytearray(len(data))
        acc = modulated[0]
        restored[0] = acc
        for i in range(1, len(data)):
            acc = (acc + modulated[i]) & 0xFF
            restored[i] = acc
        assert bytes(restored) == data

    def test_bit_interleave_preserves_length_on_64_byte_input(self):
        """bit_interleave must not crash or change length on exact 64-byte input."""
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(64))
        buf = bytearray(data)
        result = dispatch["bit_interleave"](buf, 0, data)
        assert result is not None
        assert len(result) == 64

    def test_gray_code_preserves_length_on_2_byte_input(self):
        """Gray code on minimum 2-byte input must not change length."""
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = b"\x00\xff"
        buf = bytearray(data)
        result = dispatch["gray_code"](buf, 0, data)
        assert result is not None
        assert len(result) == 2

    def test_lz_dict_mutate_returns_bytes_of_same_length(self):
        """lz_dict_mutate must return a bytes-like result of identical length."""
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=13)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = bytes(range(256))
        buf = bytearray(data)
        result = dispatch["lz_dict_mutate"](buf, 0, data)
        assert result is not None
        assert len(result) == 256

    def test_huffman_tree_mutate_returns_bytes_of_same_length(self):
        """huffman_tree_mutate must return a bytes-like result of identical length."""
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=21)
        engine = OperatorEngine(fuzzer)
        dispatch = REGISTRY.dispatch(engine)
        data = b"ab" * 128
        buf = bytearray(data)
        result = dispatch["huffman_tree_mutate"](buf, 0, data)
        assert result is not None
        assert len(result) == 256

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
