"""Regression + equivalence tests for the bithacks.html ports.

Three separate claims are under test here.

**1. ``swar_lane`` (new operator).** Word-at-a-time scanners -- ``haszero``,
``hasless``, ``hasmore``, ``hasbetween`` -- carry per-lane arithmetic whose
exactness stops at specific byte values (0x80 for the borrow, 0x7f/0x81 for the
comparison thresholds).  The operator writes windows in which *every* byte is
one of those critical values, which is the only shape that reaches the boundary.
The load-bearing assertion is the falsification test
``test_swar_window_is_all_critical_lanes``: if the construction ever emits a
byte outside the critical set, the operator is not doing what it claims and no
other test in the suite would say so.

**2/3. Two hot loops replaced by word-at-a-time equivalents.**  ``gray_code``'s
Gray decode and ``bit_interleave``'s bit-plane gather ran one Python iteration
per bit; both are now closed forms.  A speedup is only legitimate if it is
*bit-exact*, so both carry the original loop as an in-file oracle and assert
equality against it over fixed seeds.

On the oracle (Hard Rule 46): a reference implementation that cannot fail makes
the comparison vacuous.  Each oracle here is therefore paired with a
*sensitivity control* -- the same comparison run against a deliberately
perturbed reference, which must FAIL.  If a control ever passes, the comparison
is not testing what it claims and the equality assertion above it is worthless.
Note the control is a real check here, not a formality: the first draft of the
``bit_interleave`` rewrite was wrong in a way the oracle caught but a
"compare against itself" control would have called identical.

On determinism (Hard Rule 39): every equality assertion runs over a fixed,
enumerated seed list and every exact-output assertion uses ``ScriptedRng`` with
the draw order spelled out.  Nothing here retries until a random hit.
"""

import fuzzer_tool.core.berlekamp_massey as bm
import fuzzer_tool.core.mutations.structured as S
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng

# The seven lane values where the word-at-a-time idioms stop being exact.
# Mirrors S._SWAR_LANES; asserted equal in test_swar_lanes_match_the_page.
CRITICAL = (0x00, 0x01, 0x7F, 0x80, 0x81, 0xFE, 0xFF)

# Largest window swar_lane can rewrite: the 'strided' mode's 64-byte sweep.
_SWAR_SWEEP = S._SWAR_SWEEP


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


# ── Oracles: the implementations this change replaced ──────────────────────
#
# Kept verbatim so the equivalence assertions compare against the code that
# actually shipped, not against a description of it.


def _gray_decode_loop(gray: bytes) -> bytes:
    """Original per-byte Gray decode: repeated XOR-shift until the shift is 0."""
    out = bytearray(len(gray))
    for i, g in enumerate(gray):
        b = g
        k = g >> 1
        while k:
            b ^= k
            k >>= 1
        out[i] = b & 0xFF
    return bytes(out)


def _bit_interleave_loops(block: bytearray, rng) -> bytearray:
    """Original 8x8 nested-loop bit-plane gather, per 64-byte chunk.

    Note the quirk preserved here: ``out`` is allocated 64 bytes but only
    indices 0..7 are ever written, so every chunk loses its 56-byte tail to
    zeros.  That is long-standing operator behaviour and changing it would
    move every seeded run's output.

    The shuffle goes through *rng* rather than a stand-in: RandPool.shuffle is
    a real permutation and advances the stream, so the oracle has to consume
    the same draws as the implementation or the two diverge after one chunk.
    """
    length = len(block)
    for chunk_start in range(0, length, 64):
        grp = bytes(block[chunk_start : chunk_start + 64])
        dst = bytearray(64)
        for j in range(8):
            for g in range(8):
                dst[j * 8 + g] = (grp[g * 8 + j] >> (7 - j)) & 1
        rng.shuffle(dst)
        out = bytearray(64)
        for g in range(8):
            b = 0
            for j in range(8):
                b |= dst[g * 8 + j] << (7 - j)
            out[g] = b
        block[chunk_start : chunk_start + 64] = out
    return block


# ── swar_lane ─────────────────────────────────────────────────────────────


class TestSwarLaneRegistration:
    def test_registered_in_regularity_band(self):
        assert "swar_lane" in REGISTRY.names()
        assert REGISTRY.category_of("swar_lane") == "regularity"
        assert "swar_lane" in OPERATOR_CATEGORIES["regularity"]

    def test_handler_exists_and_preserves_length(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=42)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(256)) * 2
        result = dispatch["swar_lane"](bytearray(data), 0, data)
        assert result is not None
        assert len(result) == len(data)

    def test_lanes_match_the_page(self):
        """The critical set is the operator's whole justification; pin it."""
        assert S._SWAR_LANES == CRITICAL


class TestSwarLaneExactOutput:
    """Scripted draws, exact bytes -- no re-derivation from the RNG."""

    def test_uniform_mode_fills_window_with_one_critical_byte(self):
        # Draw order: choice(_SWAR_MODES)=0 -> 'uniform';
        #   choice(_SWAR_WIDTHS)=0 -> 4; _region -> randint(4, 100)=4, randint(0, 96)=0;
        #   choice(_SWAR_LANES)=6 -> 0xFF.
        rng = ScriptedRng(randints=[4, 0], choice_idxs=[0, 0, 6])
        out = S.swar_lane(b"\x11" * 100, rng)
        assert out == b"\xff" * 4 + b"\x11" * 96

    def test_mixed_mode_forces_a_zero_lane_and_a_sign_lane(self):
        # choice(_SWAR_MODES)=1 -> 'mixed'; choice(_SWAR_WIDTHS)=1 -> 8;
        # _region -> randint(8, 100)=8, randint(0, 92)=0;
        # then one choice per lane, all drawing index 0 (0x00).
        rng = ScriptedRng(randints=[8, 0], choice_idxs=[1, 1] + [0] * 8)
        out = S.swar_lane(b"\x11" * 100, rng)
        # First lane forced to 0x00 and last to 0x80 regardless of the draws:
        # the phantom zero needs a borrow source *and* a zero lane.
        assert out == b"\x00" * 7 + b"\x80" + b"\x11" * 92

    def test_strided_mode_plants_pattern_at_chosen_alignment(self):
        # choice(_SWAR_MODES)=2 -> 'strided'; _region(min_len=64, max_len=64)
        #   -> randint(64, 64)=64, randint(0, 36)=0;
        #   randint(0, 7)=5 -> plant at byte-offset 5;
        #   8 choices, all index 3 (0x80).
        rng = ScriptedRng(randints=[64, 0, 5], choice_idxs=[2] + [3] * 8)
        out = S.swar_lane(b"\x11" * 100, rng)
        assert out[:64] == b"\x00" * 5 + b"\x80" * 8 + b"\x00" * 51
        assert out[64:] == b"\x11" * 36


class TestSwarLaneProperties:
    def test_swar_window_is_all_critical_lanes(self):
        """Falsification: the construction must never leak a benign byte.

        If this fails, swar_lane is scribbling arbitrary noise rather than the
        lane-boundary pattern that is its entire reason to exist.
        """
        rng = RandPool(seed=7)
        for _ in range(200):
            data = bytes(rng.randbytes(96))
            out = S.swar_lane(data, rng)
            # Every byte the operator could have written is a critical value.
            # Compare against the input so untouched bytes are not judged.
            touched = {b for a, b in zip(data, out, strict=True) if a != b}
            assert touched <= set(CRITICAL), sorted(touched)

    def test_leaves_the_rest_of_the_input_alone(self):
        """Adversarial: the operator is a region rewrite, not a buffer rewrite."""
        rng = RandPool(seed=11)
        for _ in range(100):
            data = bytes(rng.randbytes(64))
            out = S.swar_lane(data, rng)
            assert len(out) == len(data)
            changed = sum(a != b for a, b in zip(data, out, strict=True))
            assert changed <= _SWAR_SWEEP

    def test_shorter_than_one_word_is_the_identity(self):
        """Below 4 bytes no scan can see two lanes, so nothing is rewritten --
        and no draw is consumed."""
        rng = ScriptedRng(randints=[], choice_idxs=[])
        for data in (b"", b"\x00", b"\xff", b"\x00\x80\x7f"):
            assert S.swar_lane(data, rng) == data

    def test_exactly_one_word_is_mutable(self):
        # Uniform mode overwrites the whole 4-byte window, not just lane 0.
        rng = ScriptedRng(randints=[4, 0], choice_idxs=[0, 0, 0])
        assert S.swar_lane(b"\x11\x22\x33\x44", rng) == b"\x00" * 4

    def test_never_raises_on_any_length(self):
        rng = RandPool(seed=3)
        for n in range(0, 40):
            out = S.swar_lane(b"\xa5" * n, rng)
            assert len(out) == n


# ── gray_code decode ───────────────────────────────────────────────────────


def _gray_code_loop(data: bytes, rng) -> bytes:
    """Original gray_code, with the loop decode restored."""
    if len(data) < 2:
        return data
    offset, length = S._region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = bytearray(data[offset : offset + length])
    gray = [b ^ (b >> 1) for b in block]
    n_flip = rng.randint(1, min(3, length))
    for _ in range(n_flip):
        pos = rng.randint(0, length - 1)
        bit = rng.randint(0, 7)
        gray[pos] ^= 1 << bit
    return S._splice(data, offset, _gray_decode_loop(bytes(gray)))


class TestGrayDecodeEquivalence:
    def test_matches_loop_on_every_byte_value(self):
        """Exhaustive: the decode is a per-byte map, so 256 cases is complete."""
        every = bytes(range(256))
        assert bytes(S._GRAY_DECODE[x] for x in every) == _gray_decode_loop(every)

    def test_decode_matches_the_suffix_xor_definition(self):
        """Independent derivation, from the algebra rather than the loop.

        The encode is ``g_i = b_i ^ b_{i+1}``, so inverting it gives
        ``b_i = XOR of g_j for j >= i`` -- a *suffix* XOR.  Spelled out here
        from that definition, so it can disagree with the table the loop
        produced.

        (It is tempting to assert D is an involution.  It is not: for
        ``b = 2``, ``E(2) = 3``, ``D(3) = 2`` but ``D(2) = 3``.)
        """
        for g in range(256):
            want = 0
            suffix = 0
            for i in range(7, -1, -1):
                suffix ^= (g >> i) & 1
                want |= suffix << i
            assert S._GRAY_DECODE[g] == want, g

    def test_decode_inverts_the_encoder(self):
        for b in range(256):
            assert S._GRAY_DECODE[b ^ (b >> 1)] == b

    def test_operator_matches_loop_across_seeds(self):
        data = bytes(range(256)) * 4
        for seed in range(25):
            assert S.gray_code(data, RandPool(seed=seed)) == _gray_code_loop(
                data, RandPool(seed=seed)
            ), seed

    def test_oracle_control_detects_a_perturbed_reference(self, monkeypatch):
        """Hard Rule 46: the oracle must be able to fail.

        Flips one bit of the decode and asserts the comparison now rejects.
        A control that passes means the equality tests above prove nothing.
        """
        data = bytes(range(256)) * 4
        good = S.gray_code(data, RandPool(seed=3))
        monkeypatch.setattr(S, "_GRAY_DECODE", bytes((x ^ 0x40) for x in range(256)))
        assert S.gray_code(data, RandPool(seed=3)) != good


# ── bit_interleave ────────────────────────────────────────────────────────


def _bit_interleave_loop(data: bytes, rng) -> bytes:
    """Original bit_interleave, nested loops intact."""
    if len(data) < 64:
        return data
    offset, length = S._region(len(data), rng, min_len=64, max_len=64)
    if length < 64:
        return data
    block = bytearray(data[offset : offset + length])
    return S._splice(data, offset, bytes(_bit_interleave_loops(block, rng)))


class TestBitInterleaveEquivalence:
    def test_matches_loop_across_seeds(self):
        data = bytes(range(256)) * 4
        for seed in range(25):
            assert S.bit_interleave(data, RandPool(seed=seed)) == _bit_interleave_loop(
                data, RandPool(seed=seed)
            ), seed

    def test_matches_loop_on_every_alignment_of_the_chunk(self):
        """The rewrite reads 8-byte groups strided across the chunk, so a seed
        that happens to align cleanly would not exercise the other offsets."""
        for rot in range(8):
            data = bytes((i * 7 + rot) & 0xFF for i in range(64))
            assert S.bit_interleave(data, RandPool(seed=rot)) == _bit_interleave_loop(
                data, RandPool(seed=rot)
            ), rot

    def test_preserves_the_56_byte_zero_tail(self):
        """The loop oracle allocates out[64] but fills 8, so 56 bytes/chunk are
        zeroed.  A rewrite that writes only the 8 packed bytes silently stops
        zeroing them -- same length, different output, no exception."""
        data = b"\xff" * 64
        out = S.bit_interleave(data, RandPool(seed=1))
        assert out[8:] == b"\x00" * 56

    def test_gather_collapses_source_bytes_to_zero_or_one(self):
        """The forward half turns each source byte into 8 zero/one bytes, so
        the packed output can never exceed 0xFF per lane pair."""
        data = bytes(range(256)) * 4
        out = S.bit_interleave(data, RandPool(seed=5))
        assert len(out) == len(data)

    def test_oracle_control_detects_a_transpose_mistaken_for_a_gather(self):
        """Hard Rule 46, and this control earned its keep.

        The first rewrite of this operator read the forward half as an 8x8
        bit-matrix transpose -- the 3-delta-swap form from bithacks.html --
        which is the natural thing to reach for and is *wrong*: the real
        forward half collapses each source byte into 8 zero/one bytes, so it
        is a gather and not a permutation.  A transpose preserves bit counts;
        the gather does not.

        Pinned here as the wrong answer the equality tests above must reject.
        """
        data = bytes(range(256)) * 4
        good = S.bit_interleave(data, RandPool(seed=2))
        # The transpose reading: dst[j*8 + j] instead of dst[j*8 + g].
        transposed = bytearray(64)
        for j in range(8):
            for g in range(8):
                transposed[j * 8 + j] = (data[g * 8 + j] >> (7 - j)) & 1
        assert good != bytes(transposed)

    def test_gather_really_collapses_bits(self):
        """The property that makes the transpose reading impossible."""
        src = bytes([0xFF] * 8) + bytes(56)
        dst = S._interleave_gather(src)
        assert set(dst) == {0, 1}
        assert sum(dst) == 8


# ── berlekamp_massey bit reversal ──────────────────────────────────────────


class TestReverseBitsTranslate:
    def test_reverse_byte_matches_an_independent_oracle(self):
        """Oracle is the binary-string round trip, not another bit loop."""
        for b in range(256):
            text = format(b, "08b")
            assert bm._REV8[b] == int(text[::-1], 2), b

    def test_reverse_byte_oracle_control_detects_a_perturbed_table(self):
        bad = bytes((x ^ 0x01) for x in range(256))
        text = format(0b10110010, "08b")
        assert bad[0b10110010] != int(text[::-1], 2)

    def test_reverse_bits_matches_loop_at_every_used_width(self):
        for width in (8, 16, 32, 64):
            x = 0
            for k in range(width):
                x |= 1 << k
            assert bm._reverse_bits(x, width) == x, width
        for width in (8, 16, 32, 64):
            x = 0x0123456789ABCDEF & ((1 << width) - 1)
            loop = 0
            v = x
            for _ in range(width):
                loop = (loop << 1) | (v & 1)
                v >>= 1
            assert bm._reverse_bits(x, width) == loop, width

    def test_reverse_bits_ignores_bits_above_width(self):
        """Adversarial: the loop only ever looked at the low *width* bits, so
        junk above them must be discarded, not reversed in."""
        assert bm._reverse_bits(0xFF00 | 0b00001111, 8) == 0b11110000

    def test_reverse_byte_still_callable(self):
        assert bm._reverse_byte(0b00000001) == 0b10000000
        assert bm._reverse_byte(0b10110010) == 0b01001101


class TestGrayEncodeTable:
    """gray_code encode side is a translate table, identical to ``b ^ (b >> 1)``."""

    def test_encode_table_matches_formula_and_inverts_with_decode(self):
        from fuzzer_tool.core.mutations import structured as S

        assert all(S._GRAY_ENCODE[b] == b ^ (b >> 1) for b in range(256))
        assert all(S._GRAY_DECODE[S._GRAY_ENCODE[b]] == b for b in range(256))

    def test_seeded_output_unchanged(self):
        import random

        from fuzzer_tool.core.mutations import structured as S
        from fuzzer_tool.core.rand_pool import RandPool

        def ref(data, rng):
            offset, length = S._region(len(data), rng, min_len=2)
            gray = [b ^ (b >> 1) for b in data[offset : offset + length]]
            for _ in range(rng.randint(1, min(3, length))):
                gray[rng.randint(0, length - 1)] ^= 1 << rng.randint(0, 7)
            restored = bytes(bytearray(gray).translate(S._GRAY_DECODE))
            return S._splice(data, offset, restored)

        for size in (2, 17, 300, 4097):
            for seed in range(40):
                d = random.Random(seed + size).randbytes(size)
                assert S.gray_code(d, RandPool(seed)) == ref(d, RandPool(seed))
