"""The byte-entropy input to the honggfuzz energy factor.

``SeedScorer._honggfuzz_factors`` has always had an entropy branch, and
``_hf_entropy_penalties`` has always been declared and displayed, but nothing
in the tree ever produced an ``input_entropy`` value. The parameter defaulted
to the -1.0 "unknown" sentinel, which fails the ``0 <= input_entropy`` guard,
so the branch never executed and the counter never left zero.

Testing the factor in isolation would not have caught that -- the factor
works fine when fed. What was missing was the producer, so the tests that
matter here assert the value actually arrives at the scorer from the fuzzing
loop's own call path.
"""

import math

import pytest

from fuzzer_tool.core.byte_entropy import (
    ENTROPY_SAMPLE_CAP,
    byte_entropy_bits,
    byte_entropy_pct,
)
from fuzzer_tool.core.schedules import (
    ENTROPY_RANDOM_PCT,
    ENTROPY_SPARSE_PCT,
    ENTROPY_STRUCTURED_PCT,
    SeedScorer,
)


class TestByteEntropy:
    def test_empty_input_is_zero(self):
        assert byte_entropy_bits(b"") == 0.0
        assert byte_entropy_pct(b"") == 0.0

    def test_uniform_input_is_zero_bits(self):
        assert byte_entropy_bits(b"\x00" * 1024) == pytest.approx(0.0)

    def test_all_256_bytes_equally_is_eight_bits(self):
        data = bytes(range(256)) * 4
        assert byte_entropy_bits(data) == pytest.approx(8.0, abs=1e-9)
        assert byte_entropy_pct(data) == pytest.approx(100.0, abs=1e-7)

    def test_two_symbols_is_one_bit(self):
        assert byte_entropy_bits(b"AB" * 512) == pytest.approx(1.0, abs=1e-9)

    def test_pct_is_bits_over_eight(self):
        data = bytes(range(64)) * 16
        assert byte_entropy_pct(data) == pytest.approx(
            byte_entropy_bits(data) / 8.0 * 100.0
        )

    def test_cap_bounds_the_scan(self):
        # Uniform head, high-entropy tail beyond the cap: the tail must not
        # be read, so the result stays at the head's entropy.
        data = b"\x00" * ENTROPY_SAMPLE_CAP + bytes(range(256)) * 32
        assert byte_entropy_bits(data) == pytest.approx(0.0)

    def test_matches_naive_reference(self):
        data = b"the quick brown fox jumps over the lazy dog. " * 40
        counts = {}
        for b in data[:ENTROPY_SAMPLE_CAP]:
            counts[b] = counts.get(b, 0) + 1
        total = sum(counts.values())
        expected = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        assert byte_entropy_bits(data) == pytest.approx(expected, abs=1e-9)


class TestThresholdsMatchRealData:
    """The 0-100 scale is only correct if real inputs land in the buckets
    the factor's comments name. Any other normalisation moves these."""

    def test_random_like_input_is_above_random_threshold(self):
        data = bytes(range(256)) * 16
        assert byte_entropy_pct(data) > ENTROPY_RANDOM_PCT

    def test_zeros_are_below_sparse_threshold(self):
        assert byte_entropy_pct(b"\x00" * 4096) < ENTROPY_SPARSE_PCT

    def test_text_is_in_the_structured_band(self):
        data = b"the quick brown fox jumps over the lazy dog. " * 40
        pct = byte_entropy_pct(data)
        assert ENTROPY_SPARSE_PCT <= pct < ENTROPY_STRUCTURED_PCT

    def test_json_is_in_the_structured_band(self):
        data = b'{"name":"value","id":12345,"ok":true}' * 100
        pct = byte_entropy_pct(data)
        assert ENTROPY_SPARSE_PCT <= pct < ENTROPY_STRUCTURED_PCT


class TestFactorRespondsToEntropy:
    def _factor(self, entropy):
        return SeedScorer()._honggfuzz_factors(
            bitmap_size=100, input_size=4096, input_entropy=entropy
        )

    def test_unknown_sentinel_is_inert(self):
        assert self._factor(-1.0) == self._factor(200.0)

    def test_random_and_sparse_are_penalised_relative_to_unknown(self):
        base = self._factor(-1.0)
        assert self._factor(99.0) < base
        assert self._factor(10.0) < base

    def test_structured_is_boosted_relative_to_unknown(self):
        assert self._factor(50.0) > self._factor(-1.0)


class TestEntropyReachesTheScorer:
    """The producer, which is what was actually missing."""

    def test_seed_entropy_pct_matches_helper(self, monkeypatch):
        from fuzzer_tool.services import fuzzer as fz

        seed = b"the quick brown fox " * 50
        meta = {}
        got = fz.Fuzzer._seed_entropy_pct(object(), seed, meta)
        assert got == pytest.approx(byte_entropy_pct(seed))

    def test_value_is_memoised_into_seed_meta(self):
        from fuzzer_tool.services import fuzzer as fz

        seed = b"abcdef" * 100
        meta = {}
        first = fz.Fuzzer._seed_entropy_pct(object(), seed, meta)
        assert meta["input_entropy"] == pytest.approx(first)
        # A poisoned cache value must be returned verbatim, proving the
        # second call reads the cache instead of recomputing.
        meta["input_entropy"] = 42.0
        assert fz.Fuzzer._seed_entropy_pct(object(), seed, meta) == 42.0

    def test_missing_meta_still_returns_a_real_value(self):
        from fuzzer_tool.services import fuzzer as fz

        seed = b"abcdef" * 100
        got = fz.Fuzzer._seed_entropy_pct(object(), seed, None)
        assert got == pytest.approx(byte_entropy_pct(seed))
        assert got >= 0.0

    def test_hf_kwargs_call_site_passes_input_entropy(self):
        """Guards the exact regression: input_size was passed and
        input_entropy was not, for the entire life of the feature."""
        import inspect

        from fuzzer_tool.services import fuzzer as fz

        src = inspect.getsource(fz.Fuzzer)
        idx = src.find("hf_kwargs = dict(")
        assert idx != -1, "hf_kwargs construction not found"
        # Balance parens: the first ')' closes len(seed), not the dict().
        start = src.index("(", idx)
        depth = 0
        for pos in range(start, len(src)):
            if src[pos] == "(":
                depth += 1
            elif src[pos] == ")":
                depth -= 1
                if depth == 0:
                    break
        block = src[start : pos + 1]
        assert "input_size=" in block, "block extraction is wrong"
        assert "input_entropy=" in block, (
            "hf_kwargs no longer passes input_entropy; the honggfuzz entropy "
            "factor is dead again"
        )

    def test_scorer_signature_still_accepts_input_entropy(self):
        import inspect

        assert "input_entropy" in inspect.signature(SeedScorer.score).parameters
