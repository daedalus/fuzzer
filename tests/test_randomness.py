"""Tests for randomness.py — dieharder-derived battery and region profiling.

The important tests here are the *calibration* ones: a statistical test whose
p-values are not uniform under the null is worse than no test at all, because
it silently poisons any aggregate built on top of it.  These run the battery
against os.urandom and KS the resulting p-values.
"""

import gzip
import os
import struct

import numpy as np
import pytest

from fuzzer_tool.core.randomness import (
    binary_matrix_rank,
    birthday_spacings,
    byte_chisq,
    corpus_invariants,
    fishers_method,
    invariant_mask,
    kmer_occupancy,
    ks_uniform,
    kuiper_uniform,
    lagged_autocorrelation,
    monobit,
    permutation_test,
    profile_buffer,
    repeat_test,
    runs_test,
    serial_test,
    uniformity_report,
)

WINDOW = 16384


def _lfsr(n: int, poly: int = 0x80000057, seed: int = 0xACE1ACE1) -> bytes:
    out, s = bytearray(), seed
    while len(out) < n:
        b = 0
        for _ in range(8):
            bit = s & 1
            s >>= 1
            if bit:
                s ^= poly
            b = (b << 1) | bit
        out.append(b)
    return bytes(out)


def _index_table(n: int) -> bytes:
    rng = np.random.default_rng(3)
    out, off = bytearray(), 0x1000
    while len(out) < n:
        sz = int(rng.integers(64, 4096))
        out += struct.pack("<IIHH", off, sz, int(rng.integers(0, 8)), 0)
        off += sz
    return bytes(out[:n])


class TestNullCalibration:
    """Under H0 every test must emit uniform p-values."""

    # birthday_spacings is deliberately excluded: its statistic is a small
    # Poisson count, so its p-values are discrete and cannot be uniform even
    # with the mid-p correction.  It is usable as a single rejection signal
    # but must never be fed into ks_uniform/kuiper_uniform.
    @pytest.mark.parametrize(
        "fn",
        [
            monobit,
            runs_test,
            byte_chisq,
            lambda d: serial_test(d, 8),
            lambda d: binary_matrix_rank(d, 32, 32),
            lambda d: binary_matrix_rank(d, 6, 8),
            kmer_occupancy,
            lambda d: lagged_autocorrelation(d, (1,))[1],
        ],
    )
    def test_pvalues_uniform_on_urandom(self, fn):
        ps = np.array([fn(os.urandom(WINDOW)) for _ in range(120)])
        rep = uniformity_report(ps)
        assert rep["ks"] > 0.001, f"p-values not uniform: {rep}"
        assert 0.3 < ps.mean() < 0.7

    def test_rejection_rate_near_alpha(self):
        ps = np.array([monobit(os.urandom(WINDOW)) for _ in range(400)])
        assert (ps < 0.01).mean() < 0.05


class TestDiscrimination:
    def test_rank_catches_lfsr(self):
        """The GF(2) rank test is the only one in the battery that sees an LFSR."""
        data = _lfsr(WINDOW * 4)
        assert binary_matrix_rank(data, 32, 32) < 0.01
        assert monobit(data) > 0.01  # an LFSR is balanced; monobit is blind to it

    def test_occupancy_catches_restricted_alphabet(self):
        text = (b"the quick brown fox jumps over the lazy dog " * 400)[:WINDOW]
        assert kmer_occupancy(text) < 0.01
        assert kmer_occupancy(os.urandom(WINDOW)) > 0.01

    def test_byte_chisq_catches_padding(self):
        assert byte_chisq(b"\x00" * WINDOW) < 0.01

    def test_lag_catches_fixed_stride(self):
        rec = (struct.pack("<I", 0x41414141) + b"\x00" * 12) * (WINDOW // 16)
        assert min(lagged_autocorrelation(rec).values()) < 0.01


class TestRegionProfile:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (os.urandom(WINDOW), "incompressible"),
            (gzip.compress(os.urandom(WINDOW * 2)), "incompressible"),
            ((b"the quick brown fox jumps over the lazy dog " * 500)[:WINDOW], "textual"),
            (b"\x00" * WINDOW, "repetitive"),
            (_index_table(WINDOW), "tabular"),
        ],
    )
    def test_labels(self, data, expected):
        profs = profile_buffer(data, window=4096)
        labels = [p.label for p in profs]
        assert expected in labels, f"got {labels}, wanted {expected}"

    def test_weights_downrank_incompressible(self):
        rand = profile_buffer(os.urandom(WINDOW), 4096)[0]
        table = profile_buffer(_index_table(WINDOW), 4096)[0]
        assert rand.mutation_weight() < table.mutation_weight()

    def test_short_input_yields_nothing(self):
        assert profile_buffer(b"abc", 4096) == []


class TestAggregation:
    def test_uniform_batch_passes(self):
        rng = np.random.default_rng(7)
        rep = uniformity_report(rng.uniform(size=200))
        assert rep["ks"] > 0.05 and rep["fisher"] > 0.05

    def test_biased_batch_rejected(self):
        rng = np.random.default_rng(7)
        rep = uniformity_report(rng.uniform(size=200) ** 1.35)
        assert rep["ks"] < 0.05 and rep["fisher"] < 0.05

    def test_fisher_is_the_most_powerful_aggregator(self):
        """Regression on the measurement recorded in kuiper_uniform's docstring.

        Fisher dominated KS and Kuiper on every alternative tried, which is
        why uniformity_report should be read Fisher-first.
        """
        rng = np.random.default_rng(0)
        n = 150
        rej = {"ks": 0, "kuiper": 0, "fisher": 0}
        for _ in range(n):
            ps = rng.uniform(size=200) ** 1.35
            rej["ks"] += ks_uniform(ps) < 0.05
            rej["kuiper"] += kuiper_uniform(ps) < 0.05
            rej["fisher"] += fishers_method(ps) < 0.05
        assert rej["fisher"] >= rej["ks"] >= rej["kuiper"]

    def test_aggregators_hold_their_size_under_null(self):
        rng = np.random.default_rng(1)
        n = 300
        rej = sum(fishers_method(rng.uniform(size=50)) < 0.05 for _ in range(n))
        assert rej / n < 0.12

    def test_degenerate_inputs(self):
        assert ks_uniform([0.5]) == 1.0
        assert kuiper_uniform([0.5, 0.5]) == 1.0
        assert birthday_spacings(b"") == 1.0
        assert monobit(b"ab") == 1.0


class TestCorpusInvariants:
    @staticmethod
    def _png_corpus(k=200):
        """Fixed magic + IHDR tag, varying dimensions and payload."""
        rng = np.random.default_rng(5)
        out = []
        for _ in range(k):
            w, h = int(rng.integers(1, 4096)), int(rng.integers(1, 4096))
            out.append(
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", w, h)
                + bytes([8, int(rng.integers(0, 7)), 0, 0, 0])
                + os.urandom(64)
            )
        return out

    def test_recovers_format_header(self):
        inv = corpus_invariants(self._png_corpus())
        assert set(range(16)) <= set(inv.fixed_offsets)
        header = bytes(self._png_corpus(20)[0][i] for i in range(16))
        assert header.startswith(b"\x89PNG\r\n\x1a\n")

    def test_finds_sub_byte_fixed_bits(self):
        """Dimensions below 4096 leave the top nibble of byte 18 locked."""
        inv = corpus_invariants(self._png_corpus())
        partial = dict(inv.partial_offsets)
        assert 18 in partial and partial[18] == 0xF0

    def test_undersampled_fields_look_invariant(self):
        """Documents the failure mode: the mask cannot tell a constant from an
        unexercised field, so callers must treat it as a prior."""
        inv = corpus_invariants(self._png_corpus())
        # bytes 16-17 are the high half of a 32-bit width that never exceeded 4096
        assert inv.is_structural(16) and inv.is_structural(17)

    def test_claims_nothing_below_min_samples(self):
        inv = corpus_invariants(self._png_corpus(4), min_samples=16)
        assert inv.fixed_offsets == [] and inv.locked_bit_ratio == 0.0

    def test_varied_corpus_locks_nothing(self):
        inv = corpus_invariants([os.urandom(256) for _ in range(64)])
        assert inv.locked_bit_ratio < 0.02

    def test_degenerate(self):
        assert invariant_mask([]) == b""
        assert corpus_invariants([b"abc"]).mask == bytes(3)


class TestSequenceDiagnostics:
    K = 137

    def test_null_calibration(self):
        rng = np.random.default_rng(4)
        for fn in (
            lambda d: permutation_test(d),
            lambda d: repeat_test(d, self.K),
        ):
            ps = np.array([fn(rng.integers(0, self.K, size=200000)) for _ in range(40)])
            assert 0.3 < ps.mean() < 0.7
            assert (ps < 0.01).mean() < 0.1

    def test_permutation_catches_ordering_marginals_miss(self):
        rng = np.random.default_rng(2)
        d = np.sort(rng.integers(0, self.K, size=600000).reshape(-1, 5), axis=1).ravel()
        counts = np.bincount(d, minlength=self.K)
        exp = d.size / self.K
        from fuzzer_tool.core.randomness import chisq_sf

        assert chisq_sf(float(np.sum((counts - exp) ** 2) / exp), self.K - 1) > 0.05
        assert permutation_test(d) < 0.01

    def test_repeat_catches_stickiness_permutation_misses(self):
        rng = np.random.default_rng(2)
        d = rng.integers(0, self.K, size=400000)
        keep = np.flatnonzero(rng.random(d.size) < 0.04)[1:]
        for i in keep:
            d[i] = d[i - 1]
        assert repeat_test(d, self.K) < 0.01
        assert permutation_test(d) > 0.05  # structurally blind to ties

    def test_rand_pool_is_clean(self):
        from fuzzer_tool.core.rand_pool import RandPool

        pool = RandPool()
        d = np.array([pool.randrange(self.K) for _ in range(200000)])
        assert permutation_test(d) > 0.01
        assert repeat_test(d, self.K) > 0.01

    def test_guards(self):
        rng = np.random.default_rng(1)
        assert permutation_test(rng.integers(0, 137, size=100)) == 1.0
        assert repeat_test(rng.integers(0, 137, size=10)) == 1.0
        assert repeat_test(np.zeros(5000, dtype=int)) == 1.0
        with pytest.raises(ValueError):
            permutation_test(rng.integers(0, 137, size=600000), k=9)
