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
    fishers_method,
    kmer_occupancy,
    ks_uniform,
    kuiper_uniform,
    lagged_autocorrelation,
    monobit,
    profile_buffer,
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
