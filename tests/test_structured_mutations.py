"""Tests for structured.py — the constructive inverses of the dieharder battery.

The load-bearing tests here are the *round trips*: for every construction with
a matching detector in ``core/randomness.py``, build a buffer with the operator
and assert the detector fires on it and does not fire on the noise it replaced.
A construction that fails to move its own statistic is worthless as a mutation
operator, and nothing else in the fuzz loop would notice — the buffer would
just look like slightly odd noise.

On seeding: those round trips are hypothesis tests, so run against fresh
entropy each one carries its own false-positive rate. At p < 0.01 that is
roughly one spurious failure per hundred runs *per assertion*, which on a
suite that gates every commit is a flaky test rather than a signal. Every
threshold assertion below therefore draws from a seeded RandPool and a seeded
noise source, making it a statement about a fixed, checked-in sample. The
properties that must hold for *any* draw — length preservation, no exceptions,
alphabet bounds — keep using fresh entropy, because there the whole point is
that no draw is special.
"""

import math
import os
import random
import struct

import numpy as np
import pytest

from fuzzer_tool.core import randomness as R
from fuzzer_tool.core.mutations import structured as S
from fuzzer_tool.core.rand_pool import RandPool

# Fixed seed for every statistical assertion in this file.
SEED = 20240609

# Every operator in the module. Kept as an explicit list rather than derived
# from dir() so a renamed or removed construction fails loudly here instead of
# silently dropping out of the property tests.
ALL_OPS = (
    S.fibonacci_pairs,
    S.monotone_fill,
    S.de_bruijn_fill,
    S.kmer_starve,
    S.rank_deficient,
    S.perm_lock,
    S.lag_correlate,
    S.spectral_peak,
    S.birthday_collide,
    S.degenerate_geometry,
    S.float_squeeze,
    S.popcount_lock,
)


@pytest.fixture
def rp():
    """Unseeded pool, for properties that must hold on any draw."""
    return RandPool()


@pytest.fixture
def seeded():
    """Deterministic RandPool for the statistical threshold assertions.

    RandPool refills from the numpy global generator, so seeding numpy before
    constructing it pins every draw the operators make.
    """
    np.random.seed(SEED)
    return RandPool()


@pytest.fixture
def noise():
    """Deterministic stand-in for os.urandom in threshold assertions."""
    return random.Random(SEED).randbytes


# ── Properties every construction must hold ────────────────────────────


class TestOperatorProperties:
    @pytest.mark.parametrize("op", ALL_OPS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 15, 16, 64, 300, 4097])
    def test_length_preserved(self, op, size, rp):
        """Length preservation is what lets these skip the max_len dance."""
        data = os.urandom(size)
        for _ in range(20):
            out = op(data, rng=rp)
            assert isinstance(out, bytes)
            assert len(out) == size

    @pytest.mark.parametrize("op", ALL_OPS, ids=lambda f: f.__name__)
    def test_accepts_stdlib_random(self, op):
        """The module must stay on the RandPool/stdlib-random intersection.

        Every other module under core/mutations/ works with either. An
        operator reaching for a RandPool-only method (weighted_choice,
        randint_list) would break the moment it is called from a test or a
        tool that passes a plain Random.
        """
        data = os.urandom(512)
        for _ in range(20):
            assert len(op(data, rng=random.Random(1234))) == 512

    @pytest.mark.parametrize("op", ALL_OPS, ids=lambda f: f.__name__)
    def test_empty_input_is_a_noop(self, op, rp):
        assert op(b"", rng=rp) == b""


# ── de Bruijn generator: the one real algorithm in the module ──────────


class TestDeBruijn:
    @pytest.mark.parametrize(("k", "n"), [(2, 2), (2, 3), (2, 8), (3, 3), (4, 4), (16, 2)])
    def test_is_a_de_bruijn_sequence(self, k, n):
        """Length k**n, and every cyclic n-window distinct.

        Derived independently of the generator: the window set is rebuilt from
        the output rather than compared against a stored table.
        """
        seq = S._de_bruijn_symbols(k, n)
        assert len(seq) == k**n
        cyclic = seq + seq[: n - 1]
        windows = {tuple(cyclic[i : i + n]) for i in range(len(seq))}
        assert len(windows) == k**n

    def test_symbols_span_the_alphabet(self):
        assert set(S._de_bruijn_symbols(5, 3)) == set(range(5))

    def test_cached_result_is_immutable(self):
        """The cache hands the same object to every caller.

        ``lru_cache`` over a mutable sequence would let the first operator to
        write through the result corrupt every later one — exactly the silent
        class of bug the bytes return type exists to prevent.
        """
        first = S.de_bruijn_bytes(4, 4)
        assert isinstance(first, bytes)
        assert S.de_bruijn_bytes(4, 4) is first


# ── Round trips against core/randomness.py detectors ───────────────────


class TestKmerSaturation:
    def test_de_bruijn_fill_leaves_no_missing_kmers(self, seeded, noise):
        """OPSO/bitstream inverse: occupancy lands in the extreme tail."""
        base = noise(S.MAX_REGION)
        for _ in range(20):
            assert R.kmer_occupancy(S.de_bruijn_fill(base, rng=seeded)) < 0.01

    def test_noise_passes_the_same_detector(self, noise):
        assert R.kmer_occupancy(noise(S.MAX_REGION)) > 0.01

    def test_kmer_starve_is_the_opposite_tail(self, seeded, noise):
        """Most draws, not every draw: the rewritten region is as small as
        eight bytes, and a starved run that short leaves the surrounding noise
        dominating the occupancy count. Measured 39/40 over the seeded sample.
        """
        base = noise(S.MAX_REGION)
        hits = sum(R.kmer_occupancy(S.kmer_starve(base, rng=seeded)) < 0.01 for _ in range(40))
        assert hits >= 34

    def test_starve_uses_a_tiny_alphabet(self, rp):
        """At most four symbols in the region, plus the untouched zeros."""
        for _ in range(30):
            assert len(set(S.kmer_starve(bytes(2048), rng=rp))) <= 5


class TestRankDeficient:
    @pytest.mark.parametrize(
        ("shape", "size", "dtype", "cols"),
        [((32, 32, 4), 4096, ">u4", 32), ((6, 8, 1), 600, ">u1", 8)],
    )
    def test_constructed_matrices_are_low_rank(self, shape, size, dtype, cols, rp, monkeypatch):
        """Rank stays at or below rows//2, verified with randomness.py's solver.

        Reading the buffer back at the *reader's* natural alignment rather
        than the writer's is the point: an unaligned run of dependent rows
        reads as full rank again, which is why the operator snaps to a block
        boundary.
        """
        monkeypatch.setattr(S, "_RANK_SHAPES", (shape,))
        rows = shape[0]
        worst = 0
        for _ in range(40):
            out = S.rank_deficient(bytes(size), rng=rp)
            mats = np.frombuffer(out, dtype=dtype).reshape(-1, rows).astype(">u4")
            ranks = R._batch_gf2_rank(mats, cols)
            nonzero = ranks[ranks > 0]
            if nonzero.size:
                worst = max(worst, int(nonzero.max()))
        assert 0 < worst <= rows // 2

    def test_noise_is_near_full_rank(self, noise):
        """Baseline: the same reader sees full rank on unstructured bytes."""
        mats = np.frombuffer(noise(4096), dtype=">u4").reshape(-1, 32).astype(">u4")
        assert R._batch_gf2_rank(mats, 32).min() >= 29


class TestPermLock:
    # k=3 rather than the default k=5: permutation_test returns its 1.0
    # "not enough data" sentinel below 10*k! blocks, and MAX_REGION caps the
    # rewritten span at 4096 bytes — 1024 u32 draws, far short of the 6000 a
    # k=5 test needs. k=3 needs 180 and resolves the effect just as sharply.
    K = 3

    def test_permutation_histogram_collapses(self, seeded, noise):
        """Median over trials: the operator picks its word width at random and
        only the width-4 draws line up with a u32 reader."""
        base = noise(S.MAX_REGION)
        pvals = sorted(
            R.permutation_test(np.frombuffer(S.perm_lock(base, rng=seeded), dtype="<u4"), k=self.K)
            for _ in range(40)
        )
        assert pvals[len(pvals) // 2] < 1e-6

    def test_noise_passes_the_same_detector(self, noise):
        pvals = sorted(
            R.permutation_test(np.frombuffer(noise(S.MAX_REGION), dtype="<u4"), k=self.K)
            for _ in range(40)
        )
        assert pvals[0] > 1e-4

    @pytest.mark.parametrize("mode", ["ascending", "descending", "interleave"])
    def test_shape_is_a_permutation(self, mode):
        """These three permute 1..n; organ_pipe and equal deliberately do not."""
        for n in (8, 9, 64, 65):
            assert sorted(S._sorted_shape(n, mode)) == list(range(1, n + 1))

    def test_organ_pipe_rises_then_falls(self):
        shape = S._sorted_shape(64, "organ_pipe")
        peak = shape.index(max(shape))
        assert shape[:peak] == sorted(shape[:peak])
        assert shape[peak:] == sorted(shape[peak:], reverse=True)


class TestLagCorrelate:
    def test_some_lag_reaches_the_detector_tail(self, seeded, noise):
        base = noise(S.MAX_REGION)
        hits = sum(
            min(R.lagged_autocorrelation(S.lag_correlate(base, rng=seeded)).values()) < 1e-6
            for _ in range(30)
        )
        assert hits >= 10

    def test_periodicity_is_literal(self, rp):
        """A period is really copied, not merely correlated.

        An all-zero input stays all-zero: the operator tiles a slice of the
        buffer itself rather than inventing bytes.
        """
        for _ in range(30):
            assert S.lag_correlate(bytes(1024), rng=rp) == bytes(1024)


class TestBirthdayCollide:
    def test_spacings_collapse(self, seeded, noise, monkeypatch):
        """Word width is pinned to the detector's sampling width.

        birthday_spacings slices 16-bit words; left free, the operator writes
        32- and 64-bit progressions just as often, and those are not an
        arithmetic progression when re-read 16 bits at a time.
        """
        monkeypatch.setattr(S, "_WIDTHS_WORD", (2,))
        base = noise(2048)
        pvals = sorted(
            R.birthday_spacings(S.birthday_collide(base, rng=seeded), word_bits=16, n_points=256)
            for _ in range(15)
        )
        assert pvals[len(pvals) // 2] < 0.01

    def test_noise_does_not_collapse(self, noise):
        assert R.birthday_spacings(noise(2048), word_bits=16, n_points=256) > 1e-6


class TestGcdWorstCase:
    """Independent check: count Euclid steps, do not trust the Fibonacci table.

    Consecutive Fibonacci numbers are the worst case for the Euclidean
    algorithm over operands below a bound (Lame's theorem), so a 64-bit
    consecutive pair needs ~92 steps against ~38 on average for uniform 64-bit
    operands. The word width is pinned because the operator picks it at random
    and a u64 reader sees noise in the other cases.
    """

    @staticmethod
    def _euclid_steps(a: int, b: int) -> int:
        steps = 0
        while b:
            a, b = b, a % b
            steps += 1
        return steps

    @classmethod
    def _worst_pair(cls, buf: bytes) -> int:
        """Most Euclid steps over any adjacent u64 pair, either endianness."""
        worst = 0
        for fmt in ("<Q", ">Q"):
            st = struct.Struct(fmt)
            vals = [st.unpack_from(buf, i)[0] for i in range(0, len(buf) - 7, 8)]
            for a, b in zip(vals[::2], vals[1::2], strict=False):
                if a and b:
                    worst = max(worst, cls._euclid_steps(a, b))
        return worst

    def test_produces_near_maximal_euclid_step_counts(self, seeded, monkeypatch):
        monkeypatch.setattr(S, "_WIDTHS_WORD", (8,))
        best = max(self._worst_pair(S.fibonacci_pairs(bytes(2048), rng=seeded)) for _ in range(20))
        assert best >= 80

    def test_noise_needs_far_fewer_steps(self, noise):
        assert max(self._worst_pair(noise(2048)) for _ in range(20)) < 70


class TestMonotoneFill:
    @staticmethod
    def _longest_run(buf: bytes) -> int:
        best = run = 1
        for i in range(1, len(buf)):
            run = run + 1 if buf[i] >= buf[i - 1] else 1
            best = max(best, run)
        return best

    def test_creates_long_monotone_runs(self, seeded, noise):
        """Run length is measured on the output, not read back from params."""
        base = noise(2048)
        baseline = self._longest_run(base)
        best = max(self._longest_run(S.monotone_fill(base, rng=seeded)) for _ in range(30))
        assert best > baseline * 2


class TestPopcountLock:
    def test_region_has_a_single_hamming_weight(self, rp):
        """At most two weights survive: the region's, plus the untouched zeros."""
        for _ in range(30):
            out = S.popcount_lock(bytes(2048), rng=rp)
            assert len({b.bit_count() for b in out}) <= 2

    def test_table_maps_onto_the_requested_class(self):
        for weight in (0, 1, 4, 8):
            table = S._popcount_table(weight)
            assert len(table) == 256
            assert {b.bit_count() for b in table} == {weight}


class TestFloatSqueeze:
    def test_emits_non_finite_doubles(self, seeded):
        """At least one aligned double in the region must be inf or NaN."""
        found = False
        for _ in range(40):
            out = S.float_squeeze(bytes(1024), rng=seeded)
            values = struct.unpack(f"<{len(out) // 8}d", out)
            if any(not math.isfinite(v) for v in values):
                found = True
                break
        assert found

    def test_patterns_are_paired_by_value_class(self):
        """Each (f64, f32) entry must denote the same class, not a truncation."""
        for wide, narrow in S._FLOAT_PATTERNS:
            d = struct.unpack("<d", struct.pack("<Q", wide))[0]
            f = struct.unpack("<f", struct.pack("<I", narrow))[0]
            assert math.isnan(d) == math.isnan(f)
            assert math.isinf(d) == math.isinf(f)
            if not math.isnan(d):
                assert math.copysign(1.0, d) == math.copysign(1.0, f)


class TestDegenerateGeometry:
    def test_emits_repeated_coordinate_tuples(self, seeded, monkeypatch):
        """Coincident points repeat a coordinate exactly one stride later.

        Width and dimensionality are pinned so the readback alignment matches
        what the operator wrote; left free, a u32 reader sees noise whenever
        the operator picked 16- or 64-bit coordinates.
        """
        monkeypatch.setattr(S, "_GEOMETRY_WIDTHS", (4,))
        monkeypatch.setattr(S, "_GEOMETRY_DIMS", (2,))
        found = False
        for _ in range(40):
            out = S.degenerate_geometry(bytes(1024), rng=seeded)
            words = struct.unpack(f"<{len(out) // 4}I", out)
            if any(w and w == words[i + 2] for i, w in enumerate(words[:-2])):
                found = True
                break
        assert found

    def test_collinear_points_share_a_constant_step(self, seeded, monkeypatch):
        """Assert the run directly rather than bounding distinct deltas.

        The zero tails on either side, the two region boundaries and the
        modular wrap of ``base + step*i`` each contribute their own delta, so a
        distinct-count bound would measure the buffer's edges more than the
        construction.
        """
        monkeypatch.setattr(S, "_GEOMETRY_WIDTHS", (4,))
        monkeypatch.setattr(S, "_GEOMETRY_DIMS", (2,))
        monkeypatch.setattr(S, "_GEOMETRY_MODES", ("collinear",))
        longest = 0
        for _ in range(20):
            out = S.degenerate_geometry(bytes(4096), rng=seeded)
            xs = struct.unpack(f"<{len(out) // 4}I", out)[::2]
            deltas = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            run = best = 1
            for i in range(1, len(deltas)):
                run = run + 1 if deltas[i] == deltas[i - 1] else 1
                best = max(best, run)
            longest = max(longest, best)
        assert longest >= 16


class TestSpectralPeak:
    def test_every_mode_writes_a_low_entropy_block(self, rp):
        """Against a zero buffer, the alphabet the operator adds is tiny.

        Constant and Nyquist modes add one or two values, the impulse adds
        one, and the cosine is sampled at an 8-point transform's bins so it
        can add at most eight. Measuring against zeros rather than noise makes
        this independent of how much of a buffer the region happened to cover.
        """
        for _ in range(60):
            assert len(set(S.spectral_peak(bytes(S.MAX_REGION), rng=rp))) <= 16

    def test_reduces_the_alphabet_of_a_noisy_buffer(self, seeded, noise):
        base = noise(S.MAX_REGION)
        assert len(set(base)) > 200
        assert min(len(set(S.spectral_peak(base, rng=seeded))) for _ in range(60)) < 200


# ── invariant_break measures the corpus, not the seed ──────────────────


class TestInvariantBreak:
    @staticmethod
    def _corpus(n: int = 32) -> list[bytes]:
        """Inputs sharing a fixed 8-byte header and varying afterwards."""
        rnd = random.Random(SEED)
        return [b"\x89MAGIC\x00\x01" + rnd.randbytes(56) for _ in range(n)]

    def test_writes_only_to_locked_offsets(self, rp):
        corpus = self._corpus()
        invariants = R.corpus_invariants(corpus)
        seed = corpus[0]
        locked = set(invariants.fixed_offsets) | {o for o, _m in invariants.partial_offsets}
        for _ in range(40):
            out = S.invariant_break(seed, invariants, rng=rp)
            assert len(out) == len(seed)
            assert {i for i in range(len(seed)) if out[i] != seed[i]} <= locked

    def test_preserves_the_varying_bits_of_a_partial_mask(self, rp):
        """A 0xF0 mask means only the high nibble may move."""
        corpus = self._corpus()
        invariants = R.corpus_invariants(corpus)
        seed = corpus[0]
        partial = dict(invariants.partial_offsets)
        for _ in range(40):
            out = S.invariant_break(seed, invariants, rng=rp)
            for offset, mask in partial.items():
                assert out[offset] & ~mask & 0xFF == seed[offset] & ~mask & 0xFF

    def test_actually_changes_the_header(self, rp):
        corpus = self._corpus()
        invariants = R.corpus_invariants(corpus)
        seed = corpus[0]
        assert any(S.invariant_break(seed, invariants, rng=rp) != seed for _ in range(40))

    def test_missing_invariants_is_a_noop(self, rp):
        seed = os.urandom(64)
        assert S.invariant_break(seed, None, rng=rp) == seed

    def test_structureless_corpus_is_a_noop(self, rp):
        """No shared offsets means nothing to break."""
        rnd = random.Random(SEED)
        invariants = R.corpus_invariants([rnd.randbytes(64) for _ in range(32)])
        seed = rnd.randbytes(64)
        assert S.invariant_break(seed, invariants, rng=rp) == seed
