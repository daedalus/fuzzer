"""Tests for FFT-based periodicity detection (record-size inference + spectral diagnostics)."""

import math
import random
import tempfile
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.execution_time import ExecutionTimeTracker
from fuzzer_tool.core.format_learner import FieldHypothesis, FormatLearner
from fuzzer_tool.core.grammar import Grammar, TreeMutator
from fuzzer_tool.core.mutations.generic import chunk_shuffle
from fuzzer_tool.core.periodicity import detect_periodicity, estimate_record_size, fisher_g_pvalue
from fuzzer_tool.services.fuzzer import Fuzzer


def _mk_fuzzer(**kwargs):
    tmpdir = tempfile.mkdtemp(prefix="fuzz_periodicity_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


def _records(num: int, stride: int, seed: int = 0, header: int = 2) -> bytes:
    """Buffer of ``num`` fixed-size records of ``stride`` bytes.

    Each record is ``header`` fixed magic bytes followed by randomized
    payload — the same-offset fixed bytes across records are the structural
    marker that makes the record boundary detectable in the raw bytes, while
    the payload varies per record (values don't need to repeat).
    """
    rng = random.Random(seed)
    magic = rng.randbytes(header)
    return b"".join(magic + rng.randbytes(stride - header) for _ in range(num))


class TestEstimateRecordSize:
    def test_stride8_detected(self):
        buf = _records(64, 8, seed=1)
        assert estimate_record_size(buf) == 8

    def test_stride16_detected(self):
        buf = _records(64, 16, seed=2)
        assert estimate_record_size(buf) == 16

    def test_harmonic_rejection(self):
        # Stride-8 data also correlates at lag 16/24; the smallest
        # locally-dominant peak (8) must win, never a harmonic.
        buf = _records(64, 8, seed=3)
        assert estimate_record_size(buf) == 8

    def test_random_bytes_rejected(self):
        assert estimate_record_size(random.Random(0).randbytes(256)) is None
        assert estimate_record_size(random.Random(3).randbytes(4096)) is None

    def test_short_buffer_rejected(self):
        assert estimate_record_size(b"\x01\x02" * 4) is None
        assert estimate_record_size(b"") is None

    def test_constant_buffer_rejected(self):
        assert estimate_record_size(b"\x00" * 256) is None

    def test_period_beyond_max_lag_rejected(self):
        # 900-byte buffer of 300-byte records with a 200-byte fixed prefix:
        # strong stride-300 signal, but 300 > cap(256) and > len//3.
        buf = _records(3, 300, seed=4, header=200)
        assert estimate_record_size(buf) is None

    def test_large_buffer_windowed_detection(self):
        # Regression (T2): buffers above the 4096-byte analysis window cap
        # must still infer the stride from the first window — a 2.8 MB PNG
        # seed once cost ~2.4 s of FFT per call in estimate_record_size.
        buf = _records(1250, 8, seed=7)  # 10000 bytes, stride 8
        assert len(buf) > 4096
        assert estimate_record_size(buf) == 8

    def test_large_buffer_equals_window_prefix(self):
        # The windowed path analyzes exactly data[:4096] with the same limit
        # and sigma bound, so a large buffer must match its prefix exactly.
        buf = _records(1250, 8, seed=8)
        assert len(buf) > 4096
        assert estimate_record_size(buf) == estimate_record_size(buf[:4096]) == 8


class TestChunkShuffleStride:
    def test_uses_stride(self):
        data = _records(64, 8, seed=5)
        out = chunk_shuffle(data, rng=random.Random(0), stride=8)
        assert len(out) == len(data)
        # Chunk shuffling swaps whole 8-byte chunks: multiset preserved,
        # chunk boundaries intact.
        assert sorted(out[i : i + 8] for i in range(0, len(out), 8)) == sorted(
            data[i : i + 8] for i in range(0, len(data), 8)
        )

    def test_fallback_without_stride(self):
        # len divisible by 1..4 so every legacy chunk size preserves the byte
        # multiset; stride=None keeps the old random 1-4 behavior.
        data = _records(64, 8, seed=6)
        out = chunk_shuffle(data, rng=random.Random(1), stride=None)
        assert len(out) == len(data)
        assert sorted(out) == sorted(data)

    def test_stride_ignored_when_too_large(self):
        # stride > len(data)//2 yields < 2 chunks — must fall back, not crash.
        data = _records(64, 8, seed=7)
        out = chunk_shuffle(data, rng=random.Random(2), stride=256)
        assert len(out) == len(data)

    def test_regression_small_buffer_untouched(self):
        data = b"\x01\x02\x03\x04"
        assert chunk_shuffle(data, rng=random.Random(3), stride=2) == data


class TestDetectPeriodicity:
    def test_sine_detected(self):
        rng = random.Random(1)
        series = [math.sin(2 * math.pi * k / 16) + 0.3 * rng.uniform(-1, 1) for k in range(256)]
        res = detect_periodicity(series)
        assert res.significant
        assert 15.0 <= res.dominant_period <= 17.0

    def test_noise_not_significant(self):
        rng = random.Random(2)
        series = [rng.random() for _ in range(256)]
        res = detect_periodicity(series)
        assert not res.significant

    def test_short_series_not_significant(self):
        series = [math.sin(2 * math.pi * k / 8) for k in range(30)]
        res = detect_periodicity(series, min_samples=64)
        assert not res.significant

    def test_constant_series_not_significant(self):
        res = detect_periodicity([1.0] * 128)
        assert not res.significant

    def test_dc_excluded(self):
        # A large DC offset must not be reported as a periodic component.
        series = [1000.0 + 0.01 * math.sin(2 * math.pi * k / 8) for k in range(128)]
        res = detect_periodicity(series)
        assert res.peak_bin >= 1
        assert res.significant
        assert res.dominant_period is not None

    def test_execution_time_tracker_spectral(self):
        tracker = ExecutionTimeTracker(window_size=200)
        rng = random.Random(5)
        for k in range(200):
            tracker.record(0.01 + 0.004 * math.sin(2 * math.pi * k / 16) + 0.001 * rng.random())
        res = detect_periodicity(list(tracker._times))
        assert res.significant
        assert 14.0 <= res.dominant_period <= 18.0

    def test_regression_white_noise_false_positive_rate(self):
        """Pure white noise must be flagged at ~alpha, not at a 97.5% rate.

        Regression: the old fixed peak_to_median=2.0 ratio implicitly picked
        the best of ~100 non-DC bins, so the max of ~100 exponential-family
        ordinates under the null exceeded 2x the median by chance (195/200
        trials flagged). With alpha=0.05 the nominal rate is ~5%; the bound
        of 10% absorbs binomial noise across 5x200 trials without flaking.
        """
        flagged = 0
        trials = 200
        for s in range(5):
            for t in range(trials):
                rng = random.Random(s * 1000 + t)
                series = [rng.random() for _ in range(200)]
                if detect_periodicity(series, min_samples=50).significant:
                    flagged += 1
        assert flagged / (5 * trials) <= 0.10

    def test_regression_peak_strength_bounded(self):
        """A clean signal must yield a finite strength in (0, 1), not 6e15.

        Regression: peak_strength divided by the median of the non-DC bins,
        which sits near machine-epsilon for very clean signals, blowing up
        to ~6.29e15. The g statistic divides by the total power instead, so
        it is bounded in [0, 1] by construction.
        """
        series = [math.sin(2 * math.pi * k / 16) for k in range(256)]
        res = detect_periodicity(series)
        # A pure on-bin sine puts all power in one bin -> g == 1.0 exactly;
        # the regression property is bounded/finite, not the old ~6e15 blowup.
        assert 0.0 < res.peak_strength <= 1.0
        assert math.isfinite(res.peak_strength)
        assert res.significant
        assert res.p_value < 0.05

    def test_regression_fisher_pvalue_reference(self):
        """Fisher's g p-value matches hand-computed closed-form values.

        P(G > g) = sum_k (-1)^(k-1) C(m,k) (1 - k*g)^(m-1), k <= floor(1/g).
        """
        assert fisher_g_pvalue(0.5, 10) == pytest.approx(0.01953125)  # 10 * 0.5^9
        assert fisher_g_pvalue(0.4, 10) == pytest.approx(0.10075392)
        assert fisher_g_pvalue(0.3, 10) == pytest.approx(0.39173971)
        assert fisher_g_pvalue(1.0, 10) == 0.0
        assert fisher_g_pvalue(1.0, 1) == 1.0
        assert fisher_g_pvalue(0.02, 200) == 1.0  # p1 > 1.0 -> p ~ 1 (not significant)


class TestSeedMetaStridePersistence:
    def test_stride_roundtrip_through_state(self):
        f = _mk_fuzzer(max_len=4096)
        seed = _records(64, 8, seed=10)  # 512 bytes, stride 8
        f.save_to_corpus(seed)
        assert f.seed_meta[seed]["record_stride"] == 8
        f._corpus_manager.save_state()

        f2 = _mk_fuzzer(
            resume=True,
            max_len=4096,
            corpus_dir=f.corpus_dir,
            crashes_dir=str(Path(f.crashes_dir)),
        )
        meta = f2.seed_meta.get(seed)
        assert meta is not None
        assert meta["record_stride"] == 8

    def test_random_seed_stride_none(self):
        f = _mk_fuzzer()
        seed = random.Random(11).randbytes(256)
        f.save_to_corpus(seed)
        assert f.seed_meta[seed]["record_stride"] is None
        f._corpus_manager.save_state()

        f2 = _mk_fuzzer(
            resume=True,
            corpus_dir=f.corpus_dir,
            crashes_dir=str(Path(f.crashes_dir)),
        )
        assert f2.seed_meta.get(seed)["record_stride"] is None


class TestTreeMutatorStride:
    def _tree_mutator(self):
        grammar = Grammar()
        grammar.parse("root = item+\nitem = value\nvalue = bytes")
        return TreeMutator(grammar)

    def test_parse_uses_stride_chunk_size(self):
        tm = self._tree_mutator()
        buf = _records(16, 8, seed=7)  # 128 bytes of 8-byte records
        tree = tm.parse(buf, chunk_size=8)
        assert tree.children
        assert all(len(c.data) == 8 for c in tree.children)

    def test_parse_falls_back_to_inferred_size(self):
        tm = self._tree_mutator()
        buf = _records(16, 8, seed=7)
        tree = tm.parse(buf)  # no chunk_size -> grammar-inferred default
        assert tree.children
        assert len(tree.children[0].data) == 16

    def test_parse_stride_too_large_leaves_single_node(self):
        tm = self._tree_mutator()
        buf = _records(16, 8, seed=7)
        tree = tm.parse(buf, chunk_size=4096)  # > len(data) -> whole buffer
        assert tree.is_leaf
        assert tree.data == buf


class TestFormatLearnerStride:
    def test_record_stride_in_summary(self):
        assert FormatLearner().get_format_summary()["record_stride"] is None
        fl = FormatLearner()
        fl.set_record_stride(8)
        assert fl.get_format_summary()["record_stride"] == 8

    def test_stride_aligned_hypothesis_gets_boost(self):
        fl = FormatLearner()
        fl.set_record_stride(8)
        fl.hypotheses.append(
            FieldHypothesis(offset=0, width=8, field_type="unknown", confidence=0.4, observations=3)
        )
        fl.hypotheses.append(
            FieldHypothesis(offset=1, width=4, field_type="unknown", confidence=0.4, observations=3)
        )
        fl._classify_fields()
        aligned, misaligned = fl.hypotheses
        assert aligned.confidence == 0.45  # 0.4 + 0.05 boost
        assert misaligned.confidence == 0.4  # untouched

    def test_no_boost_without_stride_prior(self):
        fl = FormatLearner()
        fl.hypotheses.append(
            FieldHypothesis(offset=0, width=8, field_type="unknown", confidence=0.4, observations=3)
        )
        fl._classify_fields()
        assert fl.hypotheses[0].confidence == 0.4


class TestBmpWfcTileBytes:
    @staticmethod
    def _bmp() -> bytes:
        """Minimal 4x4 24bpp BMP (12-byte rows, 48-byte pixel data)."""
        import struct

        header = (
            b"BM"
            + struct.pack("<I", 54 + 48)
            + b"\x00\x00\x00\x00"
            + struct.pack("<I", 54)
            + struct.pack("<I", 40)
            + struct.pack("<i", 4)
            + struct.pack("<i", 4)
            + struct.pack("<H", 1)
            + struct.pack("<H", 24)
            + struct.pack("<I", 0)
            + struct.pack("<I", 48)
            + struct.pack("<I", 2835)
            + struct.pack("<I", 2835)
            + struct.pack("<I", 0)
            + struct.pack("<I", 0)
        )
        return header + bytes(range(1, 49))

    def test_tile_bytes_aligned_preserves_row_shape(self):
        from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp

        m = BmpMutator()
        m.use_wfc = True
        m.tile_bytes = 4  # 12 % 4 == 0 -> 4-byte tiles, 3 per row
        out = m._wfc_pixels(parse_bmp(self._bmp()), 4096, 4)
        assert len(out.pixel_data) == 48  # 4 rows x 12-byte stride
        assert all(len(out.pixel_data[y * 12 : (y + 1) * 12]) == 12 for y in range(4))

    def test_tile_bytes_non_divisor_falls_back(self):
        from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp

        m = BmpMutator()
        m.use_wfc = True
        m.tile_bytes = 8  # 12 % 8 != 0 -> per-pixel tiles
        out = m._wfc_pixels(parse_bmp(self._bmp()), 4096, 8)
        assert len(out.pixel_data) == 48

    def test_mutate_with_tile_bytes(self):
        from fuzzer_tool.core.mutations.bmp import BmpMutator

        m = BmpMutator()
        m.use_wfc = True
        m.tile_bytes = 4
        out = m.mutate(self._bmp(), max_len=4096)
        assert len(out) >= 54


class TestSpectralDiagnostics:
    def test_silent_when_insufficient_data(self):
        from fuzzer_tool.services import report as report_mod

        f = _mk_fuzzer()
        assert report_mod._spectral_diagnostics(f) == ""

    def test_exec_time_periodic_flag(self):
        from fuzzer_tool.services import report as report_mod

        f = _mk_fuzzer()
        tracker = ExecutionTimeTracker(window_size=200)
        rng = random.Random(6)
        for k in range(160):
            tracker.record(0.01 + 0.004 * math.sin(2 * math.pi * k / 16) + 0.001 * rng.random())
        f._exec_time_tracker = tracker
        f._discovery_execs = array("Q")
        f._discovery_edges = array("Q")
        out = report_mod._spectral_diagnostics(f)
        assert "PERIODIC" in out
        assert "Exec time" in out

    def test_discovery_rate_periodic_flag(self):
        from fuzzer_tool.services import report as report_mod

        f = _mk_fuzzer()
        tracker = ExecutionTimeTracker(window_size=200)
        for _ in range(60):
            tracker.record(0.01)
        f._exec_time_tracker = tracker
        # Discovery bursts every 10 sync intervals
        history = []
        for i in range(150):
            burst = 8 if i % 10 == 0 else 0
            cumulative = (i + 1) * 2 + burst
            history.append((i, cumulative))
        f._discovery_execs = array("Q", (e for e, _ in history))
        f._discovery_edges = array("Q", (c for _, c in history))
        out = report_mod._spectral_diagnostics(f)
        assert "Discovery rate: PERIODIC" in out
        assert "sync artifact" in out

    def test_flat_series_not_periodic(self):
        from fuzzer_tool.services import report as report_mod

        f = _mk_fuzzer()
        tracker = ExecutionTimeTracker(window_size=200)
        for _ in range(160):
            tracker.record(0.01)
        f._exec_time_tracker = tracker
        f._discovery_execs = array("Q", (i for i in range(150)))
        f._discovery_edges = array("Q", (i for i in range(150)))
        out = report_mod._spectral_diagnostics(f)
        assert "no significant periodic component" in out
