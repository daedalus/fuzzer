"""Tests for PoolDrift: live corpus byte distribution vs the frozen seed set.

Every expected value comes from ``_ref``, an independent spelling of the
formulas over plain probability lists -- never from the tracker under test.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, CumulativeByteEntropy
from fuzzer_tool.core.pool_drift import PoolDrift
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.report import _entropy_metrics
from fuzzer_tool.services.stats import StatsReporter

TOL = 1e-9


def _dist(seeds: list[bytes]) -> list[float]:
    """Pooled byte probabilities over capped seeds."""
    freq = [0] * 256
    for seed in seeds:
        for b in seed[:ENTROPY_SAMPLE_CAP]:
            freq[b] += 1
    total = sum(freq)
    return [c / total for c in freq]


def _h(p: list[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def _ref(seeds: list[bytes], pool: list[bytes]) -> dict[str, float]:
    """Reference drift readings, spelled out from the definitions."""
    p, q = _dist(pool), _dist(seeds)
    m = [(a + b) / 2 for a, b in zip(p, q, strict=True)]
    sizes = [min(len(s), ENTROPY_SAMPLE_CAP) for s in pool]
    n = sum(sizes)
    within = sum(k * _h(_dist([s])) for s, k in zip(pool, sizes, strict=True) if k) / n
    return {
        "seed_bits": _h(q),
        "pool_bits": _h(p),
        "delta_bits": _h(p) - _h(q),
        "js_bits": _h(m) - (_h(p) + _h(q)) / 2,
        "mutual_info": _h(p) - within,
        "novel_mass": sum(a for a, b in zip(p, q, strict=True) if b == 0),
    }


def _assert_matches(drift: PoolDrift, seeds: list[bytes], pool: list[bytes]) -> None:
    reading = drift.reading()
    assert reading is not None
    for field, expected in _ref(seeds, pool).items():
        assert getattr(reading, field) == pytest.approx(expected, abs=TOL), field


def _tracker(*seeds: bytes) -> CumulativeByteEntropy:
    t = CumulativeByteEntropy()
    for s in seeds:
        t.add(s)
    return t


class TestTrackerComparisons:
    def test_js_control_identical(self):
        # Rule 46: the comparison run against itself must read zero.
        a = _tracker(b"hello world", b"\x00\x01\x02")
        assert a.js_bits(_tracker(b"hello world", b"\x00\x01\x02")) == pytest.approx(0, abs=TOL)
        assert a.js_bits(a.copy()) == pytest.approx(0, abs=TOL)

    def test_js_matches_reference(self):
        pool, seeds = [b"AAAB", b"xyz"], [b"AAAA"]
        expected = _ref(seeds, pool)["js_bits"]
        assert _tracker(*pool).js_bits(_tracker(*seeds)) == pytest.approx(expected, abs=TOL)

    def test_js_fallback_matches_reference(self, monkeypatch):
        import fuzzer_tool.core.byte_entropy as be

        monkeypatch.setattr(be, "_HAS_NUMPY", False)
        pool, seeds = [b"AAAB", b"xyz"], [b"AAAA"]
        expected = _ref(seeds, pool)["js_bits"]
        assert _tracker(*pool).js_bits(_tracker(*seeds)) == pytest.approx(expected, abs=TOL)

    def test_js_disjoint_is_one_bit_and_symmetric(self):
        a, b = _tracker(b"AAAA"), _tracker(b"BBBBBB")
        assert a.js_bits(b) == pytest.approx(1.0, abs=TOL)
        assert b.js_bits(a) == pytest.approx(a.js_bits(b), abs=TOL)

    def test_novel_mass_matches_reference(self):
        pool, seeds = [b"AAAA", b"BB"], [b"AAAA"]
        expected = _ref(seeds, pool)["novel_mass"]
        assert _tracker(*pool).novel_mass(_tracker(*seeds)) == pytest.approx(expected, abs=TOL)

    def test_copy_is_independent(self):
        a = _tracker(b"AAAA")
        b = a.copy()
        a.add(b"BBBB")
        assert len(b) == len(b"AAAA")
        assert b.bits() == 0.0

    def test_empty_side_reads_zero(self):
        assert _tracker().js_bits(_tracker(b"A")) == 0.0
        assert _tracker().novel_mass(_tracker(b"A")) == 0.0
        # Nothing in the reference: every pool byte is novel.
        assert _tracker(b"A").novel_mass(_tracker()) == 1.0


class TestPoolDrift:
    def test_no_reading_before_sync(self):
        assert PoolDrift().reading() is None

    def test_control_unchanged_corpus(self):
        # Rule 46 control: the pool compared to its own seed snapshot.
        seeds = [b"GET / HTTP/1.1", b"\x89PNG\r\n", b"zzz"]
        drift = PoolDrift()
        drift.sync(seeds)
        drift.sync(list(seeds))
        reading = drift.reading()
        assert reading.delta_bits == pytest.approx(0, abs=TOL)
        assert reading.js_bits == pytest.approx(0, abs=TOL)
        assert reading.novel_mass == 0.0
        _assert_matches(drift, seeds, seeds)

    def test_admission(self):
        a, b = b"text text text", bytes(range(64))
        drift = PoolDrift()
        drift.sync([a])
        drift.sync([a, b])
        _assert_matches(drift, [a], [a, b])

    def test_eviction(self):
        a, b = b"AAAA", b"BBBB"
        drift = PoolDrift()
        drift.sync([a, b])
        drift.sync([a])
        _assert_matches(drift, [a, b], [a])
        assert drift.reading().mutual_info == pytest.approx(0, abs=TOL)

    def test_mutual_info_separates_seeds(self):
        # Same pooled histogram {A:.5, B:.5}; only the split across seeds differs.
        split = PoolDrift()
        split.sync([b"AAAA", b"BBBB"])
        mixed = PoolDrift()
        mixed.sync([b"ABAB", b"BABA"])
        assert split.reading().mutual_info == pytest.approx(1.0, abs=TOL)
        assert mixed.reading().mutual_info == pytest.approx(0.0, abs=TOL)

    def test_falsification_noise_is_detected(self):
        # Text seeds, then the pool fills with every byte value: must register.
        seeds = [b"the quick brown fox"]
        drift = PoolDrift()
        drift.sync(seeds)
        drift.sync(seeds + [bytes(range(256))])
        reading = drift.reading()
        assert reading.delta_bits > 0
        assert reading.js_bits > 0
        assert reading.novel_mass > 0
        _assert_matches(drift, seeds, seeds + [bytes(range(256))])


class TestPoolDriftAdversarial:
    def test_empty_baseline_never_reads(self):
        drift = PoolDrift()
        drift.sync([b""])
        drift.sync([b"", b"late admission"])
        assert drift.reading() is None

    def test_empty_seed_contributes_nothing(self):
        drift = PoolDrift()
        drift.sync([b"abc", b""])
        _assert_matches(drift, [b"abc"], [b"abc"])

    def test_bytes_past_cap_are_ignored(self):
        head = b"A" * ENTROPY_SAMPLE_CAP
        drift = PoolDrift()
        drift.sync([head])
        drift.sync([head, head + b"B" * 100])
        reading = drift.reading()
        assert reading.js_bits == pytest.approx(0, abs=TOL)
        assert reading.novel_mass == 0.0

    def test_baseline_is_frozen(self):
        drift = PoolDrift()
        drift.sync([b"AAAA"])
        drift.sync([b"completely different corpus"])
        assert drift.reading().seed_bits == 0.0
        _assert_matches(drift, [b"AAAA"], [b"completely different corpus"])

    def test_churn_does_not_accumulate_error(self):
        a, b = b"steady seed", bytes(range(200))
        drift = PoolDrift()
        drift.sync([a])
        for _ in range(1000):
            drift.sync([a, b])
            drift.sync([a])
        _assert_matches(drift, [a], [a])
        assert drift.reading().mutual_info >= 0.0


class TestWiring:
    def test_init_seed_metadata_freezes_baseline(self):
        f = SimpleNamespace(corpus=[b"AAAA", b"BBBB"], map_size=64, resume=False)
        CorpusManager(f).init_seed_metadata()
        assert isinstance(f._pool_drift, PoolDrift)
        f.corpus.append(bytes(range(256)))
        _assert_matches(f._pool_drift, [b"AAAA", b"BBBB"], [b"AAAA", b"BBBB"])

    def test_status_fragment(self):
        f = SimpleNamespace(corpus=[b"AAAA"], _pool_drift=PoolDrift())
        f._pool_drift.sync(f.corpus)
        f.corpus = [b"AAAA", b"BBBB"]
        out = StatsReporter.__new__(StatsReporter)._print_stats_drift_str(f)
        ref = _ref([b"AAAA"], f.corpus)
        assert out == f" | drift: dH={ref['delta_bits']:+.2f} js={ref['js_bits']:.3f}"

    def test_status_fragment_stand_ins(self):
        reporter = StatsReporter.__new__(StatsReporter)
        assert reporter._print_stats_drift_str(MagicMock()) == ""
        assert reporter._print_stats_drift_str(SimpleNamespace(corpus=[])) == ""

    def test_report_lines(self):
        f = SimpleNamespace(corpus=[b"AAAA"], _edge_tracker=None, _pool_drift=PoolDrift())
        f._pool_drift.sync(f.corpus)
        f.corpus = [b"AAAA", b"BBBB"]
        out = _entropy_metrics(f)
        ref = _ref([b"AAAA"], f.corpus)
        assert f"JS={ref['js_bits']:.4f}" in out
        assert f"I(seed; byte)={ref['mutual_info']:.3f}" in out
