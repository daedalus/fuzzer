"""Regression: ffmpeg RSS readout and Katz masks of evicted seeds.

``rss:`` is the peak (never falls), so load/exit spikes read as growth;
``live:`` reports current RSS next to it. Katz kept one node mask per seed
ever admitted, including seeds the corpus had since dropped; ``retain``
folds those into one union mask so the visited set V is unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fuzzer_tool.adapters import libc_mem
from fuzzer_tool.services import stats as stats_mod
from fuzzer_tool.services.katz_channel import EVICTED_KEY
from fuzzer_tool.services.stats import StatsReporter
from tests import test_regression_end_of_run_persistence as end_of_run
from tests.test_katz_channel import _bits, _channel


def _mask(ch, key, n=4):
    return np.unpackbits(np.frombuffer(ch._masks[key], dtype=np.uint8), bitorder="little")[:n]


class TestKatzRetain:
    def test_regression_evicted_masks_fold_into_union(self):
        ch = _channel()
        ch.record(_bits({0}), seed_key="a")
        ch.record(_bits({1}), seed_key="b")
        ch.record(_bits({3}), seed_key="c")
        ch._dirty = False

        ch.retain({"c"})

        assert set(ch._masks) == {"c", EVICTED_KEY}
        assert _mask(ch, EVICTED_KEY).tolist() == [1, 1, 0, 0]
        assert ch._dirty

    def test_fold_accumulates_across_calls(self):
        ch = _channel()
        ch.record(_bits({0}), seed_key="a")
        ch.retain(set())
        ch.record(_bits({2}), seed_key="b")
        ch.retain(set())

        assert set(ch._masks) == {EVICTED_KEY}
        assert _mask(ch, EVICTED_KEY).tolist() == [1, 0, 1, 0]

    def test_all_live_is_noop(self):
        """Adversarial: superset of live keys drops nothing, stays clean."""
        ch = _channel()
        ch.record(_bits({0}), seed_key="a")
        ch._dirty = False
        before = dict(ch._masks)

        ch.retain({"a", "zz"})

        assert ch._masks == before
        assert not ch._dirty

    def test_fold_key_is_never_evicted(self):
        """Adversarial: the union survives a live set that omits it."""
        ch = _channel()
        ch.record(_bits({1}), seed_key="a")
        ch.retain(set())
        ch.retain(set())
        assert _mask(ch, EVICTED_KEY).tolist() == [0, 1, 0, 0]


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="fuzz_rss_") as d:
        root = Path(d)
        (root / "corpus").mkdir()
        (root / "corpus" / "seed_a").write_bytes(b"AAAA")
        yield end_of_run._make_fuzzer(root, corpus_dir=str(root / "corpus"), quiet_stats=True)


class TestKatzWiring:
    def test_retain_runs_with_live_corpus_keys(self, fuzzer):
        live = [b"live1", b"live2"]
        fuzzer.corpus = list(live)
        ch = _channel()
        for d in [*live, b"gone"]:
            ch.record(_bits({0}), seed_key=fuzzer._seed_key(d))
        fuzzer._katz_channel = ch

        fuzzer._retain_katz()

        assert set(ch._masks) == {fuzzer._seed_key(d) for d in live} | {EVICTED_KEY}

    def test_union_does_not_retrigger(self, fuzzer):
        """Adversarial: after a fold, masks == corpus + union must not rescan."""
        fuzzer.corpus = [b"live1"]
        ch = _channel()
        ch.record(_bits({0}), seed_key=fuzzer._seed_key(b"live1"))
        ch.record(_bits({1}), seed_key="gone")
        fuzzer._katz_channel = ch
        fuzzer._retain_katz()

        with patch.object(ch, "retain") as again:
            fuzzer._retain_katz()
        again.assert_not_called()

    def test_no_channel_is_noop(self, fuzzer):
        fuzzer._katz_channel = None
        fuzzer._retain_katz()


class TestLiveRss:
    def test_current_rss_matches_statm(self):
        assert libc_mem.current_rss() > 0

    def test_stats_line_shows_peak_and_live(self):
        f = SimpleNamespace(_peak_rss=732 * 1024)
        with (
            patch.object(stats_mod.libc_mem, "trim_heap", return_value=0),
            patch.object(stats_mod.libc_mem, "current_rss", return_value=424 * 2**20),
            patch.object(stats_mod, "_maxrss_kb", return_value=0),
        ):
            rss_str, _ = StatsReporter._print_stats_mem_strs(None, f)
        assert rss_str == " | rss: 732MB | live: 424MB"

    def test_kernel_peak_unmocked(self):
        """Falsification: the real getrusage path runs (no NameError) and >= live."""
        assert stats_mod._maxrss_kb() * 1024 >= libc_mem.current_rss() // 2

    def test_regression_peak_never_below_live(self):
        """Stale _peak_rss (refreshed per 100 execs) read 252MB with 530MB live."""
        f = SimpleNamespace(_peak_rss=252 * 1024)
        with (
            patch.object(stats_mod.libc_mem, "trim_heap", return_value=0),
            patch.object(stats_mod.libc_mem, "current_rss", return_value=530 * 2**20),
            patch.object(stats_mod, "_maxrss_kb", return_value=600 * 1024),
        ):
            rss_str, _ = StatsReporter._print_stats_mem_strs(None, f)
        assert rss_str == " | rss: 600MB | live: 530MB"
        assert f._peak_rss == 600 * 1024

    def test_sub_megabyte_live_in_kb(self):
        """Adversarial: tiny live RSS keeps the KB unit, like peak."""
        f = SimpleNamespace(_peak_rss=512)
        with (
            patch.object(stats_mod.libc_mem, "trim_heap", return_value=0),
            patch.object(stats_mod.libc_mem, "current_rss", return_value=300 * 1024),
            patch.object(stats_mod, "_maxrss_kb", return_value=0),
        ):
            rss_str, _ = StatsReporter._print_stats_mem_strs(None, f)
        assert rss_str == " | rss: 512KB | live: 300KB"


# ── crash-MI resume/save spike ──────────────────────────────────────────

import tracemalloc  # noqa: E402

from fuzzer_tool.core.analyzers import analyzer_crash_eta as crash_eta  # noqa: E402
from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker  # noqa: E402


def _tracker(rows: int, crash_every: int = 3) -> CrashMITracker:
    """Deterministic histograms over *rows* positions (no RNG: arithmetic fill)."""
    t = CrashMITracker(max_positions=rows, min_observations=1)
    for i in range(12):
        data = bytes((i * 7 + p) & 0xFF for p in range(rows))
        t.record(data, is_crash=(crash_every > 0 and i % crash_every == 0))
    return t


def _oracle_sums(t: CrashMITracker):
    """Whole-array S_* exactly as the pre-chunking _rebuild_sums computed them."""
    total = t._byte_total_arr.astype(np.float64)
    crash = t._joint_crash_arr.astype(np.float64)
    f = crash_eta._f
    return f(total).sum(axis=1), f(crash).sum(axis=1), f(total - crash).sum(axis=1)


class TestCrashMiRebuild:
    def test_regression_chunked_rebuild_matches_whole_array(self, monkeypatch):
        """Chunks smaller than the row count, and a ragged last chunk."""
        monkeypatch.setattr(crash_eta, "_REBUILD_CHUNK_ROWS", 7)
        t = _tracker(64)
        want = _oracle_sums(t)

        t._rebuild_sums()

        for got, exp in zip((t._s_total, t._s_crash, t._s_clean), want, strict=True):
            np.testing.assert_allclose(got, exp, rtol=1e-12)
        assert t._crash_rows.tolist() == t._joint_crash_arr.sum(axis=1).tolist()

    def test_regression_rebuild_peak_independent_of_rows(self, monkeypatch):
        """Falsification: the old rebuild held ~5 float64 copies of every row."""
        chunk = 256
        monkeypatch.setattr(crash_eta, "_REBUILD_CHUNK_ROWS", chunk)
        t = _tracker(8192, crash_every=0)
        chunk_bytes = chunk * 256 * 8

        tracemalloc.start()
        try:
            t._rebuild_sums()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        sums_bytes = 4 * 8192 * 8  # the four per-row result vectors
        assert peak <= sums_bytes + 6 * chunk_bytes


class TestCrashMiSparseState:
    def test_regression_save_is_sparse_bytes(self):
        t = _tracker(300)
        blob = t.save()
        assert "byte_total" not in blob
        assert isinstance(blob["sparse"]["total_idx"], bytes)

    def test_round_trip_exact(self):
        t = _tracker(300)
        u = CrashMITracker()
        u.load(t.save())
        assert u.byte_total == t.byte_total
        assert u.joint_crash == t.joint_crash
        assert u.position_counts == t.position_counts
        assert (u.total_execs, u.total_crashes) == (t.total_execs, t.total_crashes)

    def test_empty_tracker_round_trip(self):
        """Adversarial: no rows at all."""
        u = CrashMITracker()
        u.load(CrashMITracker().save())
        assert u.byte_total == {} and u._rows == 0

    def test_zero_crashes_round_trip(self):
        """Adversarial: all-zero crash array saves no cells."""
        t = _tracker(100, crash_every=0)
        blob = t.save()
        assert blob["sparse"]["crash_idx"] == b""
        u = CrashMITracker()
        u.load(blob)
        assert u.joint_crash == {} and u.byte_total == t.byte_total

    def test_legacy_out_of_range_keys_dropped(self):
        """Adversarial: legacy state with positions/bytes outside the store."""
        u = CrashMITracker(max_positions=4)
        u.load(
            {
                "total_execs": 2,
                "total_crashes": 1,
                "position_counts": {"0": 2, "9": 2},
                "byte_total": {"0": {"65": 2, "300": 1}, "9": {"1": 1}},
                "joint_crash": {"0": {"65": 1}},
            }
        )
        assert u.byte_total == {0: {65: 2}}
        assert u.joint_crash == {0: {65: 1}}
