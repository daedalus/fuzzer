"""Corpus bloat gate: location of recent seed sizes, not their skewness.

The gate used to be ``skewness > 2.0`` over a 200-seed RunningMoments
window. On a heavy-tailed (lognormal) size distribution the sample skewness
is driven by the window's largest members, so it fired on stationary
corpora and stayed silent on growth. core/size_bloat.py replaces it with
two location signals; see its module docstring for the measurements.

Every random stream below comes from a seeded generator, so each assertion
is on a fixed sequence (Hard Rule 39), and each old-vs-new comparison runs
the old gate on the same sequence as a control that must fail the way the
bug report says (Hard Rule 46).
"""

import math
import random
import tempfile
from array import array
from collections import deque
from pathlib import Path

import pytest

from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.size_bloat import (
    AT_CAP_SHARE,
    GROWTH_FACTOR,
    MIN_SAMPLES,
    SEGMENT,
    WINDOW,
    seed_size_bloat,
)


def _lognormal_sizes(n, sigma, seed, median=500, cap=None):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        s = max(1, int(median * math.exp(sigma * r.gauss(0.0, 1.0))))
        out.append(min(s, cap) if cap else s)
    return out


def _replay(sizes, max_len, every=5):
    """Fire rate of (old skewness gate, new gate) over checks along *sizes*,
    trimming the history 1000 -> 500 exactly as corpus_manager does."""
    moments = RunningMoments(window=200)
    hist: list[int] = []
    old = new = checks = 0
    for i, s in enumerate(sizes):
        moments.update(float(s))
        hist.append(s)
        if len(hist) > 1000:
            hist = hist[-500:]
        if i % every == 0 and i >= MIN_SAMPLES:
            checks += 1
            old += moments.count >= 50 and moments.skewness > 2.0
            new += seed_size_bloat(hist, max_len) is not None
    return old / checks, new / checks


class TestStationaryCorpusDoesNotFire:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
    def test_new_gate_quiet(self, sigma):
        _old, new = _replay(_lognormal_sizes(3000, sigma, seed=11), max_len=65536)
        assert new <= 0.01

    def test_old_gate_control_fired_on_the_same_stream(self):
        """The defect: no growth at all, yet the skewness gate fires."""
        old, _new = _replay(_lognormal_sizes(3000, 1.0, seed=11), max_len=65536)
        assert old >= 0.9


class TestGrowthFires:
    def test_drift_into_a_fixed_cap(self):
        r = random.Random(3)
        sizes = [
            min(4096, int((100 + 5900 * i / 2999) * math.exp(0.2 * r.gauss(0.0, 1.0))))
            for i in range(3000)
        ]
        old, new = _replay(sizes, max_len=4096)
        assert old == 0.0  # control: skew goes negative against the cap
        assert new >= 0.5

    def test_median_doubling(self):
        sizes = [100] * SEGMENT + [100] * (MIN_SAMPLES - 2 * SEGMENT) + [250] * SEGMENT
        reason = seed_size_bloat(sizes, max_len=0)
        assert reason is not None and "100B -> 250B" in reason

    def test_below_growth_factor(self):
        newer = int(100 * GROWTH_FACTOR) - 1
        assert seed_size_bloat([100] * SEGMENT + [newer] * SEGMENT, max_len=0) is None


class TestAtCap:
    def _sizes(self, n_at_cap):
        # Base 2000 with 4096 max_len: at-cap values 3700 >= 0.9 * 4096, and
        # the newest-segment median stays under GROWTH_FACTOR * 2000 either
        # way, so only the cap signal can fire.
        newest = [3700] * n_at_cap + [2000] * (SEGMENT - n_at_cap)
        return [2000] * SEGMENT + newest

    def test_fires_at_share(self):
        n = math.ceil(AT_CAP_SHARE * SEGMENT)
        reason = seed_size_bloat(self._sizes(n), max_len=4096)
        assert reason is not None and "max_len" in reason

    def test_quiet_just_below_share(self):
        n = math.ceil(AT_CAP_SHARE * SEGMENT) - 1
        assert seed_size_bloat(self._sizes(n), max_len=4096) is None

    def test_zero_max_len_disables_cap_signal(self):
        n = math.ceil(AT_CAP_SHARE * SEGMENT)
        assert seed_size_bloat(self._sizes(n), max_len=0) is None


class TestInputs:
    def test_too_few_samples(self):
        assert seed_size_bloat([100] * (MIN_SAMPLES - 1) + [10_000], max_len=0) is None

    def test_reads_only_the_last_window(self):
        # Huge old history outside the window must not register as a drop,
        # and a small old history outside it must not register as growth.
        sizes = [1] * 5000 + [500] * WINDOW
        assert seed_size_bloat(sizes, max_len=0) is None

    @pytest.mark.parametrize("wrap", [list, deque, lambda xs: array("I", xs)])
    def test_accepts_corpus_manager_containers(self, wrap):
        sizes = [100] * SEGMENT + [300] * SEGMENT
        assert seed_size_bloat(wrap(sizes), max_len=0) is not None


class TestWiredIntoSaveToCorpus:
    """Through CorpusManager.save_to_corpus, the path that schedules the
    minimization. Pre-fix: 251 triggers on the stationary stream, 0 on the
    growing one."""

    @staticmethod
    def _run(size_of):
        from fuzzer_tool.services.fuzzer import Fuzzer

        tmp = Path(tempfile.mkdtemp())
        (tmp / "c" / "seeds").mkdir(parents=True)
        (tmp / "x").mkdir()
        (tmp / "c" / "seeds" / "s").write_bytes(b"seed")
        f = Fuzzer(
            target="/nonexistent",
            corpus_dir=str(tmp / "c"),
            crashes_dir=str(tmp / "x"),
            max_len=65536,
        )
        r = random.Random(1)
        warned = 0
        for i in range(300):
            f.exec_count += 500  # past the rate limit every add
            before = f._last_bloat_warn_exec
            f.save_to_corpus(r.randbytes(size_of(i, r)), parent=b"seed")
            warned += f._last_bloat_warn_exec != before
        return warned

    def test_stationary_lognormal_never_warns(self):
        def size(_i, r):
            return int(min(60000, max(8, 300 * math.exp(1.5 * r.gauss(0.0, 1.0)))))

        assert self._run(size) == 0

    def test_doubling_sizes_warn(self):
        assert self._run(lambda i, r: int(50 * 2 ** (i / 60)) + r.randrange(8)) > 0
