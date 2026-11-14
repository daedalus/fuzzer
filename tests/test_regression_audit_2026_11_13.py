"""Regressions from the 2026-11-13 audit (mutators, sanitizer parse, schedulers)."""

import time

import pytest

from fuzzer_tool.core.mutations.structured import zero_run_amplify
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.schedulers.op_hierarchical import HierarchicalBanditScheduler
from tests.support.scripted_rng import ScriptedRng

_ASAN_HEAD = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000011 at pc 0x4011d6\n"
)
# Stderr capture cap (adapters/process.py): the largest report parse() sees.
_STDERR_CAP = 64 * 1024
# Quadratic parse took 6.3 s at the cap; linear takes milliseconds.
_PARSE_BUDGET_S = 1.0
_INT_MAX_STR_DIGITS = 4300


class _SampleRng(ScriptedRng):
    """ScriptedRng plus ``sample``: the first *k* items, no draw consumed."""

    def sample(self, population, k):
        return list(population)[:k]


class TestZeroRunAmplify:
    """``randint(1, 0)`` on an all-zero region killed the fuzz loop."""

    @pytest.mark.parametrize("size", [4, 8, 4096])
    def test_regression_all_zero_is_unchanged(self, size):
        data = bytes(size)

        assert zero_run_amplify(data, RandPool(seed=0)) == data

    def test_mixed_input_grows_the_run(self):
        """Falsification: with room to grow, the run extends by the scripted amount."""
        data = b"ab\x00\x00cdef"
        # _region: length 8, offset 0 (the whole buffer); then extend = 1.
        rng = _SampleRng(randints=[len(data), 0, 1])

        # The run at [2, 4) grows one byte to the left; no other zero to compensate.
        assert zero_run_amplify(data, rng) == b"a\x00\x00\x00cdef"

    def test_zero_prefix_then_one_byte(self):
        """Adversarial: one non-zero byte leaves exactly one byte of room."""
        data = bytes(15) + b"\x01"

        assert len(zero_run_amplify(data, RandPool(seed=0))) == len(data)


class TestSanitizerParse:
    """Target stderr must not stall or raise in ``SanitizerReport.parse``."""

    @staticmethod
    def _timed(text):
        t0 = time.perf_counter()
        report = SanitizerReport.parse(text)
        return report, time.perf_counter() - t0

    def test_regression_blank_lines_parse_linearly(self):
        report, took = self._timed(_ASAN_HEAD + "\n" * (_STDERR_CAP - len(_ASAN_HEAD)))

        assert report.error_type == "heap-buffer-overflow"
        assert took < _PARSE_BUDGET_S

    def test_regression_repeated_sanitizer_word(self):
        _report, took = self._timed(
            "==1==ERROR: AddressSanitizer: SEGV " + "AddressSanitizer " * 3800
        )

        assert took < _PARSE_BUDGET_S

    def test_regression_huge_access_size(self):
        text = (
            _ASAN_HEAD + "READ of size " + "9" * (_INT_MAX_STR_DIGITS + 1) + " at 0x1 thread T0\n"
        )

        assert SanitizerReport.parse(text).error_type == "heap-buffer-overflow"

    def test_frames_still_parsed(self):
        """Falsification: indented frames and the fault address survive the tighter regexes."""
        text = (
            _ASAN_HEAD
            + "READ of size 4 at 0x602000000011 thread T0\n    #0 0x4011d6 in foo a.c:3:5\n    #1 0x4011f0 in main a.c:9:3\n"
        )
        report = SanitizerReport.parse(text)

        assert report.frames[:2] == ["foo", "main"]
        assert report.fault_addr == "0x602000000011"
        assert report.access_size == 4


class TestRunningMomentsBounded:
    """Unwindowed moments kept every sample in a deque nobody read."""

    _N = 5000

    def test_regression_no_sample_buffer(self):
        rm = RunningMoments()
        for i in range(self._N):
            rm.update(float(i % 7))

        assert len(rm._buf) == 0
        assert rm.save()["buf"] == []

    def test_moments_unchanged(self):
        """Falsification: mean/variance match the closed form for 0..N-1."""
        rm = RunningMoments()
        for i in range(self._N):
            rm.update(float(i))

        assert rm.mean == pytest.approx((self._N - 1) / 2)
        assert rm.variance == pytest.approx(self._N * (self._N + 1) / 12)

    def test_windowed_buffer_still_capped(self):
        """Adversarial: window mode needs its buffer and keeps it at the window."""
        window = 16
        rm = RunningMoments(window=window)
        for i in range(self._N):
            rm.update(float(i))

        assert len(rm._buf) == window
        assert rm.mean == pytest.approx(self._N - (window + 1) / 2)


def test_regression_hierarchical_priors_flag_honest():
    """``supports_priors`` must match what ``init_arm`` accepts (Hard Rule 40)."""
    sched = HierarchicalBanditScheduler()
    if not sched.supports_priors:
        sched.init_arm("bit_flip")
        return

    sched.init_arm("bit_flip", 3.0, 1.0)
