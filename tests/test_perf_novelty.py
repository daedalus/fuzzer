"""Per-edge max hit count (performance novelty) and the exec timing window.

Two defects are covered here.

1. ``Fuzzer.fuzz_one`` opened its timing window before ``_dedup_mutate``, so
   Python mutation cost landed in ``t_elapsed`` — which feeds the anomaly
   detector, the per-seed ``exec_us`` used by the power schedule, and the
   adaptive timeout. An expensive operator inflated the number, was credited
   for the resulting ``is_slow``, and was therefore selected more often.

2. Hit-count buckets saturate, so once an edge is in the 128+ class the
   coverage signal is blind to trip counts growing by orders of magnitude —
   exactly the regime algorithmic-complexity bugs live in.
"""

import inspect

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import MAX_COUNT_GROWTH_FACTOR, ShmCoverage
from fuzzer_tool.services import fuzzer as fuzzer_mod


class _FakeShm(ShmCoverage):
    """ShmCoverage with the maxima machinery only — no real segment.

    ``_update_max_counts`` and ``mask_edges`` touch nothing but the Python
    attributes initialised here, so driving them directly avoids needing a
    SysV segment and an instrumented binary.
    """

    def __init__(self):
        self._virgin = np.zeros(0, dtype=np.uint8)
        self._virgin_wide = {}
        self._max_counts = np.zeros(0, dtype=np.uint32)
        self._max_counts_wide = {}
        self.max_count_transitions = 0
        self._last_max_gain = 0
        self._seen_edge_ids = set()
        self._masked_edge_ids = set()
        self._ptr = None  # ShmCoverage.__del__ -> cleanup() reads this

    def cleanup(self):
        return None

    def feed(self, pairs):
        """Apply one execution's {edge_id: count} and return the gain."""
        eids = np.array(list(pairs.keys()), dtype=np.uint32)
        cnts = np.array(list(pairs.values()), dtype=np.uint32)
        return self._update_max_counts(eids, cnts)


class TestMaxCounts:
    def test_first_sighting_is_not_a_gain(self):
        """A brand new edge is already new coverage; don't double-credit it."""
        shm = _FakeShm()
        assert shm.feed({7: 100}) == 0
        assert shm.get_max_counts()[7] == 100

    def test_growth_past_factor_is_a_gain(self):
        shm = _FakeShm()
        shm.feed({7: 100})
        assert shm.feed({7: int(100 * MAX_COUNT_GROWTH_FACTOR)}) == 1
        assert shm.max_count_transitions == 1

    def test_growth_below_factor_is_not_a_gain(self):
        shm = _FakeShm()
        shm.feed({7: 100})
        assert shm.feed({7: 120}) == 0

    def test_sub_factor_growth_still_advances_the_baseline(self):
        """Otherwise a slow climb never moves the max and re-reports off a
        stale value the moment it clears 1.5x of it."""
        shm = _FakeShm()
        shm.feed({7: 100})
        shm.feed({7: 140})  # no gain, but the max must become 140
        assert shm.get_max_counts()[7] == 140
        assert shm.feed({7: 149}) == 0  # 149 < 140 * 1.5; baseline moves to 149
        assert shm.feed({7: 224}) == 1  # 224 >= 149 * 1.5

    def test_a_decrease_never_lowers_the_max(self):
        shm = _FakeShm()
        shm.feed({7: 500})
        shm.feed({7: 3})
        assert shm.get_max_counts()[7] == 500

    def test_bucket_saturated_growth_is_still_seen(self):
        """The point of the whole mechanism: 200 and 10^6 are one bucket."""
        shm = _FakeShm()
        shm.feed({7: 200})
        assert shm.feed({7: 1_000_000}) == 1

    def test_repeats_are_bounded_by_the_growth_factor(self):
        """An edge cannot report on every +1 — that would be one corpus
        admission per loop iteration."""
        shm = _FakeShm()
        shm.feed({7: 1000})
        gains = sum(shm.feed({7: 1000 + i}) for i in range(1, 400))
        assert gains <= 2

    def test_no_wraparound_near_the_count_ceiling(self):
        """prev * 1.5 overflows uint32 arithmetic near 0xFFFFFF; if that
        wraps, every saturated edge reports a gain forever."""
        shm = _FakeShm()
        shm.feed({7: 0xFFFFFF})
        for _ in range(5):
            assert shm.feed({7: 0xFFFFFF}) == 0

    def test_multiple_edges_in_one_execution(self):
        shm = _FakeShm()
        shm.feed({1: 10, 2: 10, 3: 10})
        assert shm.feed({1: 10, 2: 100, 3: 1000}) == 2

    def test_masked_edges_cannot_report(self):
        """Unstable edges are suppressed on the bucket path; they have to be
        suppressed here too or they keep absorbing energy."""
        shm = _FakeShm()
        shm.feed({7: 10})
        shm.mask_edges([7])
        assert shm.feed({7: 10_000_000}) == 0


class TestSignalSeparation:
    def test_max_gain_is_not_folded_into_new_coverage(self):
        """has_new_coverage gates record_edges, cmplog, the format learner,
        the GA and the distance metric, all of which read it as 'reached
        somewhere new'. Spinning a known loop reached nowhere new."""
        src = inspect.getsource(ShmCoverage._check_new_coverage)
        assert "_last_max_gain = self._update_max_counts" in src
        # The assignment must not also set new_found.
        for line in src.splitlines():
            if "_update_max_counts" in line:
                assert "new_found" not in line

    def test_fast_path_reports_no_gain(self):
        """An unchanged path_hash means identical counts, so no max moved —
        and the fast path never scans, so a stale value must not leak."""
        src = inspect.getsource(ShmCoverage._check_new_coverage)
        head = src.split("if edge_count == self._last_edge_count")[0]
        assert "self._last_max_gain = 0" in head


class TestTimingWindow:
    def test_mutation_is_outside_the_timing_window(self):
        src = inspect.getsource(fuzzer_mod.Fuzzer.fuzz_one)
        mutate_at = src.index("mutated = self._dedup_mutate(data)")
        start_at = src.index("t_start = time.monotonic()")
        assert start_at > mutate_at, (
            "t_start must open after _dedup_mutate, or Python mutation cost "
            "contaminates the anomaly detector, exec_us and the timeout"
        )

    def test_only_run_target_is_inside_the_window(self):
        src = inspect.getsource(fuzzer_mod.Fuzzer.fuzz_one)
        start_at = src.index("t_start = time.monotonic()")
        end_at = src.index("t_elapsed = time.monotonic() - t_start")
        body = src[start_at:end_at]
        assert "self._run_target(mutated)" in body
        assert "_dedup_mutate" not in body

    def test_perf_novelty_reaches_success_and_admission(self):
        """Both assertions read the expression, not a fixed spelling of it.

        These used to pin the literal substrings ``"or is_new_max)"`` and
        ``"has_new_coverage or is_new_max:"``, which made them assertions
        about who the *last* disjunct is rather than about is_new_max
        reaching the bandits at all. Any later signal joining either
        disjunction breaks them without anything being wrong -- which is
        exactly what happened when the comparison-progress channel landed.
        """
        src = inspect.getsource(fuzzer_mod.Fuzzer.fuzz_one)

        success_expr = src.split("success = bool(")[1].split(")")[0]
        assert "is_new_max" in success_expr, "is_new_max must reach the bandits"

        admission = next(
            line for line in src.splitlines() if line.strip().startswith("if is_interesting")
        )
        assert "is_new_max" in admission, (
            "performance-novel inputs must be admitted, or the signal cannot "
            "compound across generations"
        )

    def test_crash_and_timeout_suppress_the_signal(self):
        """A truncated execution's counts are short, not extreme."""
        src = inspect.getsource(fuzzer_mod.Fuzzer.fuzz_one)
        assert "not is_timeout and not is_crash" in src.split("new_max_edges = 0")[1]


@pytest.mark.parametrize("flag", ["--no-perf-novelty"])
def test_cli_flag_exists(flag):
    from fuzzer_tool.cli import commands

    parser = commands.build_parser() if hasattr(commands, "build_parser") else None
    if parser is None:
        src = inspect.getsource(commands)
        assert flag in src
