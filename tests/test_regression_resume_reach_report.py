"""Regression: --resume hid the restored-vs-reproducible coverage gap.

The tracker's cumulative edge set is restored whole, but the SHM map is
rebuilt by re-running live seeds, so "Edges discovered" read ~11% low on
png_read_nosan (349 -> 309) with no hint why. Calibration now prints both.
"""

import re

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.fuzzer import Fuzzer


class _FakeShm:
    def __init__(self, edges: set[int]):
        self._edges = edges
        self.cumulative_edges = 0

    def is_new_coverage_with_edges(self):
        fresh = len(self._edges) - self.cumulative_edges
        self.cumulative_edges = len(self._edges)
        return fresh > 0, set(self._edges)

    def get_edge_counts(self):
        return dict.fromkeys(self._edges, 1)

    def read_stack_depth(self):
        return 0

    def read_path_hash(self):
        return 1


def _fuzzer(resume: bool, restored: set[int], live: set[int]) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f.use_coverage = True
    f._seed_calibration = True
    f.multi_targets = []
    f.target = "/x/t"
    f.resume = resume
    f.corpus = [b"seed"]
    f.shm_cov = _FakeShm(live)
    f._edge_tracker = EdgeTracker()
    f._edge_tracker.cumulative_edges = set(restored)
    f._edge_ledger = None
    f._distance = None
    f._last_perf_deltas = {}
    f.crash_count = 0
    f.timeout_count = 0
    f._run_target = lambda data: (0, "")
    f._is_crash = lambda rc, err: False
    f._confirm_new_coverage = lambda data, shm, has_new, ids: (has_new, ids)
    f._only_confirmed = lambda counts: counts
    f._seed_key = seed_key
    f._report_comparison_reach = lambda n: None
    f._report_edge_id_stability = lambda seed: None
    f._report_coverage_noise = lambda seed: None
    return f


def _line(capsys):
    m = re.search(
        r"Resume coverage: (\d+) restored \(tracker\), (\d+) reproducible.*\((\d+)%\)",
        capsys.readouterr().out,
    )
    return tuple(map(int, m.groups())) if m else None


def test_regression_resume_reach_report(capsys):
    """Falsification: 10 restored, live seeds reproduce 8 -> both counts shown."""
    f = _fuzzer(True, set(range(10)), set(range(8)))
    f._calibrate_seed_baselines()
    assert _line(capsys) == (10, 8, 80)


def test_resume_reach_silent_on_fresh_run(capsys):
    """Adversarial: no restored coverage, or not resuming, prints nothing."""
    _fuzzer(False, set(range(10)), set(range(8)))._calibrate_seed_baselines()
    assert _line(capsys) is None
    _fuzzer(True, set(), set(range(8)))._calibrate_seed_baselines()
    assert _line(capsys) is None


def test_resume_reach_never_exceeds_restored(capsys):
    """Adversarial: seeds reaching new edges cannot report >100%."""
    _fuzzer(True, set(range(4)), set(range(9)))._calibrate_seed_baselines()
    assert _line(capsys) == (4, 4, 100)
