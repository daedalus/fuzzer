"""Regression: seed calibration duplicated every seed's edge set.

``_calibrate_seed_baselines`` passed ``target_name`` unconditionally, so
``EdgeTracker.seed_target_edges`` stored a second copy of each seed's edges.
Its only reader is the multi-target weight in ``seed_picker``, and
calibration returns early in multi-target mode: the copy was never read.
Measured on ffmpeg_read_asan.so, 480 seeds: 93.5 MiB traced, ~2.5x that in
RSS under preloaded ASAN.
"""

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.fuzzer import Fuzzer

# Wide enough that a duplicated copy is not noise; ffmpeg seeds hit ~10k.
_EDGES_PER_SEED = 4096


class _FakeShm:
    """Reports one fixed edge set per execution."""

    def __init__(self, edges: set[int]):
        self._edges = edges

    def is_new_coverage_with_edges(self):
        return True, set(self._edges)

    def get_edge_counts(self):
        return dict.fromkeys(self._edges, 1)

    def read_stack_depth(self):
        return 0

    def read_path_hash(self):
        return 1


def _fuzzer(seeds: list[bytes], edges: set[int], *, multi: bool = False) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f.use_coverage = True
    f.multi_targets = ["a", "b"] if multi else []
    f.target = "/x/ffmpeg_read_asan.so"
    f.corpus = list(seeds)
    f.shm_cov = _FakeShm(edges)
    f._edge_tracker = EdgeTracker()
    f._edge_ledger = None
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
    return f


def test_regression_calibration_target_edges():
    """Falsification: single-target calibration stores no per-target copy."""
    edges = set(range(1_000_000, 1_000_000 + _EDGES_PER_SEED))
    seeds = [b"seed-a", b"seed-b"]
    f = _fuzzer(seeds, edges)

    f._calibrate_seed_baselines()

    tracker = f._edge_tracker
    assert tracker.seed_target_edges == {}
    assert tracker.target_cumulative_edges == {}

    # The coverage itself must still land.
    for s in seeds:
        assert tracker.seed_edges[seed_key(s)] == edges


def test_calibration_matches_fuzz_loop_contract():
    """Adversarial: multi-target mode skips calibration, so nothing is recorded."""
    edges = set(range(_EDGES_PER_SEED))
    f = _fuzzer([b"seed"], edges, multi=True)

    f._calibrate_seed_baselines()

    assert f._edge_tracker.seed_edges == {}
    assert f._edge_tracker.seed_target_edges == {}
