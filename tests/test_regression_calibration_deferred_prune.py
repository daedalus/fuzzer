"""Regression: seed calibration defers EdgeTracker pruning.

The calibration pass records one execution per corpus seed.  Before the
fix, ``record_edges`` called ``_maybe_prune`` on every insertion that
pushed the tracked-seed count over ``max_tracked_seeds``, so a wide
corpus paid the full prune cost many times during startup and also
evicted seeds it had just recorded.  The fix disables pruning while the
calibration loop runs and performs one prune once it finishes.
"""

from __future__ import annotations

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.fuzzer import Fuzzer


class _FakeShm:
    """Minimal SHM surface required by ``_calibrate_seed_baselines``."""

    def is_new_coverage_with_edges(self):
        return True, {1, 2, 3}

    def get_edge_counts(self):
        return {1: 1, 2: 1, 3: 1}

    def read_stack_depth(self):
        return 0

    def read_path_hash(self):
        return 0

    def compute_path_hash_from_edges(self, _edges):
        return 0


class _FakeCorpusManager:
    def save_timeout(self, _seed):
        pass


class _FakeFuzzer:
    """A Fuzzer-shaped object that drives only ``_calibrate_seed_baselines``."""

    def __init__(self, tracker: EdgeTracker, corpus: list[bytes]):
        self._seed_calibration = True
        self.use_coverage = True
        self.shm_cov = _FakeShm()
        self.multi_targets = False
        self.corpus = corpus
        self._edge_tracker = tracker
        self.crash_count = 0
        self.timeout_count = 0
        self._last_perf_deltas: dict[str, int] = {}
        self._crash_explorer = None
        self._corpus_manager = _FakeCorpusManager()

    def _run_target(self, _seed):
        return 0, b""

    def _is_crash(self, _rc, _stderr):
        return False

    def _seed_crash_path(self):
        pass

    def _tag_seed_distance(self, _seed):
        pass

    def _confirm_new_coverage(self, _seed, _shm, has_new, edges):
        return has_new, edges

    def _only_confirmed(self, counts):
        return counts

    def _strata_observe(self, _seed, _edges):
        pass

    def _seed_key(self, seed):
        return str(hash(seed))

    def _report_comparison_reach(self, _n):
        pass

    def _report_edge_id_stability(self, _seed):
        pass

    def _report_coverage_noise(self, _seed):
        pass


def test_prune_flag_blocks_eviction() -> None:
    """``_prune_enabled=False`` must leave ``seed_edges`` untouched."""
    tracker = EdgeTracker(max_tracked_seeds=2)
    tracker.seed_edges["a"] = {1, 2, 3}
    tracker.seed_edges["b"] = {1, 2}
    tracker.seed_edges["c"] = {3}

    tracker._prune_enabled = False
    tracker._maybe_prune()
    assert len(tracker.seed_edges) == 3

    tracker._prune_enabled = True
    tracker._maybe_prune()
    assert len(tracker.seed_edges) <= tracker.max_tracked_seeds


def test_calibration_defers_prune_until_after_loop(monkeypatch) -> None:
    """Prune fires with the flag restored, not while the loop is running."""
    tracker = EdgeTracker(max_tracked_seeds=1)
    fake = _FakeFuzzer(tracker, [b"a", b"b", b"c"])

    seen_true: list[bool] = []
    original = EdgeTracker._maybe_prune

    def spy(self):
        seen_true.append(self._prune_enabled)
        return original(self)

    monkeypatch.setattr(EdgeTracker, "_maybe_prune", spy)

    Fuzzer._calibrate_seed_baselines(fake)

    # The pass must have disabled pruning (we saw the flag False), and the
    # bound must be enforced exactly once after the loop (one True).
    assert False in seen_true
    assert seen_true.count(True) == 1
    assert len(tracker.seed_edges) <= tracker.max_tracked_seeds
