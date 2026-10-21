"""Regression: seed calibration batches EdgeTracker pruning and stays bounded.

The calibration pass records one execution per corpus seed.  Before the
fix, ``record_edges`` called ``_maybe_prune`` on every insertion that
pushed the tracked-seed count over ``max_tracked_seeds``, so a wide
corpus paid the full prune cost many times during startup and also
evicted seeds it had just recorded.  Disabling pruning outright fixed the
cost but let ``seed_edges`` grow without bound (~100 bytes per edge per
seed), exhausting RAM on 10^5-seed corpora.  The pass now lets the tracked
count overshoot by ``CALIBRATION_PRUNE_SLACK``, prunes early when the
per-seed edge sets exceed an entry budget, folds seeds that add no edge
into the global statistics without keeping per-seed state, and enforces
the normal bound once at the end.
"""

from __future__ import annotations

import itertools

from fuzzer_tool.core.edge_tracker import CALIBRATION_PRUNE_SLACK, EdgeTracker
from fuzzer_tool.services import fuzzer as fuzzer_mod
from fuzzer_tool.services.fuzzer import Fuzzer


class _FakeShm:
    """Minimal SHM surface required by ``_calibrate_seed_baselines``."""

    def __init__(self, edge_sets=None):
        self._sets = itertools.cycle(edge_sets or [{1, 2, 3}])
        self._cur = {1, 2, 3}

    def is_new_coverage_with_edges(self):
        self._cur = set(next(self._sets))
        return True, self._cur

    def get_edge_counts(self):
        return dict.fromkeys(self._cur, 1)

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

    def __init__(self, tracker: EdgeTracker, corpus: list[bytes], edge_sets=None):
        self._seed_calibration = True
        self.use_coverage = True
        self.shm_cov = _FakeShm(edge_sets)
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

    def _frontier_observe(self, _seed):
        pass

    def _seed_key(self, seed):
        return str(hash(seed))

    def _report_comparison_reach(self, _n):
        pass

    def _report_resume_reach(self, _restored):
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


def test_calibration_slack_then_single_bound_enforcement(monkeypatch) -> None:
    """The loop runs with slack > 1; the bound is enforced once after it."""
    tracker = EdgeTracker(max_tracked_seeds=2)
    corpus = [bytes([i]) for i in range(12)]
    fake = _FakeFuzzer(tracker, corpus, [{100 + i} for i in range(12)])

    slack_seen: list[float] = []
    peak = [0]
    original = EdgeTracker._maybe_prune

    def spy(self):
        slack_seen.append(self._prune_slack)
        peak[0] = max(peak[0], len(self.seed_edges))
        return original(self)

    monkeypatch.setattr(EdgeTracker, "_maybe_prune", spy)

    Fuzzer._calibrate_seed_baselines(fake)

    assert CALIBRATION_PRUNE_SLACK in slack_seen  # loop ran with the slack
    assert slack_seen[-1] == 1.0  # restored before the final prune
    # Bounded during the loop (slack * max + one insertion), not unbounded.
    assert peak[0] <= int(2 * CALIBRATION_PRUNE_SLACK) + 1
    assert len(tracker.seed_edges) <= tracker.max_tracked_seeds
    assert tracker._prune_slack == 1.0


def test_calibration_redundant_seeds_keep_no_per_seed_state() -> None:
    """A seed adding no edge feeds the global counters but is not tracked."""
    tracker = EdgeTracker(max_tracked_seeds=100)
    fake = _FakeFuzzer(tracker, [b"a", b"b", b"c"], [{1, 2, 3}, {1, 2}, {2, 3}])

    Fuzzer._calibrate_seed_baselines(fake)

    assert len(tracker.seed_edges) == 1  # only the first seed owns anything
    assert tracker.cumulative_edges == {1, 2, 3}
    assert tracker._global_edge_hits == {1: 2, 2: 3, 3: 2}
    assert tracker._aggregate_total_count == 7
    assert not tracker._absorb_pending


def test_calibration_entry_budget_forces_early_prune(monkeypatch) -> None:
    """Wide seeds trip the edge-entry budget before the seed-count slack."""
    monkeypatch.setattr(fuzzer_mod, "CALIBRATION_EDGE_ENTRY_BUDGET", 10)
    tracker = EdgeTracker(max_tracked_seeds=10)
    sets = [set(range(i * 4, i * 4 + 4)) for i in range(8)]
    fake = _FakeFuzzer(tracker, [bytes([i]) for i in range(8)], sets)

    pruned: list[int] = []
    original = EdgeTracker.prune_to_bound

    def spy(self):
        pruned.append(len(self.seed_edges))
        return original(self)

    monkeypatch.setattr(EdgeTracker, "prune_to_bound", spy)

    Fuzzer._calibrate_seed_baselines(fake)

    assert pruned, "entry budget never forced a prune"


def test_calibration_uses_shm_arrays_for_redundant_seeds() -> None:
    """With ``get_edge_arrays`` and no confirmation rerun, no per-seed dict is built."""
    import numpy as np

    class _ArrayShm(_FakeShm):
        def get_edge_counts(self):
            raise AssertionError("redundant seed built a dict")

        def get_edge_arrays(self):
            ids = np.array(sorted(self._cur), dtype=np.uint32)
            return ids, np.ones(len(ids), dtype=np.int64)

    tracker = EdgeTracker(max_tracked_seeds=100)
    fake = _FakeFuzzer(tracker, [b"a", b"b", b"c"], [{1, 2, 3}, {1, 2}, {2, 3}])
    fake._confirmed_edges = None
    fake.shm_cov = _ArrayShm([{1, 2, 3}, {1, 2}, {2, 3}])
    # The first seed (not redundant) still needs counts; allow it.
    first = fake.shm_cov.get_edge_counts

    calls = []

    def counts_once():
        calls.append(1)
        if len(calls) > 1:
            return first()
        return {1: 1, 2: 1, 3: 1}

    fake.shm_cov.get_edge_counts = counts_once

    Fuzzer._calibrate_seed_baselines(fake)

    assert len(calls) == 1  # only the non-redundant first seed
    assert tracker._global_edge_hits == {1: 2, 2: 3, 3: 2}
