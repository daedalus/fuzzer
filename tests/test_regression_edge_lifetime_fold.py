"""Edge last-seen comes from the SHM generation tags, folded at table loss.

``record_edge_lifetimes`` wrote ``_edge_last_seen`` per edge per exec
(~650 us at 8k edges). Every SHM entry already carries the generation tag of
the last exec that hit it, so the tracker maps generation -> exec count in
O(1) per exec and folds the tags once per 256 execs (wrap wipe), on resize,
and before anything reads last-seen.
"""

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.edge_tracker import EdgeTracker

_SIZE = 64  # table entries; edge id == slot in these tests
_N_EDGES = 40
_GEN_SPAN = 256


def _hits(c: int) -> set[int]:
    """Deterministic, shifting edge set for exec *c*."""
    return {i for i in range(1, _N_EDGES) if (i * 7 + c) % 5 != 0 and (i + c // 50) % 9 != 0}


def _plant(cov: ShmCoverage, edges: set[int]) -> None:
    """Write *edges* as the shim would in the current generation."""
    gen = cov.read_generation()
    for e in edges:
        cov._entries[e].edge_id = e
        cov._entries[e].count = (gen << 24) | 1


@pytest.fixture
def cov():
    c = ShmCoverage(size=_SIZE)
    yield c
    c.cleanup()


def _attached(cov: ShmCoverage) -> EdgeTracker:
    et = EdgeTracker(map_size=_SIZE)
    et.attach_generations(cov.read_generation, cov.entry_tags)
    cov.on_table_loss = et.fold_tags
    return et


def _run(cov, et, oracle, n_execs, rerun_every=0):
    """Drive *n_execs* execs; optional same-input reruns bump the generation."""
    for c in range(1, n_execs + 1):
        cov.reset_edge_map()
        hits = _hits(c)
        _plant(cov, hits)
        et.record_edge_lifetimes(hits, c)
        if oracle is not None:
            oracle.record_edge_lifetimes(hits, c)
        if rerun_every and c % rerun_every == 0:
            cov.reset_edge_map()  # calibration rerun of the same input
            _plant(cov, hits)


def test_control_oracle_matches_itself():
    """Rule 46: two per-exec trackers fed the same execs agree."""
    a, b = EdgeTracker(map_size=_SIZE), EdgeTracker(map_size=_SIZE)
    for c in range(1, 300):
        a.record_edge_lifetimes(_hits(c), c)
        b.record_edge_lifetimes(_hits(c), c)
    assert a._edge_last_seen == b._edge_last_seen


@pytest.mark.parametrize("n_execs", [10, 255, 256, 600])
@pytest.mark.parametrize("rerun_every", [0, 7])
def test_fold_matches_per_exec_oracle(cov, n_execs, rerun_every):
    """Falsification: across wraps (and reruns), folded last-seen equals the
    per-exec dict the old path wrote."""
    et = _attached(cov)
    oracle = EdgeTracker(map_size=_SIZE)
    _run(cov, et, oracle, n_execs, rerun_every)

    assert et.edge_lifetime_stats() == oracle.edge_lifetime_stats()
    et._sync_last_seen()
    assert et._edge_last_seen == oracle._edge_last_seen
    assert et._edge_first_seen == oracle._edge_first_seen


def test_attached_path_skips_per_edge_last_seen_writes(cov):
    """Falsification: between folds, last-seen is not touched per exec."""
    et = _attached(cov)
    _run(cov, et, None, 20)
    assert et._edge_last_seen == {}  # nothing folded yet: no wrap, no read


def test_wrap_folds_before_wipe(cov):
    """Adversarial: the wrap wipe must not lose the window."""
    et = _attached(cov)
    _run(cov, et, None, _GEN_SPAN + 3)
    assert et._edge_last_seen  # folded by the wrap, before any read
    assert max(et._edge_last_seen.values()) >= _GEN_SPAN - 1


def test_resize_folds_before_discard(cov):
    """Adversarial: resize drops the table; its window must be folded first."""
    et = _attached(cov)
    _run(cov, et, None, 30)
    cov.resize(_SIZE * 2)
    assert et._edge_last_seen
    assert set(et._edge_last_seen) == set(et._edge_first_seen)


def test_phantom_ids_excluded(cov):
    """Adversarial: a table id never reported as an edge (masked/phantom)
    must not enter the lifetime stats."""
    et = _attached(cov)
    _run(cov, et, None, 5)
    _plant(cov, {_SIZE - 1})  # in the table, never passed to the tracker
    stats = et.edge_lifetime_stats()
    oracle = EdgeTracker(map_size=_SIZE)
    for c in range(1, 6):
        oracle.record_edge_lifetimes(_hits(c), c)
    assert stats == oracle.edge_lifetime_stats()


def test_to_dict_folds_first(cov):
    """Adversarial: persisting mid-window must include the unfolded window."""
    et = _attached(cov)
    _run(cov, et, None, 12)
    saved = et.to_dict()["edge_last_seen"]
    assert saved and max(saved.values()) == 12


def test_unattached_tracker_unchanged():
    """Adversarial: non-SHM backends keep the per-exec write."""
    et = EdgeTracker(map_size=_SIZE)
    et.record_edge_lifetimes({1, 2}, 3)
    assert et._edge_last_seen == {1: 3, 2: 3}


def test_report_shows_lifetimes(cov):
    """Falsification: the run report now reads edge_lifetime_stats()."""
    from types import SimpleNamespace

    from fuzzer_tool.services.report import _edge_lifetimes

    et = _attached(cov)
    _run(cov, et, None, 30)
    stats = et.edge_lifetime_stats()
    text = _edge_lifetimes(SimpleNamespace(_edge_tracker=et))
    assert "--- Edge Lifetimes ---" in text
    assert f"max {stats['max']}" in text


def test_report_empty_without_edges():
    """Adversarial: no lifetimes recorded -> no section."""
    from types import SimpleNamespace

    from fuzzer_tool.services.report import _edge_lifetimes

    assert _edge_lifetimes(SimpleNamespace(_edge_tracker=EdgeTracker(map_size=_SIZE))) == ""
    assert _edge_lifetimes(SimpleNamespace()) == ""


@pytest.mark.parametrize("multi", [None, ["t1", "t2"]])
def test_wiring_single_target_only(cov, multi):
    """Falsification + adversarial: single-target SHM folds; multi-target
    (per-target tables, separate generations) keeps per-exec writes."""
    from types import SimpleNamespace

    from fuzzer_tool.services.corpus_manager import _wire_lifetimes

    f = SimpleNamespace(shm_cov=cov, multi_targets=multi, _edge_tracker=EdgeTracker(map_size=_SIZE))
    _wire_lifetimes(f)
    et = f._edge_tracker
    wired = multi is None
    assert (cov.on_table_loss == et.fold_tags) is wired
    assert (et.record_edge_lifetimes == et._record_first_seen) is wired


def test_wiring_without_shm():
    """Adversarial: ptrace/no-shm backends keep per-exec writes."""
    from types import SimpleNamespace

    from fuzzer_tool.services.corpus_manager import _wire_lifetimes

    f = SimpleNamespace(shm_cov=None, multi_targets=None, _edge_tracker=EdgeTracker(map_size=_SIZE))
    _wire_lifetimes(f)
    f._edge_tracker.record_edge_lifetimes({1}, 2)
    assert f._edge_tracker._edge_last_seen == {1: 2}
