"""Multi-target edge last-seen folds each target's SHM generation tags.

Single-target runs date last-seen from SHM generation tags (one fold per
256 execs). Multi-target runs kept the per-edge per-exec writes: each target
has its own table and generation counter, and the tracker had one
generation -> exec clock. Now each table gets its own clock; the active
target's clock is stamped per exec, and a fold keeps the later of the
current and folded last-seen, since targets share edge ids. Results must
equal the per-exec writes.

Also: seed calibration calls ``record_edges`` outside the fuzz round with
``len(cumulative_edges)`` as its "exec count". With generations attached
that stamped the current generation, so edges hit only by that exec got
an edge count as last-seen.
"""

import functools
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import _wire_lifetimes

_SIZE = 64
_GEN_SPAN = 256


def _hits(target: int, c: int) -> set[int]:
    """Shifting edge set for exec *c* of *target*; ids overlap across targets."""
    base = range(1, 40) if target == 0 else range(20, 60)
    return {i for i in base if (i * 7 + c) % 5 != 0 and (i + c // 50) % 9 != 0}


def _plant(cov: ShmCoverage, edges: set[int]) -> None:
    gen = cov.read_generation()
    for e in edges:
        cov._entries[e].edge_id = e
        cov._entries[e].count = (gen << 24) | 1


@pytest.fixture
def covs():
    cs = [ShmCoverage(size=_SIZE), ShmCoverage(size=_SIZE)]
    yield cs
    for c in cs:
        c.cleanup()


def _attached(covs, state):
    et = EdgeTracker(map_size=_SIZE)
    for t, cov in enumerate(covs):
        et.attach_generations(cov.read_generation, cov.entry_tags, key=t)
        cov.on_table_loss = functools.partial(et.fold_tags, key=t)
    et.follow_source(lambda: state["target"])
    return et


def _schedule(c: int) -> int:
    """Uneven, bursty target choice: generations advance at different rates."""
    return 1 if c % 3 == 0 or 400 < c < 450 else 0


def _run(covs, et, oracle, n_execs, state, rerun_every=0):
    for c in range(1, n_execs + 1):
        t = _schedule(c)
        state["target"] = t
        cov = covs[t]
        cov.reset_edge_map()
        hits = _hits(t, c)
        _plant(cov, hits)
        et.record_edge_lifetimes(hits, c)
        oracle.record_edge_lifetimes(hits, c)
        if rerun_every and c % rerun_every == 0:
            cov.reset_edge_map()  # calibration rerun of the same input
            _plant(cov, hits)


def test_control_oracle_matches_itself(covs):
    """Rule 46: two per-exec trackers fed the same execs agree."""
    a, b = EdgeTracker(map_size=_SIZE), EdgeTracker(map_size=_SIZE)
    for c in range(1, 600):
        t = _schedule(c)
        a.record_edge_lifetimes(_hits(t, c), c)
        b.record_edge_lifetimes(_hits(t, c), c)
    assert (a._edge_last_seen, a._edge_first_seen) == (b._edge_last_seen, b._edge_first_seen)


@pytest.mark.parametrize("n_execs", [10, 300, 900])
@pytest.mark.parametrize("rerun_every", [0, 7])
def test_multi_fold_matches_per_exec(covs, n_execs, rerun_every):
    """Falsification: interleaved targets (each wrapping at its own pace,
    shared edge ids) fold to exactly the per-exec last-seen."""
    state = {"target": 0}
    et = _attached(covs, state)
    oracle = EdgeTracker(map_size=_SIZE)
    _run(covs, et, oracle, n_execs, state, rerun_every)

    assert et.edge_lifetime_stats() == oracle.edge_lifetime_stats()
    assert et._edge_last_seen == oracle._edge_last_seen
    assert et._edge_first_seen == oracle._edge_first_seen


def test_shared_edge_keeps_latest_target_hit(covs):
    """Adversarial: target 1 hits edge 30 late, target 0's table still
    holds an older tag for it; the fold must not move last-seen back."""
    state = {"target": 0}
    et = _attached(covs, state)
    for c, t in ((1, 0), (2, 1), (3, 0)):
        state["target"] = t
        covs[t].reset_edge_map()
        hits = {30} if c < 3 else {5}
        _plant(covs[t], hits)
        et.record_edge_lifetimes(hits, c)
    et._sync_last_seen()
    assert et._edge_last_seen[30] == 2


def test_wiring_multi_target(covs):
    """Falsification: per-target tables are wired, one clock each."""
    f = SimpleNamespace(
        shm_cov=None,
        multi_targets=["a", "b"],
        _target_shm_covs={"a": covs[0], "b": covs[1]},
        target="a",
        _edge_tracker=EdgeTracker(map_size=_SIZE),
    )
    _wire_lifetimes(f)
    et = f._edge_tracker
    assert covs[0].on_table_loss is not None and covs[1].on_table_loss is not None
    assert et.record_edge_lifetimes != EdgeTracker(map_size=_SIZE).record_edge_lifetimes


@pytest.mark.parametrize("tables", [{}, {"a": None}])
def test_wiring_multi_without_all_tables(covs, tables):
    """Adversarial: a target without its own table (shared-SHM fallback)
    keeps per-exec writes for the whole run."""
    shm = {k: covs[0] for k, v in tables.items()}
    f = SimpleNamespace(
        shm_cov=covs[1],
        multi_targets=["a", "b"],
        _target_shm_covs=shm,
        target="a",
        _edge_tracker=EdgeTracker(map_size=_SIZE),
    )
    _wire_lifetimes(f)
    assert covs[0].on_table_loss is None and covs[1].on_table_loss is None


def test_regression_calibration_record_edges_keeps_exec_clock(covs):
    """record_edges (calibration, outside the round) must not date the
    current generation with len(cumulative_edges)."""
    cov = covs[0]
    et = EdgeTracker(map_size=_SIZE)
    et.attach_generations(cov.read_generation, cov.entry_tags)
    cov.on_table_loss = et.fold_tags

    for c in range(1, 30):
        cov.reset_edge_map()
        hits = {c % 50 + 1, 55}
        _plant(cov, hits)
        et.record_edge_lifetimes(hits, 1000 + c)
        et.record_edges(f"seed{c}", hits)  # calibration-style call
    et._sync_last_seen()
    assert et._edge_last_seen[55] == 1029
    assert et._edge_last_seen[2] == 1001
