"""Edge first-seen mixed two clocks in one map.

``record_edges`` stamped ``_edge_first_seen`` with ``len(cumulative_edges)``
(discovery ordinal); the per-exec ``record_edge_lifetimes`` stamped exec
counts into the same map. Lifetimes then read ``last_exec - ordinal`` and
the frontier compared ordinals against exec counts.

Fix: discovery ordinal stays in ``_edge_first_seen`` (frontier clock); exec
stamps go to ``_edge_first_exec`` (lifetime clock).
"""

from fuzzer_tool.core.edge_tracker import EdgeTracker

_DISCOVERY_EXEC = 5_000
_LAST_EXEC = 9_000
_LATE_EXEC = 200_000


def _fuzz_round(et: EdgeTracker, seed: str, edges: set[int], exec_count: int) -> None:
    """Mirror fuzz_round order: _record_edges, then _track_edges."""
    et.record_edges(seed, edges)
    et.record_edge_lifetimes(edges, exec_count)


def test_regression_lifetime_in_execs():
    # Falsification: lifetime must be last_exec - discovery_exec, not
    # last_exec - discovery ordinal.
    et = EdgeTracker()
    edges = {1, 2, 3}
    _fuzz_round(et, "s", edges, _DISCOVERY_EXEC)
    et.record_edge_lifetimes(edges, _LAST_EXEC)

    assert et.edge_lifetime_stats()["max"] == _LAST_EXEC - _DISCOVERY_EXEC


def test_regression_exec_stamps_keep_frontier():
    # Adversarial: a late exec stamp on an edge record_edges never saw must
    # not drag the coverage clock to exec scale and empty the frontier.
    et = EdgeTracker()
    for i in range(8):
        _fuzz_round(et, f"s{i}", {10 * i + 1}, 100 * i)
    before = set(et.discovery_frontier_edges())

    et.record_edge_lifetimes({999}, _LATE_EXEC)

    assert before
    assert et.discovery_frontier_edges() == before


def test_regression_first_exec_persists():
    et = EdgeTracker()
    _fuzz_round(et, "s", {7}, _DISCOVERY_EXEC)

    back = EdgeTracker()
    back.from_dict(et.to_dict())

    assert back._edge_first_exec == {7: _DISCOVERY_EXEC}
    assert back._edge_first_seen == et._edge_first_seen


def test_regression_legacy_state_loads():
    # State written before the split has no "edge_first_exec" key.
    et = EdgeTracker()
    et.from_dict({"edge_first_seen": {"7": 3}})

    assert et._edge_first_exec == {}
    assert et._edge_first_seen == {7: 3}
