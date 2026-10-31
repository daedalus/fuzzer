"""Edge first-seen stamps from the scan's new ids, not a superset test per exec.

``_track_edges`` handed every exec's whole edge set to
``record_edge_lifetimes``, whose ``first.keys() >= edge_set`` test hashed
every live edge (~106 us/exec on FFmpeg) to find the rare unseen one. The
only reader is the end-of-run lifetime statistic. A scanned round now passes
``last_new_ids`` (ids the coverage oracle sees for the first time, phantoms
withdrawn); unscanned rounds (timeouts, no SHM) keep the full set.
"""

from __future__ import annotations

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.test_regression_katz_resume_contract import (  # noqa: F401 - autouse fixture
    _fuzzer,
    _instrumented_target,
    _with_katz,
)

EXECS = [{1, 2, 3}, {2, 3}, {3, 4}, {1, 4}, {5}, {2, 5, 6}]


class _ScannedShm:
    def __init__(self, new_ids):
        self.last_new_ids = frozenset(new_ids)


def _round(tmp_path, monkeypatch):
    _with_katz(monkeypatch, None)
    f = _fuzzer(tmp_path)
    f.shm_cov = object()  # any coverage backend: the round tracks lifetimes
    calls = []
    monkeypatch.setattr(
        f._edge_tracker, "record_edge_lifetimes", lambda edges, n: calls.append(set(edges))
    )
    return f, FuzzRound(f, b"seed"), calls


def test_scanned_round_passes_only_new_ids(tmp_path, monkeypatch):
    f, rnd, calls = _round(tmp_path, monkeypatch)
    f._current_edges_cache = {1, 2, 3, 9}
    rnd._scanned_shm = _ScannedShm({9})

    rnd._track_edges()

    assert calls == [{9}]


def test_falsify_timeout_keeps_the_full_set(tmp_path, monkeypatch):
    """A timeout is not scanned: its edges never reach last_new_ids."""
    f, rnd, calls = _round(tmp_path, monkeypatch)
    f._current_edges_cache = None
    rnd._scanned_shm = _ScannedShm({9})
    rnd._is_timeout = True
    monkeypatch.setattr(f, "_get_current_edge_set", lambda: {1, 2})

    rnd._track_edges()

    assert calls == [{1, 2}]


def test_adversarial_round_without_edges_records_nothing(tmp_path, monkeypatch):
    f, rnd, calls = _round(tmp_path, monkeypatch)
    f._current_edges_cache = set()
    rnd._scanned_shm = _ScannedShm(())

    rnd._track_edges()

    assert calls == []


def _first_map(feed) -> dict[int, int]:
    tracker = EdgeTracker()
    for n, edges in enumerate(feed, start=1):
        tracker.record_edge_lifetimes(edges, n)
    return dict(tracker._edge_first_exec)


def test_new_ids_stamp_the_same_first_seen_as_full_sets():
    """Oracle: new ids per exec = this exec's set minus all earlier ones."""
    seen: set[int] = set()
    new_only = []
    for edges in EXECS:
        new_only.append(edges - seen)
        seen |= edges

    assert _first_map(new_only) == _first_map(EXECS)
    assert _first_map(EXECS) == _first_map(EXECS)  # Rule 46 control
