"""The fluctuation tracker's edge set is built only when the tracker is on.

``_on_boring`` / ``_on_success`` / ``_on_crash`` passed
``f._get_current_edge_set()`` as an argument, so every exec built a Python
set of its live edges (~150 us on FFmpeg, ~10k ids) for a recorder that is
off unless ``--fluctuation-theorems`` is given.
"""

from __future__ import annotations

from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.test_regression_katz_resume_contract import (  # noqa: F401 - autouse fixture
    _fuzzer,
    _instrumented_target,
    _with_katz,
)


class _Recorder:
    def __init__(self):
        self.records = []

    def observe(self, record) -> None:
        self.records.append(record)


def _counting(tmp_path, monkeypatch):
    _with_katz(monkeypatch, None)
    f = _fuzzer(tmp_path)
    calls = []
    monkeypatch.setattr(f, "_get_current_edge_set", lambda: calls.append(1) or {7, 9})
    f._last_ops_used = ["bit_flip"]
    return f, calls


def test_boring_round_builds_no_edge_set_when_tracker_off(tmp_path, monkeypatch):
    f, calls = _counting(tmp_path, monkeypatch)
    assert f._fluctuation is None

    FuzzRound(f, b"seed")._on_boring()

    assert calls == []


def test_falsify_tracker_on_still_gets_the_edges(tmp_path, monkeypatch):
    f, calls = _counting(tmp_path, monkeypatch)
    f._fluctuation = _Recorder()

    FuzzRound(f, b"seed")._on_boring()

    assert calls == [1]
    assert f._fluctuation.records[0].hit_edges == frozenset({7, 9})


def test_adversarial_explicit_edges_are_used_as_given(tmp_path, monkeypatch):
    """The Metropolis path already holds the mutant's edges: no rescan."""
    f, calls = _counting(tmp_path, monkeypatch)
    f._fluctuation = _Recorder()

    f._record_fluctuation_observation("success", {1, 2})

    assert calls == []
    assert f._fluctuation.records[0].hit_edges == frozenset({1, 2})
