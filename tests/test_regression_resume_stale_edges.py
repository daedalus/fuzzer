"""Regression: --resume restored the old binary's edge IDs into the tracker.

Edge IDs differ per build. On a target-hash mismatch ``load_state`` warned
and skipped the saved SHM novelty map, but ``_restore_subsystems`` restored
the edge tracker anyway: the old IDs stayed in ``cumulative_edges`` with no
seed attributed. GStreamer after two rebuilds: 42.6k tracked edges against
15.2k live, 12.7k "no known holder" warnings per minimize.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.edge_tracker import EDGE_TRACKER_SECTION, EdgeTracker
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import CorpusManager

OLD_EDGES = {11, 22, 33}
MAP_SIZE = 4096


class _Inert:
    def load(self, _data):
        pass


def _fuzzer(tmp_path, target_state: str):
    """Saved state holding a tracker with OLD_EDGES; a fresh tracker to restore into."""
    store = StateStore(tmp_path)
    old = EdgeTracker(map_size=MAP_SIZE)
    old.record_edges("seed", set(OLD_EDGES))
    store.set(EDGE_TRACKER_SECTION, old.to_dict())
    return SimpleNamespace(
        _state_store=store,
        _edge_tracker=EdgeTracker(map_size=MAP_SIZE),
        _resume_target_state=target_state,
        _sensitivity=_Inert(),
    )


def test_regression_resume_stale_edges(tmp_path):
    """A changed binary starts the tracker empty."""
    f = _fuzzer(tmp_path, "mismatch")
    CorpusManager(f)._restore_subsystems({})
    assert f._edge_tracker.get_cumulative_edge_count() == 0


def test_same_binary_restores(tmp_path):
    """Falsification: an unchanged binary still gets its edges back."""
    f = _fuzzer(tmp_path, "match")
    CorpusManager(f)._restore_subsystems({})
    assert f._edge_tracker.get_cumulative_edge_count() == len(OLD_EDGES)


@pytest.mark.parametrize("state", ["unknown", None])
def test_unknown_hash_keeps_restoring(tmp_path, state):
    """Adversarial: no hash to compare (old state, unreadable target) is not a mismatch."""
    f = _fuzzer(tmp_path, state)
    if state is None:
        del f._resume_target_state
    CorpusManager(f)._restore_subsystems({})
    assert f._edge_tracker.get_cumulative_edge_count() == len(OLD_EDGES)
