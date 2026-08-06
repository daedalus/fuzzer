"""Regression test: _corpus_size_history is an array.array, not a list.

AGENTS.md style rule: prefer array.array over Python lists for homogeneous
numeric data. CorpusManager.init_seed_metadata now builds array("I") and both
save_state / load_state keep the state.json format byte-identical (plain list
on disk, array in memory). Verifies:
1. init builds an array("I")
2. the >1000 -> keep-last-500 trim keeps array semantics and correct content
3. save/load round-trip: JSON holds a plain list; load rebuilds an array("I")
   with identical content (independent derivation from a reference list)
4. sorted() (the p90 consumer at corpus_manager.py:403 / report.py:822) works
"""

from __future__ import annotations

import json
from array import array
from types import SimpleNamespace

from fuzzer_tool.services.corpus_manager import CorpusManager


class _StubTracker:
    def save(self, path):
        pass

    def load(self, path):
        pass


class _StubSensitivity:
    def save(self):
        return {}

    def load(self, data):
        pass


class _StubCrashMi:
    def save(self):
        return {}


def _make_fuzzer(tmp_path, history):
    return SimpleNamespace(
        corpus_dir=tmp_path,
        _state_path=tmp_path / "state.json",
        _edge_tracker_path=tmp_path / "edge_tracker.json",
        _crash_mi_path=tmp_path / "crash_mi.json",
        corpus=[],
        seed_meta={},
        map_size=8192,
        resume=False,
        _use_elo=False,
        _edge_tracker=_StubTracker(),
        _sensitivity=_StubSensitivity(),
        _crash_mi=_StubCrashMi(),
        exec_count=0,
        crash_count=0,
        timeout_count=0,
        crash_sigs={},
        crash_frames={},
        crash_min_sizes={},
        op_counts={},
        op_success={},
        op_edges={},
        _corpus_size_history=history,
    )


def test_init_builds_array():
    """CorpusManager.init_seed_metadata constructs an array('I'), not a list."""
    f = SimpleNamespace(corpus_dir=None, corpus=[b"a"], map_size=8192, resume=False)
    # corpus_dir is used for a mkdir; give it a real temp dir.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f.corpus_dir = __import__("pathlib").Path(d)
        CorpusManager(f).init_seed_metadata()
        assert isinstance(f._corpus_size_history, array)
        assert f._corpus_size_history.typecode == "I"


def test_trim_keeps_last_500():
    """The >1000 -> keep-last-500 trim preserves content as an array."""
    sizes = list(range(1200))  # independent reference
    hist = array("I", sizes)
    if len(hist) > 1000:
        hist = hist[-500:]
    assert isinstance(hist, array)
    assert len(hist) == 500
    assert hist.tolist() == sizes[-500:]


def test_save_load_round_trip():
    """save_state writes a plain JSON list; load_state rebuilds an array('I')."""
    import tempfile
    from pathlib import Path

    sizes = [10, 20, 30, 40, 50] * 110  # 550 entries -> save keeps last 500
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        f = _make_fuzzer(tmp, array("I", sizes))
        cm = CorpusManager(f)
        cm.save_state()

        raw = json.loads((tmp / "state.json").read_text())
        expected = sizes[-500:]
        # On-disk format is a plain list (array is not JSON-serializable).
        assert raw["corpus_size_history"] == expected
        assert isinstance(raw["corpus_size_history"], list)

        # Fresh in-memory state, then load: must rebuild the array identically.
        f._corpus_size_history = array("I")
        cm.load_state()
        assert isinstance(f._corpus_size_history, array)
        assert f._corpus_size_history.typecode == "I"
        assert f._corpus_size_history.tolist() == expected


def test_load_state_sets_session_baselines():
    """load_state anchors EPS baselines to the restored exec_count.

    Regression: resumed runs divided the cumulative exec_count by fresh
    wall time (e.g. 2.3M execs / 1s of uptime), showing absurd eps.  The
    loaded count is now recorded as the session baseline and the EPS
    interval counter so display and the Kalman filter measure only this
    process.
    """
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        f = _make_fuzzer(tmp, array("I", [1, 2, 3]))
        (tmp / "state.json").write_text(json.dumps({"exec_count": 2_335_238}))
        cm = CorpusManager(f)
        cm.load_state()
        assert f.exec_count == 2_335_238
        assert f._resume_baseline_exec == 2_335_238
        assert f._last_eps_count == 2_335_238


def test_sorted_consumer():
    """sorted() over the array returns the plain list the p90 consumer needs."""
    hist = array("I", [30, 10, 20])
    assert sorted(hist) == [10, 20, 30]
    assert isinstance(sorted(hist), list)
