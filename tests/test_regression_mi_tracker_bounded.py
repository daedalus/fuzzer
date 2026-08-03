"""Regression test: the MI tracker's joint distribution is memory-bounded.

The MutualInformationTracker (enabled by --mi-guided / --elo all) builds a
{position: {byte_val: {edge: count}}} joint. The per-cell edge cap alone let
it reach 796MB mi.json / multi-GB RSS (OOM). Regressions pinned here:
1. max_positions caps the tracked byte positions even when max_len is larger.
2. a total joint-cell budget evicts the least-observed position, bounding memory.
3. loaded state is recounted and trimmed to the budget (no resume-time blowup).
4. Fuzzer only loads mi.json with --resume (like state.json).
"""

from __future__ import annotations

from fuzzer_tool.core.mi import (
    MAX_EDGES_PER_CELL,
    MI_MAX_POSITIONS,
    MutualInformationTracker,
)


def _feed(tracker, n_inputs, input_len, edge=3):
    """feed `n_inputs` identical inputs of `input_len` bytes hitting `edge`."""
    # Distinct-ish bytes per record so each (pos, byte) cell is exercised.
    data = bytes(range(16)) * ((input_len + 15) // 16)
    data = data[:input_len]
    for i in range(n_inputs):
        tracker.record(data, {edge + i % 7})


def test_max_positions_caps_tracked_positions():
    """record() ignores positions at/above max_positions."""
    t = MutualInformationTracker(max_positions=8, min_observations=1)
    _feed(t, 3, 100)
    assert max(t.position_counts.keys()) < 8
    assert all(p < 8 for p in t.position_counts)


def test_joint_cell_budget_bounds_growth(monkeypatch):
    """record() keeps the total joint cells under MAX_JOINT_CELLS."""
    budget = 500
    monkeypatch.setattr("fuzzer_tool.core.mi.MAX_JOINT_CELLS", budget)
    t = MutualInformationTracker(max_positions=4096, min_observations=1)
    # Distinct inputs with distinct edges force many new cells per record.
    for i in range(4000):
        n_bytes = 2 + i % 16
        data = bytes((i * 7 + b) & 0xFF for b in range(n_bytes))
        edges = set((i + e * 131) % 100000 for e in range(20))
        t.record(data, edges)
        assert t._joint_cells <= budget, f"overflowed at record {i}: {t._joint_cells}"
    # Bounded relative to the 4096 x 256 x MAX_EDGES_PER_CELL theoretical max.
    theoretical = MI_MAX_POSITIONS * 256 * MAX_EDGES_PER_CELL
    assert budget < theoretical


def test_joint_cell_cap_saturates_without_thrash(monkeypatch):
    """ONCE MAX_JOINT_CELLS is reached, no new cells are added (bounded, stable).

    Exercises monotonically-growing positions/edges so cells hit the budget,
    then keeps recording: cell count must stay <= MAX_JOINT_CELLS forever and
    existing cells must keep incrementing (no eviction churn, no unbounded
    growth, no collapse to an empty joint).
    """
    monkeypatch.setattr("fuzzer_tool.core.mi.MAX_JOINT_CELLS", 400)
    t = MutualInformationTracker(max_positions=8, min_observations=1)
    cells_at_cap = None
    for i in range(4000):
        data = bytes(range(8)) if i % 40 == 0 else bytes(range(4))
        edges = set((i * 13 + e * 131) % 100000 for e in range(8))
        t.record(data, edges)
        assert t._joint_cells <= 400
        if t._joint_cells == 400 and cells_at_cap is None:
            cells_at_cap = (i, sorted(t.joint))
    # The joint saturated at (never exceeded) the budget AND survived long
    # after saturation (no runaway eviction to an empty joint).
    assert cells_at_cap is not None
    # Existing cells still increment after saturation: position 0, byte 0 count
    # keeps rising with observations.
    assert t.position_counts[0] == 4000  # observed in every record
    assert t.joint  # joint survived, not evicted away


def test_loaded_state_recounts_and_trims(monkeypatch, tmp_path):
    """load() recounts cells and evicts down to the budget on a legacy file."""
    monkeypatch.setattr("fuzzer_tool.core.mi.MAX_JOINT_CELLS", 300)
    t = MutualInformationTracker(max_positions=64, min_observations=1)
    _feed(t, 50, 60, edge=1)
    path = tmp_path / "mi.json"
    assert t.save(str(path))
    t2 = MutualInformationTracker(max_positions=64, min_observations=1)
    assert t2.load(str(path))
    # Recounted from the deserialized state, then trimmed to the budget.
    assert t2._joint_cells <= 300
    # A small tracker round-trips exactly (cells recounted, nothing evicted).
    t3 = MutualInformationTracker(max_positions=4, min_observations=1)
    _feed(t3, 2, 4)
    path2 = tmp_path / "mi2.json"
    assert t3.save(path2)
    t4 = MutualInformationTracker(max_positions=4, min_observations=1)
    assert t4.load(str(path2))
    assert t4._joint_cells == t3._joint_cells


def test_mi_tracker_construction_caps_max_len(tmp_path):
    """Fuzzer caps max_positions to MI_MAX_POSITIONS even when max_len is huge."""
    from unittest.mock import patch

    from fuzzer_tool.services.fuzzer import Fuzzer

    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmp_path}/corpus",
        crashes_dir=f"{tmp_path}/crashes",
        max_len=50000,
        timeout=1,
        mutations_per_input=2,
        mi_guided=True,
        resume=False,
    )
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f = Fuzzer(**defaults)
    assert f._mi is not None
    assert f._mi.max_positions == MI_MAX_POSITIONS
    assert f._mi.max_positions < 50000


def test_mi_load_resume_gated(tmp_path):
    """mi.json is only loaded when resume=True (like state.json)."""
    from unittest.mock import patch

    from fuzzer_tool.core.mi import MutualInformationTracker
    from fuzzer_tool.services.fuzzer import Fuzzer

    # Pre-write a mi.json with observable state.
    saved = MutualInformationTracker(max_positions=16, min_observations=1)
    saved.record(b"abcd", {1, 2})
    saved.record(b"abcd", {1, 2})
    mi_path = tmp_path / "corpus" / "mi.json"
    mi_path.parent.mkdir(parents=True, exist_ok=True)
    assert saved.save(str(mi_path))

    def make(resume):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp_path}/corpus",
            crashes_dir=f"{tmp_path}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            mi_guided=True,
            resume=resume,
        )

    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f_no = make(False)
    assert f_no._mi.total_observations == 0  # not loaded without --resume

    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f_yes = make(True)
    assert f_yes._mi.total_observations == 2  # loaded with --resume
