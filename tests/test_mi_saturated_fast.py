"""Saturated ``MutualInformationTracker.record``: marginals only, no edge fold.

Once the joint hits MAX_JOINT_CELLS (every long campaign), record() still
folded every edge hash and evaluated the joint gate per byte, then updated
only the marginals: 1.6 ms/exec, ~5% of a GStreamer run.
"""

from __future__ import annotations

import copy

import pytest

from fuzzer_tool.core import mi as mi_mod
from fuzzer_tool.core.mi import MAX_JOINT_CELLS, MutualInformationTracker
from fuzzer_tool.core.rand_pool import RandPool

MAP = 1 << 16


def _ref_marginals(t: MutualInformationTracker, data: bytes) -> None:
    """Independent oracle: the marginal updates record() must make."""
    t.total_observations += 1
    for pos, bv in enumerate(data[: t.max_positions]):
        t.position_counts[pos] += 1
        t.byte_marginal[pos][bv] += 1


def _state(t):
    return (
        t.total_observations,
        dict(t.position_counts),
        {p: dict(c) for p, c in t.byte_marginal.items()},
        t._joint_cells,
        list(t.edge_marginal),
    )


def _saturated(max_positions=64) -> MutualInformationTracker:
    t = MutualInformationTracker(max_positions=max_positions, min_observations=1)
    t._joint_cells = MAX_JOINT_CELLS
    return t


def test_saturated_skips_fold(monkeypatch):
    """No joint update possible -> edge hashes are never folded."""
    t = _saturated()

    def boom(*_a):
        raise AssertionError("fold on saturated record")

    monkeypatch.setattr(mi_mod, "_fold_edges", boom)
    t.record(b"abc", {1 << 40, 7}, MAP)
    assert t.position_counts[2] == 1


@pytest.mark.parametrize("seed", range(4))
def test_saturated_matches_oracle(seed):
    """Marginals match the oracle across lengths around max_positions."""
    rng = RandPool(seed=seed)
    t = _saturated()
    ref = copy.deepcopy(t)
    for n in (0, 1, 63, 64, 65, 200):
        data = rng.randbytes(n)
        t.record(data, {rng.randint(0, 1 << 32) for _ in range(5)}, MAP)
        _ref_marginals(ref, data)
    assert _state(t) == _state(ref)


def test_oracle_matches_itself():
    """Control (Hard Rule 46): two oracle runs agree."""
    a, b = _saturated(), _saturated()
    for t in (a, b):
        _ref_marginals(t, b"\x00\xffxyz" * 20)
    assert _state(a) == _state(b)


def test_unsaturated_still_joins():
    """Falsification: below the cap, edges still reach the joint."""
    t = MutualInformationTracker(max_positions=8, min_observations=1)
    t.record(b"ab", {3, 5}, MAP)
    assert t._joint_cells > 0
    assert t.edge_marginal[3] > 0


def test_no_edges_marginals_only():
    """Adversarial: empty edge set and empty input never touch the joint."""
    t = MutualInformationTracker(max_positions=8, min_observations=1)
    t.record(b"", set(), MAP)
    t.record(b"\x01" * 50, set(), MAP)
    assert t._joint_cells == 0
    assert t.position_counts[7] == 1 and 8 not in t.position_counts


def test_saturated_defers_dicts():
    """Saturated records land in dense buffers; reading the dicts folds them in."""
    t = _saturated()
    t.record(b"ab", {1}, MAP)
    assert not t._position_counts
    assert t.position_counts == {0: 1, 1: 1}
    assert t.byte_marginal[1] == {ord("b"): 1}


def test_round_trip_keeps_pending():
    """to_dict includes not-yet-folded records."""
    t = _saturated()
    t.record(b"xyz", {1}, MAP)
    u = MutualInformationTracker()
    u.from_dict(t.to_dict())
    assert u.position_counts == {0: 1, 1: 1, 2: 1}


def test_load_discards_pending():
    """Adversarial: from_dict replaces state; stale pending must not leak in."""
    t = _saturated()
    t.record(b"q" * 10, {1}, MAP)
    t.from_dict({"max_positions": 4, "position_counts": {"0": 7}, "byte_marginal": {}})
    t._joint_cells = MAX_JOINT_CELLS
    t.record(b"zzzzzz", {1}, MAP)  # max_positions shrank to 4
    assert t.position_counts == {0: 8, 1: 1, 2: 1, 3: 1}


def test_stale_wp_sees_pending():
    """Adversarial: weighted_position cache drops when a pending count crosses min_obs."""
    t = _saturated(max_positions=8)
    t.min_observations = 2
    t._wp_sorted_pos = [0]
    for _ in range(2):  # the check reads the pre-record count: 0, then 1
        t.record(b"abcd", {1}, MAP)
    assert t._wp_sorted_pos == [0]
    t.record(b"abcd", {1}, MAP)  # pre-record count 2 = min_obs, pos 3 not cached
    assert t._wp_sorted_pos is None
