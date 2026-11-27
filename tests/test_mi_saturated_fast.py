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
