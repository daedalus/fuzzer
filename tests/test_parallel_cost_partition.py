"""Tests for core/parallel_cost_partition.py (P3-3, step 5 of 6)."""

import pytest

from fuzzer_tool.core.parallel_cost_partition import (
    CostPartition,
    compute_partition,
    maybe_repartition,
)


# --------------------------------------------------------------------------
# compute_partition
# --------------------------------------------------------------------------


def test_compute_partition_assigns_every_id():
    costs = {"a": 10.0, "b": 20.0, "c": 5.0, "d": 15.0}
    part = compute_partition(costs, m=2)
    assert set(part.assignment) == set(costs)
    assert all(0 <= w < 2 for w in part.assignment.values())


def test_compute_partition_balances_load_better_than_arbitrary_grouping():
    # Four items, three heavy + one light -- a naive split-in-half would put
    # two heavies on one worker; Multifit should not do that when a better
    # balance exists.
    costs = {"heavy1": 100.0, "heavy2": 100.0, "heavy3": 100.0, "light": 1.0}
    part = compute_partition(costs, m=2)
    per_worker: dict[int, float] = {}
    for item_id, w in part.assignment.items():
        per_worker[w] = per_worker.get(w, 0.0) + costs[item_id]
    # Best achievable split is 200/101; a bad split would be 300/1.
    assert max(per_worker.values()) <= 201.0


def test_compute_partition_single_worker_gets_everything():
    costs = {"a": 1.0, "b": 2.0, "c": 3.0}
    part = compute_partition(costs, m=1)
    assert set(part.assignment.values()) == {0}
    assert part.makespan == pytest.approx(6.0)


def test_compute_partition_stores_costs_snapshot_and_m():
    costs = {"a": 1.0, "b": 2.0}
    part = compute_partition(costs, m=3)
    assert part.costs == costs
    assert part.m == 3
    # Snapshot is a copy, not the same dict -- mutating the caller's dict
    # afterward must not silently rewrite what the partition was computed from.
    costs["a"] = 999.0
    assert part.costs["a"] == 1.0


def test_compute_partition_rejects_non_positive_cost():
    with pytest.raises(ValueError):
        compute_partition({"a": 0.0}, m=1)
    with pytest.raises(ValueError):
        compute_partition({"a": -1.0}, m=1)


def test_compute_partition_rejects_empty_costs():
    with pytest.raises(ValueError):
        compute_partition({}, m=1)


def test_compute_partition_rejects_non_positive_m():
    with pytest.raises(ValueError):
        compute_partition({"a": 1.0}, m=0)


# --------------------------------------------------------------------------
# maybe_repartition: hysteresis
# --------------------------------------------------------------------------


def test_maybe_repartition_computes_fresh_when_previous_is_empty():
    empty = CostPartition()
    costs = {"a": 1.0, "b": 2.0}
    result = maybe_repartition(empty, costs, m=2)
    assert set(result.assignment) == set(costs)
    assert result.m == 2


def test_maybe_repartition_returns_same_object_below_threshold():
    costs = {"a": 10.0, "b": 20.0, "c": 30.0}
    first = compute_partition(costs, m=2)
    # Tiny drift, well under the default 25% threshold.
    nudged = {"a": 10.5, "b": 20.2, "c": 29.8}
    result = maybe_repartition(first, nudged, m=2)
    assert result is first  # identity, not just equality -- no repack happened


def test_maybe_repartition_repacks_above_threshold():
    costs = {"a": 10.0, "b": 20.0, "c": 30.0}
    first = compute_partition(costs, m=2)
    # "a" quadruples -- well past the default 25% threshold.
    drifted = {"a": 40.0, "b": 20.0, "c": 30.0}
    result = maybe_repartition(first, drifted, m=2)
    assert result is not first
    assert result.costs == drifted


def test_maybe_repartition_one_outlier_triggers_even_if_average_is_calm():
    # Large corpus, everything stable except one seed that changed a lot --
    # the max-drift rule must not let it get diluted by the calm majority.
    costs = {f"seed{i}": 10.0 for i in range(50)}
    first = compute_partition(costs, m=4)
    drifted = dict(costs)
    drifted["seed0"] = 100.0  # 9x -- far past threshold for this one id
    result = maybe_repartition(first, drifted, m=4)
    assert result is not first


def test_maybe_repartition_new_id_counts_as_maximal_drift():
    costs = {"a": 10.0, "b": 20.0}
    first = compute_partition(costs, m=2)
    with_new = dict(costs)
    with_new["c"] = 15.0
    result = maybe_repartition(first, with_new, m=2)
    assert result is not first
    assert "c" in result.assignment


def test_maybe_repartition_removed_id_counts_as_maximal_drift():
    costs = {"a": 10.0, "b": 20.0, "c": 15.0}
    first = compute_partition(costs, m=2)
    without_c = {"a": 10.0, "b": 20.0}
    result = maybe_repartition(first, without_c, m=2)
    assert result is not first
    assert "c" not in result.assignment


def test_maybe_repartition_worker_count_change_always_repacks():
    costs = {"a": 10.0, "b": 20.0}
    first = compute_partition(costs, m=2)
    # Identical costs, only m changes -- must still repack since the old
    # assignment's worker indices are not meaningful under a new m.
    result = maybe_repartition(first, costs, m=3)
    assert result is not first
    assert result.m == 3


def test_maybe_repartition_custom_threshold_is_honored():
    costs = {"a": 10.0, "b": 20.0}
    first = compute_partition(costs, m=2)
    nudged = {"a": 11.0, "b": 20.0}  # ~9% drift on "a"
    # Below a loose threshold: no repack.
    assert maybe_repartition(first, nudged, m=2, drift_threshold=0.5) is first
    # Above a tight threshold: repacks.
    assert maybe_repartition(first, nudged, m=2, drift_threshold=0.05) is not first
