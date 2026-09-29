"""Fair-queue primitives: smooth WRR, DRR, WFQ (SCFQ clock)."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.fair_queue import DeficitRR, SmoothWRR, WeightedFairQueue

NAN = float("nan")
INF = float("inf")


def _counts(picks):
    return {k: picks.count(k) for k in sorted(set(picks))}


# --- SmoothWRR --------------------------------------------------------------


def test_wrr_share_matches_weights_exactly():
    """Falsification: 5:1:1 over one period of 7 picks -> exactly 5,1,1."""
    q = SmoothWRR()
    picks = [q.pick({"a": 5, "b": 1, "c": 1}) for _ in range(7)]

    assert _counts(picks) == {"a": 5, "b": 1, "c": 1}


def test_wrr_interleaves_not_bursts():
    """Smooth variant: the heavy flow is never served more than its ceil share in a row."""
    q = SmoothWRR()
    picks = [q.pick({"a": 5, "b": 1, "c": 1}) for _ in range(7)]

    assert "aaaa" not in "".join(picks)


def test_wrr_deterministic():
    a, b = SmoothWRR(), SmoothWRR()
    w = {"x": 3, "y": 2}

    assert [a.pick(w) for _ in range(20)] == [b.pick(w) for _ in range(20)]


def test_wrr_fractional_weights():
    """1/edges-style weights: 0.5 : 0.25 -> 2:1 over 30 picks."""
    q = SmoothWRR()
    picks = [q.pick({"a": 0.5, "b": 0.25}) for _ in range(30)]

    assert _counts(picks) == {"a": 20, "b": 10}


def test_wrr_weight_change_takes_effect():
    q = SmoothWRR()
    for _ in range(10):
        q.pick({"a": 1, "b": 1})
    picks = [q.pick({"a": 1, "b": 9}) for _ in range(100)]

    assert picks.count("b") == 90


@pytest.mark.parametrize("bad", [0, -3, NAN, INF, -INF])
def test_wrr_bad_weight_excluded(bad):
    """Adversarial: zero/negative/NaN/inf weight never wins while a valid flow exists."""
    q = SmoothWRR()
    picks = [q.pick({"a": 1, "bad": bad}) for _ in range(20)]

    assert set(picks) == {"a"}


def test_wrr_all_bad_falls_back_to_rotation():
    q = SmoothWRR()
    picks = [q.pick({"a": 0, "b": NAN, "c": -1}) for _ in range(6)]

    assert picks == ["a", "b", "c", "a", "b", "c"]


def test_wrr_empty_raises():
    with pytest.raises(ValueError):
        SmoothWRR().pick({})


def test_wrr_dropped_flow_state_does_not_leak():
    """Adversarial: a flow that vanishes and returns starts clean, no hoarded credit."""
    q = SmoothWRR()
    for _ in range(50):
        q.pick({"a": 1, "b": 1})
    for _ in range(50):
        q.pick({"a": 1})
    picks = [q.pick({"a": 1, "b": 1}) for _ in range(10)]

    assert _counts(picks) == {"a": 5, "b": 5}


# --- DeficitRR --------------------------------------------------------------


def _drr_run(q, flows, cost, weight, n):
    return [q.pick(flows, cost, weight) for _ in range(n)]


def test_drr_equal_cost_equal_weight_is_round_robin():
    """Falsification: flat cost and weight reduce DRR to plain RR."""
    q = DeficitRR(quantum=1.0)
    picks = _drr_run(q, ["a", "b", "c"], lambda k: 1.0, lambda k: 1.0, 9)

    assert picks == ["a", "b", "c"] * 3


def test_drr_cost_share_is_equal_in_time():
    """Cost 1 vs 4, equal weight: served *time* is equal, so picks are 4:1."""
    cost = {"fast": 1.0, "slow": 4.0}
    q = DeficitRR(quantum=4.0)
    picks = _drr_run(q, ["fast", "slow"], cost.get, lambda k: 1.0, 500)
    time = {k: picks.count(k) * cost[k] for k in cost}

    assert abs(time["fast"] - time["slow"]) <= max(cost.values())


def test_drr_weight_scales_time_share():
    """Weight 3:1 at equal cost -> 3:1 picks within one quantum."""
    q = DeficitRR(quantum=1.0)
    w = {"a": 3.0, "b": 1.0}
    picks = _drr_run(q, ["a", "b"], lambda k: 1.0, w.get, 400)

    assert abs(picks.count("a") - 300) <= 3


def test_drr_carries_deficit_over_rounds():
    """Cost above one quantum is still served: deficit accumulates, no starvation."""
    q = DeficitRR(quantum=1.0)
    picks = _drr_run(q, ["a", "b"], {"a": 1.0, "b": 3.0}.get, lambda k: 1.0, 40)

    assert picks.count("b") > 0


def test_drr_terminates_on_bad_inputs():
    """Adversarial: zero/NaN/inf/negative cost and weight neither hang nor raise."""
    q = DeficitRR(quantum=1.0)
    bad_cost = {"a": 0.0, "b": NAN, "c": INF, "d": -5.0}
    bad_weight = {"a": 0.0, "b": NAN, "c": -1.0, "d": INF}
    picks = _drr_run(q, list(bad_cost), bad_cost.get, bad_weight.get, 50)

    assert len(picks) == 50
    assert all(p in bad_cost for p in picks)


def test_drr_huge_cost_is_bounded():
    """Adversarial: cost >> quantum must not spin; forced service after the round cap."""
    q = DeficitRR(quantum=1.0)
    picks = _drr_run(q, ["a", "b"], lambda k: 1e18, lambda k: 1.0, 6)

    assert picks == ["a", "b"] * 3


def test_drr_empty_and_single():
    q = DeficitRR(quantum=1.0)

    assert q.pick([], lambda k: 1.0, lambda k: 1.0) == ""
    assert q.pick(["only"], lambda k: 1.0, lambda k: 1.0) == "only"


def test_drr_new_flows_join_without_reset():
    q = DeficitRR(quantum=1.0)
    _drr_run(q, ["a", "b"], lambda k: 1.0, lambda k: 1.0, 6)
    picks = _drr_run(q, ["a", "b", "c"], lambda k: 1.0, lambda k: 1.0, 12)

    assert _counts(picks) == {"a": 4, "b": 4, "c": 4}


def test_drr_removed_flow_never_returned():
    q = DeficitRR(quantum=1.0)
    _drr_run(q, ["a", "b", "c"], lambda k: 1.0, lambda k: 1.0, 9)
    picks = _drr_run(q, ["a", "c"], lambda k: 1.0, lambda k: 1.0, 20)

    assert "b" not in picks


def test_drr_cache_invalidates_on_same_size_swap():
    """Adversarial: one flow replaced by another (same length) must not serve the stale set."""
    q = DeficitRR(quantum=1.0)
    _drr_run(q, ["a", "b", "c"], lambda k: 1.0, lambda k: 1.0, 6)
    picks = _drr_run(q, ["a", "b", "d"], lambda k: 1.0, lambda k: 1.0, 9)

    assert "c" not in picks
    assert "d" in picks


def test_drr_cache_invalidates_on_reorder_and_prune():
    """Adversarial: a shuffled list and a mass departure both rebuild the active set."""
    q = DeficitRR(quantum=1.0)
    many = [f"s{i}" for i in range(40)]
    _drr_run(q, many, lambda k: 1.0, lambda k: 1.0, 80)
    picks = _drr_run(q, ["s3", "s1"], lambda k: 1.0, lambda k: 1.0, 10)

    assert set(picks) == {"s1", "s3"}


# --- WeightedFairQueue ------------------------------------------------------


def _wfq_run(q, weights, cost, n):
    out = []
    for _ in range(n):
        k = q.pick(weights)
        q.charge(k, cost[k], weights[k])
        out.append(k)
    return out


def test_wfq_time_share_follows_weight():
    """Falsification: equal cost, weights 3:1 -> 3:1 picks within 1."""
    q = WeightedFairQueue()
    picks = _wfq_run(q, {"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, 400)

    assert abs(picks.count("a") - 300) <= 1


def test_wfq_charges_time_not_counts():
    """Equal weight, cost 1 vs 3: picks are 3:1 so served time is equal. WRR would give 1:1."""
    cost = {"fast": 1.0, "slow": 3.0}
    q = WeightedFairQueue()
    picks = _wfq_run(q, {"fast": 1.0, "slow": 1.0}, cost, 400)
    time = {k: picks.count(k) * cost[k] for k in cost}

    assert abs(time["fast"] - time["slow"]) <= max(cost.values())
    assert picks.count("fast") > 2 * picks.count("slow")


def test_wfq_idle_flow_gets_no_burst_credit():
    """Adversarial: a flow absent for 1000 charges returns without monopolising service."""
    q = WeightedFairQueue()
    cost = {"a": 1.0, "b": 1.0}
    _wfq_run(q, {"a": 1.0}, cost, 1000)
    picks = _wfq_run(q, {"a": 1.0, "b": 1.0}, cost, 20)

    assert abs(picks.count("a") - picks.count("b")) <= 1


def test_wfq_bad_weight_excluded_and_all_bad_rotates():
    q = WeightedFairQueue()

    assert {q.pick({"a": 1.0, "b": NAN, "c": 0.0}) for _ in range(5)} == {"a"}
    assert [q.pick({"a": 0.0, "b": NAN}) for _ in range(4)] == ["a", "b", "a", "b"]


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF])
def test_wfq_bad_cost_is_floored(bad):
    """Adversarial: a garbage cost neither poisons the clock nor starves other flows."""
    q = WeightedFairQueue()
    q.charge("a", bad, 1.0)
    picks = _wfq_run(q, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, 20)

    assert math.isfinite(q.virtual_time)
    assert _counts(picks) == {"a": 10, "b": 10}


def test_wfq_empty_raises():
    with pytest.raises(ValueError):
        WeightedFairQueue().pick({})


def test_wfq_unseen_cost_estimate_falls_back():
    """A never-charged flow is still pickable (estimate falls back to the global mean / 1.0)."""
    q = WeightedFairQueue()

    assert q.pick({"a": 1.0, "b": 1.0}) in {"a", "b"}
