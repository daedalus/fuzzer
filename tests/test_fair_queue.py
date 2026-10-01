"""Fair-queue primitives: smooth WRR, DRR, WFQ (SCFQ clock)."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.fair_queue import EEVDF, DeficitRR, SmoothWRR, Stride, WeightedFairQueue

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


# --- Stride -----------------------------------------------------------------


def _unit(_k):
    return 1.0


def test_stride_flat_weight_is_round_robin():
    """Falsification: equal tickets reduce stride to plain cycling in list order."""
    q = Stride()

    assert [q.pick(["a", "b", "c"], _unit) for _ in range(6)] == ["a", "b", "c"] * 2


def test_stride_share_matches_weights():
    """3:1 tickets -> 3:1 picks, exact over whole periods (deterministic)."""
    w = {"a": 3.0, "b": 1.0}
    q = Stride()
    picks = [q.pick(["a", "b"], w.get) for _ in range(400)]

    assert picks.count("a") == pytest.approx(300, abs=1)


def test_stride_late_joiner_banks_no_credit():
    """Adversarial: a flow joining late starts at the current pass, not at zero."""
    q = Stride()
    for _ in range(100):
        q.pick(["a", "b"], _unit)
    picks = [q.pick(["a", "b", "c"], _unit) for _ in range(30)]

    assert _counts(picks) == {"a": 10, "b": 10, "c": 10}


def test_stride_departed_flow_never_picked():
    q = Stride()
    for _ in range(5):
        q.pick(["a", "b", "c"], _unit)

    assert "b" not in [q.pick(["a", "c"], _unit) for _ in range(20)]


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF])
def test_stride_bad_weight_is_neutral(bad):
    """Adversarial: garbage tickets count as 1, never divide by zero or starve."""
    w = {"a": bad, "b": 1.0}
    q = Stride()
    picks = [q.pick(["a", "b"], w.get) for _ in range(20)]

    assert _counts(picks) == {"a": 10, "b": 10}


def test_stride_empty_and_single():
    assert Stride().pick([], _unit) == ""
    assert Stride().pick(["x"], _unit) == "x"


# --- EEVDF ------------------------------------------------------------------


def test_eevdf_flat_is_round_robin():
    """Falsification: unit cost and weight reduce EEVDF to plain cycling."""
    q = EEVDF()

    assert [q.pick(["a", "b", "c"], _unit, _unit) for _ in range(6)] == ["a", "b", "c"] * 2


def test_eevdf_equal_time_for_unequal_cost():
    """Cost 4 vs 1 at equal weight: the slow flow gets a quarter of the picks."""
    cost = {"fast": 1.0, "slow": 4.0}
    q = EEVDF()
    picks = [q.pick(["fast", "slow"], cost.get, _unit) for _ in range(500)]

    assert picks.count("fast") == pytest.approx(4 * picks.count("slow"), abs=4)


def test_eevdf_weight_scales_share():
    w = {"a": 2.0, "b": 1.0}
    q = EEVDF()
    picks = [q.pick(["a", "b"], _unit, w.get) for _ in range(300)]

    assert picks.count("a") == pytest.approx(2 * picks.count("b"), abs=2)


def test_eevdf_late_joiner_gets_fair_share_not_catch_up():
    """Adversarial: a new flow joins at lag 0 (V), so it cannot monopolise."""
    q = EEVDF()
    for _ in range(100):
        q.pick(["a", "b"], _unit, _unit)
    picks = [q.pick(["a", "b", "c"], _unit, _unit) for _ in range(30)]

    assert _counts(picks) == {"a": 10, "b": 10, "c": 10}


def test_eevdf_flow_ahead_of_clock_is_ineligible():
    """An expensive service puts b ahead of V; it waits until the clock catches up.

    a=cost 1, b=cost 8, equal weight: after a, b the lag of b is -7, so a
    runs 7 times in a row (its ve climbs 1..8 while V = (ve_a + 8) / 2).
    """
    cost = {"a": 1.0, "b": 8.0}
    q = EEVDF()
    head = [q.pick(["a", "b"], cost.get, _unit) for _ in range(2)]

    assert head == ["a", "b"]
    assert [q.pick(["a", "b"], cost.get, _unit) for _ in range(7)] == ["a"] * 7


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF])
def test_eevdf_bad_cost_and_weight_are_neutral(bad):
    """Adversarial: garbage from a corrupted ledger is neutral, never a hang or NaN clock."""
    q = EEVDF()
    bad_fn = {"a": bad, "b": 1.0}.get
    picks = [q.pick(["a", "b"], bad_fn, bad_fn) for _ in range(20)]

    assert _counts(picks) == {"a": 10, "b": 10}
    assert math.isfinite(q.virtual_time)


def test_eevdf_departed_flow_never_picked():
    q = EEVDF()
    for _ in range(5):
        q.pick(["a", "b", "c"], _unit, _unit)

    assert "b" not in [q.pick(["a", "c"], _unit, _unit) for _ in range(20)]


def test_eevdf_empty_and_single():
    assert EEVDF().pick([], _unit, _unit) == ""
    assert EEVDF().pick(["x"], _unit, _unit) == "x"


def test_eevdf_no_eligible_flow_serves_lowest_ve():
    """Adversarial: a clock corrupted past every flow still serves, never IndexError."""
    q = EEVDF()
    q.pick(["a", "b"], _unit, _unit)
    q._sum_wve = -1e9  # V far below every ve: nobody eligible

    assert q.pick(["a", "b"], _unit, _unit) == "b"


def test_eevdf_pick_does_not_scan_ineligible_flows(monkeypatch):
    """Adversarial (PR #44 review): deadline order != eligibility order.

    One flow at ve=0 (w=1) and 4999 at ve=0.49 (w=2): V ~= 0.48995, so only
    the first is eligible while the others hold earlier deadlines. A pick
    must not pop every ineligible flow on the way (O(n log n)).
    """
    import fuzzer_tool.core.fair_queue as fq

    flows = ["a"] + [f"f{i}" for i in range(4999)]
    weight = {k: 2.0 for k in flows}
    weight["a"] = 1.0
    q = EEVDF()  # slice 1: deadlines a=1.0, others=0.99 -- the ineligible ones first
    q._ve = {k: 0.49 for k in flows}
    q._ve["a"] = 0.0
    q._w = dict(weight)

    pops = []
    real_pop = fq.heapq.heappop
    monkeypatch.setattr(fq.heapq, "heappop", lambda h: pops.append(1) or real_pop(h))

    assert q.pick(flows, _unit, weight.get) == "a"
    assert len(pops) <= 8
