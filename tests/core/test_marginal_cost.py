"""Tests for core/marginal_cost.py (handover §1: Δcost/Δoutput estimator)."""

from fuzzer_tool.core.marginal_cost import MarginalCostTracker


def test_no_snapshot_yet_returns_none():
    t = MarginalCostTracker()
    assert t.marginal_cost("a") is None


def test_single_snapshot_returns_none():
    t = MarginalCostTracker()
    t.record_snapshot("a", cumulative_cost=10, cumulative_output=2)
    assert t.marginal_cost("a") is None


def test_two_snapshots_basic_ratio():
    t = MarginalCostTracker()
    t.record_snapshot("a", cumulative_cost=0, cumulative_output=0)
    t.record_snapshot("a", cumulative_cost=100, cumulative_output=20)
    assert t.marginal_cost("a") == 5.0


def test_marginal_cost_diffs_only_the_last_two_snapshots():
    t = MarginalCostTracker()
    t.record_snapshot("a", cumulative_cost=0, cumulative_output=0)
    t.record_snapshot("a", cumulative_cost=100, cumulative_output=20)  # MC=5
    t.record_snapshot("a", cumulative_cost=120, cumulative_output=40)  # Δ=20/20=1
    assert t.marginal_cost("a") == 1.0


def test_zero_delta_output_returns_none_not_zero_or_inf():
    t = MarginalCostTracker()
    t.record_snapshot("a", cumulative_cost=0, cumulative_output=0)
    t.record_snapshot("a", cumulative_cost=50, cumulative_output=0)
    assert t.marginal_cost("a") is None


def test_negative_delta_output_returns_none():
    # Cumulative output should never decrease, but the tracker is
    # defensive rather than assuming callers never regress a counter.
    t = MarginalCostTracker()
    t.record_snapshot("a", cumulative_cost=0, cumulative_output=10)
    t.record_snapshot("a", cumulative_cost=50, cumulative_output=5)
    assert t.marginal_cost("a") is None


def test_population_average_mc_skips_undefined_keys():
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 100, 20)  # MC=5
    t.record_snapshot("b", 0, 0)
    t.record_snapshot("b", 30, 0)  # MC undefined (no output)
    assert t.population_average_mc(["a", "b"]) == 5.0


def test_population_average_mc_none_when_all_undefined():
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 30, 0)
    assert t.population_average_mc(["a"]) is None


def test_should_stop_true_when_far_above_multiplier():
    # Two cheap operators and one expensive one, so the population
    # average (which includes the expensive one itself) is still
    # dominated by the cheap majority -- a single expensive key can't
    # clear a self-inclusive average on its own with only two keys total.
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 100, 20)  # MC=5 (cheap)
    t.record_snapshot("c", 0, 0)
    t.record_snapshot("c", 100, 20)  # MC=5 (cheap)
    t.record_snapshot("b", 0, 0)
    t.record_snapshot("b", 100, 1)  # MC=100 (expensive)
    assert t.should_stop("b", multiplier=2.0, keys=["a", "b", "c"]) is True
    assert t.should_stop("a", multiplier=2.0, keys=["a", "b", "c"]) is False


def test_should_stop_false_when_insufficient_data():
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    # Only one snapshot -- no MC defined yet.
    assert t.should_stop("a", multiplier=1.5) is False


def test_should_stop_true_when_cost_spent_with_zero_output():
    # "b" incurs cost but produces nothing this window -- marginal_cost("b")
    # is None, but should_stop must not read that as "no signal, keep going"
    # as long as some other key has a defined average to compare against.
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 100, 20)  # MC=5, defines the population average
    t.record_snapshot("b", 0, 0)
    t.record_snapshot("b", 50, 0)  # cost spent, zero output
    assert t.marginal_cost("b") is None
    assert t.should_stop("b", multiplier=2.0, keys=["a", "b"]) is True


def test_should_stop_false_zero_output_but_no_cost_spent_either():
    # Nothing happened for "b" this window at all -- no cost, no output --
    # so there's nothing to flag even though the ratio is still undefined.
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 100, 20)
    t.record_snapshot("b", 10, 5)
    t.record_snapshot("b", 10, 5)  # unchanged
    assert t.should_stop("b", multiplier=2.0, keys=["a", "b"]) is False


def test_reset_drops_both_snapshots():
    t = MarginalCostTracker()
    t.record_snapshot("a", 0, 0)
    t.record_snapshot("a", 100, 20)
    assert t.marginal_cost("a") == 5.0
    t.reset("a")
    assert t.marginal_cost("a") is None
    # A single snapshot recorded after reset should again read as "not
    # enough data yet", not silently diff against the pre-reset history.
    t.record_snapshot("a", 500, 999)
    assert t.marginal_cost("a") is None
