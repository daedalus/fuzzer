"""Tests for core/job_scheduling.py (P3-3, step 1 -- pure functions, no wiring)."""

import pytest

from fuzzer_tool.core.job_scheduling import (
    Job,
    edf_order,
    ffd_pack,
    lawler_order,
    mdd_order,
    multifit,
    wmdd_order,
)


# --------------------------------------------------------------------------
# edf_order
# --------------------------------------------------------------------------


def test_edf_order_sorts_by_due_date():
    jobs = [Job("a", 3, due_date=10), Job("b", 1, due_date=5), Job("c", 2, due_date=7)]
    order = edf_order(jobs)
    assert [j.id for j in order] == ["b", "c", "a"]


def test_edf_order_ties_broken_by_id():
    jobs = [Job("b", 1, due_date=5), Job("a", 1, due_date=5)]
    order = edf_order(jobs)
    assert [j.id for j in order] == ["a", "b"]


def test_edf_order_negative_processing_time_rejected():
    with pytest.raises(ValueError):
        edf_order([Job("a", -1, due_date=5)])


def test_edf_order_is_optimal_for_lmax_no_precedence():
    # Jackson's rule: EDF minimizes max lateness with no precedence.
    # Brute-force check against all permutations for a small instance.
    import itertools

    jobs = [Job(0, 3, due_date=6), Job(1, 2, due_date=4), Job(2, 4, due_date=10)]

    def lmax(seq):
        t = 0.0
        worst = float("-inf")
        for j in seq:
            t += j.processing_time
            worst = max(worst, t - j.due_date)
        return worst

    best_possible = min(lmax(seq) for seq in itertools.permutations(jobs))
    assert lmax(edf_order(jobs)) == best_possible


# --------------------------------------------------------------------------
# mdd_order
# --------------------------------------------------------------------------


def test_mdd_order_no_precedence_reduces_to_something_sane():
    jobs = [Job("a", 3, due_date=10), Job("b", 1, due_date=5), Job("c", 2, due_date=7)]
    order = mdd_order(jobs)
    assert {j.id for j in order} == {"a", "b", "c"}
    assert len(order) == 3


def test_mdd_order_respects_precedence():
    # c must come after a, even though c's due date is far earlier.
    jobs = [Job("a", 5, due_date=100), Job("c", 1, due_date=1)]
    precedence = {"c": {"a"}}
    order = mdd_order(jobs, precedence)
    assert [j.id for j in order] == ["a", "c"]


def test_mdd_order_cycle_raises():
    jobs = [Job("a", 1), Job("b", 1)]
    precedence = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(ValueError, match="cycle"):
        mdd_order(jobs, precedence)


def test_mdd_order_unknown_predecessor_raises():
    jobs = [Job("a", 1)]
    precedence = {"a": {"ghost"}}
    with pytest.raises(ValueError, match="unknown"):
        mdd_order(jobs, precedence)


def test_mdd_order_deterministic_tie_break():
    jobs = [Job("b", 1, due_date=5), Job("a", 1, due_date=5)]
    order = mdd_order(jobs)
    assert [j.id for j in order] == ["a", "b"]


# --------------------------------------------------------------------------
# wmdd_order
# --------------------------------------------------------------------------


def test_wmdd_order_equals_mdd_when_all_weights_equal():
    jobs = [Job("a", 5, due_date=100, weight=2.0), Job("c", 1, due_date=1, weight=2.0)]
    precedence = {"c": {"a"}}
    assert wmdd_order(jobs, precedence) == mdd_order(jobs, precedence)


def test_wmdd_order_higher_weight_sorts_earlier_when_tied_on_urgency():
    # Same processing time and due date, only weight differs -> the
    # higher-weight job should be treated as more urgent.
    jobs = [Job("low", 2, due_date=10, weight=1.0), Job("high", 2, due_date=10, weight=5.0)]
    order = wmdd_order(jobs)
    assert order[0].id == "high"


def test_wmdd_order_nonpositive_weight_rejected():
    with pytest.raises(ValueError):
        wmdd_order([Job("a", 1, weight=0.0)])


# --------------------------------------------------------------------------
# lawler_order
# --------------------------------------------------------------------------


def _lateness_cost(job: Job, completion_time: float) -> float:
    return completion_time - job.due_date


def test_lawler_order_matches_edf_when_cost_is_lateness_no_precedence():
    jobs = [Job(0, 3, due_date=6), Job(1, 2, due_date=4), Job(2, 4, due_date=10)]
    order, max_cost = lawler_order(jobs, None, _lateness_cost)
    assert [j.id for j in order] == [j.id for j in edf_order(jobs)]

    def lmax(seq):
        t = 0.0
        worst = float("-inf")
        for j in seq:
            t += j.processing_time
            worst = max(worst, t - j.due_date)
        return worst

    assert max_cost == lmax(order)


def test_lawler_order_respects_precedence():
    jobs = [Job("a", 5, due_date=100), Job("c", 1, due_date=1)]
    precedence = {"c": {"a"}}
    order, _ = lawler_order(jobs, precedence, _lateness_cost)
    assert [j.id for j in order] == ["a", "c"]


def test_lawler_order_is_exact_against_brute_force():
    import itertools

    jobs = [Job(0, 2, due_date=5), Job(1, 3, due_date=8), Job(2, 1, due_date=3), Job(3, 4, due_date=12)]
    # Precedence: job 2 must follow job 0.
    precedence = {2: {0}}

    def lmax(seq):
        t = 0.0
        worst = float("-inf")
        for j in seq:
            t += j.processing_time
            worst = max(worst, t - j.due_date)
        return worst

    def respects(seq):
        seen = set()
        for j in seq:
            if j.id in precedence and not precedence[j.id] <= seen:
                return False
            seen.add(j.id)
        return True

    valid_perms = [seq for seq in itertools.permutations(jobs) if respects(seq)]
    best_possible = min(lmax(seq) for seq in valid_perms)

    order, max_cost = lawler_order(jobs, precedence, _lateness_cost)
    assert max_cost == best_possible


def test_lawler_order_cycle_raises():
    jobs = [Job("a", 1), Job("b", 1)]
    precedence = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(ValueError, match="cycle"):
        lawler_order(jobs, precedence, _lateness_cost)


# --------------------------------------------------------------------------
# ffd_pack
# --------------------------------------------------------------------------


def test_ffd_pack_basic():
    items = [("a", 4), ("b", 8), ("c", 1), ("d", 4), ("e", 2), ("f", 1)]
    bins = ffd_pack(items, capacity=10)
    # All items placed exactly once.
    placed = [i for b in bins for i in b]
    assert sorted(placed) == sorted(i for i, _ in items)
    # No bin exceeds capacity.
    sizes = dict(items)
    for b in bins:
        assert sum(sizes[i] for i in b) <= 10


def test_ffd_pack_rejects_oversized_item():
    with pytest.raises(ValueError):
        ffd_pack([("a", 20)], capacity=10)


def test_ffd_pack_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        ffd_pack([("a", 0)], capacity=10)


def test_ffd_pack_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        ffd_pack([("a", 1)], capacity=0)


def test_ffd_pack_deterministic_order():
    items = [("z", 5), ("a", 5)]
    bins1 = ffd_pack(items, capacity=5)
    bins2 = ffd_pack(list(reversed(items)), capacity=5)
    assert bins1 == bins2


# --------------------------------------------------------------------------
# multifit
# --------------------------------------------------------------------------


def test_multifit_all_items_placed():
    items = [(i, size) for i, size in enumerate([9, 8, 7, 6, 5, 4, 3, 2, 1])]
    bins, makespan = multifit(items, m=3)
    placed = [i for b in bins for i in b]
    assert sorted(placed) == sorted(i for i, _ in items)
    assert len(bins) <= 3
    sizes = dict(items)
    for b in bins:
        assert sum(sizes[i] for i in b) <= makespan + 1e-6


def test_multifit_single_machine_is_everything_in_one_bin():
    items = [(0, 3), (1, 5), (2, 2)]
    bins, makespan = multifit(items, m=1)
    assert len(bins) == 1
    assert makespan == pytest.approx(10)


def test_multifit_more_machines_than_items():
    items = [(0, 3), (1, 5)]
    bins, makespan = multifit(items, m=10)
    # Never opens more bins than items, regardless of m.
    assert len(bins) <= 2
    placed = [i for b in bins for i in b]
    assert sorted(placed) == [0, 1]


def test_multifit_rejects_bad_m():
    with pytest.raises(ValueError):
        multifit([(0, 1)], m=0)


def test_multifit_rejects_empty_items():
    with pytest.raises(ValueError):
        multifit([], m=2)


def test_multifit_makespan_within_classic_bound_of_optimal():
    # Optimal makespan for this instance (found by brute-force partition
    # search) is known; Multifit's guarantee is <= 1.22 * OPT after enough
    # iterations (default 25 is far beyond what's needed here).
    items = [(i, s) for i, s in enumerate([6, 5, 5, 4, 4, 3, 3, 2])]
    sizes = dict(items)

    def best_partition_makespan(ids, m):
        # Brute force over all assignments for a small instance.
        import itertools

        best = float("inf")
        for assignment in itertools.product(range(m), repeat=len(ids)):
            loads = [0] * m
            for idx, machine in zip(ids, assignment):
                loads[machine] += sizes[idx]
            best = min(best, max(loads))
        return best

    opt = best_partition_makespan([i for i, _ in items], m=3)
    _, makespan = multifit(items, m=3)
    assert makespan <= 1.22 * opt + 1e-6
