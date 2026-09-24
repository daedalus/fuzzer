"""SJT brute-force oracle for core/job_scheduling (FINDINGS P4-14).

The oracle walks every permutation by adjacent transposition, so each step
recosts only the two swapped jobs. Control first (Hard Rule 46): the SJT
optimum must equal an itertools.permutations brute force.
"""

from __future__ import annotations

import itertools
import math

import pytest

from fuzzer_tool.core.job_scheduling import Job, edf_order, lawler_order
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.sjt import sjt_min_fmax, sjt_swaps

N_INSTANCES = 25


def _lateness(job: Job, c: float) -> float:
    return c - job.due_date


def _weighted_tardiness(job: Job, c: float) -> float:
    return job.weight * max(0.0, c - job.due_date)


def _fmax(seq, cost) -> float:
    t, worst = 0.0, -math.inf
    for j in seq:
        t += j.processing_time
        worst = max(worst, cost(j, t))
    return worst


def _respects(seq, prec) -> bool:
    seen: set = set()
    for j in seq:
        if not prec.get(j.id, set()) <= seen:
            return False
        seen.add(j.id)
    return True


def _brute(jobs, prec, cost) -> float:
    return min(
        (_fmax(p, cost) for p in itertools.permutations(jobs) if _respects(p, prec)),
        default=math.inf,
    )


def _instance(rng: RandPool, n: int, with_prec: bool) -> tuple[list[Job], dict]:
    jobs = [
        Job(i, rng.randint(0, 9), due_date=rng.randint(0, 30), weight=rng.randint(1, 4))
        for i in range(n)
    ]
    prec: dict = {}
    if with_prec:
        # Edges only from lower to higher id: acyclic by construction.
        for b in range(n):
            preds = {a for a in range(b) if rng.random() < 0.25}
            if preds:
                prec[b] = preds
    return jobs, prec


# ---------------------------------------------------------------------------
# The walk itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(0, 7))
def test_sjt_visits_every_permutation_once(n):
    perm = list(range(n))
    seen = {tuple(perm)}
    for i in sjt_swaps(n):
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        seen.add(tuple(perm))
    assert len(seen) == math.factorial(n)


# ---------------------------------------------------------------------------
# Oracle vs brute force (control)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_prec", [False, True])
def test_control_sjt_matches_itertools(with_prec):
    rng = RandPool(seed=11)
    for _ in range(N_INSTANCES):
        jobs, prec = _instance(rng, 6, with_prec)
        for cost in (_lateness, _weighted_tardiness):
            best, order = sjt_min_fmax(jobs, prec, cost)
            assert best == _brute(jobs, prec, cost)
            assert _fmax(order, cost) == best
            assert _respects(order, prec)


# ---------------------------------------------------------------------------
# job_scheduling vs oracle
# ---------------------------------------------------------------------------


def test_edf_is_optimal_for_lmax():
    rng = RandPool(seed=3)
    for _ in range(N_INSTANCES):
        jobs, _ = _instance(rng, 7, with_prec=False)
        best, _ = sjt_min_fmax(jobs, {}, _lateness)
        assert _fmax(edf_order(jobs), _lateness) == best


@pytest.mark.parametrize("cost", [_lateness, _weighted_tardiness])
def test_lawler_is_optimal_with_precedence(cost):
    rng = RandPool(seed=5)
    for _ in range(N_INSTANCES):
        jobs, prec = _instance(rng, 7, with_prec=True)
        best, _ = sjt_min_fmax(jobs, prec, cost)
        order, value = lawler_order(jobs, prec, cost)
        assert value == best
        assert _respects(order, prec)


def test_oracle_rejects_a_suboptimal_rule():
    """Falsification: latest-due-first must lose to the oracle here."""
    jobs = [Job(0, 3, due_date=3), Job(1, 3, due_date=6)]
    best, _ = sjt_min_fmax(jobs, {}, _lateness)
    ldd = sorted(jobs, key=lambda j: -j.due_date)
    assert _fmax(ldd, _lateness) > best


def test_oracle_infeasible_precedence():
    """Adversarial: a 2-cycle has no feasible order."""
    jobs = [Job(0, 1), Job(1, 1)]
    best, order = sjt_min_fmax(jobs, {0: {1}, 1: {0}}, _lateness)
    assert best == math.inf
    assert order == []


def test_oracle_empty_and_zero_times():
    assert sjt_min_fmax([], {}, _lateness) == (-math.inf, [])
    jobs = [Job(0, 0, due_date=0), Job(1, 0, due_date=0)]
    assert sjt_min_fmax(jobs, {}, _lateness)[0] == 0.0
