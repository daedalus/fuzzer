"""Tests for ``core/schedulers/op_ant_colony.py`` -- MAX-MIN ant system over op chains.

Read first:

- ``test_draw_follows_the_aco_rule`` -- weight tau(prev, op)^alpha *
  eta(op)^beta against a plain-Python transcription.
- ``test_without_pheromone_it_is_the_heuristic`` (falsification) -- alpha = 0
  must ignore every deposit.
- ``test_evaporation_floors_at_tau_min`` (adversarial) -- after 10^5
  evaporations (and the scale renormalisation they force) an edge reads
  exactly tau_min, never 0: the MMAS anti-stagnation bound.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_ant_colony import AntColonyScheduler
from tests.support.scripted_rng import ScriptedRng


def _sched(**kw) -> AntColonyScheduler:
    kw.setdefault("rng", RandPool(seed=31))
    return AntColonyScheduler(**kw)


def test_declares_prior_support():
    assert AntColonyScheduler.supports_priors is True


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1.0},
        {"beta": -1.0},
        {"rho": 0.0},
        {"rho": 1.0},
        {"tau_min": 0.0},
        {"tau_min": 2.0, "tau_max": 1.0},
        {"alpha": math.nan},
    ],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        AntColonyScheduler(**kwargs)


def test_empty_candidate_list_is_not_a_crash():
    assert _sched().select_op([]) == ""


def test_prior_seeds_the_heuristic():
    s = _sched()
    s.init_arm("x", 3.0, 1.0)
    s.init_arm("x", 1.0, 9.0)  # idempotent
    assert s.heuristic("x") == pytest.approx(3.0 / 4.0)


# --------------------------------------------------------------------------
# Pheromone mechanics
# --------------------------------------------------------------------------


def test_fresh_edges_start_at_tau_max():
    s = _sched(tau_max=2.0)
    s.init_arm("a")
    s.init_arm("b")
    assert s.pheromone("a", "b") == 2.0
    assert s.pheromone(None, "a") == 2.0


def test_evaporation_is_geometric_until_the_floor():
    rho, tau_max, tau_min = 0.1, 1.0, 0.05
    s = _sched(rho=rho, tau_max=tau_max, tau_min=tau_min)
    s.init_arm("a")
    s.init_arm("b")
    for n in range(1, 40):
        s.record("b", False)
        assert s.pheromone("a", "b") == pytest.approx(max(tau_max * (1 - rho) ** n, tau_min))


def test_evaporation_floors_at_tau_min():
    s = _sched(rho=0.5, tau_min=0.01)
    s.init_arm("a")
    for _ in range(100_000):
        s.record("a", False)
    assert s.pheromone("a", "a") == 0.01
    assert math.isfinite(s._scale) and s._scale > 0.0


def test_deposit_follows_record_order_and_is_capped():
    """Evaporate, then deposit r on edge prev -> op, starting from the
    tau_min floor. a then b succeeds: a->b gains r; b->a does not."""
    rho, tau_min, r = 0.5, 0.01, 0.3
    s = _sched(rho=rho, tau_max=1.0, tau_min=tau_min)
    s.init_arm("a")
    s.init_arm("b")
    for _ in range(12):  # drain every edge to the floor
        s.record("a", False)
    before = s.pheromone("a", "b")
    assert before == tau_min
    s.record("b", True, weight=r)
    assert s.pheromone("a", "b") == pytest.approx(max(before * (1 - rho), tau_min) + r)
    assert s.pheromone("b", "a") == tau_min
    for _ in range(20):
        s.record("a", False)
        s.record("b", True)
    assert s.pheromone("a", "b") <= 1.0


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_draw_follows_the_aco_rule():
    """First draw is from the virtual start node: inverse CDF of
    tau(start, op)^alpha * eta(op)^beta, normalised."""
    alpha, beta, u = 1.5, 2.0, 0.55
    s = AntColonyScheduler(alpha=alpha, beta=beta, rng=ScriptedRng(randoms=[u]))
    ops = ["a", "b", "c"]
    for op, wins in (("a", 1), ("b", 3), ("c", 0)):
        for i in range(4):
            s.record(op, i < wins)
    w = [s.pheromone(None, op) ** alpha * s.heuristic(op) ** beta for op in ops]
    cum, want = 0.0, ops[-1]
    for op, x in zip(ops, w, strict=True):
        cum += x / sum(w)
        if u < cum:
            want = op
            break
    assert s.select_op(ops) == want


def test_without_pheromone_it_is_the_heuristic():
    """Falsification: alpha = 0 makes tau^alpha = 1 whatever was deposited."""
    s = _sched(alpha=0.0)
    s.init_arm("a")
    s.init_arm("b")
    for _ in range(30):
        s.record("a", False)
        s.record("b", True)
    w = s.weights(None, ["a", "b"])
    assert w == pytest.approx([s.heuristic("a") ** s.beta, s.heuristic("b") ** s.beta])


def test_learns_a_chain():
    """b pays only right after a; c never. P(b | prev a) must beat P(b | prev c).

    Mechanism, not defaults: at the default rho 0.05 / beta 4 the colony
    settles on b -> b, the a -> b edge evaporates to the floor and the chain
    is forgotten. Slow evaporation and a weak heuristic let pheromone lead."""
    s = _sched(rho=0.005, beta=1.0)
    ops = ["a", "b", "c"]
    env = random.Random(5)
    prev = None
    for _ in range(4000):
        op = s.select_op(ops)
        s.record(op, op == "b" and prev == "a" and env.random() < 0.8)
        prev = op
    wa = s.weights("a", ops)
    wc = s.weights("c", ops)
    assert wa[1] / sum(wa) > wc[1] / sum(wc)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_stay_bounded(weight):
    s = _sched()
    for _ in range(50):
        s.record(s.select_op(["a", "b"]), True, weight=weight)
    for e in (("a", "b"), ("b", "a"), ("a", "a")):
        assert s.tau_min <= s.pheromone(*e) <= s.tau_max


def test_a_productive_operator_dominates():
    s = _sched()
    arms = [f"op{i}" for i in range(12)]
    env = random.Random(7)
    picks = 0
    for t in range(3000):
        op = s.select_op(arms)
        s.record(op, env.random() < (0.3 if op == arms[5] else 0.03))
        picks += t >= 2000 and op == arms[5]
    assert picks > 300


def test_bandit_stats_reports_pulls():
    s = _sched()
    s.record(s.select_op(["a", "b"]), True)
    stats = s.bandit_stats()
    assert stats["ant_colony_pulls"] == 1
    assert stats["ant_colony_arms"] == 2
