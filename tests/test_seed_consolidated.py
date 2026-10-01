"""SeedConsolidatedScheduler: p2c + cost/weight + AIMD decay + sweep + flows.

Covers core/schedulers/seed_consolidated.py. Shared OS-arm interface tests
(empty/single, clamping, ledger bound, no starvation) run through
``tests/test_os_net_schedulers.py::SEED_FACTORIES``.
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_consolidated import (
    COST_FLOOR,
    LOSS_DECAY,
    LOSS_RUN,
    SWEEP_EVERY,
    SeedConsolidatedScheduler,
)
from tests.support.scripted_rng import ScriptedRng

IDS = ["a", "b", "c"]


def _unit(_key):
    return 1.0


def _sched(randints=()):
    return SeedConsolidatedScheduler(rng=ScriptedRng(randints=randints))


def _mean(s, f):
    """Beta(1, 1) posterior mean, derived independently of the class."""
    return (s + 1.0) / (s + f + 2.0)


class TestTwoChoices:
    def test_higher_posterior_wins(self):
        s = _sched(randints=[0, 1])
        s.record("b", success=True)

        assert s.select_seed(IDS) == "b"

    def test_falsification_no_signal_is_first_draw(self):
        """Equal scores: the first uniform draw stands, as plain p2c."""
        s = _sched(randints=[2, 0])

        assert s.select_seed(IDS) == "c"

    def test_cheap_seed_beats_slow_one(self):
        """EEVDF/DRR feature: yield per unit of target time."""
        s = _sched(randints=[0, 1])
        cost = {"a": 4.0, "b": 1.0, "c": 1.0}.get

        assert s.select_seed(IDS, cost, _unit) == "b"

    def test_favored_seed_beats_plain_one(self):
        """Stride/BFQ feature: favored seeds weigh more."""
        s = _sched(randints=[1, 0])
        weight = {"a": 2.0, "b": 1.0, "c": 1.0}.get

        assert s.select_seed(IDS, _unit, weight) == "a"

    @pytest.mark.parametrize("bad", [0.0, -3.0, math.nan, math.inf])
    def test_adversarial_cost_and_weight_sanitised(self, bad):
        """Zero cost floors at COST_FLOOR; NaN/inf/negative are neutral."""
        s = _sched(randints=[0, 1])
        s.record("b", success=True)
        hostile = {"a": bad, "b": 1.0, "c": 1.0}.get

        picked = s.select_seed(IDS, hostile, hostile)
        a_score = _mean(0, 0) * (1.0 / COST_FLOOR if bad == 0.0 else 1.0)
        b_score = _mean(1, 0)

        assert picked == ("a" if a_score > b_score else "b")


class TestAIMDDecay:
    def test_loss_run_decays_success_evidence(self):
        s = _sched()
        for _ in range(3):
            s.record("a", success=True)
        for _ in range(LOSS_RUN):
            s.record("a", success=False)

        assert s.bandit_stats()["a"] == (3.0 * LOSS_DECAY, float(LOSS_RUN))

    def test_falsification_find_resets_the_run(self):
        s = _sched()
        s.record("a", success=True)
        for _ in range(LOSS_RUN - 1):
            s.record("a", success=False)
        s.record("a", success=True)
        for _ in range(LOSS_RUN - 1):
            s.record("a", success=False)

        assert s.bandit_stats()["a"] == (2.0, float(2 * (LOSS_RUN - 1)))


class TestSweep:
    def test_every_nth_pick_is_round_robin(self):
        """Anti-starvation: sweep picks consume no RNG and walk the corpus."""
        s = _sched(randints=[0, 0] * (2 * SWEEP_EVERY - 2))
        s.record("a", success=True)
        picks = [s.select_seed(IDS) for _ in range(2 * SWEEP_EVERY)]

        assert picks[SWEEP_EVERY - 1] == "a"
        assert picks[2 * SWEEP_EVERY - 1] == "b"
        assert {p for i, p in enumerate(picks) if (i + 1) % SWEEP_EVERY} == {"a"}

    def test_adversarial_worst_seed_still_served(self):
        """A seed that never wins a duel is reached by the sweep."""
        s = SeedConsolidatedScheduler(rng=RandPool(seed=7))
        ids = [f"s{i}" for i in range(8)]
        for k in ids[1:]:
            for _ in range(5):
                s.record(k, success=True)
        picks = [s.select_seed(ids) for _ in range(SWEEP_EVERY * len(ids))]

        assert "s0" in picks


class TestFlows:
    def test_siblings_share_one_flow(self):
        """SFQ feature: flow drawn first, member second."""
        kids = [f"k{i}" for i in range(6)]
        ids = [*kids, "solo"]
        flow = {**dict.fromkeys(kids, "P"), "solo": None}.get
        # pick 1: flow 1 (solo, no member draw), flow 0 + member 3.
        s = _sched(randints=[1, 0, 3])

        assert s.select_seed(ids, _unit, _unit, flow) == "solo"

    def test_singleton_flows_draw_once_each(self):
        """Without lineage every seed is its own flow: one draw per candidate."""
        s = _sched(randints=[1, 2])
        s.record("c", success=True)

        assert s.select_seed(IDS, _unit, _unit, lambda _k: None) == "c"


class TestBounds:
    def test_regression_miss_runs_bounded_by_corpus(self):
        s = _sched(randints=[0, 1])
        for i in range(1000):
            s.record(f"gone{i}", success=False)
        s.select_seed(IDS)

        assert len(s._misses) <= 2 * len(IDS) + 8

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            SeedConsolidatedScheduler(rng=None)

    def test_supports_priors_declared(self):
        assert SeedConsolidatedScheduler.__dict__["supports_priors"] is False
