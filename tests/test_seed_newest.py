"""SeedNewestScheduler: Growing Tree 'newest' policy as a seed arm
(docs/handover/handover_maze_algorithms_2026-09-24.md, item 2).

With probability ``p_newest`` pick the most recently registered live seed
(DFS / recursive backtracker); otherwise a uniform live seed (Prim). All
draws go through the injected RNG, so every test scripts the exact sequence
(Hard Rule 39).
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers.seed_newest import SeedNewestScheduler
from tests.support.scripted_rng import ScriptedRng

IDS = ["a", "b", "c", "d"]


def _sched(p: float = 0.5, **rng_kw) -> SeedNewestScheduler:
    return SeedNewestScheduler(rng=ScriptedRng(**rng_kw), p_newest=p)


class TestPolicy:
    def test_coin_below_p_picks_newest(self):
        s = _sched(0.5, randoms=[0.49])

        assert s.select_seed(IDS) == "d"

    def test_coin_at_or_above_p_picks_uniform(self):
        s = _sched(0.5, randoms=[0.5], choice_idxs=[1])

        assert s.select_seed(IDS) == "b"

    def test_newest_is_registration_order_not_list_order(self):
        s = _sched(1.0, randoms=[0.0, 0.0])
        s.select_seed(["a", "b"])

        # "z" registers after "b"; the caller's list order is irrelevant.
        assert s.select_seed(["z", "a", "b"]) == "z"

    def test_newest_moves_to_next_newest_when_it_leaves_the_corpus(self):
        s = _sched(1.0, randoms=[0.0, 0.0])
        s.select_seed(IDS)

        assert s.select_seed(["a", "b", "c"]) == "c"


class TestFalsification:
    """The two knob endpoints reduce to the policies they are named after."""

    def test_p_one_is_always_newest(self):
        s = _sched(1.0, randoms=[0.0, 0.5, 0.999999])

        assert [s.select_seed(IDS) for _ in range(3)] == ["d", "d", "d"]

    def test_p_zero_is_uniform_and_never_newest_by_coin(self):
        s = _sched(0.0, randoms=[0.0, 0.0], choice_idxs=[0, 3])

        assert [s.select_seed(IDS) for _ in range(2)] == ["a", "d"]

    def test_outcome_signal_never_changes_the_pick(self):
        a = _sched(0.5, randoms=[0.9], choice_idxs=[2])
        b = _sched(0.5, randoms=[0.9], choice_idxs=[2])
        for key in IDS:
            b.record(key, success=True, weight=1.0)

        assert a.select_seed(IDS) == b.select_seed(IDS) == "c"


class TestAdversarial:
    def test_empty_returns_empty_string(self):
        assert _sched().select_seed([]) == ""

    def test_single_candidate_draws_nothing(self):
        """No scripted draws: a draw here would raise StopIteration."""
        assert _sched().select_seed(["only"]) == "only"

    @pytest.mark.parametrize("p", [-0.1, 1.1, math.nan, math.inf])
    def test_rejects_p_outside_unit_interval(self, p):
        with pytest.raises(ValueError, match="p_newest"):
            SeedNewestScheduler(rng=ScriptedRng(), p_newest=p)

    def test_requires_a_randpool(self):
        with pytest.raises(ValueError, match="RandPool"):
            SeedNewestScheduler(rng=None)

    def test_reregistering_never_makes_a_seed_newer(self):
        s = _sched(1.0, randoms=[0.0, 0.0])
        s.select_seed(["a", "b"])
        s.init_arm("a")

        assert s.select_seed(["a", "b"]) == "b"

    def test_ledger_stays_bounded_while_the_corpus_turns_over(self):
        s = SeedNewestScheduler(rng=ScriptedRng(randoms=[0.0] * 400), p_newest=1.0)
        for i in range(400):
            s.select_seed([f"k{i}", f"k{i + 1}"])

        assert len(s.bandit_stats()) <= 2 * 2 + 8 + 2
        assert len(s._born) <= len(s.bandit_stats())

    def test_departed_seed_that_returns_is_newest(self):
        s = _sched(1.0, randoms=[0.0] * 3)
        s.select_seed(["a", "b", "c"])
        s._counts.pop("a")
        s._born.pop("a")

        assert s.select_seed(["a", "b", "c"]) == "a"
