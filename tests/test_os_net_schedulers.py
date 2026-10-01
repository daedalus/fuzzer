"""OS / network schedulers ported to the seed and operator arenas.

Seed: mlfq, stride, eevdf, bfq, sfq, codel, aimd, p2c. Operator: op_stride,
op_p2c. Each reduces to a known baseline (round robin or uniform) under a
stated condition -- that is the falsification test for each.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_p2c import OpP2CScheduler
from fuzzer_tool.core.schedulers.op_stride import OpStrideScheduler
from fuzzer_tool.core.schedulers.seed_aimd import SeedAIMDScheduler
from fuzzer_tool.core.schedulers.seed_bfq import SeedBFQScheduler
from fuzzer_tool.core.schedulers.seed_codel import SeedCoDelScheduler
from fuzzer_tool.core.schedulers.seed_consolidated import SeedConsolidatedScheduler
from fuzzer_tool.core.schedulers.seed_eevdf import SeedEEVDFScheduler
from fuzzer_tool.core.schedulers.seed_mlfq import BASE_ALLOTMENT, SeedMLFQScheduler
from fuzzer_tool.core.schedulers.seed_p2c import SeedP2CScheduler
from fuzzer_tool.core.schedulers.seed_sfq import SeedSFQScheduler, bucket_of
from fuzzer_tool.core.schedulers.seed_stride import SeedStrideScheduler
from tests.support.scripted_rng import ScriptedRng

NAN = float("nan")
IDS = ["a", "b", "c"]


def _unit(_key):
    return 1.0


def _run(s, ids, n, outcome=lambda k: False):
    """Pick n times, recording outcome(picked) after each pick."""
    picks = []
    for _ in range(n):
        k = s.select_seed(ids)
        s.record(k, success=outcome(k))
        picks.append(k)
    return picks


SEED_FACTORIES = {
    "mlfq": SeedMLFQScheduler,
    "stride": SeedStrideScheduler,
    "eevdf": SeedEEVDFScheduler,
    "bfq": SeedBFQScheduler,
    "sfq": lambda: SeedSFQScheduler(rng=RandPool(seed=1)),
    "codel": SeedCoDelScheduler,
    "aimd": SeedAIMDScheduler,
    "p2c": lambda: SeedP2CScheduler(rng=RandPool(seed=1)),
    "consolidated": lambda: SeedConsolidatedScheduler(rng=RandPool(seed=1)),
}
OP_FACTORIES = {
    "op_stride": OpStrideScheduler,
    "op_p2c": lambda: OpP2CScheduler(rng=RandPool(seed=1)),
}


# --- Shared bandit interface ------------------------------------------------


@pytest.mark.parametrize("name", sorted(SEED_FACTORIES))
class TestSeedInterface:
    def test_empty_and_single(self, name):
        s = SEED_FACTORIES[name]()

        assert s.select_seed([]) == ""
        assert s.select_seed(["only"]) == "only"

    def test_supports_priors_is_false(self, name):
        assert type(SEED_FACTORIES[name]()).__dict__["supports_priors"] is False

    def test_record_clamps_and_registers(self, name):
        """Adversarial: NaN / >1 weights clamp; unknown keys register on record."""
        s = SEED_FACTORIES[name]()
        s.record("x", success=True, weight=5.0)
        s.record("x", success=True, weight=NAN)
        s.record("x", success=False)

        assert s.bandit_stats() == {"x": (1.0, 2.0)}

    def test_init_arm_never_resets(self, name):
        s = SEED_FACTORIES[name]()
        s.record("x", success=True)
        s.init_arm("x", prior_alpha=9.0, prior_beta=9.0)

        assert s.bandit_stats() == {"x": (1.0, 0.0)}

    def test_removed_seed_never_picked(self, name):
        s = SEED_FACTORIES[name]()
        _run(s, IDS, 6)

        assert "b" not in _run(s, ["a", "c"], 30)

    def test_regression_ledger_bounded_by_corpus(self, name):
        """Adversarial (PR #44 review): churned-out keys do not stay in the ledger."""
        s = SEED_FACTORIES[name]()
        for i in range(1000):
            s.record(f"gone{i}", success=bool(i % 2))
        s.select_seed(IDS)

        assert len(s.bandit_stats()) <= 2 * len(IDS) + 8

    def test_every_seed_served(self, name):
        """No starvation: all live seeds appear, even when all always fail."""
        s = SEED_FACTORIES[name]()

        assert set(_run(s, IDS, 300)) == set(IDS)


@pytest.mark.parametrize("name", sorted(OP_FACTORIES))
class TestOpInterface:
    def test_empty_and_single(self, name):
        s = OP_FACTORIES[name]()

        assert s.select_op([]) == ""
        assert s.select_op(["only"]) == "only"

    def test_supports_priors_is_false(self, name):
        assert type(OP_FACTORIES[name]()).__dict__["supports_priors"] is False

    def test_record_clamps(self, name):
        s = OP_FACTORIES[name]()
        s.record("x", True, weight=5.0)
        s.record("x", False)

        assert s.bandit_stats() == {"x": (1.0, 1.0)}

    def test_result_is_a_candidate(self, name):
        s = OP_FACTORIES[name]()
        s.record("zzz", True)

        assert {s.select_op(IDS) for _ in range(50)} <= set(IDS)


# --- MLFQ -------------------------------------------------------------------


class TestMLFQ:
    def test_one_level_is_round_robin(self):
        """Falsification: a single queue cannot demote -> plain cycling."""
        s = SeedMLFQScheduler(levels=1)

        assert _run(s, IDS, 6) == IDS * 2

    def test_fruitless_seed_demoted_below_fresh(self):
        s = SeedMLFQScheduler()
        s.select_seed(["a", "b"])
        for _ in range(BASE_ALLOTMENT):
            s.record("a", success=False)

        assert [s.select_seed(["a", "b"]) for _ in range(6)] == ["b"] * 6

    def test_success_resets_allotment(self):
        s = SeedMLFQScheduler()
        s.select_seed(["a", "b"])
        for _ in range(BASE_ALLOTMENT - 1):
            s.record("a", success=False)
        s.record("a", success=True)
        for _ in range(BASE_ALLOTMENT - 1):
            s.record("a", success=False)

        assert set(s.select_seed(["a", "b"]) for _ in range(4)) == {"a", "b"}

    def test_boost_returns_demoted_seed(self):
        """Anti-starvation: the periodic boost brings every seed back to the top."""
        s = SeedMLFQScheduler(boost_period=8)
        s.select_seed(["a", "b"])
        for _ in range(BASE_ALLOTMENT):
            s.record("a", success=False)
        picks = [s.select_seed(["a", "b"]) for _ in range(12)]

        assert "a" not in picks[:6]
        assert "a" in picks[6:]

    def test_bottom_level_is_round_robin(self):
        s = SeedMLFQScheduler(levels=2)
        s.select_seed(["a", "b"])
        for _ in range(50 * BASE_ALLOTMENT):
            s.record("a", success=False)
            s.record("b", success=False)

        assert [s.select_seed(["a", "b"]) for _ in range(4)] in (["a", "b"] * 2, ["b", "a"] * 2)

    def test_state_is_pruned(self):
        """Memory bound: churned-out seeds do not accumulate queue state."""
        s = SeedMLFQScheduler()
        for i in range(500):
            s.select_seed([f"k{i}", "x"])

        assert len(s._level) <= 16


# --- Stride / EEVDF (seed) ------------------------------------------------


class TestSeedStride:
    def test_flat_is_round_robin(self):
        assert _run(SeedStrideScheduler(), IDS, 6) == IDS * 2

    def test_weight_scales_share(self):
        w = {"fav": 2.0, "std": 1.0}
        s = SeedStrideScheduler()
        picks = [s.select_seed(["fav", "std"], w.get) for _ in range(300)]

        assert picks.count("fav") == pytest.approx(200, abs=1)


class TestSeedEEVDF:
    def test_flat_is_round_robin(self):
        assert _run(SeedEEVDFScheduler(), IDS, 6) == IDS * 2

    def test_slow_seed_gets_equal_time(self):
        cost = {"fast": 1.0, "slow": 4.0}
        s = SeedEEVDFScheduler()
        picks = [s.select_seed(["fast", "slow"], cost.get, _unit) for _ in range(500)]

        assert picks.count("fast") == pytest.approx(4 * picks.count("slow"), abs=4)

    def test_record_does_not_affect_selection(self):
        a, b = SeedEEVDFScheduler(), SeedEEVDFScheduler()
        b.record("a", success=True)

        assert [a.select_seed(IDS) for _ in range(9)] == [b.select_seed(IDS) for _ in range(9)]


# --- BFQ --------------------------------------------------------------------


class TestBFQ:
    def test_unit_budget_is_round_robin(self):
        """Falsification: min = max = init = 1 -> one pick per service -> cycling."""
        s = SeedBFQScheduler(min_budget=1, max_budget=1, init_budget=1)

        assert _run(s, IDS, 6) == IDS * 2

    def test_service_is_sticky_for_one_budget(self):
        s = SeedBFQScheduler(init_budget=4)

        assert _run(s, ["a", "b"], 8) == ["a"] * 4 + ["b"] * 4

    def test_budget_grows_when_productive_and_shrinks_when_idle(self):
        s = SeedBFQScheduler(init_budget=4)
        picks = _run(s, ["a", "b"], 18, outcome=lambda k: k == "a")

        assert picks[8:16] == ["a"] * 8
        assert picks[16:18] == ["b"] * 2

    def test_long_run_share_is_fair_despite_budgets(self):
        """Budgets change burst length, not share: finish tags charge every pick."""
        s = SeedBFQScheduler()
        picks = _run(s, ["a", "b"], 600, outcome=lambda k: k == "a")

        assert abs(picks.count("a") - picks.count("b")) <= s.max_budget

    def test_in_service_seed_removed(self):
        """Adversarial: the corpus drops the seed mid-budget."""
        s = SeedBFQScheduler(init_budget=4)
        s.select_seed(["a", "b"])

        assert s.select_seed(["b"]) == "b"
        assert s.select_seed(["b", "c"]) in {"b", "c"}

    def test_garbage_weight_is_neutral(self):
        w = {"a": NAN, "b": -3.0}
        s = SeedBFQScheduler(min_budget=1, max_budget=1, init_budget=1)

        assert [s.select_seed(["a", "b"], w.get) for _ in range(4)] == ["a", "b"] * 2


# --- SFQ --------------------------------------------------------------------


class TestSFQ:
    def test_one_bucket_is_round_robin(self):
        """Falsification: every flow hashes to bucket 0 -> plain cycling."""
        s = SeedSFQScheduler(rng=ScriptedRng(randints=[7]), buckets=1)

        assert _run(s, IDS, 6) == IDS * 2

    def test_family_cannot_crowd_out_a_singleton(self):
        """100 siblings share one flow; the lone seed still gets half the picks."""
        salt = 11
        family = {f"f{i}": "fam" for i in range(100)}
        flows = {**family, "solo": "solo"}
        assert bucket_of(salt, "fam", 1024) != bucket_of(salt, "solo", 1024)  # precondition

        s = SeedSFQScheduler(rng=ScriptedRng(randints=[salt]))
        picks = [s.select_seed(list(flows), flows.get) for _ in range(200)]

        assert picks.count("solo") == 100

    def test_siblings_round_robin_within_their_flow(self):
        family = {f"f{i}": "fam" for i in range(4)}
        s = SeedSFQScheduler(rng=ScriptedRng(randints=[3]))
        picks = [s.select_seed(list(family), family.get) for _ in range(8)]

        assert picks == list(family) * 2

    def test_salt_is_perturbed_every_period(self):
        """Exactly one fresh salt per period: the scripted rng runs dry on the 12th pick."""
        s = SeedSFQScheduler(rng=ScriptedRng(randints=[1, 2, 3]), perturb_period=4)
        for _ in range(11):
            s.select_seed(IDS)

        with pytest.raises(StopIteration):
            s.select_seed(IDS)

    def test_none_flow_falls_back_to_the_seed(self):
        """Adversarial: a flow function returning None keys the seed on itself."""
        s = SeedSFQScheduler(rng=ScriptedRng(randints=[0]), buckets=1)

        assert [s.select_seed(IDS, lambda _k: None) for _ in range(3)] == IDS

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            SeedSFQScheduler(rng=None)

    def test_bucket_is_process_stable(self):
        """crc32, not hash(): PYTHONHASHSEED must not move a flow between runs."""
        import zlib

        assert bucket_of(5, "fam", 1024) == zlib.crc32(b"5:fam") % 1024


# --- CoDel ------------------------------------------------------------------


class TestCoDel:
    def test_unreachable_target_is_round_robin(self):
        s = SeedCoDelScheduler(target=10**9)

        assert _run(s, IDS, 6) == IDS * 2

    def test_drop_schedule(self):
        """a stale after target+interval fruitless visits; served every 1+isqrt(count) landings."""
        s = SeedCoDelScheduler(target=1, interval=1)
        picks = _run(s, ["a", "b"], 12, outcome=lambda k: k == "b")

        assert picks == list("ababbabbabba")

    def test_success_leaves_dropping_state(self):
        s = SeedCoDelScheduler(target=1, interval=1)
        _run(s, ["a", "b"], 12, outcome=lambda k: k == "b")
        s.record("a", success=True)

        assert _run(s, ["a", "b"], 4, outcome=lambda k: True) in (["a", "b"] * 2, ["b", "a"] * 2)

    def test_stale_seed_deprioritized_never_starved(self):
        s = SeedCoDelScheduler(target=1, interval=1)
        picks = _run(s, ["a", "b"], 400, outcome=lambda k: k == "b")

        assert 0 < picks.count("a") < picks.count("b") // 2

    def test_all_seeds_stale_still_serves_each(self):
        """Adversarial: every seed dropping -> bounded pick, nobody starved."""
        s = SeedCoDelScheduler(target=1, interval=1)

        assert set(_run(s, IDS, 3 * 3 * s.max_gap)) == set(IDS)


# --- AIMD -------------------------------------------------------------------


class TestAIMD:
    def test_no_increase_no_decrease_is_round_robin(self):
        """Falsification: alpha=0, beta=1 freezes every window -> stride on equal tickets."""
        s = SeedAIMDScheduler(alpha=0.0, beta=1.0)

        assert _run(s, IDS, 6, outcome=lambda k: k == "a") == IDS * 2

    def test_productive_seed_window_grows(self):
        s = SeedAIMDScheduler()
        picks = _run(s, ["a", "b"], 200, outcome=lambda k: k == "a")

        assert picks.count("a") > 2 * picks.count("b")

    def test_loss_run_halves_window(self):
        s = SeedAIMDScheduler(alpha=1.0, beta=0.5, loss_run=3)
        for _ in range(7):
            s.record("a", success=True)
        before = s._window["a"]
        for _ in range(3):
            s.record("a", success=False)

        assert s._window["a"] == pytest.approx(before * 0.5)

    def test_window_bounded(self):
        """Adversarial: endless wins or losses stay inside [w_min, w_max]."""
        s = SeedAIMDScheduler()
        for _ in range(10_000):
            s.record("up", success=True)
            s.record("down", success=False)

        assert s._window["up"] == s.w_max
        assert s._window["down"] == s.w_min


# --- Power of two choices ---------------------------------------------------


class TestSeedP2C:
    def test_equal_scores_take_the_first_draw(self):
        """Falsification: no signal -> the first uniform draw wins -> uniform selection."""
        s = SeedP2CScheduler(rng=ScriptedRng(choice_idxs=[2, 0]))

        assert s.select_seed(IDS) == "c"

    def test_better_second_draw_wins(self):
        s = SeedP2CScheduler(rng=ScriptedRng(choice_idxs=[0, 1]))
        s.record("b", success=True)

        assert s.select_seed(IDS) == "b"

    def test_same_seed_drawn_twice(self):
        s = SeedP2CScheduler(rng=ScriptedRng(choice_idxs=[1, 1]))

        assert s.select_seed(IDS) == "b"

    def test_worse_second_draw_loses(self):
        s = SeedP2CScheduler(rng=ScriptedRng(choice_idxs=[0, 1]))
        s.record("b", success=False)

        assert s.select_seed(IDS) == "a"

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            SeedP2CScheduler(rng=None)


# --- Operator arms ----------------------------------------------------------


class TestOpStride:
    def test_no_signal_is_round_robin(self):
        s = OpStrideScheduler()

        assert [s.select_op(IDS) for _ in range(6)] == IDS * 2

    def test_share_follows_posterior_mean(self):
        """3 wins vs 3 losses: means 4/5 vs 1/5 -> 4:1 deterministic share."""
        s = OpStrideScheduler()
        for _ in range(3):
            s.record("a", True)
            s.record("b", False)
        picks = [s.select_op(["a", "b"]) for _ in range(500)]

        assert picks.count("a") == pytest.approx(400, abs=2)


class TestOpP2C:
    def test_equal_scores_take_the_first_draw(self):
        s = OpP2CScheduler(rng=ScriptedRng(choice_idxs=[2, 0]))

        assert s.select_op(IDS) == "c"

    def test_better_second_draw_wins(self):
        s = OpP2CScheduler(rng=ScriptedRng(choice_idxs=[0, 1]))
        s.record("b", True)

        assert s.select_op(IDS) == "b"

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpP2CScheduler(rng=None)
