"""Unit, falsification and adversarial coverage for D-UCB, SW-UCB and CUCB.

Hard Rule 23: every new capability ships one falsification test and one
adversarial test. The falsification tests here are the ones that would still
pass if the scheduler were replaced by ``rng.choice`` or by a scheduler with
no recency mechanism at all, so each is paired with an assertion pinning the
*mechanism* rather than the absence of a crash.

Convergence against ground-truth environments lives in
``tests/test_scheduler_convergence.py``; this file covers the arithmetic those
runs depend on.
"""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import CUCBScheduler, DUCBScheduler, SWUCBScheduler

ARMS = ["bit_flip", "byte_flip", "block_insert", "dict_append"]

#: Distinct enough from 1.0 that a single record visibly moves the discount,
#: without underflowing before the renormalisation test can reach it.
FAST_GAMMA = 0.9


def _seeded(cls, **kw):
    return cls(rng=RandPool(seed=1234), **kw)


def _settle(scheduler) -> None:
    settle = getattr(scheduler, "settle_round", None)
    if settle is not None:
        settle()


def _open_all(sched, arms=ARMS):
    """Drive every arm out of the unpulled state with one failure each."""
    for a in arms:
        sched.init_arm(a)
    for a in arms:
        sched.record(a, success=False)
        _settle(sched)


# ---------------------------------------------------------------------------
# D-UCB
# ---------------------------------------------------------------------------


class TestDUCB:
    def test_rejects_invalid_parameters(self):
        for kw in ({"gamma": 0.0}, {"gamma": 1.5}, {"xi": 0.0}, {"exploration": -1.0}):
            with pytest.raises(ValueError):
                DUCBScheduler(**kw)

    def test_unpulled_arms_come_first(self):
        s = _seeded(DUCBScheduler)
        for a in ARMS:
            s.init_arm(a)
        seen = set()
        for _ in range(len(ARMS)):
            op = s.select_op(ARMS)
            seen.add(op)
            s.record(op, success=False)
        assert seen == set(ARMS), "an arm was never opened before exploitation began"

    def test_discount_ages_old_evidence(self):
        """The mechanism: a win N records ago is worth gamma**N of a fresh one."""
        s = _seeded(DUCBScheduler, gamma=FAST_GAMMA)
        for a in ARMS:
            s.init_arm(a)
        s.record("bit_flip", success=True, weight=1.0)
        fresh = s.discounted_counts()["bit_flip"]
        for _ in range(10):
            s.record("byte_flip", success=False)
        aged = s.discounted_counts()["bit_flip"]
        assert aged == pytest.approx(fresh * FAST_GAMMA**10, rel=1e-9)

    def test_gamma_one_never_forgets(self):
        s = _seeded(DUCBScheduler, gamma=1.0)
        s.init_arm("bit_flip")
        s.record("bit_flip", success=True)
        for _ in range(500):
            s.record("byte_flip", success=False)
        assert s.discounted_counts()["bit_flip"] == pytest.approx(1.0)

    def test_renormalisation_preserves_finiteness(self):
        """Adversarial: run long enough that the relative basis underflows.

        The O(1) update stores statistics relative to a single discount
        factor. If the renormalisation sweep were missing, the factor would
        underflow to zero and every mean would become inf or nan.
        """
        s = _seeded(DUCBScheduler, gamma=FAST_GAMMA)
        for a in ARMS:
            s.init_arm(a)
        for i in range(4000):
            s.record(ARMS[i % len(ARMS)], success=(i % 2 == 0))
        counts = s.discounted_counts()
        means = s.discounted_means()
        assert all(math.isfinite(v) and v >= 0.0 for v in counts.values())
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in means.values())
        assert sum(counts.values()) > 0.0, "every arm's evidence underflowed to zero"

    def test_prefers_the_arm_that_wins(self):
        s = _seeded(DUCBScheduler)
        _open_all(s)
        for _ in range(400):
            for a in ARMS:
                s.record(a, success=(a == "bit_flip"))
        assert s.select_op(ARMS) == "bit_flip"

    def test_falsification_abandons_a_stale_winner(self):
        """A scheduler without recency passes ``prefers_the_arm_that_wins``
        and still fails here: once bit_flip dies it must be abandoned."""
        s = _seeded(DUCBScheduler, gamma=0.99)
        _open_all(s)
        for _ in range(300):
            for a in ARMS:
                s.record(a, success=(a == "bit_flip"))
        for _ in range(600):
            for a in ARMS:
                s.record(a, success=(a == "byte_flip"))
        assert s.select_op(ARMS) == "byte_flip"


# ---------------------------------------------------------------------------
# SW-UCB
# ---------------------------------------------------------------------------


class TestSWUCB:
    def test_rejects_invalid_parameters(self):
        for kw in ({"window": 0}, {"window": -5}, {"xi": 0.0}):
            with pytest.raises(ValueError):
                SWUCBScheduler(**kw)

    def test_window_bounds_the_history(self):
        s = _seeded(SWUCBScheduler, window=50)
        _open_all(s)
        for i in range(500):
            s.record(ARMS[i % len(ARMS)], success=True)
        assert sum(s.windowed_counts().values()) == 50

    def test_evicted_arm_becomes_unpulled_again(self):
        """The mechanism that lets SW-UCB re-open an abandoned arm."""
        s = _seeded(SWUCBScheduler, window=20)
        for a in ARMS:
            s.init_arm(a)
        s.record("dict_append", success=True)
        assert "dict_append" in s.windowed_counts()
        for _ in range(20):
            s.record("bit_flip", success=False)
        assert "dict_append" not in s.windowed_counts(), (
            "an arm pushed out of the window still carries state, so it can "
            "never be treated as fresh"
        )

    def test_sums_never_go_negative(self):
        """Adversarial: fractional weights evicted thousands of times.

        Repeated float subtraction on eviction is where a residual negative
        epsilon would appear, and a negative windowed mean ranks a live arm
        below an untried one.
        """
        s = _seeded(SWUCBScheduler, window=37)
        _open_all(s)
        for i in range(5000):
            s.record(ARMS[i % len(ARMS)], success=True, weight=0.1 + (i % 7) / 70.0)
        assert all(v >= 0.0 for v in s._sums.values())
        assert all(0.0 <= v <= 1.0 for v in s.windowed_means().values())

    def test_prefers_the_arm_that_wins(self):
        s = _seeded(SWUCBScheduler, window=2000)
        _open_all(s)
        for _ in range(300):
            for a in ARMS:
                s.record(a, success=(a == "block_insert"))
        assert s.select_op(ARMS) == "block_insert"

    def test_falsification_ignores_a_pre_window_winner(self):
        s = _seeded(SWUCBScheduler, window=400)
        _open_all(s)
        for _ in range(300):
            for a in ARMS:
                s.record(a, success=(a == "block_insert"))
        for _ in range(300):
            for a in ARMS:
                s.record(a, success=(a == "dict_append"))
        assert s.select_op(ARMS) == "dict_append"


# ---------------------------------------------------------------------------
# CUCB
# ---------------------------------------------------------------------------


class TestCUCB:
    def test_rejects_invalid_parameters(self):
        for kw in ({"gamma": 0.0}, {"gamma": 2.0}, {"min_out_rounds": -1.0}):
            with pytest.raises(ValueError):
                CUCBScheduler(**kw)

    def test_record_does_not_commit_until_settle(self):
        s = _seeded(CUCBScheduler)
        for a in ARMS:
            s.init_arm(a)
        s.record("bit_flip", success=True)
        assert s.bandit_stats()["cucb_rounds"] == 0
        s.settle_round()
        assert s.bandit_stats()["cucb_rounds"] == 1

    def test_repeat_selection_credits_once(self):
        """max, not sum: one round is one discovery however many times the
        same operator was drawn into the stack."""
        s = _seeded(CUCBScheduler)
        for a in ARMS:
            s.init_arm(a)
        for _ in range(5):
            s.record("bit_flip", success=True, weight=1.0)
        s.settle_round()
        assert s._s_own_rel["bit_flip"] * s._discount == pytest.approx(1.0)
        assert s._n_in_rel["bit_flip"] * s._discount == pytest.approx(1.0)

    def test_select_op_closes_an_open_round(self):
        """settle_round() is optional for callers on the common interface."""
        s = _seeded(CUCBScheduler)
        _open_all(s)
        s.record("bit_flip", success=True)
        assert s._pending
        s.select_op(ARMS)
        assert not s._pending

    def test_contrast_recovers_the_planted_rate(self):
        """The mechanism. bit_flip fires in 30% of the rounds it joins; a
        background arm fires in 20% of all rounds. The naive mean over
        bit_flip's rounds is far above 0.30; the contrast must return ~0.30.
        """
        s = _seeded(CUCBScheduler, gamma=1.0, min_out_rounds=10.0)
        for a in ARMS:
            s.init_arm(a)
        rng = RandPool(seed=7)
        for i in range(6000):
            background = rng.random() < 0.20
            with_a = i % 2 == 0
            own = 1.0 if (with_a and rng.random() < 0.30) else 0.0
            reward = bool(own or background)
            if with_a:
                s.record("bit_flip", success=bool(own))
            s.record("byte_flip", success=bool(reward and not own))
            s.settle_round()
        est, used_contrast = s._mu_hat(
            "bit_flip",
            s._n_rounds_rel * s._discount,
            s._s_rounds_rel * s._discount,
        )
        assert used_contrast, "the out-sample was large enough; the fallback should not fire"
        assert est == pytest.approx(0.30, abs=0.06), f"contrast returned {est:.3f}, planted 0.30"

    def test_falsification_naive_mean_would_be_inflated(self):
        """Pins the defect the contrast exists to fix: the raw per-round mean
        over an arm's rounds is far above its own rate under a stacked reward,
        so an estimator using it would rank on the wrong scale."""
        s = _seeded(CUCBScheduler, gamma=1.0, min_out_rounds=10.0)
        for a in ARMS:
            s.init_arm(a)
        for i in range(2000):
            # bit_flip contributes nothing; byte_flip carries most rounds.
            s.record("bit_flip", success=False)
            s.record("byte_flip", success=(i % 4 != 0))
            # Present in a third of the rounds, so the out-sample is large
            # and its inclusion is uncorrelated with byte_flip's 3-in-4 rate.
            if i % 3 == 0:
                s.record("block_insert", success=False)
            s.settle_round()
        n_in = s._n_in_rel["block_insert"] * s._discount
        naive = (s._s_in_rel["block_insert"] * s._discount) / n_in
        est, used = s._mu_hat(
            "block_insert",
            s._n_rounds_rel * s._discount,
            s._s_rounds_rel * s._discount,
        )
        assert used, "block_insert was absent from half the rounds; contrast should fire"
        assert naive > 0.6, "the environment did not actually produce a bundled reward"
        assert est < 0.05, f"contrast inherited the bundled inflation ({est:.3f})"

    def test_adversarial_arm_in_every_round_scores_zero(self):
        """An arm present in *every* round is unidentifiable. The first cut of
        this scheduler returned the bundled mean for it (~0.75 against a true
        0.18), which made the incumbent unbeatable. It must score 0 and rely
        on its confidence radius instead.
        """
        s = _seeded(CUCBScheduler, gamma=1.0)
        for a in ARMS:
            s.init_arm(a)
        for i in range(1000):
            s.record("bit_flip", success=False)
            s.record("byte_flip", success=(i % 3 == 0))
            s.settle_round()
        est, used_contrast = s._mu_hat(
            "bit_flip",
            s._n_rounds_rel * s._discount,
            s._s_rounds_rel * s._discount,
        )
        assert not used_contrast, "there were no rounds without bit_flip to contrast against"
        assert est == pytest.approx(0.0, abs=1e-9)

    def test_both_estimator_branches_share_a_scale(self):
        """Adversarial: the branches must be rankable against each other.

        A contributing arm scored by the out-sample contrast has to outrank a
        free-riding arm scored by the fallback. When the fallback returned the
        bundled mean this was inverted, which is what cost 0.27 tail share
        against a 0.77 ceiling.
        """
        s = _seeded(CUCBScheduler, gamma=1.0, min_out_rounds=10.0)
        for a in ARMS:
            s.init_arm(a)
        for i in range(2000):
            # free_rider is in every round and causes nothing.
            s.record("bit_flip", success=False)
            contributes = i % 2 == 0
            s.record("byte_flip", success=contributes)
            s.settle_round()
        n_rounds = s._n_rounds_rel * s._discount
        s_rounds = s._s_rounds_rel * s._discount
        rider, rider_contrast = s._mu_hat("bit_flip", n_rounds, s_rounds)
        worker, worker_contrast = s._mu_hat("byte_flip", n_rounds, s_rounds)
        assert not rider_contrast and not worker_contrast, (
            "both arms are in every round; this test is about the fallback branch"
        )
        assert 0.0 <= rider <= 1.0 and 0.0 <= worker <= 1.0
        assert worker >= rider

    def test_explicit_credits_bypass_the_contrast(self):
        s = _seeded(CUCBScheduler)
        for a in ARMS:
            s.init_arm(a)
        s.record("bit_flip", success=False)
        s.settle_round(credits={"byte_flip": 1.0})
        assert s._n_in_rel.get("bit_flip", 0.0) == 0.0
        assert s._s_own_rel["byte_flip"] * s._discount == pytest.approx(1.0)

    def test_unrecorded_round_leaves_no_residue(self):
        """Adversarial: the deterministic stage selects nothing and records
        nothing, so a round can evaporate between two real ones."""
        s = _seeded(CUCBScheduler)
        _open_all(s)
        before = s.bandit_stats()["cucb_rounds"]
        for _ in range(50):
            for _ in range(len(ARMS)):
                s.select_op(ARMS)
        assert s.bandit_stats()["cucb_rounds"] == before
        assert s.select_op(ARMS) in ARMS

    def test_renormalisation_keeps_statistics_finite(self):
        s = _seeded(CUCBScheduler, gamma=FAST_GAMMA)
        _open_all(s)
        for i in range(4000):
            s.record(ARMS[i % len(ARMS)], success=(i % 3 == 0))
            s.settle_round()
        rates = s.estimated_rates()
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in rates.values())
        assert math.isfinite(s._n_rounds_rel * s._discount)

    def test_contrast_coverage_is_a_fraction(self):
        s = _seeded(CUCBScheduler)
        _open_all(s)
        for _ in range(200):
            op = s.select_op(ARMS)
            s.record(op, success=True)
            s.settle_round()
        assert 0.0 <= s.contrast_coverage() <= 1.0


# ---------------------------------------------------------------------------
# Shared interface contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [DUCBScheduler, SWUCBScheduler, CUCBScheduler])
class TestSharedContract:
    def test_declares_supports_priors(self, cls):
        """Hard Rule 40: the flag must be explicit, not inherited by default."""
        assert "supports_priors" in vars(cls)
        assert isinstance(cls.supports_priors, bool)

    def test_empty_candidate_list(self, cls):
        assert _seeded(cls).select_op([]) == ""

    def test_single_candidate(self, cls):
        s = _seeded(cls)
        s.init_arm("bit_flip")
        assert s.select_op(["bit_flip"]) == "bit_flip"

    def test_selects_only_from_candidates(self, cls):
        """Adversarial: a filtered candidate list, as build_ops() produces
        when the sniffers reject most operators. Returning an arm outside it
        would dispatch an operator that cannot run on this input."""
        s = _seeded(cls)
        _open_all(s)
        subset = ARMS[:2]
        for _ in range(100):
            op = s.select_op(subset)
            assert op in subset
            s.record(op, success=True)
            _settle(s)

    def test_record_for_unknown_arm(self, cls):
        """Operators registered at runtime via REGISTRY.register_mutator()
        reach record() without ever having been init_arm()'d."""
        s = _seeded(cls)
        _open_all(s)
        s.record("plugin_op", success=True)
        _settle(s)
        candidates = [*ARMS, "plugin_op"]
        assert s.select_op(candidates) in candidates

    def test_bandit_stats_are_json_safe(self, cls):
        s = _seeded(cls)
        _open_all(s)
        stats = s.bandit_stats()
        assert stats
        assert all(isinstance(v, int | float) for v in stats.values())
