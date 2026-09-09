"""Falsification and adversarial coverage for C2UCBScheduler.

Hard Rule 23: every new capability ships one falsification test and one
adversarial test. This file additionally pins the documented context-
dilution limitation (see c2ucb.py's module docstring) as an explicit
regression: a future change to the credit arithmetic that accidentally
"fixes" or further breaks that gap should be visible here, not discovered
in production.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.c2ucb import C2UCBScheduler
from fuzzer_tool.core.schedulers.cucb import CUCBScheduler

ARMS = ["bit_flip", "byte_flip"]


def _seeded(**kw) -> C2UCBScheduler:
    return C2UCBScheduler(dim=2, **kw)


class TestSharedContract:
    """Mirrors TestSharedContract in test_recency_and_combinatorial_schedulers.py."""

    def test_declares_supports_priors(self):
        assert C2UCBScheduler.supports_priors is False

    def test_empty_candidate_list(self):
        s = _seeded()
        assert s.select_op([], [1.0, 0.0]) == ""

    def test_single_candidate(self):
        s = _seeded()
        assert s.select_op(["only"], [1.0, 0.0]) == "only"

    def test_record_for_unknown_arm(self):
        """record()/settle_round() must not require a prior init_arm() call."""
        s = _seeded()
        s.record("never_initialized", [1.0, 0.0], True)
        s.settle_round()  # must not raise
        assert s.bandit_stats()["c2ucb_arms"] == 1

    def test_bandit_stats_are_json_safe(self):
        import json

        s = _seeded()
        s.init_arm("a")
        s.record("a", [1.0, 0.0], True)
        s.settle_round()
        json.dumps(s.bandit_stats())  # must not raise


class TestRoundMechanics:
    """CUCBScheduler's own round-boundary contract, reused verbatim here."""

    def test_record_does_not_commit_until_settle(self):
        s = _seeded()
        s.init_arm("a")
        s.record("a", [1.0, 0.0], True)
        assert s.bandit_stats()["c2ucb_linucb_pulls"] == 0
        s.settle_round()
        assert s.bandit_stats()["c2ucb_linucb_pulls"] == 1

    def test_repeat_selection_credits_once(self):
        """Duplicate record() calls in one round keep the higher reward."""
        s = _seeded()
        s.init_arm("a")
        s.record("a", [1.0, 0.0], False)
        s.record("a", [1.0, 0.0], True)  # same arm, higher reward, same round
        s.settle_round()
        assert s._n_in["a"] == 1.0, "one round with two record() calls is one pull, not two"
        assert s._s_in["a"] == 1.0, "the higher of the two rewards must win"

    def test_select_op_closes_an_open_round(self):
        """settle_round() is optional for callers on the common interface."""
        s = _seeded()
        s.init_arm("a")
        s.init_arm("b")
        s.record("a", [1.0, 0.0], True)
        s.select_op(["a", "b"], [1.0, 0.0])  # must close the round for "a"
        assert s.bandit_stats()["c2ucb_rounds"] == 1

    def test_unrecorded_round_leaves_no_residue(self):
        s = _seeded()
        s.init_arm("a")
        s.settle_round()  # nothing pending
        assert s.bandit_stats()["c2ucb_rounds"] == 0


class TestCreditArithmeticMatchesCUCB:
    """The no-attribution path must be CUCB's own arithmetic, not a variant.

    Composition, not reimplementation, is the entire design premise of this
    module (see its docstring) -- if these ever diverge with context held
    fixed, the composition itself is broken, independent of anything about
    context.
    """

    def test_matches_cucb_when_context_is_constant(self):
        cucb = CUCBScheduler(gamma=1.0, min_out_rounds=10.0, exploration=0.15)
        c2 = _seeded(min_out_rounds=10.0)
        rng = RandPool(seed=7)
        x = [1.0, 0.0]
        for a in ARMS:
            cucb.init_arm(a)
            c2.init_arm(a)

        for _i in range(4000):
            a_in = rng.random() < 0.6
            b_in = rng.random() < 0.6
            a_ok = a_in and rng.random() < 0.4
            b_ok = b_in and rng.random() < 0.1
            if a_in:
                cucb.record("bit_flip", success=bool(a_ok))
                c2.record("bit_flip", x, bool(a_ok))
            if b_in:
                cucb.record("byte_flip", success=bool(b_ok))
                c2.record("byte_flip", x, bool(b_ok))
            cucb.settle_round()
            c2.settle_round()

        n_rounds_cucb = cucb._n_rounds_rel * cucb._discount
        s_rounds_cucb = cucb._s_rounds_rel * cucb._discount
        n_rounds_c2 = c2._n_rounds
        s_rounds_c2 = c2._s_rounds

        for arm in ARMS:
            cucb_est, cucb_used = cucb._mu_hat(arm, n_rounds_cucb, s_rounds_cucb)
            c2_est, c2_used = c2._credit(arm, n_rounds_c2, s_rounds_c2)
            assert cucb_used == c2_used, f"{arm}: contrast-vs-fallback branch differs"
            assert c2_est == pytest.approx(cucb_est), (
                f"{arm}: C2UCB credit {c2_est} != CUCB credit {cucb_est} "
                "with identical round history and constant context"
            )


class TestExplicitCredits:
    def test_explicit_credits_bypass_the_contrast(self):
        s = _seeded()
        s.init_arm("byte_flip")
        s.record("byte_flip", [1.0, 0.0], False)  # own outcome says failure
        s.settle_round(credits={"byte_flip": 1.0})  # true attribution says success
        # The regressor must have been trained on 1.0, not on a contrast
        # estimate derived from the recorded False.
        assert s._linucb.score("byte_flip", [1.0, 0.0]) > 0.5

    def test_credits_only_apply_to_named_arms(self):
        """An arm present in the round but absent from credits still falls
        back to the contrast, rather than silently getting no update."""
        s = _seeded(min_out_rounds=1.0)
        s.init_arm("a")
        s.init_arm("b")
        s.record("a", [1.0, 0.0], True)
        s.record("b", [1.0, 0.0], True)
        s.settle_round(credits={"a": 1.0})  # "b" not named
        assert s._n_in["b"] == 1.0, "b should still have been counted into round stats"


class TestContextDependence:
    """The module's central claim, and its documented limit.

    Environment: two contexts, two arms, arm quality flips by context
    (rate 0.40 for the context-correct arm, 0.05 for the other). "Correct
    arm for context" accuracy is the metric in both directions.
    """

    ENV_ROUNDS = 3000
    CTX_A = [1.0, 0.0]
    CTX_B = [0.0, 1.0]

    def _run_with_attribution(self, seed: int) -> float:
        s = C2UCBScheduler(dim=2, alpha=1.0, lambda_reg=1.0, min_out_rounds=10.0)
        rng = RandPool(seed=seed)
        correct = 0
        for _ in range(self.ENV_ROUNDS):
            if rng.random() < 0.5:
                ctx, good = self.CTX_A, "x"
            else:
                ctx, good = self.CTX_B, "y"
            chosen = s.select_op(["x", "y"], ctx)
            if chosen == good:
                correct += 1
            ok = rng.random() < (0.8 if chosen == good else 0.1)
            s.record(chosen, ctx, ok)
            s.settle_round(credits={chosen: 1.0 if ok else 0.0})
        return correct / self.ENV_ROUNDS

    def _run_without_attribution(self, seed: int) -> float:
        s = C2UCBScheduler(dim=2, alpha=1.0, lambda_reg=1.0, min_out_rounds=10.0)
        rng = RandPool(seed=seed)
        correct = 0
        for _ in range(self.ENV_ROUNDS):
            if rng.random() < 0.5:
                ctx, good = self.CTX_A, "x"
            else:
                ctx, good = self.CTX_B, "y"
            chosen = s.select_op(["x", "y"], ctx)
            if chosen == good:
                correct += 1
            ok = rng.random() < (0.8 if chosen == good else 0.1)
            s.record(chosen, ctx, ok)
            s.settle_round()  # no credits: forced through the global contrast
        return correct / self.ENV_ROUNDS

    def test_adversarial_context_dependent_arm_credits_bypass_the_contrast(self):
        """The falsification test: a scheduler with no learning mechanism
        (or one that ignored context) would score ~0.5 here regardless of
        how long it ran. Scoring far above that, and specifically climbing
        with attribution supplied, pins that the regressor is genuinely
        conditioning its choice on context.
        """
        acc = self._run_with_attribution(seed=1)
        assert acc >= 0.90, (
            f"context-conditioned accuracy only {acc:.3f} with true per-arm "
            "attribution supplied; expected near-ceiling performance"
        )

    def test_falsification_context_blind_contrast_dilutes_signal(self):
        """Pins the documented limitation (see c2ucb.py's module docstring,
        'Context dilution'): without attribution, the global inclusion
        contrast hands the regressor one blended scalar per arm, largely
        independent of which context was actually active. Accuracy must
        stay near chance -- if this test starts failing because accuracy
        rose, the credit arithmetic changed in a way the docstring's
        numbers no longer describe and both need re-checking together.
        """
        acc = self._run_without_attribution(seed=1)
        assert acc <= 0.65, (
            f"context-blind contrast scored {acc:.3f}, above the documented "
            "near-chance ceiling -- the 'Context dilution' section of "
            "c2ucb.py's docstring is now inaccurate and needs updating "
            "alongside whatever changed this"
        )

    def test_attribution_beats_no_attribution_on_the_same_environment(self):
        """Direct comparison, same seed, isolating the one variable that
        changed: whether settle_round() received true credits."""
        with_credits = self._run_with_attribution(seed=42)
        without_credits = self._run_without_attribution(seed=42)
        assert with_credits - without_credits >= 0.25, (
            f"attribution ({with_credits:.3f}) did not meaningfully beat "
            f"no attribution ({without_credits:.3f}) on an identical "
            "context-dependent environment"
        )


class TestConstructorValidation:
    def test_rejects_negative_min_out_rounds(self):
        with pytest.raises(ValueError):
            C2UCBScheduler(dim=2, min_out_rounds=-1.0)

    def test_accepts_zero_min_out_rounds(self):
        C2UCBScheduler(dim=2, min_out_rounds=0.0)  # must not raise


class TestDiagnostics:
    def test_contrast_coverage_is_a_fraction(self):
        s = _seeded(min_out_rounds=5.0)
        s.init_arm("a")
        s.init_arm("b")
        rng = RandPool(seed=3)
        for _i in range(200):
            a_in = rng.random() < 0.6
            b_in = rng.random() < 0.6
            if a_in:
                s.record("a", [1.0, 0.0], rng.random() < 0.3)
            if b_in:
                s.record("b", [1.0, 0.0], rng.random() < 0.3)
            s.settle_round()
        cov = s.contrast_coverage()
        assert 0.0 <= cov <= 1.0

    def test_linucb_pulls_match_settled_rounds_arm_count(self):
        s = _seeded()
        s.init_arm("a")
        s.init_arm("b")
        s.record("a", [1.0, 0.0], True)
        s.record("b", [1.0, 0.0], False)
        s.settle_round()
        stats = s.bandit_stats()
        assert stats["c2ucb_linucb_pulls"] == 2
        assert stats["c2ucb_rounds"] == 1
        assert stats["c2ucb_dim"] == 2
