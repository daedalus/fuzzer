"""Convergence tests for the operator-selection schedulers.

Ten bandits live in ``core/schedulers/``. Before this module, nothing asserted
that any of them *converges*: the existing tests pin the import graph
(``test_regression_scheduler_independence``), fallback ordering
(``test_regression_scheduler_fallback_precedence``), and per-scheduler
structural properties. All of those pass for a scheduler that selects
uniformly at random forever -- and, as it turns out, two of them do.

Method, from TigerBeetle's ARR fuzzer (matklad, *A Tale Of Four Fuzzers*):
build a deliberately idealized environment where the best arm is known by
construction, so the acceptance criterion can be *strict* rather than merely
"doesn't crash". See ``tests/support/bandit_env.py``.

Seeding follows the same post's two-run pattern. Statistics are asserted
against a fixed seed (92) so the thresholds mean something and cannot flake;
a second run on a genuinely random per-session seed accumulates state-space
coverage over the suite's lifetime but asserts only a weak invariant, because
a strict assertion on a random seed is a debugging session five years from
now. Reproduce a random-seed failure with the ``--fuzz-seed`` printed in the
pytest session header.

WHAT THIS FOUND
---------------
Four defects, all reproducible, none of which any existing test notices. See
``docs/learnings/2026-08-21-scheduler-convergence.md`` for the full write-up.

1. ``MOptScheduler`` never converges -- indistinguishable from uniform random
   selection (tail share 0.080 vs. a 0.083 uniform baseline; regret slope
   1.02, i.e. linear). ``_normalize_to_simplex`` softmaxes a vector that is
   already a probability distribution, compressing every particle to within
   ``exp(max-min) <= exp(1)`` of uniform. PSO cannot concentrate.

2. ``CMAESScheduler`` diverges numerically and then locks onto an arbitrary
   arm (tail share 0.005 -- *worse* than uniform). σ grows from 0.3 to ~72
   and the mean vector reaches ~1e14, at which point the softmax is a hard
   one-hot on whichever logit happened to be largest.

3. ``GPUCBScheduler`` starves the best arm on ~42% of seeds. Once an arm has
   ``min_samples`` observations that are all zero, its score is
   ``mean + beta*stddev == 0 + 2e-6`` forever: there is no count-based
   exploration bonus, so a UCB algorithm treats an under-sampled arm as a
   *certain* one. Separately, ``select_op`` never calls the RBF kernel the
   class docstring is about.

4. ``HierarchicalBanditScheduler`` starves the best arm's whole *category* at
   the top level on ~1.5% of seeds.

Rather than delete the broken schedulers or paper over the flaky ones, the
failure *rates* are pinned as assertions with documented bands. A fix will
turn those tests red and say so.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import (
    CMAESScheduler,
    ContextualLinUCBScheduler,
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    GPUCBScheduler,
    HierarchicalBanditScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
)
from tests.support.bandit_env import (
    DecayingBest,
    StationaryBernoulli,
    adapt,
    run,
    uniform_baseline,
)

#: matklad's fixed seed, kept literally: statistics are asserted here.
FIXED_SEED = 92

#: Enough rounds for the reliable schedulers to converge, few enough to keep
#: the default suite under a couple of seconds. The 20k-round variants are
#: marked slow.
ROUNDS = 6_000


# ---------------------------------------------------------------------------
# Reliable schedulers: strict convergence
# ---------------------------------------------------------------------------

#: (factory, min tail share on the best arm, max regret log-log slope).
#: Thresholds sit below the observed minimum over 100 seeds at ROUNDS, with
#: margin -- measured, not guessed. Observed minima:
#:   Contextual 0.968 | EpsGreedy 0.920 | Exp3 0.850
#:   MonteCarlo 0.971 | Hierarchical 0.986 (median; see known-defects below)
RELIABLE = {
    "ContextualLinUCB": (lambda seed: ContextualLinUCBScheduler(dim=4), 0.90, 0.40),
    "EpsilonGreedy": (lambda seed: EpsilonGreedyScheduler(), 0.85, 0.45),
    "Exp3": (lambda seed: Exp3Scheduler(), 0.78, 0.70),
    "MonteCarlo": (lambda seed: MonteCarloScheduler(), 0.90, 0.45),
}


@pytest.mark.parametrize("name", sorted(RELIABLE))
def test_converges_to_best_arm(name):
    """The best arm dominates the tail of the campaign at the fixed seed."""
    factory, min_share, max_slope = RELIABLE[name]
    env = StationaryBernoulli.build()
    c = run(factory(FIXED_SEED), env, seed=FIXED_SEED, rounds=ROUNDS)

    share = c.tail_share(env.best)
    assert share >= min_share, (
        f"{name} spent only {share:.3f} of the campaign tail on {env.best!r} "
        f"(p={env.probs[env.best]}); uniform selection would give "
        f"{uniform_baseline(env, ROUNDS):.3f}"
    )


@pytest.mark.parametrize("name", sorted(RELIABLE))
def test_regret_is_sublinear(name):
    """Cumulative regret grows slower than linearly -- the bandit property.

    A scheduler that never learns pulls a constant-suboptimality arm forever
    and gives slope 1.0. Anything meaningfully below that is learning.
    """
    factory, _, max_slope = RELIABLE[name]
    env = StationaryBernoulli.build()
    c = run(factory(FIXED_SEED), env, seed=FIXED_SEED, rounds=ROUNDS)

    slope = c.regret_slope()
    assert slope < max_slope, f"{name} regret slope {slope:.2f} is not sublinear enough"


@pytest.mark.parametrize("name", sorted(RELIABLE))
def test_converges_on_random_seed(name, random_seed):
    """Weak invariant on a genuinely random seed -- coverage, not statistics.

    Deliberately *not* the strict threshold above. A strict assertion driven
    by a random seed is a one-in-a-thousand CI failure waiting to happen; the
    point of this run is to walk parts of the state space the fixed seed never
    reaches. Beating uniform selection by 3x is a floor no working bandit can
    plausibly miss.
    """
    factory, _, _ = RELIABLE[name]
    env = StationaryBernoulli.build()
    c = run(factory(random_seed), env, seed=random_seed, rounds=ROUNDS)

    floor = 3.0 * uniform_baseline(env, ROUNDS)
    assert c.tail_share(env.best) >= floor, (
        f"{name} failed to beat 3x uniform on seed 0x{random_seed:x}"
    )


# ---------------------------------------------------------------------------
# Known defects, pinned rather than hidden
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MOptScheduler does not converge: _normalize_to_simplex softmaxes an "
        "already-normalized distribution, so no particle can concentrate. "
        "Fix: project onto the simplex by clipping and dividing by the sum, "
        "or keep particle positions as unconstrained logits and softmax only "
        "at sampling time (as CMAESScheduler does)."
    ),
)
def test_mopt_converges():
    env = StationaryBernoulli.build()
    c = run(MOptScheduler(), env, seed=FIXED_SEED, rounds=ROUNDS)
    assert c.tail_share(env.best) >= 0.50


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CMAESScheduler diverges: sigma has no upper clamp and the CSA update "
        "is a positive feedback loop here, so sigma reaches ~72 and the mean "
        "reaches ~1e14, hard-one-hotting the softmax onto an arbitrary arm. "
        "Contributing: _update_cmaes zeroes _eval_count before using it in the "
        "hsigma damping heuristic, so the heuristic is evaluated at a constant "
        "and never damps."
    ),
)
def test_cmaes_converges():
    env = StationaryBernoulli.build()
    c = run(CMAESScheduler(rng=RandPool(FIXED_SEED)), env, seed=FIXED_SEED, rounds=ROUNDS)
    assert c.tail_share(env.best) >= 0.50


def test_mopt_is_indistinguishable_from_uniform():
    """Pin *how* MOpt fails, so a partial fix is still visible as progress."""
    env = StationaryBernoulli.build()
    c = run(MOptScheduler(), env, seed=FIXED_SEED, rounds=20_000)
    baseline = uniform_baseline(env, 20_000)

    assert abs(c.tail_share(env.best) - baseline) < 0.02, (
        "MOpt no longer matches uniform selection -- if this is a fix, update "
        "the xfail on test_mopt_converges and delete this test"
    )
    assert c.regret_slope() > 0.95, "MOpt regret is no longer linear"


def test_replicator_identifies_best_arm_but_concentrates_weakly():
    """Replicator is slow, not broken -- assert what it actually guarantees.

    It reaches the correct ``dominant_operator()`` and beats uniform by ~5x,
    but plateaus near 0.42 population share rather than concentrating. That is
    a defensible design point (the mutation_rate floor guarantees exploration),
    so it is asserted as-is rather than xfailed. It needs the full 20k rounds:
    with window_size=200 it only gets 30 replicator updates in 6k.
    """
    env = StationaryBernoulli.build()
    sched = ReplicatorScheduler()
    c = run(sched, env, seed=FIXED_SEED, rounds=20_000)

    assert sched.dominant_operator() == env.best
    assert sched.is_converged()
    assert c.tail_share(env.best) >= 4.0 * uniform_baseline(env, 20_000)


@pytest.mark.slow
def test_gpucb_starvation_rate_is_pinned():
    """GP-UCB abandons the best arm on ~42% of seeds. Pin the rate.

    Root cause: in ``select_op``, an arm past ``min_samples`` scores
    ``mean + beta * max(stddev, 1e-6)``. An arm whose observations are all
    zero has ``mean == 0`` *and* ``stddev == 0``, so it scores 2e-6 forever --
    a UCB algorithm treating an under-sampled arm as a certain one. With
    p_best=0.3 and min_samples=3 the best arm draws three zeros with
    probability 0.7**3 = 0.34, which is the floor of the observed rate.

    A fix must add a count-based bonus (``beta * sqrt(log(t)/n)``), which is
    what the missing kernel machinery was presumably meant to supply --
    ``select_op`` never calls ``_rbf`` or ``_kernel_row`` at all, so the class
    is not GP-UCB, it is empirical-stddev UCB.
    """
    env = StationaryBernoulli.build()
    failures = 0
    for seed in range(1, 61):
        c = run(GPUCBScheduler(), env, seed=seed, rounds=ROUNDS)
        if c.tail_share(env.best) < 0.5:
            failures += 1

    assert 15 <= failures <= 40, (
        f"GP-UCB starved the best arm on {failures}/60 seeds; the documented "
        "band is 15-40 (observed 42% over 100 seeds). Outside it means the "
        "behaviour changed -- if this is the count-based-bonus fix, update or "
        "delete this test and add GPUCB to RELIABLE."
    )


@pytest.mark.slow
def test_hierarchical_category_starvation_rate_is_pinned():
    """The top-level category posterior collapses before the best arm is found.

    Rarer than GP-UCB's failure (~1.5% of seeds) but the same shape one level
    up: Thompson sampling over categories can abandon the category containing
    the best operator, after which the bottom level never sees it. 1.5% is
    well inside the range that produces intermittent CI failures, which is why
    the strict threshold above runs only at the fixed seed.
    """
    env = StationaryBernoulli.build()
    failures = sum(
        run(HierarchicalBanditScheduler(), env, seed=seed, rounds=ROUNDS).tail_share(env.best) < 0.5
        for seed in range(1, 201)
    )
    assert failures <= 8, (
        f"Hierarchical starved the best category on {failures}/200 seeds "
        "(documented: ~3). A jump means the top-level posterior got more brittle."
    )


# ---------------------------------------------------------------------------
# Non-stationary: the regime fuzzing actually runs in
# ---------------------------------------------------------------------------

#: Measured tail share on the *new* best arm after the old best decays, at
#: FIXED_SEED over 20k rounds with the switch at 10k. Only two schedulers
#: recover at all. Asserted as a floor with margin, because "which schedulers
#: survive coverage saturation" is the property that matters in a real
#: campaign and it should not silently regress.
RECOVERS = {
    "ContextualLinUCB": (lambda: ContextualLinUCBScheduler(dim=4), 0.35),
    "Exp3": (Exp3Scheduler, 0.30),
}

#: Schedulers that stay locked on the dead arm for the whole second half.
#: This is a real limitation, not a test bug: EpsilonGreedy uses uniform
#: sample averages with no recency weighting, and GP-UCB has no exploration
#: bonus to re-open an abandoned arm.
STUCK = {
    "EpsilonGreedy": EpsilonGreedyScheduler,
    "GPUCB": GPUCBScheduler,
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(RECOVERS))
def test_recovers_after_best_arm_decays(name):
    factory, floor = RECOVERS[name]
    env = DecayingBest.build(switch_at=10_000)
    c = run(factory(), env, seed=FIXED_SEED, rounds=20_000)

    assert c.tail_share(env.best_late) >= floor, (
        f"{name} failed to re-converge on {env.best_late!r} after {env.best_early!r} decayed"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(STUCK))
def test_stays_stuck_on_decayed_arm(name):
    """Pin the known-bad behaviour so a fix shows up as a failure here.

    Not an xfail: the assertion is that these schedulers *do* get stuck, which
    is information a campaign operator needs when choosing a scheduler.
    """
    env = DecayingBest.build(switch_at=10_000)
    c = run(STUCK[name](), env, seed=FIXED_SEED, rounds=20_000)

    assert c.tail_share(env.best_early) > 0.80, (
        f"{name} now escapes the decayed arm -- good news; move it to RECOVERS"
    )


# ---------------------------------------------------------------------------
# Harness self-checks
# ---------------------------------------------------------------------------


class TestHarnessCoverage:
    """The harness must be shown to exercise what it claims to.

    matklad's warning: a negative-space test whose generator never produces a
    valid case passes happily while testing nothing. The same applies to a
    bandit environment that never rewards anything, or one whose "best" arm is
    reachable by luck.
    """

    def test_both_outcomes_occur(self):
        env = StationaryBernoulli.build()
        c = run(EpsilonGreedyScheduler(), env, seed=FIXED_SEED, rounds=ROUNDS)
        assert c.successes > 100, "environment never rewarded anything"
        assert c.failures > 100, "environment never withheld a reward"

    def test_uniform_selection_does_not_pass(self):
        """A feedback-ignoring scheduler must fail the strict threshold.

        Without this, the thresholds could be satisfied by chance and the
        whole module would prove nothing.
        """

        class Ignorant:
            def init_arm(self, name):
                pass

            def select_op(self, ops):
                return random.choice(ops)

            def record(self, name, success, weight=1.0):
                pass

        env = StationaryBernoulli.build()
        c = run(Ignorant(), env, seed=FIXED_SEED, rounds=ROUNDS)

        assert c.tail_share(env.best) < 0.20
        assert c.regret_slope() > 0.95
        for _, min_share, _ in RELIABLE.values():
            assert c.tail_share(env.best) < min_share

    def test_arms_are_real_operator_names(self):
        """Synthetic names would silently disable two schedulers.

        Hierarchical ignores ``init_arm``/``record`` for names outside
        ``OPERATOR_CATEGORIES``, and GP-UCB builds its features from the same
        table. A harness using ``"op0".."op11"`` would still pass while
        testing a scheduler that had received no data at all.
        """
        from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES

        known = {op for ops in OPERATOR_CATEGORIES.values() for op in ops}
        env = StationaryBernoulli.build()
        assert set(env.arms) <= known
        assert len({HierarchicalBanditScheduler()._op_to_cat[a] for a in env.arms}) >= 3

    def test_best_arm_shares_a_category_with_weak_arms(self):
        """Otherwise the hierarchy hands Hierarchical a free win at level one."""
        h = HierarchicalBanditScheduler()
        env = StationaryBernoulli.build()
        best_cat = h._op_to_cat[env.best]
        siblings = [a for a in env.arms if a != env.best and h._op_to_cat[a] == best_cat]
        assert siblings, "best arm is alone in its category"
        assert all(env.probs[s] < env.probs[env.best] for s in siblings)

    def test_every_exported_operator_scheduler_is_adaptable(self):
        """Adding a scheduler must not silently skip convergence testing.

        MCTSSeedScheduler is excluded by name: it schedules *seeds* against a
        LineageTree, not operators, and needs its own environment.
        """
        import fuzzer_tool.core.schedulers as pkg

        env = StationaryBernoulli.build()
        for cls_name in pkg.__all__:
            if cls_name == "MCTSSeedScheduler":
                continue
            cls = getattr(pkg, cls_name)
            sched = cls(dim=4) if cls_name == "ContextualLinUCBScheduler" else cls()
            a = adapt(sched, env.arms)
            op = a.select()
            assert op in env.arms, f"{cls_name} returned {op!r}"
            a.update(op, True)


class TestSchedulerSeedability:
    """Reproducibility properties the ``--seed`` CLI flag implicitly promises."""

    def test_cmaes_is_not_reproducible_without_an_explicit_rng(self):
        """``services/fuzzer.py`` constructs CMAESScheduler with no ``rng``.

        ``CMAESScheduler.__init__`` falls back to ``RandPool()``, which seeds
        from OS entropy, so ``--seed`` does not determine CMA-ES behaviour and
        a crash found under CMA-ES scheduling cannot be replayed. Every other
        scheduler draws from the module-level ``random`` and is therefore
        covered by a global ``random.seed()``.

        Fix: thread the campaign seed through at the construction site
        (``services/fuzzer.py``, the ``CMAESScheduler(...)`` call).
        """
        env = StationaryBernoulli.build()
        a = run(CMAESScheduler(), env, seed=FIXED_SEED, rounds=400)
        b = run(CMAESScheduler(), env, seed=FIXED_SEED, rounds=400)
        assert a.picks != b.picks, (
            "CMAESScheduler is now reproducible from the global seed -- if the "
            "rng was wired up, delete this test and its counterpart below"
        )

    def test_cmaes_is_reproducible_with_an_explicit_rng(self):
        env = StationaryBernoulli.build()
        a = run(CMAESScheduler(rng=RandPool(7)), env, seed=FIXED_SEED, rounds=400)
        b = run(CMAESScheduler(rng=RandPool(7)), env, seed=FIXED_SEED, rounds=400)
        assert a.picks == b.picks

    @pytest.mark.parametrize(
        "factory",
        [
            EpsilonGreedyScheduler,
            Exp3Scheduler,
            MonteCarloScheduler,
            MOptScheduler,
            ReplicatorScheduler,
            HierarchicalBanditScheduler,
        ],
        ids=lambda f: f.__name__,
    )
    def test_reproducible_from_global_seed(self, factory):
        env = StationaryBernoulli.build()
        a = run(factory(), env, seed=FIXED_SEED, rounds=1_000)
        b = run(factory(), env, seed=FIXED_SEED, rounds=1_000)
        assert a.picks == b.picks
