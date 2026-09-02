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

WHAT THIS FOUND, AND WHAT WAS DONE ABOUT IT
-------------------------------------------
The harness found four defects on first contact, none of which any of the
4,767 existing tests noticed, because every one of those asks a structural
question. Three are now fixed and the schedulers are asserted here as working;
the fourth is fixed numerically but the algorithm remains a poor fit for the
problem and stays xfailed. Full write-up and measurements in
``docs/learnings/2026-08-21-scheduler-convergence.md``.

* ``MOptScheduler`` was indistinguishable from ``random.choice`` (tail share
  0.080 against a 0.083 uniform baseline). Five stacked bugs, each hidden by
  the one above it. Now identifies the best arm on 100/100 seeds. It allocates
  proportionally to measured efficiency rather than converging on the argmax,
  so it is asserted like ``ReplicatorScheduler``, not like a UCB bandit.

* ``GPUCBScheduler`` starved the best arm on 42% of seeds; its score had no
  count term, so an all-zero arm looked *certain* rather than *unsampled*.
  Now 0/100 with a proper ``sqrt(2 log t / n)`` width, and it recovers from
  arm decay instead of locking on forever.

* ``HierarchicalBanditScheduler`` starved the best arm's whole category on
  ~1.5% of seeds and could not leave a decayed arm. Capping the Beta
  pseudocount took starvation to ~0.25% and decay recovery from 0.23 to 0.99.

* ``CMAESScheduler`` diverged numerically (sigma 0.3 -> 72, mean -> 1e14) and
  committed to an arbitrary arm, scoring *worse* than uniform. Six separate
  numerical errors fixed; it no longer diverges and does converge given
  ~100k executions, but remains bimodal and sample-inefficient. Still
  xfailed, deliberately, rather than tuned until this file goes green.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import (
    CMAESScheduler,
    ContextualLinUCBScheduler,
    CUCBScheduler,
    DUCBScheduler,
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    GPUCBScheduler,
    HierarchicalBanditScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
    SWUCBScheduler,
)
from tests.support.bandit_env import (
    DecayingBest,
    StationaryBernoulli,
    adapt,
    run,
    uniform_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

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
#:   Contextual 0.968 | EpsGreedy 0.920 | Exp3 0.850 | GPUCB 0.694
#:   MonteCarlo 0.971 | Hierarchical 0.994 median (rare starvation: see below)
RELIABLE = {
    "ContextualLinUCB": (lambda seed: ContextualLinUCBScheduler(dim=4), 0.90, 0.40),
    "EpsilonGreedy": (lambda seed: EpsilonGreedyScheduler(), 0.85, 0.45),
    "Exp3": (lambda seed: Exp3Scheduler(), 0.78, 0.70),
    "GPUCB": (lambda seed: GPUCBScheduler(), 0.62, 0.75),
    "Hierarchical": (lambda seed: HierarchicalBanditScheduler(), 0.90, 0.40),
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


@pytest.mark.slow
def test_cmaes_convergence_rate_is_pinned():
    """CMA-ES converges on some seeds and not others. Pin the rate.

    Not an xfail: the outcome is bimodal, so a single-seed xfail would XPASS
    or fail depending on which seed it happened to draw. Counting is the
    honest form of the assertion -- the same reason the harness counts both
    outcomes of its own generator rather than trusting one run.

    The six numerical errors are fixed and it does reach ~0.95 on good seeds
    given ~100k executions, but CMA-ES is a continuous black-box optimizer
    being asked to optimize a 12-dimensional categorical distribution from
    Bernoulli feedback: each generation spends ``generation_size`` executions
    to buy one very noisy ranking of ``pop_size`` near-identical candidates.
    Left as-is rather than tuned until this file goes green -- the constants
    that would do it are not derived from anything.
    """
    env = StationaryBernoulli.build()
    converged = sum(
        run(CMAESScheduler(rng=RandPool(seed)), env, seed=seed, rounds=ROUNDS).tail_share(env.best)
        >= 0.5
        for seed in range(1, 41)
    )
    assert converged <= 12, (
        f"CMA-ES converged on {converged}/40 seeds, above the documented band. "
        "If this is a real improvement, tighten or replace this test and "
        "consider promoting CMAES to RELIABLE."
    )


def test_cmaes_no_longer_diverges():
    """The actual bug was divergence, and that part is fixed.

    Before: sigma ran from 0.3 to ~72 and the mean vector to ~1e14, at which
    point the softmax is a hard one-hot and the scheduler is committed to an
    arbitrary operator forever -- scoring *worse* than uniform selection
    because it was reliably wrong rather than merely uninformed.
    """
    env = StationaryBernoulli.build()
    sched = CMAESScheduler(rng=RandPool(FIXED_SEED))
    c = run(sched, env, seed=FIXED_SEED, rounds=20_000)

    assert sched.sigma_min <= sched._sigma <= sched.sigma_max
    assert max(abs(float(x)) for x in sched._mean) <= sched.logit_clip
    # No longer systematically worse than ignoring feedback entirely.
    assert c.tail_share(env.best) >= 0.5 * uniform_baseline(env, 20_000)


def test_mopt_identifies_the_best_arm():
    """MOpt allocates proportionally to efficiency; assert that, not argmax.

    The efficiency attractor is normalized reward rate, so the best arm's
    share is bounded near ``p_best / sum(p)`` ~ 0.31 by construction -- the
    same soft-allocation design point as ReplicatorScheduler. Asserting a
    UCB-style 0.9 here would be asserting that MOpt is a different algorithm.

    Before the fixes this was 0.080 against a 0.083 uniform baseline, with the
    most-selected arm being whichever operator ``init_arm`` happened to
    register first.
    """
    env = StationaryBernoulli.build()
    c = run(MOptScheduler(), env, seed=FIXED_SEED, rounds=ROUNDS)

    assert c.tail_picks.most_common(1)[0][0] == env.best
    assert c.tail_share(env.best) >= 3.0 * uniform_baseline(env, ROUNDS)


def test_mopt_swarm_is_not_degenerate():
    """Guard the three structural defects that made PSO a no-op.

    Each of these passed silently before, and each on its own is enough to
    reduce the scheduler to uniform sampling.
    """
    env = StationaryBernoulli.build()
    sched = MOptScheduler()
    run(sched, env, seed=FIXED_SEED, rounds=ROUNDS)

    positions = [tuple(p.pos) for p in sched.particles]
    assert len(set(positions)) > 1, "particles are identical -- PSO cannot generate a gradient"
    assert any(any(v != 0.0 for v in p.vel) for p in sched.particles), "swarm never moved"
    assert sum(1 for p in sched.particles if p.fitness > 0.0) >= 2, (
        "particle fitnesses collapsed to a single survivor"
    )


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
def test_gpucb_no_longer_starves_the_best_arm():
    """Was 42/100 seeds. The bound now has a count term.

    ``mean + beta * max(stddev, 1e-6)`` gave an all-zero arm a score of 2e-6
    forever: zero mean *and* zero empirical stddev read as certainty rather
    than as absence of evidence. ``mean + beta * sqrt(2 log t / n)`` is UCB1,
    where the width is governed by how little the arm has been sampled.
    """
    env = StationaryBernoulli.build()
    failures = sum(
        run(GPUCBScheduler(), env, seed=seed, rounds=ROUNDS).tail_share(env.best) < 0.5
        for seed in range(1, 61)
    )
    assert failures == 0, f"GP-UCB starved the best arm on {failures}/60 seeds"


@pytest.mark.slow
def test_hierarchical_category_starvation_is_rare():
    """Was ~1.5% of seeds (3/200); now ~0.25% (1/400).

    The top-level Thompson sample over categories could drive the posterior of
    the category holding the best operator to ~0.02 before the bottom level
    ever saw it. Capping ``alpha + beta`` preserves the posterior mean while
    stopping its variance from shrinking, so a wrong early verdict stays
    revisable. Not zero, so the strict thresholds still run only at the fixed
    seed.
    """
    env = StationaryBernoulli.build()
    failures = sum(
        run(HierarchicalBanditScheduler(), env, seed=seed, rounds=ROUNDS).tail_share(env.best) < 0.5
        for seed in range(1, 201)
    )
    assert failures <= 2, (
        f"Hierarchical starved the best category on {failures}/200 seeds (documented: ~0-1)"
    )


# ---------------------------------------------------------------------------
# Non-stationary: the regime fuzzing actually runs in
# ---------------------------------------------------------------------------

#: Measured tail share on the *new* best arm after the old best decays, at
#: FIXED_SEED over 20k rounds with the switch at 10k. Four schedulers now
#: recover; Hierarchical recovers essentially completely (0.994) after the
#: pseudocount cap, having been at 0.139 before it. Asserted as a floor with margin, because "which schedulers
#: survive coverage saturation" is the property that matters in a real
#: campaign and it should not silently regress.
RECOVERS = {
    "ContextualLinUCB": (lambda: ContextualLinUCBScheduler(dim=4), 0.35),
    "Exp3": (Exp3Scheduler, 0.30),
    "GPUCB": (GPUCBScheduler, 0.35),
    "Hierarchical": (HierarchicalBanditScheduler, 0.90),
    # The three schedulers built for this regime. Floors sit below the
    # observed minimum over 20 seeds at 20k rounds: D-UCB 0.739,
    # SW-UCB 0.806, CUCB 0.910.
    "DUCB": (lambda: DUCBScheduler(rng=RandPool(FIXED_SEED)), 0.60),
    "SWUCB": (lambda: SWUCBScheduler(rng=RandPool(FIXED_SEED)), 0.65),
    "CUCB": (lambda: CUCBScheduler(rng=RandPool(FIXED_SEED)), 0.80),
}


#: Stationary tail share for the recency-weighted family, asserted on its own
#: rather than through :data:`RELIABLE`.
#:
#: They are excluded from RELIABLE's regret-slope bound deliberately. A
#: sliding window and a per-round discount both pay a *permanent* exploration
#: tax: an arm whose evidence ages out is re-opened, so per-round regret
#: plateaus instead of decaying and the log-log slope sits near or above 1.0
#: by construction (measured p95 over 100 seeds at ROUNDS: D-UCB 0.659,
#: SW-UCB 1.656, CUCB 0.894). That is the price of the recovery the RECOVERS
#: entries above assert, not a failure to converge -- the same runs put
#: 78-90% of tail pulls on the best arm. Bounding a slope that is linear by
#: design would assert nothing, so the share floor is asserted instead.
#:
#: Floors sit below the observed minimum over 100 seeds at ROUNDS:
#: D-UCB 0.892, SW-UCB 0.895, CUCB 0.779.
RECENCY_STATIONARY = {
    "DUCB": (DUCBScheduler, 0.80),
    "SWUCB": (SWUCBScheduler, 0.80),
    "CUCB": (CUCBScheduler, 0.65),
}


@pytest.mark.parametrize("name", sorted(RECENCY_STATIONARY))
def test_recency_family_converges_on_stationary(name):
    """Forgetting must not cost convergence when there is nothing to forget."""
    factory, floor = RECENCY_STATIONARY[name]
    env = StationaryBernoulli.build()
    c = run(factory(rng=RandPool(FIXED_SEED)), env, seed=FIXED_SEED, rounds=ROUNDS)

    assert c.tail_share(env.best) >= floor, (
        f"{name} spent only {c.tail_share(env.best):.3f} of its tail on the best arm"
    )
    assert c.tail_share(env.best) > uniform_baseline(env, ROUNDS), (
        f"{name} did no better than ignoring feedback entirely"
    )


#: Schedulers that stay locked on the dead arm for the whole second half.
#: A real limitation, not a test bug: both accumulate uniform sample averages
#: with no recency weighting, so evidence from before the decay never ages out.
#: GPUCB used to be the worst offender here (0.998 on the dead arm); the
#: count-based width lets it re-open an abandoned arm, and it has moved to
#: RECOVERS.
STUCK = {
    "EpsilonGreedy": EpsilonGreedyScheduler,
    "MonteCarlo": MonteCarloScheduler,
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

    assert c.tail_share(env.best_early) > c.tail_share(env.best_late), (
        f"{name} now prefers the live arm over the decayed one -- good news; move it to RECOVERS"
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

        MCTSSeedScheduler and AlphaBetaMCTSSeedScheduler are excluded by name:
        they schedule *seeds* against a LineageTree, not operators, and need
        their own environment.
        """
        import fuzzer_tool.core.schedulers as pkg

        env = StationaryBernoulli.build()
        for cls_name in pkg.__all__:
            if cls_name in ("MCTSSeedScheduler", "AlphaBetaMCTSSeedScheduler"):
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
            HierarchicalBanditScheduler,
            GPUCBScheduler,
        ],
        ids=lambda f: f.__name__,
    )
    def test_reproducible_across_hash_seeds(self, factory):
        """A fixed --seed must fix behaviour, whatever PYTHONHASHSEED is.

        The operator taxonomy maps categories to *sets*. Both of these
        schedulers used to iterate them directly, so the order of their
        random draws depended on hash randomization. With the RNG seed pinned
        at 92, Hierarchical's tail share on the best arm ranged from 0.001 to
        0.998 across hash seeds -- 4 of 26 starved the best arm completely --
        which means a crash found under hierarchical scheduling could not be
        replayed from the seed alone.

        Checked by running a subprocess under two different hash seeds rather
        than by inspecting the code, since new set iteration can appear
        anywhere on the selection path.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            sys.path.insert(0, {str(_REPO_ROOT / "src")!r})
            from tests.support.bandit_env import StationaryBernoulli, run
            from fuzzer_tool.core.schedulers import {factory.__name__}
            env = StationaryBernoulli.build()
            c = run({factory.__name__}(), env, seed=92, rounds=1500)
            print(sorted(c.picks.items()))
        """)
        outs = []
        for hash_seed in ("0", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                check=True,
            )
            outs.append(proc.stdout)
        assert outs[0] == outs[1], (
            f"{factory.__name__} behaviour depends on PYTHONHASHSEED -- "
            "something on the selection path iterates a set"
        )

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
