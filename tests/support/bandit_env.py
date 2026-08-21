"""Ground-truth bandit environments for scheduler convergence testing.

Every scheduler in ``core/schedulers/`` is a multi-armed bandit, but nothing
in the suite asserts that any of them *converges*. The existing tests check
import-graph independence, fallback precedence, and structural invariants --
all necessary, none of which would notice a scheduler that selects uniformly
at random forever.

This module supplies the missing piece: an environment where the best arm is
known **by construction** rather than inferred from a campaign, so the
acceptance criterion can be strict. The technique is lifted from TigerBeetle's
Adaptive Replication Routing fuzzer (matklad, *A Tale Of Four Fuzzers*): build
a deliberately idealized lab -- stationary rewards, no noise beyond the
Bernoulli draw itself, no operator interaction -- precisely because a clear
ground truth is what lets you assert *optimality* rather than merely
*doesn't crash*. Realism is a different fuzzer's job; see
``tools/gen_synthetic_target.py`` for the realistic end.

The value is not primarily in catching coding bugs. It is in catching bugs in
the *reward definition*: ``record(name, success, weight)`` attributes a
discovery to a single operator name, and whether that attribution means what
the schedulers assume is exactly the kind of thing a ground-truth harness
answers and a live campaign cannot.

Two environments:

* :class:`StationaryBernoulli` -- fixed per-arm success probabilities. The
  textbook case; a scheduler that fails here is broken.
* :class:`DecayingBest` -- the best arm's yield decays to below the runner-up
  partway through. This is the regime fuzzing actually operates in (an
  operator that was productive saturates its region of the coverage map), and
  it is where bandits without a recency mechanism get permanently stuck.

Arm names are drawn from the real ``OPERATOR_CATEGORIES`` taxonomy, not
synthetic strings: :class:`HierarchicalBanditScheduler` silently ignores
``init_arm``/``record`` for names it cannot map to a category, and
:class:`GPUCBScheduler` builds its RBF features from the same table. Feeding
either one ``"op0"``..``"op11"`` would produce a test that passes while
exercising nothing.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES

# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------

#: Categories used to build arm sets, in a fixed order so arm selection is a
#: pure function of (n_arms, spread) and does not drift when the taxonomy
#: gains operators.
_CATEGORY_ORDER = ("bit", "byte", "block", "dict", "structural", "radamsa")


def build_arms(n_arms: int = 12, n_categories: int = 4) -> list[str]:
    """Return *n_arms* real operator names spread over *n_categories*.

    Deterministic given the arguments: names are sorted within each category
    and taken round-robin, so adding an operator to the taxonomy can shift
    membership but never randomizes it between runs.
    """
    cats = _CATEGORY_ORDER[:n_categories]
    pools = [sorted(OPERATOR_CATEGORIES[c]) for c in cats]
    arms: list[str] = []
    depth = 0
    while len(arms) < n_arms:
        progressed = False
        for pool in pools:
            if depth < len(pool) and len(arms) < n_arms:
                arms.append(pool[depth])
                progressed = True
        if not progressed:  # pragma: no cover - only if taxonomy shrinks
            raise ValueError(f"cannot draw {n_arms} arms from {cats}")
        depth += 1
    return arms


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


@dataclass
class StationaryBernoulli:
    """Fixed-probability arms with one unambiguously best arm.

    The best arm is deliberately placed in the *same category* as several
    near-worst arms. A naive placement -- best arm alone in its category --
    would hand :class:`HierarchicalBanditScheduler` the answer at the top
    level for free and make its result incomparable with the flat bandits.
    """

    arms: list[str]
    probs: dict[str, float]
    best: str

    @classmethod
    def build(
        cls,
        n_arms: int = 12,
        p_best: float = 0.30,
        p_runner_up: float = 0.18,
        p_base: float = 0.05,
    ) -> StationaryBernoulli:
        arms = build_arms(n_arms)
        # Best arm sits mid-list, sharing its category with base-rate arms.
        best = arms[len(arms) // 2]
        runner_up = arms[0]
        probs = dict.fromkeys(arms, p_base)
        probs[runner_up] = p_runner_up
        probs[best] = p_best
        return cls(arms=arms, probs=probs, best=best)

    def p(self, arm: str, t: int) -> float:  # noqa: ARG002 - stationary
        return self.probs[arm]

    def p_max(self, t: int) -> float:  # noqa: ARG002 - stationary
        return self.probs[self.best]

    def best_at(self, t: int) -> str:  # noqa: ARG002 - stationary
        return self.best


@dataclass
class DecayingBest:
    """The initially-best arm's yield collapses at *switch_at*.

    Models coverage saturation: an operator that was finding new edges runs
    out of new edges to find. After the switch the runner-up is strictly
    best, and a scheduler with no recency mechanism (uniform sample averages,
    undecayed Beta posteriors) will keep exploiting a dead arm.
    """

    arms: list[str]
    best_early: str
    best_late: str
    p_high: float
    p_low: float
    p_base: float
    switch_at: int

    @classmethod
    def build(
        cls,
        n_arms: int = 12,
        p_high: float = 0.30,
        p_low: float = 0.02,
        p_base: float = 0.05,
        switch_at: int = 10_000,
    ) -> DecayingBest:
        arms = build_arms(n_arms)
        return cls(
            arms=arms,
            best_early=arms[len(arms) // 2],
            best_late=arms[0],
            p_high=p_high,
            p_low=p_low,
            p_base=p_base,
            switch_at=switch_at,
        )

    def p(self, arm: str, t: int) -> float:
        if arm == self.best_early:
            return self.p_high if t < self.switch_at else self.p_low
        if arm == self.best_late:
            # Runner-up throughout; strictly best once best_early collapses.
            return 0.18
        return self.p_base

    def p_max(self, t: int) -> float:
        return max(self.p(a, t) for a in self.arms)

    def best_at(self, t: int) -> str:
        return max(self.arms, key=lambda a: self.p(a, t))


# ---------------------------------------------------------------------------
# Scheduler adapters
# ---------------------------------------------------------------------------


class Adapter:
    """Uniform ``select()`` / ``update(success)`` facade over a scheduler.

    The schedulers do *not* share one interface, despite the docstrings
    implying they do:

    * seven expose ``select_op(ops) -> str`` and ``record(name, success, weight)``
    * ``MOptScheduler.select_op`` returns ``(op, particle_idx)`` and
      ``record`` takes ``particle_id``
    * ``ContextualLinUCBScheduler`` needs a context vector and takes a float
      ``reward`` rather than a bool ``success``
    * ``MCTSSeedScheduler`` schedules *seeds*, not operators, and is out of
      scope here

    Papering over that divergence in the adapter is the point: it keeps the
    divergence visible in one place instead of spread across every call site.
    """

    needs_context = False

    def __init__(self, scheduler, arms: list[str]):
        self.s = scheduler
        self.arms = arms
        init = getattr(scheduler, "init_arm", None)
        if init is not None:
            for a in arms:
                init(a)

    def select(self) -> str:
        return self.s.select_op(self.arms)

    def update(self, name: str, success: bool) -> None:
        self.s.record(name, success)


class MOptAdapter(Adapter):
    """MOpt returns and requires a particle index alongside the operator."""

    def __init__(self, scheduler, arms: list[str]):
        super().__init__(scheduler, arms)
        self._particle = 0

    def select(self) -> str:
        op, self._particle = self.s.select_op(self.arms)
        return op

    def update(self, name: str, success: bool) -> None:
        self.s.record(name, success, particle_id=self._particle)


class ContextualAdapter(Adapter):
    """LinUCB needs a feature vector; supply a constant one.

    A constant context reduces LinUCB to a ridge-regression UCB bandit, which
    is the fair comparison against the context-free schedulers. Testing its
    *contextual* behaviour needs a different environment (reward depending on
    the feature vector) and is deliberately not attempted here.
    """

    needs_context = True
    _CTX = [1.0, 0.0, 0.0, 0.0]

    def select(self) -> str:
        return self.s.select_op(self.arms, self._CTX)

    def update(self, name: str, success: bool) -> None:
        self.s.record(name, self._CTX, 1.0 if success else 0.0)


def adapt(scheduler, arms: list[str]) -> Adapter:
    """Wrap *scheduler* in the right adapter, chosen by class name."""
    cls = type(scheduler).__name__
    if cls == "MOptScheduler":
        return MOptAdapter(scheduler, arms)
    if cls == "ContextualLinUCBScheduler":
        return ContextualAdapter(scheduler, arms)
    return Adapter(scheduler, arms)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


@dataclass
class Campaign:
    """Outcome of one run. All fields are ground-truth-relative."""

    picks: Counter = field(default_factory=Counter)
    tail_picks: Counter = field(default_factory=Counter)
    successes: int = 0
    failures: int = 0
    regret: float = 0.0
    regret_trace: list[float] = field(default_factory=list)
    rounds: int = 0
    tail_rounds: int = 0

    def tail_share(self, arm: str) -> float:
        """Fraction of the final segment spent on *arm*."""
        return self.tail_picks[arm] / max(self.tail_rounds, 1)

    def regret_slope(self) -> float:
        """Log-log slope of cumulative regret against round number.

        Sublinear regret -- the defining property of a working bandit -- shows
        up as a slope below 1.0. A scheduler that never learns pulls a
        constant-suboptimality arm forever and gives slope 1.0. Fitted over
        the second half only, so the unavoidable linear burn-in of the
        exploration phase does not dominate.
        """
        pts = [
            (math.log(t), math.log(r))
            for t, r in enumerate(self.regret_trace, start=1)
            if t > len(self.regret_trace) // 2 and r > 0
        ]
        if len(pts) < 2:
            return float("nan")
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        denom = sum((x - mx) ** 2 for x, _ in pts)
        if denom == 0:
            return float("nan")
        return sum((x - mx) * (y - my) for x, y in pts) / denom


def run(
    scheduler,
    env,
    seed: int,
    rounds: int = 20_000,
    tail_frac: float = 0.2,
    trace_every: int = 25,
) -> Campaign:
    """Drive *scheduler* against *env* for *rounds* pulls.

    ``random.seed`` is set globally because six of the schedulers draw from
    the module-level ``random`` rather than an injected generator (see
    ``test_scheduler_convergence.py::TestSchedulerSeedability``). The
    environment uses its own :class:`random.Random` so that changing the
    number of environment draws cannot perturb the scheduler's stream.
    """
    random.seed(seed)
    env_rng = random.Random(seed ^ 0x5EED)

    a = adapt(scheduler, env.arms)
    c = Campaign(rounds=rounds)
    tail_start = int(rounds * (1.0 - tail_frac))

    for t in range(rounds):
        op = a.select()
        # A scheduler returning something outside the candidate list is a bug
        # regardless of convergence, so check it on the hot path.
        assert op in env.arms, f"{type(scheduler).__name__} returned {op!r}, not a candidate"
        success = env_rng.random() < env.p(op, t)
        a.update(op, success)

        c.picks[op] += 1
        c.successes += success
        c.failures += not success
        c.regret += env.p_max(t) - env.p(op, t)
        if t % trace_every == 0:
            c.regret_trace.append(c.regret)
        if t >= tail_start:
            c.tail_picks[op] += 1
            c.tail_rounds += 1

    return c


def uniform_baseline(env, rounds: int, tail_frac: float = 0.2) -> float:
    """Tail share a scheduler that ignores all feedback would achieve.

    Any convergence threshold must clear this, or the test proves nothing.
    """
    del rounds, tail_frac
    return 1.0 / len(env.arms)
