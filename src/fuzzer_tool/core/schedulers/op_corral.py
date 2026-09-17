"""CorralScheduler: log-barrier OMD over mutation operators.

Corral (Agarwal, Luo, Neyshabur & Schapire, *Corralling a Band of Bandit
Algorithms*, COLT 2017) is stated as a master over base algorithms, but its
engine is a bandit in its own right: online mirror descent under the
log-barrier regulariser ``psi(p) = sum_i (1/eta_i) ln(1/p_i)``, fed
importance-weighted loss estimates, with per-arm learning rates that only
ever increase. That engine is what this module applies to the operator
arms, so it competes on the ballot like any other scheduler and Elo
arbitrates above it.

Why this is a family the tree does not already have
---------------------------------------------------
The existing twenty schedulers are UCB confidence widths (``ucb_common``,
``ducb``, ``swucb``, ``kl_*``, ``cucb``, ``c2ucb``, ``moss``, ``gp_ucb``,
``cusum_ucb``, ``contextual``), posterior sampling (``monte_carlo``,
``consolidated``), exponential weights (``exp3``, ``fpl``), population
methods (``cmaes``, ``mopt``, ``replicator``) and tree search (``mcts``).
Log-barrier OMD is none of those, and the difference is not cosmetic:

- **EXP3 uses entropy**, so its update is multiplicative and a single large
  importance-weighted loss can drive an arm's weight down by an unbounded
  factor. The log barrier updates ``1/p_i`` *additively*, which is why its
  probabilities decay polynomially rather than exponentially -- the
  property the best-of-both-worlds literature is built on.
- **UCB widths need a variance model**; this needs none.
- The per-arm learning rate with the doubling trick makes an arm that has
  been beaten down *cheaper to revisit over time*, rather than relying on a
  global forgetting factor. That matters here for the reason
  ``docs/handover/handover_non_ucb_schedulers_2026-09-13.md`` §2 sets out:
  every recency mechanism in the tree discounts on the global round counter,
  and an operator's yield actually falls as a function of its *own* pulls.
  The doubling trick is indexed per arm. It is not a rotting-bandit
  estimator and does not claim to be one -- it raises an arm's learning
  rate, not its mean -- but it is the one mechanism here whose clock is the
  arm's own history.

What is claimed, and what is not
--------------------------------
Claimed: an operator scheduler whose update is unbiased for the arm it drew
and whose probabilities cannot collapse exponentially.

Not claimed: Corral's regret theorem. That result is about corralling *base
algorithms* and additionally requires each base to be stable in its sense
and to receive losses scaled by the master's ``rho_i``. Used as a plain
bandit over operators there are no bases, so there is nothing to be stable;
what ships is the estimator and the update rule. Anything written about
this should say "log-barrier OMD over operators", not "Corral's guarantee".

The baseline term, and why it is on by default
----------------------------------------------
OMD is stated over losses in ``[0, 1]``. The signal here is a reward, and
across the ~155 registered arms the reward is zero on the overwhelming
majority of rounds, so ``loss = 1 - reward`` sits near 1 almost always.
The textbook estimate ``loss / p`` then hands the arm that was just drawn a
value near ``1/p`` -- at uniform over 155 arms that is ~155 -- *every*
round. Whoever played is punished hardest, which is churn, not learning,
and it is the same pathology that makes the Elo fan-out prefer whoever
played least recently (measured in
``core/schedulers/op_consolidated.py``). Unbiasedness does not save it; the
variance is what does the damage.

The fix is the standard loss shift. With running mean loss ``b``,

    est_i = b + (loss - b) / p_drawn   for the arm that was drawn
    est_i = b                          for every other arm

``E[est_i] = (loss_i - b) + b = loss_i``, so it stays unbiased for every arm
(asserted numerically by ``test_baseline_estimator_is_unbiased``), while the
spike scales with the *deviation* from a typical round instead of with the
loss. When nothing is working the estimates collapse to a common ``b``,
every arm moves together, and the distribution correctly stays flat: there
is no signal to concentrate on. Set ``baseline=False`` to get the textbook
estimator back; it is kept so the choice stays falsifiable, not because it
is usable (``test_textbook_estimator_punishes_whoever_played``).

Measured limitation: spurious concentration on equal arms
---------------------------------------------------------
With genuinely equal arms the distribution does not stay flat. The estimate
has variance of order ``1/p``, and at the default ``eta`` a lucky round moves
the winner far enough that it is sampled more often, which compounds.
Measured over 20 seeds and 3000 rounds with every arm at 3%: normalised
entropy 0.300/0.547/0.926 (min/median/max) at 4 arms and 0.559/0.672/0.836 at
12 arms; at eta=0.3, 0.490/0.828 and 0.694/0.889. So it is the learning rate
as much as the estimator.

That costs no regret when the arms really are equal, but it is the same
mechanism as the recovery fragility on the convergence harness's
``DecayingBest`` (best_late tail share min 0.006, median 0.777 over 12
seeds): the doubling trick raises a starved arm's learning *rate* without
getting it drawn, so whether it comes back depends on the mixing floor
sampling it soon enough. Both are why this is an Elo-only arm and not a
member of the fallback chain. ``tests/test_corral_scheduler.py`` pins the
entropy bound so that a future fix shows up as a failure there.

On-policy only
--------------
``record`` credits an arm only against a probability this scheduler itself
drew it with, because that is the only distribution the importance weight
is unbiased against. Operators recorded from another scheduler's round are
counted as orphans and dropped, and ``Fuzzer._record_outcome`` gates the
fan-out on ``selector == "corral"`` for the same reason it already does for
``exp3`` and ``cmaes``. A silently on-by-default fan-out here would not just
add noise -- it would divide someone else's outcome by a probability that
never applied to it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

#: Floor on any arm's probability. The doubling trick bounds probabilities
#: away from zero in theory; this is the numerical guard that keeps ``1 / p``
#: finite over a long campaign.
MIN_PROB = 1e-9

#: Safeguarded-Newton steps for the log-barrier normaliser. Convex and
#: increasing, warm-started from the previous round, so the residual is
#: under _SOLVE_TOL in single digits of iterations; the cap only bounds the
#: pathological case.
_NEWTON_ITERS = 24

#: Residual on ``sum(p) - 1`` that ends the solve. The probabilities are
#: renormalised afterwards regardless, so this only has to be tight enough
#: that renormalising is a correction rather than the answer.
_SOLVE_TOL = 1e-12

#: Margin kept between the search bracket and the nearest pole of the
#: normaliser, where a denominator reaches zero.
_POLE_MARGIN = 1e-12

#: Cap on remembered unconsumed draws. A ``select_op`` whose operator is
#: never recorded (the round threw it away) would otherwise leak one entry;
#: the oldest are dropped and counted.
_PENDING_MAX = 256


class CorralScheduler:
    """Log-barrier OMD bandit over mutation operators.

    State is held as parallel numpy arrays indexed by ``_idx``, not as dicts.
    That is not premature: the update is O(K) over *every registered arm*
    every round, because lambda normalises across the whole simplex. Rebuilding
    three K-vectors from dicts per round measured 153 us at the ~155 registered
    operators -- against ~800 us for one real in-process ffmpeg execution --
    and nearly all of it was the rebuild, not the solve. Arrays are grown on
    ``init_arm`` only, which happens once per operator.

    Args:
        eta: Base learning rate, shared by every arm at registration and
            raised per arm thereafter by the doubling trick. The default was
            measured, and the measurement contradicted the obvious guess:
            the step on ``1/p_i`` is ``eta * (est_i - lambda)`` and ``est``
            scales with ``1/p``, hence with the arm count, so a smaller
            ``eta`` at larger K looks right and is wrong. At 6000 rounds
            over 155 arms each arm is pulled ~39 times, and a small ``eta``
            simply never learns. Best-arm tail share, ``mix`` at its default,
            on a 155-arm synthetic with one 10% arm against a 0.2-2.2% tail
            (4 seeds, uniform share 0.0065) and on the 12-arm stationary
            harness (20 seeds, 6000 rounds):

            ===== ================== =================
            eta   155 arms min/med   12 arms min/med
            ===== ================== =================
            0.05  0.005 / 0.009      --
            0.1   0.006 / 0.012      0.853 / 0.885
            0.3   0.012 / 0.292      0.918 / 0.931
            0.6   0.007 / 0.523      0.925 / 0.943
            1.0   0.000 / 0.003      0.752 / 0.948
            2.0   0.000 / 0.000      0.943 / 0.952
            ===== ================== =================

            So 0.6 is the joint optimum and the collapse above it is lock-in
            at high K, not a smooth degradation: at 1.0 the 155-arm median
            drops by two orders of magnitude while the 12-arm median still
            rises. Tune this per target if at all, and read the ``entropy``
            line in the convergence report before trusting a raised value.
        horizon: Expected round count, used only for the doubling factor
            ``beta = exp(1 / ln(horizon))``. Wrong values change how fast
            rates grow, not correctness. Must exceed ``e``: at or below it
            ``ln(horizon) <= 1``, so one underflow would multiply a learning
            rate by at least e.
        baseline: Subtract the running mean loss before importance
            weighting. See the module docstring.
        mix: Uniform mixing floor in [0, 0.5), applied at selection so every
            offered arm keeps at least ``mix / len(ops)`` of the draw. This
            is a departure from the literature, taken on measurement:
            log-barrier OMD is supposed to need no floor because the
            doubling trick keeps arms recoverable, but the trick only raises
            a starved arm's learning *rate* -- the arm still has to be drawn
            to produce an estimate, and at p ~ 1e-3 that is a thousand-round
            wait. On the stationary 12-arm harness an over-large eta locked
            onto a wrong arm and never left (best-arm tail share 0.000 at
            eta=1.0 over 12 seeds); mix=0.05 recovered it to 0.938 at a cost
            of 0.973 -> 0.932 in the well-tuned case. The floor is insurance
            against a mis-set eta, which is a user-facing flag.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: Log-barrier OMD has no Beta-Bernoulli prior to seed (Hard Rule 40).
    supports_priors = False

    def __init__(
        self,
        eta: float = 0.6,
        horizon: int = 100_000,
        baseline: bool = True,
        mix: float = 0.05,
        rng: RandPool | None = None,
    ) -> None:
        if eta <= 0.0:
            raise ValueError(f"eta must be positive, got {eta!r}")
        if horizon <= math.e:
            raise ValueError(f"horizon must be > e, got {horizon!r}")
        if not 0.0 <= mix < 0.5:
            raise ValueError(f"mix must be in [0, 0.5), got {mix!r}")
        self.eta = float(eta)
        self.horizon = int(horizon)
        self.baseline = bool(baseline)
        self.mix = float(mix)
        self._rng = rng if rng is not None else RandPool()

        # Array-backed state. _names[i] <-> _idx[name] == i for every arm.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._pv = np.zeros(0, dtype=np.float64)  # the distribution
        self._etav = np.zeros(0, dtype=np.float64)  # per-arm learning rate
        self._rhov = np.zeros(0, dtype=np.float64)  # 1 / recorded lower bound
        self._pullv = np.zeros(0, dtype=np.int64)
        self._winv = np.zeros(0, dtype=np.float64)

        # Operator -> probability it was drawn with, awaiting its reward. The
        # probability must come from the draw: an importance weight is only
        # unbiased against the distribution the sample came from, and several
        # operators can be drawn in one round (an operator stack), so this
        # cannot collapse to a single "last pick".
        self._pending: dict[str, float] = {}

        # Arms offered by the most recent select_op. bandit_stats() reports
        # over these, not over every registered arm: ~155 operators are
        # registered but a round offers the handful that passed the sniffers,
        # and averaging in arms that could not be drawn would turn the
        # concentration readout into a statement about dead weight.
        self._ballot: list[str] = []

        # Previous round's normaliser, used to warm-start the solve. Pure
        # optimisation: it changes the iteration count, never the root, which
        # the bracket pins.
        self._last_lambda: float | None = None

        self._rounds = 0
        # Running mean loss as sum/count rather than an EWMA, so the baseline
        # is exactly reproducible from the same round sequence.
        self._loss_sum = 0.0
        self._loss_count = 0
        # record() calls for arms this scheduler did not draw. Dropped, not
        # applied -- but a large count means the on-policy gate upstream is
        # not holding and the distribution is learning from less than it looks.
        self._orphans = 0
        self._expired_draws = 0
        self._rate_increases = 0
        self._degenerate_steps = 0

    # -- properties ---------------------------------------------------

    @property
    def beta(self) -> float:
        """Corral's doubling factor ``exp(1 / ln(horizon))``."""
        return math.exp(1.0 / math.log(self.horizon))

    # -- arm registration ---------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator at the current uniform share.

        Idempotent, and callable mid-run: ``REGISTRY.register_mutator`` adds
        operators after import, which is exactly the case Hierarchical used
        to drop silently. A new arm enters at ``1/(M+1)`` and incumbents
        scale by ``M/(M+1)``, so the simplex constraint holds without
        reordering anyone.
        """
        if name in self._idx:
            return
        m = len(self._names)
        self._idx[name] = m
        self._names.append(name)
        share = 1.0 if m == 0 else 1.0 / (m + 1)
        if m > 0:
            self._pv *= 1.0 - share
        self._pv = np.append(self._pv, share)
        self._etav = np.append(self._etav, self.eta)
        self._rhov = np.append(self._rhov, 2.0 * (m + 1))
        self._pullv = np.append(self._pullv, 0)
        self._winv = np.append(self._winv, 0.0)

    # -- selection ----------------------------------------------------

    def probabilities(self, ops: list[str]) -> dict[str, float]:
        """Draw distribution restricted to *ops*, registering anything new."""
        if not ops:
            return {}
        for op in ops:
            self.init_arm(op)
        self._ballot = list(ops)
        raw = [float(self._pv[self._idx[op]]) for op in ops]
        total = sum(raw)
        n = len(ops)
        if total <= 0.0:
            return dict.fromkeys(ops, 1.0 / n)
        if self.mix > 0.0:
            keep = 1.0 - self.mix
            floor = self.mix / n
            return {op: keep * (q / total) + floor for op, q in zip(ops, raw, strict=True)}
        return {op: q / total for op, q in zip(ops, raw, strict=True)}

    def select_op(self, ops: list[str]) -> str:
        """Draw one operator and remember the probability it was drawn with."""
        probs = self.probabilities(ops)
        if not probs:
            return ""
        r = self._rng.random()
        cumulative = 0.0
        chosen = ops[-1]
        for op in ops:
            cumulative += probs[op]
            if r <= cumulative:
                chosen = op
                break
        if len(self._pending) >= _PENDING_MAX and chosen not in self._pending:
            # Drop the oldest unconsumed draw rather than grow without bound.
            # dicts keep insertion order, so next(iter(...)) is the oldest.
            self._pending.pop(next(iter(self._pending)))
            self._expired_draws += 1
        self._pending[chosen] = probs[chosen]
        return chosen

    # -- learning -----------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Apply one OMD step for an operator this scheduler drew.

        Args:
            name: The operator that ran.
            success: Whether the round produced new coverage.
            weight: Cost-adjusted reward magnitude for a successful round,
                clamped into [0, 1]. A failure is reward 0 whatever the
                weight, matching every other scheduler's ``record``.
        """
        p_drawn = self._pending.pop(name, None)
        if p_drawn is None or p_drawn <= 0.0:
            # Either another scheduler's round, or a second reward for a draw
            # already settled. Both must be dropped: the weight would be taken
            # against a distribution that did not produce the sample.
            self._orphans += 1
            return
        self.init_arm(name)
        j = self._idx[name]

        reward = min(1.0, max(0.0, float(weight))) if success else 0.0
        loss = 1.0 - reward
        b = (self._loss_sum / self._loss_count) if (self.baseline and self._loss_count) else 0.0
        self._loss_sum += loss
        self._loss_count += 1
        self._rounds += 1
        self._pullv[j] += 1
        self._winv[j] += reward

        est = np.full(len(self._names), b, dtype=np.float64)
        est[j] = b + (loss - b) / p_drawn

        if self._omd_step(est):
            self._apply_doubling()

    def _omd_step(self, est: np.ndarray) -> bool:
        """Solve ``sum_i 1 / (1/p_i + eta_i (est_i - lam)) = 1`` and move p.

        The left side is strictly increasing and convex in ``lam`` -- every
        denominator falls linearly -- tending to 0 as ``lam -> -inf`` and to
        ``+inf`` at the smallest pole, so a sign-change bracket always
        exists. Safeguarded Newton inside that bracket converges in single
        digits of iterations; the bracket is kept only to clamp a step that
        would land on or past a pole, where the derivative is meaningless.

        Returns False without touching ``p`` when the problem is numerically
        degenerate. Leaving the distribution alone is the safe outcome: a
        renormalised guess here would be a silent unbounded jump, and the
        count surfaces in ``bandit_stats``.
        """
        p = self._pv
        if not np.all(p > 0.0):
            self._degenerate_steps += 1
            return False
        inv = 1.0 / p
        eta = self._etav
        slack = inv / eta

        hi = float((est + slack).min()) - _POLE_MARGIN
        lo = float(est.min() - slack.max()) - 1.0

        def residual(lam: float) -> float:
            denom = inv + eta * (est - lam)
            if not np.all(denom > 0.0):
                return math.inf
            return float((1.0 / denom).sum()) - 1.0

        if residual(lo) > 0.0 or residual(hi) < 0.0:
            self._degenerate_steps += 1
            return False

        lam = 0.5 * (lo + hi)
        if self._last_lambda is not None and lo < self._last_lambda < hi:
            # Consecutive rounds differ in one arm's estimate, so the previous
            # root is usually a better guess than the midpoint.
            lam = self._last_lambda
        denom = inv + eta * (est - lam)
        for _ in range(_NEWTON_ITERS):
            denom = inv + eta * (est - lam)
            if not np.all(denom > 0.0):
                lam = 0.5 * (lo + hi)
                continue
            recip = 1.0 / denom
            f = float(recip.sum()) - 1.0
            if -_SOLVE_TOL < f < _SOLVE_TOL:
                break
            if f > 0.0:
                hi = lam
            else:
                lo = lam
            # d/dlam sum(1/d_i) = sum(eta_i / d_i^2) > 0
            fp = float((eta * recip * recip).sum())
            step = (lam - f / fp) if fp > 0.0 else 0.5 * (lo + hi)
            lam = step if lo < step < hi else 0.5 * (lo + hi)
        self._last_lambda = lam

        denom = inv + eta * (est - lam)
        out = np.where(denom > 0.0, 1.0 / np.maximum(denom, MIN_PROB), MIN_PROB)
        np.maximum(out, MIN_PROB, out=out)
        out /= out.sum()
        self._pv = out
        return True

    def _apply_doubling(self) -> None:
        """Corral's trick: when an arm falls below its recorded lower bound,
        halve the bound and raise *that arm's* learning rate.

        This is the one mechanism here indexed on an arm's own history rather
        than on the global round counter, and it is what makes a beaten-down
        operator cheaper to re-learn -- the arm whose probability shrank is
        the one whose next update moves furthest. It does not, on its own,
        get that arm drawn again; see ``mix``.
        """
        under = self._pv < 1.0 / self._rhov
        if not under.any():
            return
        count = int(under.sum())
        self._rhov[under] = 2.0 / np.maximum(self._pv[under], MIN_PROB)
        self._etav[under] *= self.beta
        self._rate_increases += count

    # -- reporting ----------------------------------------------------

    def bandit_stats(self) -> dict[str, Any]:
        """Convergence stats, reported over the arms the last round offered.

        ``entropy`` is normalised to [0, 1] so "flat" is one number to read.
        For this scheduler that is the diagnostic that matters: a log-barrier
        distribution that never leaves uniform is indistinguishable in
        outcome from the arbiter failure this family was proposed to avoid.
        """
        live = [a for a in (self._ballot or self._names) if a in self._idx]
        idx = [self._idx[a] for a in live]
        total = float(self._pv[idx].sum()) if idx else 0.0
        probs = (
            {a: float(self._pv[i]) / total for a, i in zip(live, idx, strict=True)}
            if total > 0.0
            else {}
        )
        n = len(probs)
        ent = -sum(q * math.log(q) for q in probs.values() if q > 0.0)
        return {
            "rounds": self._rounds,
            "arms": n,
            "registered": len(self._names),
            "probs": probs,
            "pulls": {a: int(self._pullv[i]) for a, i in zip(live, idx, strict=True)},
            "wins": {a: float(self._winv[i]) for a, i in zip(live, idx, strict=True)},
            "concentration": max(probs.values()) if probs else 0.0,
            "entropy": (ent / math.log(n)) if n > 1 else 0.0,
            "mean_loss": (self._loss_sum / self._loss_count) if self._loss_count else 0.0,
            "rate_increases": self._rate_increases,
            "orphan_records": self._orphans,
            "expired_draws": self._expired_draws,
            "degenerate_steps": self._degenerate_steps,
        }
