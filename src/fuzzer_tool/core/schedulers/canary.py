"""CanaryScheduler: a deliberately worst-possible operator scheduler.

Why a scheduler that tries to lose
-----------------------------------
The Elo meta-scheduler (``--elo``) arbitrates among every enabled operator
scheduler by playing whichever one was selected against every other
enabled scheduler each round (see ``analyzer_registry._activate_elo`` and
``services.operators.operator_strategy_pool``). That tournament only
produces *relative* standing -- it has no independent floor. If every real
scheduler regressed to the same degenerate policy (a bug that makes every
scheduler fall back to uniform random, or one that always reselects the
same operator), the tournament would still produce a ranking and Elo
would still happily arbitrate over it, because every entrant is equally
bad and nothing in the ratings says so.

CanaryScheduler exists to be a known, deliberately pessimal floor: given
the same win/loss signal every other scheduler in the pool receives
(``record(name, success, weight)``), it always selects the candidate with
the LOWEST posterior success-rate estimate -- the opposite of the
argmax/Thompson-draw every real scheduler here performs. Provided the
candidates are not all tied, its picks should discover fewer edges than
any scheduler that is even minimally exploiting the same signal, so it
should end up with the lowest rating in the strategy pool and get chosen
by ``select_strategy`` the least of anyone.

If canary is ever NOT the bottom entry -- a real scheduler's posterior
mean drops to or below canary's -- that is not canary doing well, it is a
real scheduler doing pathologically. See
``BayesianEloTracker.strategies_below_canary`` and
``Fuzzer._check_canary_inspection``, which log that condition for
inspection rather than silently letting Elo route around it.
"""

from __future__ import annotations


class CanaryScheduler:
    """Deliberately worst-in-class operator scheduler; a floor for the Elo pool.

    Tracks a Beta(alpha, beta) posterior per operator from the same
    ``record(name, success, weight)`` signal every other scheduler in the
    pool receives, then always selects the operator with the LOWEST
    posterior mean success rate -- an intentional argmin instead of the
    argmax/Thompson-draw every real scheduler here performs. Ties (most
    commonly at the shared Beta(1, 1) prior, before any candidate has been
    recorded) go to whichever candidate appears first in ``ops`` --
    deterministic, no randomness anywhere in this path, so canary never
    gets an accidental assist from luck either.

    This is not a fuzzing strategy. It exists purely as an instrumented
    floor for the Elo meta-scheduler's tournament: see the module
    docstring for why an intentionally-bad, signal-driven baseline is more
    useful here than round-robin's signal-blind one.
    """

    #: No meaningful priors: seeding it with a "good" prior would work
    #: against the one property that matters -- being worst.
    supports_priors = False

    def __init__(self) -> None:
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        self._order: list[str] = []

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *name* with a uniform Beta(1, 1) prior.

        ``prior_alpha``/``prior_beta`` are accepted for interface parity
        with the other schedulers' ``init_arm`` but ignored (see
        ``supports_priors``). Re-registering an arm never resets it.
        """
        if name not in self._alpha:
            self._alpha[name] = 1.0
            self._beta[name] = 1.0
            self._order.append(name)

    def _posterior_mean(self, name: str) -> float:
        a = self._alpha.get(name, 1.0)
        b = self._beta.get(name, 1.0)
        return a / (a + b)

    def select_op(self, ops: list[str]) -> str:
        """Select the candidate with the lowest posterior success rate.

        Unregistered candidates are registered on the fly (same
        just-in-time behavior as the other schedulers' ``select_op``).
        Ties go to whichever candidate came first in ``ops``.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        for op in ops:
            self.init_arm(op)
        worst = ops[0]
        worst_mean = self._posterior_mean(worst)
        for op in ops[1:]:
            mean = self._posterior_mean(op)
            if mean < worst_mean:
                worst = op
                worst_mean = mean
        return worst

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """One fractional-Bernoulli observation -- the same signal every other scheduler gets."""
        self.init_arm(name)
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        self._alpha[name] += r
        self._beta[name] += 1.0 - r

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return (alpha, beta) pseudocounts for each registered arm, prior included."""
        return {name: (self._alpha[name], self._beta[name]) for name in sorted(self._order)}
