"""SeedCanaryScheduler: a deliberately worst-possible seed scheduler.

The seed-arena counterpart of ``core/schedulers/op_canary.py``. The Elo
meta-scheduler does not run one tournament, it runs two: alongside the
operator-scheduler pool arbitrated under plain strategy names (``mopt``,
``exp3``, ``canary``, ...), ``services/seed_picker.py::_pick_seed_elo``
arbitrates a *separate* pool of seed-selection strategies (``weighted``,
``mcts``, ``ecofuzz``, ...) under ``seed_``-prefixed keys, with no cross
competition between the two arenas (see
``core/elo.py::BayesianEloTracker._record_seed_strategy_matches`` /
``_record_operator_strategy_matches``, each restricted to its own pool).

Each arena needs its own floor for the same reason: a tournament among
seed strategies only produces *relative* standing. If every real seed
strategy regressed to the same degenerate policy (a bug that makes seed
selection fall back to uniform-random, or one that always reselects the
same seed), the tournament would still produce a ranking and Elo would
still happily arbitrate over it, because every entrant is equally bad
and nothing in the ratings says so. ``op_canary`` cannot serve this role
for the seed arena: it is registered and rated under a plain (unprefixed)
name in the operator pool, and it argmin-selects over *operators*, not
over *seeds* -- it has no posterior over which corpus entry is worth
fuzzing.

SeedCanaryScheduler closes that gap. Registered as the strategy named
``"canary"`` in ``_pick_seed_elo``'s available list, it is offered to
``select_strategy`` under the same ``seed_canary`` key every other seed
strategy is offered under (the ``seed_`` prefix is applied uniformly by
the caller, not by this module). Given the same signal every real seed
strategy's outcome is judged by -- did fuzzing this seed key produce new
coverage, and at what surprisal weight -- it always selects the seed with
the LOWEST posterior success-rate estimate, the opposite of every real
seed strategy's argmax/Thompson pick. Provided the candidates are not all
tied, fuzzing canary's picks should discover less than any strategy that
is even minimally exploiting the same per-seed signal, so it should end
up with the lowest rating in the seed strategy pool and get chosen by
``select_strategy`` the least of anyone.

If canary is ever NOT the bottom entry of the seed arena -- a real seed
strategy's posterior mean drops to or below canary's -- that is not
canary doing well, it is a real seed strategy doing pathologically. See
``BayesianEloTracker.strategies_below_canary`` (already generic over
``canary_name``, so ``strategies_below_canary("seed_canary")`` reuses it
unchanged) and ``Fuzzer._check_canary_inspection``, which should log that
condition for the seed arena the same way it already does for the
operator arena.
"""

from __future__ import annotations


class SeedCanaryScheduler:
    """Deliberately worst-in-class seed scheduler; a floor for the seed Elo pool.

    Tracks a Beta(alpha, beta) posterior per seed key from the same
    ``record(seed_id, success, weight)`` signal ``core/seed_quality.py``'s
    ``BayesianSeedQuality.record_outcome`` receives off-policy every round
    (the outcome of whichever seed was actually fuzzed, regardless of
    which strategy picked it -- a Beta posterior does not care who pulled
    the arm), then always selects the seed with the LOWEST posterior mean
    success rate -- an intentional argmin instead of the
    argmax/Thompson-draw every real seed strategy here performs. Ties
    (most commonly at the shared Beta(1, 1) prior, before any candidate
    has been recorded) go to whichever candidate appears first in
    ``seed_ids`` -- deterministic, no randomness anywhere in this path,
    so canary never gets an accidental assist from luck either.

    This is not a seed-scheduling strategy. It exists purely as an
    instrumented floor for the seed arena's tournament: see the module
    docstring for why an intentionally-bad, signal-driven baseline is
    more useful here than a signal-blind one.
    """

    #: No meaningful priors: seeding it with a "good" prior would work
    #: against the one property that matters -- being worst. Mirrors
    #: CanaryScheduler.supports_priors.
    supports_priors = False

    def __init__(self) -> None:
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        self._order: list[str] = []

    def init_arm(self, seed_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *seed_id* with a uniform Beta(1, 1) prior.

        ``prior_alpha``/``prior_beta`` are accepted for interface parity
        with ``BayesianSeedQuality.init_seed`` but ignored (see
        ``supports_priors``). Re-registering a seed never resets it.
        """
        if seed_id not in self._alpha:
            self._alpha[seed_id] = 1.0
            self._beta[seed_id] = 1.0
            self._order.append(seed_id)

    def _posterior_mean(self, seed_id: str) -> float:
        a = self._alpha.get(seed_id, 1.0)
        b = self._beta.get(seed_id, 1.0)
        return a / (a + b)

    def select_seed(self, seed_ids: list[str]) -> str:
        """Select the candidate with the lowest posterior success rate.

        Unregistered candidates are registered on the fly (same
        just-in-time behavior as ``CanaryScheduler.select_op`` and every
        real seed strategy's selection path). Ties go to whichever
        candidate came first in ``seed_ids``.
        """
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        for seed_id in seed_ids:
            self.init_arm(seed_id)
        worst = seed_ids[0]
        worst_mean = self._posterior_mean(worst)
        for seed_id in seed_ids[1:]:
            mean = self._posterior_mean(seed_id)
            if mean < worst_mean:
                worst = seed_id
                worst_mean = mean
        return worst

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """One fractional-Bernoulli observation -- the same signal ``BayesianSeedQuality``
        gets on every round, fed regardless of which strategy picked the seed.
        """
        self.init_arm(seed_id)
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        self._alpha[seed_id] += r
        self._beta[seed_id] += 1.0 - r

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return (alpha, beta) pseudocounts for each registered arm, prior included."""
        return {
            seed_id: (self._alpha[seed_id], self._beta[seed_id]) for seed_id in sorted(self._order)
        }
