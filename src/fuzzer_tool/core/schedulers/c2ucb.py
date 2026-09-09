"""C2UCBScheduler: contextual combinatorial UCB over the round's stack.

Qin, Chen & Zhu, *Contextual Combinatorial Bandit and its Application on
Diversified Online Recommendation* (SDM 2014). This is not a third
implementation of a bandit index -- it is ``CUCBScheduler`` and
``ContextualLinUCBScheduler`` fused, because each already solves half of
what this scheduler needs and neither solves the other half:

    CUCBScheduler:        knows the round is a superarm with one shared
                           semi-bandit signal (see its module docstring for
                           the measured posterior collapse this causes every
                           *other* scheduler), and recovers a per-arm credit
                           from it by inclusion contrast. Blind to context:
                           mu_hat_i is one scalar per arm forever, regardless
                           of which seed produced the round.

    ContextualLinUCBScheduler: knows the best operator depends on the seed
                           being mutated, and learns a ridge regressor per
                           arm over the seed-context vector. Blind to the
                           superarm: it is wired into ``_record_outcome``
                           *outside* the shared per-operator loop precisely
                           because it has no notion of a round, and the
                           reward it is handed is the same raw broadcast
                           outcome CUCB's docstring measures collapsing under
                           a stack -- so on ``mutations_per_input`` > 1 its
                           regressor is fitting a diluted signal exactly like
                           every scalar bandit in the package.

C2UCB is neither on its own: it is LinUCB's regressor, selected and scored
by LinUCB's own rule (``theta_a.x + alpha*sqrt(x^T A_inv x)``, entirely
unmodified -- see ``contextual.py``), but *updated* from CUCB's per-round
inclusion-contrast credit instead of the raw per-pull outcome. Composition
over reimplementation: both halves are already correct and tested in
isolation, so this module is the round bookkeeping that turns one CUCB
round into per-arm ``(context, credit)`` pairs and hands each to the
existing ``ContextualLinUCBScheduler.record()`` unchanged.

Interface
---------
Same shape as CUCBScheduler's: ``record(op, x, success, weight)``
accumulates into an open round; ``settle_round()`` closes it, computing
each present arm's credit by inclusion contrast and feeding
``(x, credit)`` into that arm's ridge regressor. A caller that never calls
``settle_round()`` still behaves correctly -- the next ``select_op()``
closes any open round first, exactly as CUCB's does.

As with CUCB, a caller that has true per-arm attribution -- this fuzzer's
own ``_track_op_effect``/``_last_ops_effective`` mechanism, when enabled --
can pass it as ``settle_round(credits={op: reward})``, bypassing the
contrast entirely. This is not a minor convenience here the way it is for
CUCB: it is the difference between this scheduler actually working and
merely running. See "Context dilution", below.

Context dilution in the no-attribution path
--------------------------------------------
The inclusion contrast (``_credit``, reused byte-for-byte from
``CUCBScheduler._mu_hat`` via its ``_invert_or`` static method) estimates
one scalar per arm from *global* round statistics -- it has no notion of
context at all. Feeding that scalar to the regressor as the per-round
training target is correct on average (it is an unbiased estimate of the
arm's marginal rate) but destroys exactly the signal this scheduler exists
to capture: if arm A's true rate is 0.40 under context X and 0.05 under
context Y, the contrast converges to one blended number near the
context-weighted average of the two, and every round hands the regressor
approximately that same number regardless of which context was active.
``theta_A`` then learns almost nothing context-dependent -- confirmed
empirically: a synthetic two-context/two-arm environment (rates 0.40/0.05,
flipped by context) reaches ~54% accuracy on the correct-arm-for-context
question, indistinguishable from chance, when trained purely through the
no-attribution contrast path. The same environment reaches ~99.9% once
``settle_round(credits=...)`` supplies true per-arm attribution instead
(see ``test_c2ucb_context_dependent_arm_credits_bypass_the_contrast`` for
the reproducible version of both numbers).

This means the honest value proposition is narrower than "context-aware
CUCB" on its own: without attribution, this scheduler's regressor update
is not meaningfully better context-conditioned than feeding
``ContextualLinUCBScheduler`` the raw per-round broadcast outcome would be
-- both are context-blind at the point where context would matter, one
just also has a correctly-debiased *scale*. The scheduler earns its
existence when paired with genuine attribution, where it is strictly
better than either half alone (CUCB has nowhere to put attributed
per-arm evidence except a single running mean; ``ContextualLinUCBScheduler``
has no round/superarm model to attribute against in the first place). A
future extension -- estimating each arm's contribution to *this specific
round's* outcome from the co-present arms' own current context-conditioned
predictions, rather than from global n_out/s_out statistics -- would close
this gap for the no-attribution case too, but that is a genuinely new
piece of math (context-conditioned responsibility assignment under a
noisy-OR model) and is deliberately out of scope here; ``contrast_coverage()``
plus the dilution numbers above are meant to make the gap legible rather
than silently present it as solved.

Scope: no decay
----------------
CUCBScheduler discounts its round statistics by ``gamma`` per round, both
to model coverage-saturation drift and because its own docstring measures
the failure mode of not doing so. ``ContextualLinUCBScheduler`` has no
decay at all -- it assumes a stationary seed/operator relationship, full
stop. This scheduler follows the *contextual* half's assumption, not the
combinatorial half's: decaying a ridge regressor's design matrix while
preserving Sherman-Morrison's O(d^2) update (no direct matrix inversion on
the hot path -- the specific property ``contextual.py`` exists to have)
is a real technique (see e.g. discounted/windowed LinUCB in the
non-stationary bandit literature) but a distinct one, with its own
numerical-stability surface this module does not take on. The round-level
credit bookkeeping below is undiscounted for the same reason. If the
seed/operator relationship itself turns out to drift in practice, that is
the "SW-LinUCB" extension this module deliberately leaves for a later,
separately-measured change -- see the D-UCB/SW-UCB pair in this package
for the shape that split would take.

Why no ``rng``
--------------
``ContextualLinUCBScheduler`` -- the scheduler this one delegates
selection to -- takes no ``RandPool`` and is deliberately excluded from
``test_regression_scheduler_rand_pool.py``'s Rule 16 coverage: LinUCB's
selection is a plain ``argmax`` with no tie-break draw and no separate
"unpulled arm" branch, because an unpulled arm's confidence term is
already large under ``A_inv = (1/lambda_reg) I``. This scheduler adds no
randomness of its own on top of that -- ``select_op`` is a pure delegation
-- so it takes no ``rng`` for the same reason.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.schedulers.contextual import ContextualLinUCBScheduler
from fuzzer_tool.core.schedulers.cucb import MIN_CONTRAST_DENOM, CUCBScheduler


class C2UCBScheduler:
    """Contextual combinatorial UCB: LinUCB regressor, CUCB round credit.

    Args:
        dim: Dimensionality of the context feature vector. Forwarded to
            the internal ``ContextualLinUCBScheduler`` unchanged.
        alpha: LinUCB exploration weight. Forwarded unchanged.
        lambda_reg: LinUCB ridge regularization. Forwarded unchanged.
        min_out_rounds: Minimum out-sample mass before the inclusion
            contrast is trusted over the global-mean fallback. Same role
            as ``CUCBScheduler.min_out_rounds``; see that class's
            docstring for the degenerate-arm reasoning. Undiscounted here
            (see module docstring), so this is a plain round count rather
            than ``CUCBScheduler``'s discounted mass.
    """

    supports_priors = False

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        min_out_rounds: float = 30.0,
    ):
        if min_out_rounds < 0.0:
            raise ValueError(f"min_out_rounds must be non-negative, got {min_out_rounds!r}")
        self.dim = dim
        self.min_out_rounds = min_out_rounds

        # The regressor and its selection rule are entirely delegated;
        # this class never touches A_inv/b/scores directly.
        self._linucb = ContextualLinUCBScheduler(dim=dim, alpha=alpha, lambda_reg=lambda_reg)

        # Round bookkeeping, same shape as CUCBScheduler's but undiscounted
        # (no _discount/_renormalise -- see module docstring).
        self._n_in: dict[str, float] = {}
        self._s_in: dict[str, float] = {}
        self._n_rounds: float = 0.0
        self._s_rounds: float = 0.0
        # Open round: op -> (own credited reward this round, its context).
        self._pending: dict[str, tuple[float, np.ndarray]] = {}
        self._rounds_settled: int = 0
        self._contrast_used: int = 0
        self._fallback_used: int = 0

    # -- arm bookkeeping ------------------------------------------------------

    def init_arm(self, name: str) -> None:
        self._linucb.init_arm(name)
        self._n_in.setdefault(name, 0.0)
        self._s_in.setdefault(name, 0.0)

    # -- credit estimation (CUCBScheduler._mu_hat, undiscounted) ---------------

    def _credit(self, op: str, n_rounds: float, s_rounds: float) -> tuple[float, bool]:
        """Return (estimated per-arm credit, used_out_sample_contrast).

        Identical reasoning to ``CUCBScheduler._mu_hat`` -- see that
        docstring for why the fallback branch must land on the same scale
        as the contrast branch rather than the raw bundled mean. The
        static ``_invert_or`` solver is reused directly rather than
        copied, so a correctness fix there does not need to be mirrored
        here by hand.
        """
        n_in = self._n_in.get(op, 0.0)
        if n_in <= 0.0:
            return 0.0, False

        s_in = self._s_in.get(op, 0.0)
        mu_in = s_in / n_in
        n_out = n_rounds - n_in

        if n_out < self.min_out_rounds:
            mu_ref = s_rounds / n_rounds if n_rounds > 0.0 else 0.0
            return CUCBScheduler._invert_or(mu_in, mu_ref), False

        return CUCBScheduler._invert_or(mu_in, (s_rounds - s_in) / n_out), True

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str], context) -> str:
        """Select the next arm of the current superarm by LinUCB score.

        Closes any open round first, matching CUCBScheduler's contract:
        a caller that only ever calls select_op()/record() still gets a
        correct round boundary without an explicit settle_round().

        Args:
            ops: Candidate operator names.
            context: Forwarded to ``ContextualLinUCBScheduler.select_op``
                unchanged -- either a single feature vector shared by all
                arms, or a callable ``op -> list[float]``.
        """
        if self._pending:
            self.settle_round()
        if not ops:
            return ""
        for op in ops:
            self.init_arm(op)
        return self._linucb.select_op(ops, context)

    # -- update -----------------------------------------------------------

    def record(self, name: str, x, success: bool, weight: float = 1.0) -> None:
        """Accumulate one operator's outcome and context into the open round.

        Nothing is fed to the regressor until ``settle_round()`` -- the
        credit depends on the whole round's inclusion statistics, which
        are not known until every arm in the stack has reported in.

        Mirrors ``CUCBScheduler.record``'s max-not-sum duplicate rule: a
        stack that pulls the same arm twice credits it once, from
        whichever pull scored higher. In practice ``x`` will not differ
        across duplicates within one round (it is the context of the seed
        being mutated, not of the pull), so this rule only ever resolves
        which *reward* wins, never which context does.
        """
        self.init_arm(name)
        reward = weight if success else 0.0
        prev_reward, _ = self._pending.get(name, (-1.0, None))
        if name not in self._pending or reward > prev_reward:
            self._pending[name] = (reward, np.asarray(x, dtype=float))

    def settle_round(self, credits: dict[str, float] | None = None) -> None:
        """Close the open round: estimate credit, update each arm's regressor.

        Args:
            credits: Optional true per-arm outcomes (this fuzzer's
                ``_last_ops_effective`` attribution, when enabled). When
                given for an arm, its value replaces the inclusion-contrast
                estimate for that arm; arms present in the round but absent
                from ``credits`` still fall back to the contrast.
        """
        if not self._pending:
            return
        pending = self._pending
        self._pending = {}

        own_rewards = [own for own, _x in pending.values()]
        round_reward = max(own_rewards) if own_rewards else 0.0

        self._n_rounds += 1.0
        if round_reward:
            self._s_rounds += round_reward
        for op in pending:
            self._n_in[op] = self._n_in.get(op, 0.0) + 1.0
            if round_reward:
                self._s_in[op] = self._s_in.get(op, 0.0) + round_reward

        n_rounds = self._n_rounds
        s_rounds = self._s_rounds
        for op, (_own, x) in pending.items():
            if credits is not None and op in credits:
                credit = credits[op]
            else:
                credit, used = self._credit(op, n_rounds, s_rounds)
                if used:
                    self._contrast_used += 1
                else:
                    self._fallback_used += 1
            self._linucb.record(op, x, credit)

        self._rounds_settled += 1

    # -- diagnostics ------------------------------------------------------

    def contrast_coverage(self) -> float:
        """Fraction of credit estimates using the out-sample contrast.

        Same interpretation as ``CUCBScheduler.contrast_coverage``: near
        zero means the superarm never varies enough for the contrast to
        identify anything for the arms not passed explicit credits.
        """
        total = self._contrast_used + self._fallback_used
        return self._contrast_used / total if total else 0.0

    def bandit_stats(self) -> dict:
        """Return C2UCB diagnostics, plus the wrapped LinUCB's own."""
        inner = self._linucb.bandit_stats()
        return {
            "c2ucb_rounds": self._rounds_settled,
            "c2ucb_arms": len(self._n_in),
            "c2ucb_contrast_coverage": round(self.contrast_coverage(), 4),
            "c2ucb_dim": self.dim,
            "c2ucb_linucb_pulls": inner["contextual_pulls"],
        }


# MIN_CONTRAST_DENOM is re-exported for tests that want to construct the
# same degenerate-denominator scenario CUCBScheduler's own tests use.
__all__ = ["C2UCBScheduler", "MIN_CONTRAST_DENOM"]
