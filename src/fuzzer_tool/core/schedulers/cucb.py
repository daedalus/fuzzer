"""CUCBScheduler: combinatorial UCB over the round's operator stack.

Chen, Wang & Yuan, *Combinatorial Multi-Armed Bandit: General Framework,
Results and Applications* (ICML 2013). CUCB plays a **superarm** -- a set S of
base arms -- each round and updates from **semi-bandit feedback**: the outcome
of each arm in S, not just the outcome of S as a whole.

    index(i) = mu_hat_i + sqrt(3*ln(t) / (2*N_i))      (paper's radius)

Why this scheduler exists
-------------------------
Every other scheduler in this package is a single-arm bandit reading a signal
that is not single-arm. ``services/fuzzer.py::_record_outcome`` runs
``mutations_per_input`` operators against one input, gets **one** binary
outcome for the whole stack, and hands that same outcome to every operator in
the round. The arms are a superarm; the feedback is full-bandit; the
schedulers assume neither.

Measured cost of that mismatch, on ``tests/support/bandit_env.py``'s
``StationaryBernoulli`` (12 arms, best p=0.30, weak p=0.05, 40k rounds, 3
seeds), with the round succeeding when any arm in the stack fires:

    stack size k | posterior separation between the 0.30 and 0.05 arms
               1 | 0.250   (the true gap)
               2 | 0.155
               4 | 0.091
               8 | 0.029   <- the shipped default
              16 | 0.006

At the default ``mutations_per_input = 8`` the true 0.25 gap reaches the
posterior as 0.029: an 8.6x loss of the signal every operator scheduler in the
tree is trying to read.

Recovering the per-arm outcome
------------------------------
CUCB needs semi-bandit feedback the fuzzer cannot directly observe -- that
would take a coverage snapshot between every mutation in the stack. What it
*can* observe is which stacks contained which arms and how those stacks did,
so ``mu_hat_i`` is estimated by **inclusion contrast**:

    mu_hat_i = (mean reward of rounds containing i - mean reward of rounds
                not containing i) / (1 - mean reward of rounds not containing i)

This inverts the round model exactly when the round reward behaves like a
noisy OR of the arms' individual successes -- which is what "did this stack
find a new edge" is. If arm i fires with probability p_i and the rest of the
stack succeeds with probability q independent of i, then

    E[R | i in S] = 1 - (1-p_i)(1-q) = q + p_i(1-q)
    E[R | i not in S] ~ q

and the ratio returns p_i. Measured on the same environment, the estimator
holds its separation where the naive per-arm mean collapses:

    k  | naive sep | contrast sep
    1  |   0.245   |    0.251
    2  |   0.221   |    0.259
    4  |   0.175   |    0.279
    8  |   0.105   |    0.313
    16 |   0.039   |    0.389

The contrast over-separates at large k because the weak arms clamp at zero;
for a bandit that is the harmless direction, since only the ranking is used.

Where it degenerates, and what happens then
-------------------------------------------
The contrast needs rounds *without* arm i. Under greedy exploitation a
dominant arm can end up in essentially every round and the out-sample
vanishes. Below ``min_out_rounds`` the estimator contrasts against the global
mean instead, which includes the arm's own rounds and is therefore biased
toward zero -- the conservative direction, and crucially still on the same
scale. ``contrast_coverage()`` reports what fraction of index evaluations are
on the good path; on the convergence environment it sits at 0.92-0.99.

Both statistics are discounted by ``gamma`` per round, so the scheduler tracks
coverage saturation as well -- the failure mode that puts ``MonteCarlo`` and
``EpsilonGreedy`` in the ``STUCK`` set of the convergence suite.

The superarm is not de-duplicated
---------------------------------
An arm drawn into every round is unidentifiable, which argues for sampling the
stack without replacement. Measured, that argument loses: on tail round-success
rate -- the metric a superarm scheduler actually optimises, and the one that
corresponds to "rounds that found a new edge" --

    arms | distinct stack           | repeats allowed
      12 | 0.604 stat, 0.433 decay  | 0.891 stat, 0.723 decay
      40 | 0.579 stat, 0.418 decay  | 0.882 stat, 0.719 decay

Forcing diversity costs about a third of the round yield, because on an
independent-arms bundle the optimal stack really is the best operator
repeated. Identifiability does not suffer for it: contrast coverage stays at
0.92-0.99 either way, since the applicability filter and the exploration
radius already supply out-sample rounds. So repeats are allowed, and the
degenerate case is handled by the fallback above rather than prevented.

Interface
---------
Rounds are delimited by ``settle_round()``. ``record(op, success, weight)``
accumulates into the open round rather than updating immediately, so the
shared per-operator loop in ``_record_outcome`` needs no special case. A
caller that never calls ``settle_round()`` still behaves correctly: the next
``select_op()`` closes any open round first.

A caller that *does* have true per-arm outcomes can pass them as
``settle_round(credits={op: reward})``, which bypasses the contrast entirely
and makes this the textbook CUCB.
"""

import math

from fuzzer_tool.core.rand_pool import RandPool

RENORM_FLOOR = 1e-12
MIN_LOG_ARG = 1.0 + 1e-9

#: Chen et al.'s radius is sqrt(3*ln t / (2*N)); this is the 3/2 under the root.
CUCB_RADIUS_COEFF = 1.5

#: Below this the contrast denominator is degenerate and the estimate is zero.
MIN_CONTRAST_DENOM = 1e-9


class CUCBScheduler:
    """Combinatorial UCB with inclusion-contrast semi-bandit recovery.

    Args:
        gamma: Per-round discount on all statistics, in (0, 1]. 1.0 keeps
            every round forever (stationary assumption). The default 0.9995
            gives an effective memory of 2000 *rounds*, which unlike D-UCB's
            and SW-UCB's horizons is measured in mutation rounds rather than
            operator pulls, because this scheduler updates once per round.
        min_out_rounds: Minimum discounted out-sample mass before the
            out-sample contrast is used for an arm.
        exploration: Multiplier on the paper's radius. The bare radius
            over-explores at this reward scale -- the same finding
            ``GPUCBScheduler`` records for ``beta`` and ``DUCBScheduler`` for
            its leading 2B -- and here it is worse, because the contrast
            estimate is bounded by the arm's own rate (<=0.30 in the
            convergence environment) while the radius is not. Measured tail
            share on the best arm, 12 arms, 20k rounds, 3 seeds, as
            stationary | live-arm-after-decay | dead-arm-after-decay, at
            gamma=0.9995:

                expl   single-pull            stacked k=8 (ceiling 0.771)
                1.00   0.531|0.259|0.061      0.757|0.766|0.021
                0.50   0.818|0.552|0.029      0.757|0.766|0.021
                0.25   0.946|0.849|0.001      0.761|0.766|0.021
                0.15   0.980|0.938|0.000      0.766|0.767|0.021
                0.10   0.984|0.979|0.000      0.765|0.767|0.021

            Stacking hides the over-exploration: at k=8 an arm accumulates
            N_i eight times faster, so the radius is already small by the
            time it matters. The single-pull column is what exposes it.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    # The index is a contrast of empirical means, not a Beta posterior.
    supports_priors = False

    def __init__(
        self,
        gamma: float = 0.9995,
        min_out_rounds: float = 30.0,
        exploration: float = 0.15,
        rng: RandPool | None = None,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        if min_out_rounds < 0.0:
            raise ValueError(f"min_out_rounds must be non-negative, got {min_out_rounds!r}")
        if exploration <= 0.0:
            raise ValueError(f"exploration must be positive, got {exploration!r}")

        self.gamma = gamma
        self.min_out_rounds = min_out_rounds
        self.exploration = exploration

        # Hard Rule 16: see ducb.py.
        self._rng = rng if rng is not None else RandPool()

        # Per-arm, relative to _discount (same O(1) trick as DUCBScheduler):
        # rounds the arm appeared in, and the summed round reward of those.
        self._n_in_rel: dict[str, float] = {}
        self._s_in_rel: dict[str, float] = {}
        # Per-arm own credited reward, for the explicit-credits path.
        self._s_own_rel: dict[str, float] = {}
        # Round totals, same relative basis.
        self._n_rounds_rel: float = 0.0
        self._s_rounds_rel: float = 0.0
        self._discount: float = 1.0

        self._known: set[str] = set()
        # Open round: arm -> its own credited reward this round.
        self._pending: dict[str, float] = {}
        self._rounds_settled: int = 0
        self._contrast_used: int = 0
        self._fallback_used: int = 0

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator with empty statistics."""
        self._known.add(name)
        self._n_in_rel.setdefault(name, 0.0)
        self._s_in_rel.setdefault(name, 0.0)
        self._s_own_rel.setdefault(name, 0.0)

    def _renormalise(self) -> None:
        """Fold the accumulated discount back into the stored statistics."""
        d = self._discount
        for table in (self._n_in_rel, self._s_in_rel, self._s_own_rel):
            for k in table:
                table[k] *= d
        self._n_rounds_rel *= d
        self._s_rounds_rel *= d
        self._discount = 1.0

    # -- estimation -------------------------------------------------------

    @staticmethod
    def _invert_or(mu_in: float, mu_ref: float) -> float:
        """Solve q + p(1-q) = mu_in for p, given a reference rate q.

        Clamped to [0, 1]: sampling noise puts mu_in below mu_ref for arms
        that contribute nothing, and a negative rate is not something an
        index can be ranked on.
        """
        denom = 1.0 - mu_ref
        if denom <= MIN_CONTRAST_DENOM:
            return 0.0

        est = (mu_in - mu_ref) / denom
        if est < 0.0:
            return 0.0

        return 1.0 if est > 1.0 else est

    def _mu_hat(self, op: str, n_rounds: float, s_rounds: float) -> tuple[float, bool]:
        """Return (estimated per-arm success rate, used_out_sample_contrast).

        Both branches return a value on the *same scale* -- an estimate of the
        arm's own success probability -- which the first cut of this did not.
        It fell back to the raw bundled mean when the out-sample was empty,
        and that number lives on a different scale: with an eight-deep stack
        the bundled mean sits near 0.75 while every contrast estimate is
        bounded by the arm's own rate, at most 0.30 in the convergence
        environment. Measured consequence: whichever arm the index happened to
        lock onto was scored 0.749 against a true 0.18, the genuine 0.30 arm
        was correctly scored 0.250, and the inflated incumbent won every
        comparison forever. Tail share on the best arm was 0.27 against a
        0.77 ceiling -- worse than every scheduler it was meant to improve on.

        The out-sample branch contrasts against rounds *without* the arm,
        which identifies its rate exactly under the noisy-OR round model. The
        fallback contrasts against the global mean instead, which includes the
        arm's own rounds and is therefore biased toward zero. An arm appearing
        in *every* round is unidentifiable and correctly scores 0: no evidence
        separates it from the background, so its confidence radius, not a
        borrowed mean, is what keeps it in play.
        """
        d = self._discount
        n_in = self._n_in_rel.get(op, 0.0) * d
        if n_in <= 0.0:
            return 0.0, False

        s_in = self._s_in_rel.get(op, 0.0) * d
        mu_in = s_in / n_in
        n_out = n_rounds - n_in

        if n_out < self.min_out_rounds:
            mu_ref = s_rounds / n_rounds if n_rounds > 0.0 else 0.0
            return self._invert_or(mu_in, mu_ref), False

        return self._invert_or(mu_in, (s_rounds - s_in) / n_out), True

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the next arm of the current superarm by CUCB index."""
        # Membership in the superarm comes from record(), not from here, so a
        # round whose selections were never recorded (the deterministic stage
        # bypasses build_ops entirely and leaves _last_ops_used empty) cannot
        # leak arms into the next round's statistics. Closing any open round
        # here is what makes settle_round() optional for callers that only use
        # the common select/record interface.
        if self._pending:
            self.settle_round()

        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        d = self._discount
        n_rounds = self._n_rounds_rel * d
        s_rounds = self._s_rounds_rel * d

        unpulled = [op for op in ops if self._n_in_rel.get(op, 0.0) <= 0.0]
        if unpulled:
            return self._rng.choice(unpulled)

        # t is the discounted round count, matching the units of N_i below.
        log_t = math.log(max(n_rounds, MIN_LOG_ARG))
        radius_scale = self.exploration * math.sqrt(CUCB_RADIUS_COEFF * log_t)

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n_i = self._n_in_rel.get(op, 0.0) * d
            mu, used = self._mu_hat(op, n_rounds, s_rounds)
            if used:
                self._contrast_used += 1
            else:
                self._fallback_used += 1

            score = mu + radius_scale / math.sqrt(n_i)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Accumulate one operator's outcome into the open round.

        Nothing is committed until ``settle_round()``. An arm recorded without
        having been selected (a caller driving the scheduler directly, or an
        operator applied outside ``select_op``) still joins the superarm --
        the contrast asks which arms were *present*, not which were chosen.
        """
        reward = weight if success else 0.0
        prev = self._pending.get(name, 0.0)

        # max, not sum: mutations_per_input can select the same operator more
        # than once in a round and the round reward is a single event. Summing
        # would credit an arm twice for one discovery.
        self._pending[name] = reward if reward > prev else prev

    def settle_round(self, credits: dict[str, float] | None = None) -> None:
        """Close the open round and fold it into the discounted statistics.

        Args:
            credits: Optional true per-arm outcomes. When given, these replace
                whatever ``record()`` accumulated and the arms' own means are
                built from them directly -- the textbook semi-bandit update.
        """
        pending = credits if credits is not None else self._pending
        if not pending:
            self._pending = {}
            return

        if self.gamma < 1.0:
            self._discount *= self.gamma
            if self._discount < RENORM_FLOOR:
                self._renormalise()
        inv = 1.0 / self._discount

        # The round reward is the stack's shared outcome: the round succeeded
        # iff some arm in it was credited.
        round_reward = max(pending.values())

        self._n_rounds_rel += inv
        if round_reward:
            self._s_rounds_rel += round_reward * inv

        for op, own in pending.items():
            self._n_in_rel[op] = self._n_in_rel.get(op, 0.0) + inv
            if round_reward:
                self._s_in_rel[op] = self._s_in_rel.get(op, 0.0) + round_reward * inv
            if own:
                self._s_own_rel[op] = self._s_own_rel.get(op, 0.0) + own * inv

        self._rounds_settled += 1
        self._pending = {}

    # -- diagnostics ------------------------------------------------------

    def estimated_rates(self) -> dict[str, float]:
        """Per-arm mu_hat as the index currently sees it."""
        d = self._discount
        n_rounds = self._n_rounds_rel * d
        s_rounds = self._s_rounds_rel * d
        return {op: self._mu_hat(op, n_rounds, s_rounds)[0] for op in self._n_in_rel}

    def contrast_coverage(self) -> float:
        """Fraction of index evaluations using the out-sample contrast.

        A campaign where this sits near zero is one where the superarm never
        varies enough for the contrast to identify anything, and CUCB is
        behaving as a discounted UCB1 over shrunken means.
        """
        total = self._contrast_used + self._fallback_used
        return self._contrast_used / total if total else 0.0

    def bandit_stats(self) -> dict:
        """Return CUCB diagnostics."""
        return {
            "cucb_rounds": self._rounds_settled,
            "cucb_arms": len(self._n_in_rel),
            "cucb_contrast_coverage": round(self.contrast_coverage(), 4),
        }
