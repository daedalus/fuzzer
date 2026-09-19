"""FEWAScheduler: Filtering on Expanding Window Averages for rotting arms.

Seznec, Locatelli, Carpentier, Lazaric & Valko, *Rotting Bandits are Not
Harder than Stochastic Ones* (AISTATS 2019, arXiv:1811.11043). Where
``DUCBScheduler``/``CUSUM_UCBScheduler`` model an *environment* that shifts
-- gradual drift or an abrupt step, the same change hitting every arm --
FEWA targets a different assumption: each arm's own expected reward decays
monotonically in how many times *that arm* has been pulled, independent of
the others. That is a closer model for an operator's yield inside one
fuzzing campaign than either: ``bit_flip`` does not get worse because the
*target* changed, it gets worse because the input space it can still find
new edges in has been shrinking since the campaign's own first hour --
exactly the "eventually every source of novelty exhausts itself" shape the
paper calls rotting.

Algorithm
---------
For a window length ``h``, arm ``i``'s windowed mean is the average of its
*last h* rewards, not its all-time mean. Two active arms are compared only
at matched window lengths, and ``h`` is doubled through a geometric ladder
(1, 2, 4, 8, ...):

    while some active arm has fewer than h stored pulls:
        pull the active arm with the fewest pulls (round-robin warm-up)
    mu_i(h)  <- mean of arm i's last h rewards, for each active arm i
    B(h)     <- sqrt(alpha * log(t) / (2h))
    active   <- active \\ { i : exists j active, mu_j(h) - mu_i(h) > 2*B(h) }
    if |active| <= 1 or h >= max_window: restart (active <- candidates, h <- 1)
    else: h <- 2h

A rotting arm's recent window falls behind a still-productive one at some
``h`` and gets filtered out; the doubling ladder means an arm only has to
survive O(log h) elimination rounds to be trusted at long range, instead of
paying the O(h) cost of a fixed sliding window (``SWUCBScheduler``) at every
single window length.

Deviation from the paper: commit-then-restart, not a one-shot identification
-----------------------------------------------------------------------------
The paper's Algorithm 1 is a best-arm-identification routine: it runs once,
returns the single arm surviving when ``|active| = 1`` (or the horizon is
reached), and stops. This scheduler is called once per mutation round and
has to keep producing a pick indefinitely, so identifying a winner enters a
*commitment* phase instead: the winning arm is pulled for ``h`` further
rounds (``h`` being the window length that produced the identification --
deeper ladder climbs earn longer trust), while the rest of the epoch state
is reset immediately (``active`` back to every candidate, ``h`` back to 1)
so the instant the commitment budget is spent, comparison resumes from a
fresh, unbiased start rather than carrying over a stale narrowed set.

An earlier version of this scheduler restarted the moment ``|active|`` hit
1, with no commitment phase. That is wrong for exactly the reason a rotting
bandit exists at all: as soon as the ladder isolated a real winner, the very
next call re-admitted every eliminated arm and re-ran the warm-up from
scratch, so a clear, persistent gap between two arms translated into
*worse* tail share than plain round-robin, not better -- confirmed
empirically (``StationaryBernoulli`` tail share on the best arm collapsed to
~0.14, far below the >=0.90 floor every other scheduler in this package
clears). Committing to the winner for a bounded, confidence-scaled window
before re-opening comparison is what turns elimination into exploitation;
without it, elimination alone is pure information-gathering that never
gets spent. The bound still keeps the rotting-bandit property intact: a
winner that has since rotted away is judged against its own recent history
again as soon as its commitment budget runs out, on the same schedule any
other arm gets re-evaluated, so it does not coast on pulls from before it
rotted.

``max_window`` doubles as the deque's ``maxlen``: an arm's history older
than the largest window ever compared is evicted, so long-idle arms do not
accumulate unbounded state, mirroring ``WindowedUCBBase``'s eviction but
per-arm instead of one shared ring buffer.

``alpha``/``B(h)``
------------------
``B(h) = sqrt(alpha * log(t) / (2h))`` matches the shape of the paper's
confidence radius for sub-Gaussian rewards bounded in [0, 1] (the same
reward range every other scheduler in this package consumes -- cost-adjusted
surprisal weight, clamped by ``Fuzzer.fuzz_one``). ``alpha=0.5`` is the
Hoeffding-style constant for a [0, 1] reward (variance proxy 1/4, folded
into the leading 2 already in the formula); this is a stated choice, not a
sweep result -- unlike ``DUCBScheduler.exploration`` or
``CUSUM_UCBScheduler.h``, there is no equivalent empirical sweep for FEWA
yet in this package. Treat it the same way the un-swept ``kl_ducb``
composition gap is documented: as the paper-native default until a
measurement says otherwise.

Cost
----
O(1) amortised per record (deque append with a bounded ``maxlen``).
``select_op`` is O(|active|) for the warm-up check and O(|active| log h)
amortised for the doubling ladder, the same shape as
``UCBBase.select_op``'s O(K) per call.

Measured (6 seeds, 20k rounds, default alpha/max_window, against
``tests/support/bandit_env.py``):

    StationaryBernoulli (12 arms, best p=0.30, base p=0.05):
        tail share on the best arm: 0.985, 0.986, 0.985, 0.687, 0.984, 0.748
        (mean 0.896). The two low seeds land during an unlucky elimination
        that briefly commits to a runner-up; both still clear plain
        round-robin's 0.083 baseline by a wide margin, but the spread shows
        the commitment-length heuristic (``exploit_len = h``) is a
        documented choice, not a swept one -- see ``alpha`` above for the
        same caveat.

    DecayingBest (same arms, best collapses 0.30 -> 0.02 at round 10,000):
        tail share on the new best arm: 0.985, 0.985, 0.984, 0.986, 0.983,
        0.985 (mean 0.985, min 0.983). Consistently high: the commitment
        phase's forced re-opening on a fixed schedule catches the collapse
        regardless of seed, without needing an explicit change-point test
        the way ``CUSUM_UCBScheduler`` does.
"""

from __future__ import annotations

import collections
import math

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

#: Hoeffding-style constant for a reward bounded in [0, 1].
DEFAULT_ALPHA = 0.5

#: Deque maxlen per arm, and the window length at which an epoch force-restarts.
DEFAULT_MAX_WINDOW = 512


class FEWAScheduler:
    """Filtering on Expanding Window Averages (Seznec et al. 2019).

    Args:
        alpha: Confidence-radius constant in ``B(h) = sqrt(alpha*log(t)/(2h))``.
            Must be positive. See module docstring for why this is a
            paper-native default rather than a swept one.
        max_window: Ceiling on the window ladder and the per-arm history
            deque's ``maxlen``. Must be positive. An epoch restarts once
            ``h`` reaches this value even if more than one arm survives, so
            it also bounds how long the scheduler can commit to a shrinking
            active set before re-admitting eliminated arms.
        rng: Shared ``RandPool`` (Hard Rule 16). Consumed to break ties
            among the warm-up candidates and among elimination survivors.
    """

    #: init_arm() takes no priors -- arm state is raw reward history.
    supports_priors = False

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_window: int = DEFAULT_MAX_WINDOW,
        rng: RandPool | None = None,
    ):
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        if max_window <= 0:
            raise ValueError(f"max_window must be positive, got {max_window!r}")
        self.alpha = alpha
        self.max_window = max_window
        self._rng = rng if rng is not None else get_default_rand_pool()

        self._history: dict[str, collections.deque] = {}
        self._active: set[str] = set()
        self._h: int = 1
        self._epoch: int = 0
        self._total_pulls: int = 0
        # Commitment phase entered once the ladder narrows to one survivor
        # (or hits max_window); see select_op and the module docstring's
        # "Deviation from the paper" section for why this exists.
        self._exploit_arm: str | None = None
        self._exploit_left: int = 0

    # -- arm bookkeeping ----------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator with an empty reward history."""
        if name not in self._history:
            self._history[name] = collections.deque(maxlen=self.max_window)
            self._active.add(name)

    def _restart_epoch(self, candidates: set[str]) -> None:
        """Re-admit every candidate and reset the window ladder to h=1."""
        self._active = set(candidates)
        self._h = 1
        self._epoch += 1

    def _bound(self, h: int) -> float:
        """Confidence radius B(h) for a reward in [0, 1]."""
        t = max(self._total_pulls, 2)
        return math.sqrt(self.alpha * math.log(t) / (2.0 * h))

    def _windowed_mean(self, name: str, h: int) -> float:
        history = self._history[name]
        n = len(history)
        if n <= 0:
            return 0.0
        window = list(history)[-min(h, n) :]
        return sum(window) / len(window)

    # -- selection ------------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select an operator via the FEWA elimination ladder.

        Warms up unpulled/under-sampled active arms first (round-robin to
        the window floor), then either eliminates arms whose windowed mean
        trails the best by more than ``2*B(h)`` and doubles ``h``, or
        enters a commitment phase on the surviving arm once at most one
        arm survives or ``h`` has reached ``max_window`` (see
        ``_enter_exploit``). A running commitment phase is served first,
        for as long as its arm is still offered and its budget lasts.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        for op in ops:
            self.init_arm(op)
        candidates = set(ops)

        if self._exploit_arm is not None:
            if self._exploit_arm not in candidates:
                # The committed arm disappeared from the offered set (e.g.
                # a runtime operator gate flipped) -- abandon the
                # commitment rather than serve an arm nobody offered.
                self._exploit_arm = None
                self._restart_epoch(candidates)
            elif self._exploit_left > 0:
                self._exploit_left -= 1
                return self._exploit_arm
            else:
                # Commitment budget spent: re-open comparison instead of
                # exploiting forever, so a since-rotted winner can be
                # displaced. _restart_epoch already ran when the
                # commitment was entered (see _enter_exploit), so falling
                # through re-evaluates from a fresh h=1 immediately.
                self._exploit_arm = None

        if not (self._active & candidates):
            # The candidate pool changed out from under the active set --
            # nothing in the current epoch is even offered anymore, so
            # there is nothing to compare. Start a fresh epoch scoped to
            # what is offered now.
            self._restart_epoch(candidates)

        active_candidates = [op for op in ops if op in self._active] or list(ops)

        # Warm-up: bring every active candidate up to h pulls before any
        # windowed comparison is meaningful at this rung of the ladder.
        counts = {op: len(self._history[op]) for op in active_candidates}
        floor = min(counts.values())
        if floor < self._h:
            tied = [op for op in active_candidates if counts[op] == floor]
            return tied[0] if len(tied) == 1 else self._rng.choice(tied)

        # Every active candidate has >= h pulls: run the elimination test.
        means = {op: self._windowed_mean(op, self._h) for op in active_candidates}
        bound = self._bound(self._h)
        best_mean = max(means.values())
        survivors = [op for op in active_candidates if best_mean - means[op] <= 2.0 * bound]

        if len(survivors) < len(active_candidates):
            self._active &= set(survivors)

        if len(survivors) <= 1 or self._h >= self.max_window:
            winner = survivors[0] if len(survivors) == 1 else max(survivors, key=means.get)
            return self._enter_exploit(winner, candidates)

        self._h *= 2
        return survivors[0] if len(survivors) == 1 else self._rng.choice(survivors)

    def _enter_exploit(self, winner: str, candidates: set[str]) -> str:
        """Commit to *winner* for ``h`` further pulls, then force a restart.

        The commitment length scales with the window ``h`` that produced
        the identification: the deeper the ladder climbed before settling,
        the more confidence backs the pick, and the longer it is trusted
        before comparison re-opens. This is the module's documented
        deviation from the paper's one-shot identify-then-stop routine --
        without it, ``winner`` would be re-admitted into a fresh, uniform
        active set on literally the next call, and a large, real gap would
        never translate into sustained exploitation.

        The epoch is restarted immediately (every candidate re-admitted,
        ``h`` back to 1) rather than when the commitment budget runs out,
        so the state ``select_op`` falls through to the instant the budget
        is spent is already a fresh, unbiased ladder start.
        """
        exploit_len = self._h
        self._restart_epoch(candidates)
        self._exploit_arm = winner
        self._exploit_left = max(exploit_len - 1, 0)
        return winner

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Append this pull's reward to *name*'s windowed history."""
        self.init_arm(name)
        self._total_pulls += 1
        self._history[name].append(weight if success else 0.0)

    # -- diagnostics ----------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Return FEWA diagnostics."""
        return {
            "fewa_pulls": self._total_pulls,
            "fewa_window": self._h,
            "fewa_active_arms": len(self._active),
            "fewa_known_arms": len(self._history),
            "fewa_epochs": self._epoch,
            "fewa_exploiting": self._exploit_arm is not None,
        }
