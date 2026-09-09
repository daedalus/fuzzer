"""CUSUM_UCBScheduler: change-point-detection UCB for abruptly-changing arms.

Liu, Lee & Shroff, *A Change-Detection based Framework for Piecewise-
Stationary Multi-Armed Bandit Problem* (AAAI 2018, arXiv:1711.03539). Where
``DUCBScheduler``/``SWUCBScheduler`` (Garivier & Moulines) assume drift is
continuous and forget the past on a fixed schedule -- a gamma decay or a
sliding window that keeps ticking whether or not anything actually changed --
this scheduler runs an explicit statistical test per arm and only forgets when
the test fires:

    warm-up:  mu0_a <- mean of arm a's first M pulls since the last reset
    per pull: s_pos = x_t - mu0_a - epsilon      s_neg = mu0_a - x_t - epsilon
              g_pos <- max(0, g_pos + s_pos)      g_neg <- max(0, g_neg + s_neg)
    trigger:  g_pos > h  or  g_neg > h  =>  change detected

    index(i) = mean_i(t-tau) + b*sqrt(xi * log(t-tau) / N_i(t-tau))

``tau`` is the last reset time. A detection in *any* arm resets *every* arm's
statistics globally, on the assumption -- shared with the paper's own
experiments -- that what moved is the environment, not one arm in isolation.
For fuzzing that assumption is the realistic one: a corpus discovering a new
coverage island, a target flag flipping a code path, or a schedule ablation
changing the mutation mix all shift every operator's yield at once, not just
the operator that happened to trigger the observation.

Why this and not D-UCB/SW-UCB
------------------------------
Both existing switching-bandit schedulers forget continuously, which is the
right model for *gradual* drift (coverage saturation decaying an operator's
yield over thousands of pulls -- exactly ``DecayingBest`` in
``tests/support/bandit_env.py``) but the wrong one for a *step* change: they
pay a constant "recency tax" even on a perfectly stationary target, because
D-UCB's discount and SW-UCB's window both erase old evidence on a clock that
runs regardless of whether the environment moved. CUSUM-UCB pays that tax
only after evidence of an actual change accumulates, so on a stationary
target it converges close to plain UCB1 and only starts forgetting once
something has demonstrably moved.

The trade-off is detection delay: CUSUM-UCB needs ``M`` samples per arm to
re-establish a baseline before it can detect anything again, so it reacts to
a step change more slowly than SW-UCB reacts once its window has slid past
the change point. Which one wins is the same "empirical question per target"
call already made for D-UCB vs SW-UCB.

Defaults
--------
``m=30, epsilon=0.1, h=40.0, xi=0.6`` were picked by the same kind of sweep
``DUCBScheduler.exploration`` documents, against the same two environments.
Measured over 20 independent seeds, 20k rounds each:

    StationaryBernoulli (12 arms, best p=0.30, base p=0.05):
        7 false resets total across 20 campaigns (~1 per 57k pulls),
        mean best-arm tail share 0.983 (uniform baseline: 0.083).

    DecayingBest (same arms, best collapses 0.30 -> 0.02 at round 10,000):
        every one of the 20 campaigns detected the collapse (>=1 reset),
        mean tail share on the new-best arm 0.890, mean tail share
        remaining on the collapsed arm 0.007.

``h`` was swept from 8.0 (17 false resets per 5 campaigns, unusable) up to
80.0 (no further reduction in false-reset rate, so no benefit to going
higher); 40.0 sits past the point of diminishing returns on false positives
without measurably hurting detection sensitivity on ``DecayingBest``. It is
the one paper-native free parameter with no closed form for a target reward
scale that is not literally Bernoulli-in-[0,1] with a known gap -- see
``epsilon``, below, for why the same is true of the CUSUM margin. Both are
therefore tuned empirically rather than derived, exactly as ``exploration``
is for D-UCB. Re-run the sweep (``StationaryBernoulli``/``DecayingBest`` in
``tests/support/bandit_env.py``) before changing either on a target whose
reward distribution looks materially different from cost-adjusted surprisal
in [0, 1].

``epsilon`` is the minimum mean shift the test is built to catch; it must
stay below the true gap between pre- and post-change means or the test never
fires, and above the noise floor of a stationary run or it fires
spontaneously. The reward scale here is the same cost-adjusted surprisal
weight in [0, 1] that every other scheduler in this package consumes, so
``epsilon=0.1`` is one tenth of the full reward range -- the same role
``DUCBScheduler.b`` plays for its Gaussian width, chosen for the same
reason: no other scale-free default is available without knowing the
target's actual gap in advance.

Cost
----
O(1) per record outside a reset: one comparison against ``m``, two
CUSUM updates, two threshold checks. A reset is O(K) (every arm's state
zeroed), the same amortised cost ``DiscountedUCBBase._renormalise`` pays,
and firing on every arm's *first* post-reset pull is intentionally avoided by
only testing the CUSUM once ``n > m`` (see ``_arm_update``).
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.ucb_common import MIN_LOG_ARG, UCBBase

#: Reward range default, matching DUCBScheduler.b's role.
DEFAULT_B = 1.0


class CUSUM_UCBScheduler(UCBBase):
    """CUSUM-based change-point-detection UCB (Liu, Lee & Shroff 2018).

    Args:
        m: Warm-up pull count per arm used to estimate the pre-change mean
            ``mu0``. Must be positive. No CUSUM statistic is computed for an
            arm until it has been pulled more than ``m`` times since the
            last reset, so a larger ``m`` gives a more stable baseline at
            the cost of a longer blind spot right after every reset.
        epsilon: Minimum-detectable-mean-shift margin subtracted from both
            CUSUM slopes. Must be positive. See module docstring for the
            trade-off against the reward scale.
        h: CUSUM decision threshold. Must be positive. Lower triggers
            resets more eagerly (shorter detection delay, more false
            positives on a stationary target); higher is the reverse.
        xi: Exploration constant inside the post-reset UCB1-style
            confidence width.
        b: Reward range, matching ``DUCBScheduler.b``'s role. Rewards
            handed to ``record()`` are cost-adjusted surprisal weights,
            which stay in [0, 1] by construction.
        rng: Shared ``RandPool`` (Hard Rule 16). Only consumed when opening
            an arm with zero pulls since the last reset.
    """

    supports_priors = False

    def __init__(
        self,
        m: int = 30,
        epsilon: float = 0.1,
        h: float = 40.0,
        xi: float = 0.6,
        b: float = DEFAULT_B,
        rng: RandPool | None = None,
    ):
        if m <= 0:
            raise ValueError(f"m must be positive, got {m!r}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}")
        if h <= 0.0:
            raise ValueError(f"h must be positive, got {h!r}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")
        self.m = m
        self.epsilon = epsilon
        self.h = h
        self.xi = xi
        self.b = b

        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        # None until the arm's baseline mean is frozen at n == m.
        self._mu0: dict[str, float | None] = {}
        self._g_pos: dict[str, float] = {}
        self._g_neg: dict[str, float] = {}
        self._reset_count: int = 0

        super().__init__(rng=rng)

    # -- arm bookkeeping ----------------------------------------------------

    def init_arm(self, name: str) -> None:
        self._counts.setdefault(name, 0)
        self._sums.setdefault(name, 0.0)
        self._mu0.setdefault(name, None)
        self._g_pos.setdefault(name, 0.0)
        self._g_neg.setdefault(name, 0.0)

    # -- UCBBase hooks --------------------------------------------------------

    def _arm_count(self, op: str) -> float:
        return float(self._counts.get(op, 0))

    def _arm_mean(self, op: str, n: float) -> float:
        if n <= 0.0:
            return 0.0
        return self._sums.get(op, 0.0) / n

    def _width(self, mean: float, n: float, log_n: float) -> float:  # noqa: ARG002
        """UCB1-style Gaussian width, reset relative: b*sqrt(xi*log_n / n).

        ``log_n`` is ``log(t - tau)`` for free: ``UCBBase.select_op`` sums
        ``_arm_count`` over the candidate arms to compute it, and every
        arm's count is zeroed by the same global reset, so that sum is
        exactly the pulls elapsed since ``tau`` with no separate clock
        needed.
        """
        return self.b * math.sqrt(self.xi * max(log_n, 0.0) / n)

    def _arm_update(self, name: str, reward: float) -> None:
        self.init_arm(name)
        self._counts[name] += 1
        self._sums[name] += reward
        n = self._counts[name]

        if n < self.m:
            return
        if n == self.m:
            self._mu0[name] = self._sums[name] / n
            return

        mu0 = self._mu0.get(name)
        if mu0 is None:  # pragma: no cover - defensive, see class docstring
            # Should not happen once n > m outside a corrupted partial
            # reset; re-freeze rather than propagate a None into arithmetic.
            self._mu0[name] = self._sums[name] / n
            return

        s_pos = reward - mu0 - self.epsilon
        s_neg = mu0 - reward - self.epsilon
        self._g_pos[name] = max(0.0, self._g_pos.get(name, 0.0) + s_pos)
        self._g_neg[name] = max(0.0, self._g_neg.get(name, 0.0) + s_neg)

        if self._g_pos[name] > self.h or self._g_neg[name] > self.h:
            self._reset_all()

    def _reset_all(self) -> None:
        """Global reset: every arm's statistics zeroed, one arm's evidence."""
        for k in self._counts:
            self._counts[k] = 0
            self._sums[k] = 0.0
            self._mu0[k] = None
            self._g_pos[k] = 0.0
            self._g_neg[k] = 0.0
        self._reset_count += 1

    # -- diagnostics ----------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Return CUSUM-UCB diagnostics."""
        pulls_since_reset = sum(self._counts.values())
        log_n = math.log(max(pulls_since_reset, MIN_LOG_ARG))
        return {
            "cusum_pulls": self._total_pulls,
            "cusum_resets": self._reset_count,
            "cusum_arms": len(self._counts),
            "cusum_pulls_since_reset": pulls_since_reset,
            "cusum_log_n": round(log_n, 3),
        }
