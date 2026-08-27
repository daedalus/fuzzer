"""DUCBScheduler: Discounted UCB for abruptly-changing reward distributions.

Garivier & Moulines, *On Upper-Confidence Bound Policies for Switching Bandit
Problems* (arXiv:0805.3415). Every statistic is an exponentially discounted
sum, so evidence from before a change point decays out of the index on its own:

    N_t(i) = sum_{s<=t} gamma^(t-s) * 1{I_s = i}
    X_t(i) = sum_{s<=t} gamma^(t-s) * r_s * 1{I_s = i}
    n_t    = sum_i N_t(i)

    index(i) = X_t(i)/N_t(i) + 2B*sqrt(xi * log(n_t) / N_t(i))

Why this and not ``arm_decay``
------------------------------
``MonteCarloScheduler`` and ``HierarchicalBanditScheduler`` already multiply
their posteriors by ``arm_decay`` every ``decay_interval`` records. That is a
coarser version of the same idea and it is measurably too weak: at the default
0.999 per 100 pulls it removes about 6% of the accumulated mass over a
6000-pull campaign, which is why ``MonteCarloScheduler`` sits in the ``STUCK``
set of ``tests/test_scheduler_convergence.py`` -- it never leaves an arm whose
yield has collapsed. D-UCB discounts on every record and puts the discount
inside the confidence width as well as the mean, so an abandoned arm's width
re-widens and the arm is retried. That second half is what ``arm_decay``
cannot express.

Cost
----
The naive form is O(K) per record: every arm's statistics decay each round.
This stores values relative to a single global discount factor, exactly as
``Exp3Scheduler`` does with its weights, so a record is O(1) and the O(K)
renormalisation sweep only runs as the factor approaches underflow.

Bundled rewards
---------------
``services/fuzzer.py::_record_outcome`` calls ``record()`` once per operator
per round, so a "round" here is one pull, not one mutation round. With
``mutations_per_input = 8`` the effective horizon of gamma is ~8x shorter in
rounds than it looks in pulls. ``CUCBScheduler`` models the round as a unit;
this one deliberately does not.
"""

import math

from fuzzer_tool.core.rand_pool import RandPool

#: Below this the relative statistics are rescaled back to an absolute basis.
#: 1e-12 leaves ~4 orders of float64 headroom above the point where
#: 1/_discount stops being exactly representable.
RENORM_FLOOR = 1e-12

#: The index is only defined for n_t >= 1. Clamping the log argument is what
#: keeps the width real rather than NaN over the first few pulls.
MIN_LOG_ARG = 1.0 + 1e-9

#: Paper's leading coefficient on the confidence width.
UCB_WIDTH_COEFF = 2.0


class DUCBScheduler:
    """Discounted UCB (Garivier & Moulines) over mutation operators.

    Args:
        gamma: Discount factor per record, in (0, 1]. 1.0 degenerates to
            UCB1. The default 0.9999 gives an effective memory of
            1/(1-gamma) = 10,000 pulls, ~1,250 mutation rounds at the default
            ``mutations_per_input = 8``. The horizon has to stay well above
            the number of candidate arms or every arm's discounted count sits
            near zero and the index is pure exploration: with 147 registered
            operators, 10,000 pulls is ~68 per arm.
        xi: Exploration constant inside the confidence width.
        b: Reward range. Rewards handed to ``record()`` are cost-adjusted
            surprisal weights, which ``_cost_adjusted_weight`` keeps in [0, 1].
        exploration: Multiplier on the whole confidence width. The paper's
            index has a hard 2B in front, which is a large over-exploration at
            this reward scale -- the same finding ``GPUCBScheduler`` already
            records for its ``beta``. Measured tail share on the best arm,
            12 arms, 20k rounds, 3 seeds, as
            stationary | live-arm-after-decay | dead-arm-after-decay:

                gamma   coef 2.0        coef 1.0        coef 0.5        coef 0.25
                0.995   0.141|0.11|.08  0.239|0.16|.07  0.516|0.26|.05  0.787|0.59|.03
                0.999   0.261|0.15|.07  0.545|0.31|.05  0.829|0.63|.02  0.949|0.88|.01
                0.9995  0.358|0.18|.06  0.680|0.41|.05  0.904|0.75|.02  0.959|0.92|.01
                0.9999  0.691|0.34|.11  0.895|0.64|.12  0.979|0.91|.03  0.992|0.97|.02

            At the paper's own coefficient the index is barely above the
            1/12 = 0.083 uniform baseline.
        rng: Shared ``RandPool`` (Hard Rule 16). Only consumed when opening
            an arm with no discounted evidence.
    """

    # Beta-Bernoulli priors have no meaning for a discounted-sum index: there
    # is no (alpha, beta) pair to seed.
    supports_priors = False

    def __init__(
        self,
        gamma: float = 0.9999,
        xi: float = 0.6,
        b: float = 1.0,
        exploration: float = 0.25,
        rng: RandPool | None = None,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")
        if exploration <= 0.0:
            raise ValueError(f"exploration must be positive, got {exploration!r}")

        self.gamma = gamma
        self.xi = xi
        self.b = b
        self.exploration = exploration

        # Hard Rule 16: all randomness comes from RandPool, so --seed
        # determines which unpulled arm is opened first and a crash found
        # under this scheduler replays. Same convention as cmaes.py.
        self._rng = rng if rng is not None else RandPool()

        # Relative statistics: the true discounted value is the stored value
        # times self._discount. Keeping the discount in one place is what
        # makes record() O(1) instead of O(K).
        self._n_rel: dict[str, float] = {}
        self._x_rel: dict[str, float] = {}
        self._discount: float = 1.0
        self._total_pulls: int = 0

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator with zero discounted count and reward."""
        self._n_rel.setdefault(name, 0.0)
        self._x_rel.setdefault(name, 0.0)

    def _renormalise(self) -> None:
        """Fold the accumulated discount back into the per-arm statistics."""
        d = self._discount
        for k in self._n_rel:
            self._n_rel[k] *= d
        for k in self._x_rel:
            self._x_rel[k] *= d
        self._discount = 1.0

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the operator with the highest discounted-UCB index.

        Arms with zero discounted count are opened first. That is the standard
        UCB initialisation, and it also handles operators registered at runtime
        via ``REGISTRY.register_mutator()``: an arm this scheduler has never
        seen is indistinguishable from one whose evidence has fully decayed,
        and both should be tried.
        """
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        d = self._discount

        # n_t is the discounted total over the *candidate* arms, matching the
        # index's own denominator. Summing over every registered arm instead
        # would inflate log(n_t) on every build_ops() call that filters the
        # operator list by sniffer applicability.
        n_total = 0.0
        unpulled = []
        for op in ops:
            n = self._n_rel.get(op, 0.0) * d
            if n <= 0.0:
                unpulled.append(op)
                continue
            n_total += n

        if unpulled:
            return self._rng.choice(unpulled)

        log_n = math.log(max(n_total, MIN_LOG_ARG))
        width_scale = self.exploration * UCB_WIDTH_COEFF * self.b * math.sqrt(self.xi * log_n)

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._n_rel.get(op, 0.0) * d
            mean = (self._x_rel.get(op, 0.0) * d) / n
            score = mean + width_scale / math.sqrt(n)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Discount every arm, then credit *name* with this round's reward."""
        self._total_pulls += 1
        reward = weight if success else 0.0

        if self.gamma < 1.0:
            self._discount *= self.gamma
            if self._discount < RENORM_FLOOR:
                self._renormalise()

        # Adding an absolute 1 to a value stored relative to _discount means
        # adding 1/_discount in relative space.
        inv = 1.0 / self._discount
        self._n_rel[name] = self._n_rel.get(name, 0.0) + inv
        if reward:
            self._x_rel[name] = self._x_rel.get(name, 0.0) + reward * inv

    # -- diagnostics ------------------------------------------------------

    def discounted_counts(self) -> dict[str, float]:
        """Per-arm discounted pull count N_t(i), in absolute units."""
        d = self._discount
        return {k: v * d for k, v in self._n_rel.items()}

    def discounted_means(self) -> dict[str, float]:
        """Per-arm discounted empirical mean X_t(i)/N_t(i)."""
        d = self._discount
        out = {}
        for k, n_rel in self._n_rel.items():
            n = n_rel * d
            out[k] = (self._x_rel.get(k, 0.0) * d / n) if n > 0 else 0.0
        return out

    def bandit_stats(self) -> dict:
        """Return D-UCB diagnostics."""
        counts = self.discounted_counts()
        return {
            "ducb_pulls": self._total_pulls,
            "ducb_effective_n": round(sum(counts.values()), 3),
            "ducb_arms": len(self._n_rel),
        }
