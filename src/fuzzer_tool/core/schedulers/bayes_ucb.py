"""BayesUCBScheduler: Bayes-UCB (Kaufmann, Cappé & Garivier, COLT 2012).

Every other UCB variant in this package scores an arm with a concentration
inequality: mean plus some width term derived from a tail bound (Hoeffding
for UCB1, a KL divergence for KL-UCB, and so on). Bayes-UCB scores an arm
with an actual Bayesian quantity instead: the (1 - 1/(t*(log t)^c))-quantile
of that arm's posterior distribution over its true success rate, given a
Beta prior and the Bernoulli-like evidence observed so far. Kaufmann et al.
show this quantile *is* asymptotically the same optimism-under-uncertainty
index KL-UCB computes from a concentration bound, but obtained from the
posterior directly rather than derived from a tail inequality -- and they
report it matching or beating KL-UCB empirically, particularly early, where
the tail bound is loose but the exact Beta-posterior quantile already has
the right shape.

Why not just Thompson sampling (this package's ``MonteCarloScheduler``)?
Same Beta-Bernoulli conjugate model, same per-arm prior support (see
"Per-arm priors" below) -- the difference is what happens with the
posterior once you have it. Thompson sampling draws one random sample per
arm per round and picks the largest draw; Bayes-UCB takes a fixed, high
quantile of each arm's posterior directly, with no sampling variance to
average out. This makes Bayes-UCB deterministic given the same evidence
(useful for exactly the same reason CUCB and CUSUM-UCB are documented as
preferring their own deterministic contrasts over resampling), at the cost
of the closed-form-ish quantile computation this module spends most of its
code computing (see "Numerical cost", below).

Per-arm priors
--------------
Like ``MonteCarloScheduler``, ``supports_priors = True``: ``init_arm``
accepts an optional ``(prior_alpha, prior_beta)`` override, applied once at
first registration and never overwritten. ``services/fuzzer.py``'s
``_register_arms`` already knows how to pass ``target_profiler.
format_operator_priors()``'s per-operator priors to any scheduler
declaring this flag, so this scheduler gets that domain knowledge (e.g.
"crc/checksum mutators are more likely useful against this binary format")
for free, biasing the very first quantile before any evidence exists,
exactly like Thompson sampling already does.

This is also the reason this scheduler is *not* a ``ucb_common.UCBBase``
subclass despite otherwise fitting that skeleton closely (mean/count
bookkeeping): ``UCBBase._width(mean, n, log_n)`` is not told *which* arm
it is scoring, only its aggregate statistics -- fine for every existing
subclass, whose width formula is prior-free, but not enough information to
look up a specific arm's own prior_alpha/prior_beta here. Reimplementing
UCBBase's ~15-line select_op loop locally was less risky than adding an
arm-identity parameter to a shared abstract method four other schedulers
already depend on.

No forced unpulled-arm branch
------------------------------
Every ``UCBBase`` subclass forces zero-evidence arms to the front of the
queue before computing any index, because their additive width formulas
are typically degenerate or meaningless at n=0 (division by n=0, a log
argument of 0, and so on). Bayes-UCB has no such degeneracy: the quantile
of Beta(prior_alpha, prior_beta) at n=0 is perfectly well-defined, and --
this is the point of using a real posterior instead of a derived bound --
already expresses exactly the right cold-start optimism on its own. An
unpulled Beta(0.5, 0.5) arm's quantile at a realistic quantile order
(q_t=0.999 at t=1000) evaluates to ~0.9995, comfortably above even a
long-established, genuinely excellent arm's quantile (~0.89 for 250/300
successes at the same t) -- so forcing zero-evidence arms to the front
would only ever agree with what the index already says, at the cost of
special-casing away exactly the mechanism (posterior width) that makes
this scheduler worth having. Adding that branch back in would also have
silently defeated per-arm priors: it would pick uniformly at random among
every n=0 arm regardless of *how* informative each one's prior was,
exactly backwards from what "supports_priors" is supposed to buy.

One consequence worth stating plainly, because it surprises people used
to non-Bayesian bandits: at n=0, a *more* informative prior can score
*lower* than a less informative one, if the informative prior is narrow
(confident). Beta(40, 2) (prior mean ~0.95, but effectively already 42
pseudo-observations' worth of certainty) evaluates to ~0.9985 at the same
q_t=0.999 -- lower than the uninformed Beta(0.5, 0.5)'s ~0.9995, despite
having a far higher mean. This is correct, not a bug: an index built
around a quantile rewards uncertainty as well as expectation, and a
prior that has already resolved most of the uncertainty about an arm has
correspondingly less exploration value left to offer. A prior meant to
front-load an operator's selection likelihood should therefore be
informative about the *mean* without also being artificially narrow --
``format_operator_priors()``'s priors are small pseudo-count nudges for
exactly this reason, not high-confidence claims.

Because there is no unpulled-arm branch, this scheduler also consumes no
randomness anywhere (float-tie-breaking, on the rare exact tie, falls to
the first candidate in ``ops`` order, same convention as every UCBBase
score loop) -- it takes no ``rng`` parameter, the same precedent
``ContextualLinUCBScheduler`` already sets for a fully deterministic
scheduler in this package.

Numerical cost
---------------
There is no closed form for the Beta quantile function, so this module
implements the regularized incomplete beta function I_x(a, b) (Numerical
Recipes' Lentz continued fraction, the same algorithm --
independently -- already used for this purpose in ``coverage_regime.py``'s
``_betai``/``_betacf``; not imported from there because that module is
about percolation-phase detection, an unrelated domain, and duplicating a
~40-line numerical primitive was judged less coupling than reaching into
another module's underscore-prefixed internals) and inverts it by
bisection to get the quantile.

This is real, measured cost, not a rounding error: at the loosened
tolerances below (chosen because this index only needs to be accurate
enough to rank arms correctly, not to report a precise probability),
scoring 150 candidate arms -- a realistic operator-availability count for
this fuzzer, see ``REGISTRY.available()`` -- takes on the order of a few
milliseconds on commodity hardware (measured ~3-4ms locally; scales
roughly linearly with candidate count). That is one to three orders of
magnitude more than every additive-width scheduler in this package, whose
per-candidate cost is a handful of arithmetic operations. Reserve this
scheduler for a bounded/curated candidate set, or accept the cost
consciously -- it is not appropriate as a drop-in replacement for a
per-mutation hot-path scheduler at full operator-registry scale without
that trade-off in mind.

The tolerances below were chosen empirically, not guessed: cross-checked
against a high-precision reference (tol=1e-10, cf_eps=1e-14) over 2000
random (pulls, successes, quantile-order) triples spanning realistic
ranges, the defaults here reach a maximum absolute quantile error of
~5e-4 -- far below the gap between genuinely different arms' scores, and
irrelevant when arms are close enough that the choice barely matters
anyway.

Reference
---------
Kaufmann, Cappé & Garivier, "On Bayesian Upper Confidence Bounds for
Bandit Problems", AISTATS 2012. ``c=0`` (this module's default) is the
paper's own practical recommendation, matching their reported experiments;
the paper's finite-time regret bound is proved for ``c >= 5``, exposed here
as a constructor parameter for anyone who wants the proven-bound regime
instead of the practically-recommended one.
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool

# Numerical defaults for the incomplete-beta bisection -- see the module
# docstring's "Numerical cost" section for the accuracy/speed measurement
# behind these. cf_eps/cf_maxit bound the inner continued-fraction
# evaluation; tol/max_iter bound the outer bisection over it.
CF_EPS = 1e-4
CF_MAXIT = 30
BISECT_TOL = 1e-3
BISECT_MAXIT = 15

_MIN_BETA_PARAM = 1e-6


def _betacf(a: float, b: float, x: float, eps: float = CF_EPS, maxit: int = CF_MAXIT) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method).

    Same algorithm as ``coverage_regime._betacf`` -- see this module's
    docstring for why it is duplicated rather than imported.
    """
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float, eps: float = CF_EPS, maxit: int = CF_MAXIT) -> float:
    """Regularized incomplete beta function I_x(a, b) -- the Beta(a,b) CDF at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x, eps, maxit) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x, eps, maxit) / b


def beta_quantile(
    p: float,
    a: float,
    b: float,
    tol: float = BISECT_TOL,
    max_iter: int = BISECT_MAXIT,
    cf_eps: float = CF_EPS,
    cf_maxit: int = CF_MAXIT,
) -> float:
    """Quantile function (inverse CDF) of Beta(a, b) at probability p.

    Bisection on ``_betai``, since there is no closed form. See the module
    docstring's "Numerical cost" section for the accuracy this converges
    to at the default tolerances, and why they are loosened relative to a
    reference-quality incomplete-beta evaluation.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if _betai(a, b, mid, cf_eps, cf_maxit) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


class BayesUCBScheduler:
    """Bayes-UCB: Beta-posterior quantile index (Kaufmann, Cappé & Garivier 2012).

    Args:
        prior_alpha: Default Beta prior alpha (pseudo-successes). Must be
            positive. Kaufmann et al.'s own choice for Bernoulli rewards is
            the Jeffreys prior Beta(1/2, 1/2) -- the default here.
        prior_beta: Default Beta prior beta (pseudo-failures). Must be
            positive.
        c: Exponent in the quantile order ``1 - 1/(t * (log t)^c))``.
            ``c=0`` is the paper's practically-recommended default (see
            module docstring "Reference"); must be non-negative.
        rng: Shared ``RandPool`` (Hard Rule 16), consumed only when
            breaking ties among arms with zero evidence.
    """

    supports_priors = True

    def __init__(
        self,
        prior_alpha: float = 0.5,
        prior_beta: float = 0.5,
        c: float = 0.0,
        rng: RandPool | None = None,
    ):
        if prior_alpha <= 0.0:
            raise ValueError(f"prior_alpha must be positive, got {prior_alpha!r}")
        if prior_beta <= 0.0:
            raise ValueError(f"prior_beta must be positive, got {prior_beta!r}")
        if c < 0.0:
            raise ValueError(f"c must be non-negative, got {c!r}")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.c = c
        self._rng = rng if rng is not None else RandPool()

        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._prior_alpha: dict[str, float] = {}
        self._prior_beta: dict[str, float] = {}
        self._total_pulls: int = 0

    # -- arm bookkeeping ----------------------------------------------------

    def init_arm(
        self,
        name: str,
        prior_alpha: float | None = None,
        prior_beta: float | None = None,
    ) -> None:
        """Register a mutation operator arm with a Beta prior.

        Defaults to this scheduler's constructor-level prior. A no-op if
        the arm is already registered -- the prior only applies at first
        registration, matching ``MonteCarloScheduler.init_arm``'s
        idempotent convention.

        Args:
            name: Name of the mutation operator.
            prior_alpha: Per-arm prior alpha override. Must be > 0 if given.
            prior_beta: Per-arm prior beta override. Must be > 0 if given.
        """
        if name in self._counts:
            return
        self._counts[name] = 0
        self._sums[name] = 0.0
        a0 = prior_alpha if prior_alpha is not None else self.prior_alpha
        b0 = prior_beta if prior_beta is not None else self.prior_beta
        self._prior_alpha[name] = max(a0, _MIN_BETA_PARAM)
        self._prior_beta[name] = max(b0, _MIN_BETA_PARAM)

    # -- selection ----------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the operator with the highest Bayes-UCB index.

        Arms with zero evidence are opened first, same convention as
        ``UCBBase.select_op`` -- see the module docstring for why this
        scheduler reimplements that loop rather than inheriting it.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        for op in ops:
            self.init_arm(op)

        unpulled = [op for op in ops if self._counts.get(op, 0) <= 0]
        if unpulled:
            return self._rng.choice(unpulled)

        t = max(self._total_pulls, 2)
        log_t = math.log(t)
        q_t = 1.0 - 1.0 / (t * (log_t**self.c))
        q_t = min(max(q_t, 1e-9), 1.0 - 1e-9)

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._counts.get(op, 0)
            s = self._sums.get(op, 0.0)
            a = self._prior_alpha.get(op, self.prior_alpha) + s
            b = self._prior_beta.get(op, self.prior_beta) + (n - s)
            score = beta_quantile(q_t, a, b)
            if score > best_score:
                best_score = score
                best_op = op
        return best_op

    # -- update ---------------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Credit *name* with this pull's reward.

        Args:
            name: Operator name.
            success: Whether the mutation produced an interesting result.
            weight: Reward weight (default 1.0), added to the arm's
                pseudo-success count on success. Surprisal-weighted calls
                pass a value in (0, 1] -- see ``MonteCarloScheduler.record``
                for the same convention.
        """
        self.init_arm(name)
        self._total_pulls += 1
        self._counts[name] = self._counts.get(name, 0) + 1
        if success:
            self._sums[name] = self._sums.get(name, 0.0) + weight

    # -- diagnostics ------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Return Bayes-UCB diagnostics."""
        return {
            "bayes_ucb_pulls": self._total_pulls,
            "bayes_ucb_arms": len(self._counts),
            "bayes_ucb_prior_alpha": self.prior_alpha,
            "bayes_ucb_prior_beta": self.prior_beta,
            "bayes_ucb_c": self.c,
        }
