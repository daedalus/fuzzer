# KL_DUCBScheduler: fidelity to the cited papers

**Status:** analysis only, no code change. HEAD at time of writing: `d1c13b7`.

## Trigger

Live campaign Elo convergence showed `kl_ducb` (1497, -3, 1546 matches) sitting
at/below `canary` (1499, -1) — the scheduler `strategies_below_canary` check
exists precisely to flag this. Its sibling `ducb`, sharing the same
`DiscountedUCBBase` skeleton and the same `gamma/xi/exploration` defaults, sits
comfortably above canary (1513, +13). This doc checks whether `kl_ducb`'s
confidence-width formula is faithful to the papers its own docstrings cite,
since the two schedulers differ only in `_width()`.

## What's cited, and what it actually says

`op_kl_ducb.py` cites Garivier & Moulines (arXiv:0805.3415, the same D-UCB paper
`op_ducb.py` implements) for the discounted structure, and `_kl_ucb.py`'s module
docstring cites Cappé, Garivier, Maillard, Munos, Stoltz (2013) for the KL
bound — in practice the same result as Garivier & Cappé, *The KL-UCB Algorithm
for Bounded Stochastic Bandits and Beyond*, COLT 2011 (arXiv:1102.2490), which
is the paper that actually derives the Bernoulli-KL upper-confidence index.

Two separate problems, both verified against the primary sources this turn:

### 1. The KL budget is over-shrunk relative to what its own paper recommends

Garivier & Cappé's index (Algorithm 1, COLT 2011) is

    u_a(t) = max { q > mean_a : N_a(t)*d(mean_a, q) <= log(t) + c*log(log(t)) }

with `c = 3` for the proof and, verbatim, **"in practice, however, we
recommend to take c = 0 for optimal performance"** (Remark 5). Either way the
budget is `log(t)` (plus a non-negative log-log correction) — there is no
free shrinking multiplier in the theorem at all. That's the entire point of
KL-UCB: unlike the Hoeffding/Gaussian bound, it needs no fudge factor to be
tight.

`op_kl_ducb.py`'s `_width()` instead computes the budget as
`exploration * xi * log_n / n = 0.25 * 0.6 * log_n / n = 0.15 * log_n/n` —
i.e. it reuses the `exploration=0.25, xi=0.6` pair verbatim from
`op_ducb.py`. That pair is a deliberate, empirically-measured correction
documented in `op_ducb.py`'s own docstring (a 12-arm sweep showing the paper's
literal `2B` coefficient is a massive over-exploration at this reward scale).
But that correction was derived *for the Gaussian bound's own inflated
leading constant* — it has no independent justification against the KL
bound, whose paper explicitly says the untouched `log(t)` is already the
right tuning. `op_kl_ducb.py`'s own docstring acknowledges this reasoning
("the KL theorem uses a leading constant of 1, so `exploration` is the
tuning knob here") but the reasoning doesn't hold: cutting the *budget*
inside a KL divergence by 85% is not equivalent to cutting a linear
multiplicative *coefficient* by 85% the way it is for the Gaussian width,
because `d(p,q)` is not linear in the budget outside a narrow regime.

### 2. Measured effect: over-shrinking the budget still leaves `kl_ducb` wider than `ducb`, not narrower

Traced `kl_upper_bound()`'s branch selection directly (not just the formula)
across `n` in the range this campaign actually produced (10 to 50,000) and
`p` from 0.001 to 0.99. Result: for every case except `p >= 0.9`, the
function takes its own documented Gaussian-approximation shortcut
(`p + sqrt(2*budget)`) rather than the true KL bisection — because at these
budgets (~1e-2 to 1e-4) the quadratic approximation to `d(p,q)` is accurate
to well within the `1e-12` tolerance for any `p` not near the boundary. Since
almost no fuzzing operator succeeds on 90%+ of pulls, `kl_ducb` is in
practice *always* running the Gaussian-shortcut branch, not the asymmetric
KL bound the docstring advertises.

In that branch, `kl_ducb`'s width is `sqrt(2*budget)` and `ducb`'s is
`exploration*2*b*sqrt(xi*log_n/n)`. Both reduce to `sqrt(c*log_n/n)` for a
constant `c`; plugging in the actual defaults gives `c_kl = 0.30` vs.
`c_ducb = 0.15`, a fixed **√2 ≈ 1.41x wider** width for `kl_ducb` than
`ducb`, independent of `n` and `p`, confirmed numerically:

    n=1000, p=0.10: ducb_width=0.0831  kl_ducb_width=0.1175  ratio=1.41
    (constant 1.41 across every (n,p) tried outside p>=0.9)

So the net, measured effect of points 1 and 2 together is the *opposite* of
what `op_kl_ducb.py`'s docstring claims ("the KL bound is tighter, so it
explores less"): in this operating regime `kl_ducb` explores **more** than
`ducb`, not less, because the under-tuned budget still produces a wider
Gaussian-shortcut width than `ducb`'s own already-corrected one. That's a
plausible direct explanation for `kl_ducb` sitting below `ducb` (and at the
canary floor) in the live convergence table.

### 3. A structural gap neither cited paper actually covers

Independent of the tuning question: Garivier & Moulines' Theorem 18 (the
result `op_ducb.py` relies on) is a Hoeffding-type self-normalized deviation
bound for **exponentially discounted** sums with a random number of
summands. Garivier & Cappé's Theorem 10 (the result behind `kl_upper_bound`)
proves the analogous statement for the **KL** divergence, but only for an
*undiscounted* indicator sum (`N(t) = sum(eps_s)`, no `gamma^(t-s)` weight).
Neither paper proves a KL-analog of Theorem 18 for the discounted case. That
means composing "discounted counts/means" (`DiscountedUCBBase`) with
`kl_upper_bound()` — as `op_kl_ducb.py` does — has no proof behind it in either
source; it's a reasonable-looking heuristic combination of two papers, not
the peer-reviewed algorithm the docstrings imply. This doesn't make it
*wrong*, but "faithful to the paper" is not an accurate description of what
`KL_DUCBScheduler` currently is.

## Bottom line

`kl_ducb` is not a bug in the sense of a coding mistake — `kl_upper_bound()`
is correctly implemented against its own stated contract, and `_width()`
correctly implements the formula its docstring describes. The problem is
that formula: it inherits Gaussian-bound tuning constants that the KL-UCB
paper explicitly says are unnecessary (`c=0`, i.e. no shrinkage), the
un-shrunk-vs-shrunk comparison still comes out wider than `ducb` rather than
narrower as claimed, and the whole discounted+KL combination is an
unproven composition of two separate papers' results.

## Update: tried the paper-literal fix, it made things worse

Implemented candidate (1) first: dropped `exploration`/`xi` to `1.0` each
(budget = `log(n)/N`, Garivier & Cappé's own `c=0` recommendation) and
re-ran `tools/measure_klucb_signal.py`. Result: stationary tail share
**dropped from 0.943 to 0.657** — the paper-literal fix made `kl_ducb`
dramatically worse, not better. That's the opposite of the hypothesis in
the original version of this doc.

Followed up with a shrinkage sweep (`xi` from 1.0 down to 0.01,
`exploration` fixed at 1.0), averaged over 5 seeds on both a stationary
environment (tail share) and a decaying-best environment (post-decay
recovery), matching the two-metric methodology `op_ducb.py`'s own docstring
uses:

    xi      stationary   recovery
    0.60    0.864        0.475
    0.30    0.950        0.640
    0.20    0.975        0.772
    0.15    0.971        0.824
    0.10    0.982        0.854   <- picked
    0.075   1.000        0.556   (unstable across seeds)
    0.05    0.400        0.783   (unstable across seeds)
    DUCB reference: 0.978 / 0.858

`xi=0.10` (with `exploration=1.0`) tracks `DUCBScheduler` on both metrics
within noise. Below ~0.075 the measurement gets visibly unstable across
seeds (not a smooth curve) — a further sign that this discounted+KL
composition doesn't have the well-behaved, monotone tradeoff either
source paper would lead you to expect near its own recommended constant.

**Applied**: `KL_DUCBScheduler` now defaults to `xi=0.10, exploration=1.0`
instead of `xi=0.6, exploration=0.25`. All 4 existing
`tests/test_kl_ducb_scheduler.py` tests still pass unchanged (the
adversarial test pins its own explicit `xi=0.6, exploration=0.25` and is
checking the formula, not the default). The rest of the suite
(`test_regression_scheduler_fallback_precedence.py`,
`test_regression_operator_ballot_symmetry.py`,
`test_regression_elo_all.py`, etc.) has 63 pre-existing failures in this
environment verified identical with and without this change (`git stash`
diff), so they're unrelated to this fix.

## Remaining open question

`xi=0.10` is an empirical fit to two synthetic environments, not a
theoretical constant — nobody has proven a discounted-KL-UCB regret bound
that this value would follow from. The honest characterization is still
what section 3 above says: this scheduler is a heuristic composition of
two papers' techniques, now re-tuned by measurement rather than by
borrowing an unrelated constant. Re-measuring directly against a live
fuzzing campaign (not just the two synthetic bandit environments here)
would be the natural next check, and `kl_swucb` — which uses no
`exploration` multiplier and a single `xi=0.15` for the *windowed* rather
than discounted composition, and already measured well above `swucb`
(0.997) without needing this fix — is worth a similar audit for
completeness even though nothing currently flags it as underperforming.
