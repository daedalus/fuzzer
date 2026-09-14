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

`kl_ducb.py` cites Garivier & Moulines (arXiv:0805.3415, the same D-UCB paper
`ducb.py` implements) for the discounted structure, and `_kl_ucb.py`'s module
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

`kl_ducb.py`'s `_width()` instead computes the budget as
`exploration * xi * log_n / n = 0.25 * 0.6 * log_n / n = 0.15 * log_n/n` —
i.e. it reuses the `exploration=0.25, xi=0.6` pair verbatim from
`ducb.py`. That pair is a deliberate, empirically-measured correction
documented in `ducb.py`'s own docstring (a 12-arm sweep showing the paper's
literal `2B` coefficient is a massive over-exploration at this reward scale).
But that correction was derived *for the Gaussian bound's own inflated
leading constant* — it has no independent justification against the KL
bound, whose paper explicitly says the untouched `log(t)` is already the
right tuning. `kl_ducb.py`'s own docstring acknowledges this reasoning
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
what `kl_ducb.py`'s docstring claims ("the KL bound is tighter, so it
explores less"): in this operating regime `kl_ducb` explores **more** than
`ducb`, not less, because the under-tuned budget still produces a wider
Gaussian-shortcut width than `ducb`'s own already-corrected one. That's a
plausible direct explanation for `kl_ducb` sitting below `ducb` (and at the
canary floor) in the live convergence table.

### 3. A structural gap neither cited paper actually covers

Independent of the tuning question: Garivier & Moulines' Theorem 18 (the
result `ducb.py` relies on) is a Hoeffding-type self-normalized deviation
bound for **exponentially discounted** sums with a random number of
summands. Garivier & Cappé's Theorem 10 (the result behind `kl_upper_bound`)
proves the analogous statement for the **KL** divergence, but only for an
*undiscounted* indicator sum (`N(t) = sum(eps_s)`, no `gamma^(t-s)` weight).
Neither paper proves a KL-analog of Theorem 18 for the discounted case. That
means composing "discounted counts/means" (`DiscountedUCBBase`) with
`kl_upper_bound()` — as `kl_ducb.py` does — has no proof behind it in either
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

## Candidate directions (not implemented)

1. Drop `exploration`/`xi` from `kl_ducb` entirely and use the paper's own
   recommendation (`budget = log_n / n`, i.e. `c=0`), then re-measure against
   `ducb` and `canary` empirically — this is what the cited paper actually
   prescribes.
2. If (1) still underperforms in the discounted, high-arm-count regime this
   fuzzer runs in (147 operators, `gamma=0.9999`), that would itself be a
   useful negative result: it would suggest the undiscounted KL-UCB
   guarantee genuinely doesn't transfer to the discounted setting, matching
   the gap identified in section 3.
3. Either way, `kl_ducb.py`'s docstring claim ("explores less, exploits more
   aggressively") should be corrected or removed — it's not supported by
   what the code measurably does.

No patch attached; next step would be trying (1) and re-running the
convergence measurement.
