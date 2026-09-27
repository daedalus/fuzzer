"""Discrete power-law (Zipf) tail fit and Heaps' law fit for coverage spectra.

Edge frequencies in fuzzing are rank-frequency Zipfian: a few hot edges, a
long tail of rare ones. Chao2 reads only that tail's first two cells (Q1, Q2)
and is a lower bound under it; this module measures the tail itself.

    seeds-per-edge counts ──► distinct values + multiplicities
        (capped at m seeds)            │
                                       ▼
                 for each candidate xmin: MLE alpha (golden section)
                                          KS distance on observed support
                                       │  keep min-KS xmin
                                       ▼
                 KS misfit + Vuong LR vs geometric + tail guard ──► TailLaw

    coverage timeline ──► log D = log K + beta * log N ──► HeapsFit

Rank-frequency exponent s and spectrum exponent alpha: alpha = 1 + 1/s.
Heaps: distinct edges D(N) ~ K * N^beta, beta = 1/s when execs sample Zipf.

References: Clauset, Shalizi & Newman (2009) "Power-law distributions in
empirical data"; Vuong (1989) likelihood-ratio test for non-nested models.
"""

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# ── Hurwitz zeta (Euler-Maclaurin) ─────────────────────────────────────────
# Direct-sum terms before the Euler-Maclaurin tail; with B2..B10 the error is
# below 1e-12 for s in (1, 6] and q >= 1.
EM_TERMS = 10
BERNOULLI = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66)  # B2, B4, ..., B10

# ── Tail fit ───────────────────────────────────────────────────────────────
ALPHA_LO = 1.01  # MLE bracket; alpha <= 1 has no normalisable tail
ALPHA_HI = 6.0  # an alpha pinned here means "decays faster than any power law"
GOLDEN_TOL = 1e-5  # bracket width at which the golden-section search stops
AT_BOUND = 1e-3  # distance from ALPHA_HI that counts as pinned
MAX_XMIN = 32  # bounded xmin scan (Hard Rule 54)
MIN_TAIL = 50  # tail points needed before any verdict
MIN_TAIL_FRAC = 0.2  # lognormal guard: the power law must describe most data
Z_95 = 1.96  # two-sided 95% standard-normal quantile
# Kolmogorov 95% critical value: reject when KS > KS_95 / sqrt(n). Fitted
# parameters shrink the true KS distribution, so this errs toward accepting.
KS_95 = 1.36

# ── Heaps fit ──────────────────────────────────────────────────────────────
HEAPS_TAIL = 0.5  # fit the most recent half of the timeline
HEAPS_MIN_POINTS = 3

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
# Bracket shrinks by _GOLDEN per step: steps to go from (ALPHA_HI - ALPHA_LO) to GOLDEN_TOL.
_GOLDEN_STEPS = math.ceil(math.log(GOLDEN_TOL / (ALPHA_HI - ALPHA_LO)) / math.log(_GOLDEN))


_Floats = npt.NDArray[np.float64]
_Ints = npt.NDArray[np.int64]
_Real = float | _Floats  # scalar or elementwise array


class TailLaw(enum.Enum):
    """Verdict on whether the spectrum's tail is a power law."""

    POWER_LAW = "power_law"
    NOT_POWER_LAW = "not_power_law"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class ZipfFit:
    """Power-law fit of a frequency spectrum's tail (x >= xmin)."""

    alpha: float
    xmin: int
    n_tail: int
    tail_frac: float
    ks: float
    vuong: float
    law: TailLaw

    @property
    def s(self) -> float:
        """Rank-frequency Zipf exponent, 1 / (alpha - 1)."""
        return 1.0 / (self.alpha - 1.0)


@dataclass(frozen=True, slots=True)
class HeapsFit:
    """Heaps' law D(N) = k * N^beta fitted to a discovery curve."""

    beta: float
    k: float
    r2: float

    def project(self, execs: int) -> float:
        """Distinct edges the fit expects after ``execs`` executions."""
        return self.k * math.pow(execs, self.beta)

    @property
    def doubling_gain(self) -> float:
        """Fractional edge gain from doubling execs: 2^beta - 1 (beta 0.5 -> +41%)."""
        return math.pow(2.0, self.beta) - 1.0


_INSUFFICIENT = ZipfFit(math.nan, 0, 0, 0.0, math.nan, math.nan, TailLaw.INSUFFICIENT)


def hurwitz(s: float, q: float) -> float:
    """Hurwitz zeta: sum_{k>=0} (k + q)^-s, for s > 1 and q > 0.

    Example: hurwitz(2, 1) = pi^2 / 6.
    """
    if s <= 1.0 or q <= 0.0:
        raise ValueError(f"hurwitz needs s > 1 and q > 0, got s={s}, q={q}")
    return float(_hurwitz_arr(np.float64(s), np.float64(q)))


def _hurwitz_arr(s: _Real, q: _Real) -> _Real:
    """Elementwise Hurwitz zeta; ``s`` and ``q`` broadcast (scalars or arrays)."""
    # Direct head: sum of the first EM_TERMS terms.
    total: _Real = 0.0
    for k in range(EM_TERMS):
        total = total + (q + k) ** -s

    # Euler-Maclaurin tail from a = q + EM_TERMS:
    #   a^(1-s)/(s-1) + a^-s/2 + sum_j B_2j/(2j)! * s(s+1)..(s+2j-2) * a^(-s-2j+1)
    a = q + EM_TERMS
    total = total + a ** (1.0 - s) / (s - 1.0) + 0.5 * a**-s
    fact = 1.0
    rising: _Real = s
    power = a ** (-s - 1.0)
    for j, b in enumerate(BERNOULLI):
        two_j = 2 * (j + 1)
        fact *= (two_j - 1) * two_j
        total = total + b / fact * rising * power
        rising = rising * (s + two_j - 1) * (s + two_j)
        power = power / (a * a)
    return total


def fit_zipf(counts: Iterable[int], xmax: int = 0) -> ZipfFit:
    """Fit a discrete power law to per-item counts.

    Args:
        counts: One count per item (e.g. seeds covering each edge). Values
            below 1 are ignored.
        xmax: Upper support bound (0 = unbounded). Pass the corpus size for
            seeds-per-edge counts: no edge can have more owners than seeds.

    Returns:
        The min-KS fit over xmin candidates, with a TailLaw verdict.
    """
    vals, mult = _aggregate(counts)
    total = int(mult.sum())
    if len(vals) < 2 or total < MIN_TAIL:
        return _INSUFFICIENT

    best = _scan_xmin(vals, mult, xmax)
    if best is None:
        return _INSUFFICIENT

    alpha, xmin, n_tail, ks = best
    tail = vals >= xmin
    vuong = _vuong(vals[tail], mult[tail], alpha, xmin, xmax)
    return _classify(alpha, xmin, n_tail, total, ks, vuong)


def _aggregate(counts: Iterable[int]) -> tuple[_Ints, _Ints]:
    """Distinct positive values (ascending) and their multiplicities."""
    arr = np.fromiter(counts, dtype=np.int64)
    arr = arr[arr > 0]
    vals, mult = np.unique(arr, return_counts=True)
    return vals, mult.astype(np.int64)


def _scan_xmin(vals: _Ints, mult: _Ints, xmax: int) -> tuple[float, int, int, float] | None:
    """Fit alpha at each candidate xmin; keep the one with the smallest KS."""
    # Suffix sums give each candidate's tail size and log-sum in O(1).
    logs: _Floats = np.log(vals.astype(np.float64)) * mult
    n_suffix = np.cumsum(mult[::-1])[::-1]
    log_suffix: _Floats = np.cumsum(logs[::-1])[::-1]

    # Candidates form a prefix: xmin <= MAX_XMIN, >= MIN_TAIL points, and a
    # tail of >= 2 distinct values. All three shrink monotonically with i.
    ok = (vals[:-1] <= MAX_XMIN) & (n_suffix[:-1] >= MIN_TAIL)
    m = len(ok) if ok.all() else int(np.argmin(ok))
    if m == 0:
        return None

    alphas = _fit_alpha(n_suffix[:m], log_suffix[:m], vals[:m], xmax)
    best: tuple[float, int, int, float] | None = None
    for i in range(m):
        xmin = int(vals[i])
        alpha = float(alphas[i])
        ks = _ks(vals[i:], mult[i:], alpha, xmin, xmax)
        if best is None or ks < best[3]:
            best = (alpha, xmin, int(n_suffix[i]), ks)
    return best


def _log_norm(alpha: float, xmin: int, xmax: int) -> float:
    """log of the (possibly truncated) normaliser zeta(alpha, xmin) - zeta(alpha, xmax+1)."""
    z = _hurwitz_arr(alpha, float(xmin))
    if xmax:
        z = z - _hurwitz_arr(alpha, float(xmax + 1))
    return math.log(float(z))


def _fit_alpha(n: _Ints, sum_log: _Floats, xmin: _Ints, xmax: int) -> _Floats:
    """Golden-section MLE of alpha, one lane per xmin candidate.

    Minimises n * log Z(alpha) + alpha * sum(log x). Every lane shrinks its
    bracket by the same ratio, so a fixed step count reaches GOLDEN_TOL in
    all of them at once (1.5x faster than a per-candidate scalar loop).
    """
    q = xmin.astype(np.float64)
    tail_q = float(xmax + 1)

    def nll(a: _Floats) -> _Floats:
        z = np.asarray(_hurwitz_arr(a, q), dtype=np.float64)
        if xmax:
            z = z - _hurwitz_arr(a, tail_q)
        out: _Floats = n * np.log(z) + a * sum_log
        return out

    lo = np.full(len(q), ALPHA_LO)
    hi = np.full(len(q), ALPHA_HI)
    c = hi - _GOLDEN * (hi - lo)
    d = lo + _GOLDEN * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(_GOLDEN_STEPS):
        # left: minimum lies in [lo, d]; the old c becomes the new d.
        left = fc < fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        new_c = np.where(left, hi - _GOLDEN * (hi - lo), d)
        new_d = np.where(left, c, lo + _GOLDEN * (hi - lo))

        # Only one probe per lane is new; the other reuses a known value.
        f_new = nll(np.where(left, new_c, new_d))
        fc, fd = np.where(left, f_new, fd), np.where(left, fc, f_new)
        c, d = new_c, new_d
    mid: _Floats = (lo + hi) / 2.0
    return mid


def _ks(vals: _Ints, mult: _Ints, alpha: float, xmin: int, xmax: int) -> float:
    """Discrete KS distance, evaluated on the observed support only.

    Model CDF: F(v) = (zeta(a, xmin) - zeta(a, v + 1)) / Z. Between observed
    values the empirical CDF is flat while the model keeps rising, so the gap
    just below each observed v, |F_emp(prev) - F(v - 1)|, is checked too.
    """
    z = math.exp(_log_norm(alpha, xmin, xmax))
    head = _hurwitz_arr(alpha, float(xmin))
    v = vals.astype(np.float64)

    model = (head - _hurwitz_arr(alpha, v + 1.0)) / z
    model_below = model - v**-alpha / z
    emp = np.cumsum(mult) / mult.sum()
    emp_prev = np.concatenate(([0.0], emp[:-1]))

    return float(max(np.max(np.abs(emp - model)), np.max(np.abs(emp_prev - model_below))))


def _vuong(vals: _Ints, mult: _Ints, alpha: float, xmin: int, xmax: int) -> float:
    """Normalised log-likelihood ratio, power law vs geometric; > Z_95 favours the power law."""
    w = mult.astype(np.float64)
    n = w.sum()
    shift = vals.astype(np.float64) - xmin

    # Geometric on the shifted tail, MLE p = 1 / (1 + mean shift).
    p = 1.0 / (1.0 + float((shift * w).sum() / n))
    ll_geo = math.log(p) + shift * math.log1p(-p) if p < 1.0 else np.zeros_like(shift)
    ll_pl = -alpha * np.log(vals.astype(np.float64)) - _log_norm(alpha, xmin, xmax)

    d = ll_pl - ll_geo
    mean = float((d * w).sum() / n)
    sd = math.sqrt(float(((d - mean) ** 2 * w).sum() / n))
    if sd == 0.0:
        return 0.0
    return mean * math.sqrt(n) / sd


def _classify(alpha: float, xmin: int, n_tail: int, total: int, ks: float, vuong: float) -> ZipfFit:
    """Reject a pinned alpha, a KS misfit, a minority tail, or a lost Vuong test.

    Vuong only says the power law beats a geometric; a spectrum that is
    neither (png_read: decaying head plus a hot core every seed owns) still
    wins it, so the absolute KS fit is checked too.
    """
    frac = n_tail / total
    pinned = alpha >= ALPHA_HI - AT_BOUND or alpha <= ALPHA_LO + AT_BOUND
    # xmin is the min-KS candidate of up to MAX_XMIN, which biases KS low:
    # another reason the guard errs toward accepting, never toward rejecting.
    misfit = ks > KS_95 / math.sqrt(n_tail)
    law = TailLaw.POWER_LAW
    if pinned or misfit or frac < MIN_TAIL_FRAC or vuong < Z_95:
        law = TailLaw.NOT_POWER_LAW
    return ZipfFit(alpha, xmin, n_tail, frac, ks, vuong, law)


def fit_heaps(execs: Sequence[int], edges: Sequence[int]) -> HeapsFit | None:
    """Least-squares fit of log D = log k + beta * log N over the recent timeline.

    Args:
        execs: Execution count per snapshot (ascending).
        edges: Cumulative distinct edges per snapshot.

    Returns:
        The fit, or None with fewer than HEAPS_MIN_POINTS usable points.
    """
    if len(execs) != len(edges):
        raise ValueError(f"length mismatch: {len(execs)} execs vs {len(edges)} edges")

    x = np.asarray(execs, dtype=np.float64)
    y = np.asarray(edges, dtype=np.float64)
    keep = (x > 0) & (y > 0)  # log undefined at 0
    x, y = x[keep], y[keep]
    k = max(HEAPS_MIN_POINTS, int(len(x) * HEAPS_TAIL))
    if len(x) < HEAPS_MIN_POINTS:
        return None

    lx, ly = np.log(x[-k:]), np.log(y[-k:])
    if np.ptp(lx) == 0.0:
        return None

    beta, log_k = np.polyfit(lx, ly, 1)
    var = float(ly.var())
    r2 = 1.0 if var == 0.0 else 1.0 - float((ly - (beta * lx + log_k)).var()) / var
    return HeapsFit(float(beta), math.exp(float(log_k)), r2)
