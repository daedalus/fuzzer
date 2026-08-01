"""Chi-squared tests — goodness-of-fit, homogeneity, independence.

Pure Python, no scipy dependency.  Uses Lentz's continued fraction for the
regularized incomplete gamma function (the χ² CDF), matching the project's
approach of self-contained statistical implementations (see ``edge_tracker.py``
for the equivalent KS test pattern).

Reference
---------
Abramowitz & Stegun, "Handbook of Mathematical Functions", §6.5.
Press et al., "Numerical Recipes", §6.2 (incomplete gamma).
"""

from __future__ import annotations

import math
from collections import defaultdict

# ── helpers ────────────────────────────────────────────────────────────

_LOG_PI = math.log(math.pi)
_LOG_2 = math.log(2.0)


def _log_gamma(x: float) -> float:
    """Natural log of the Gamma function via Stirling's approximation + Lanczos.

    Accuracy ~2e-10 for x > 0, more than sufficient for p-value work.
    Uses the standard 7-term Lanczos approximation (the same one behind
    ``math.lgamma`` on most platforms; provided here as a cross-platform
    stable fallback).
    """
    if x <= 0.0:
        return float("inf")
    # Lanczos coefficients (7-term, from GNU Scientific Library)
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    x -= 1.0
    y = coeffs[0]
    for i in range(1, 9):
        y += coeffs[i] / (x + i)
    t = x + 7.5
    return 0.5 * _LOG_2 * 7.0 + 0.9189385332046727 + (x + 0.5) * math.log(t) - t + math.log(y)


# Use math.lgamma when available (faster, native), fall back to Lanczos.
_lgamma: callable
_lgamma = math.lgamma if hasattr(math, "lgamma") else _log_gamma  # type: ignore[attr-defined]


# ── incomplete gamma (regularized lower) ───────────────────────────────


def _reg_lower_incomplete_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a).

    Uses the series expansion when x < a + 1 (small-x regime) and Lentz's
    continued fraction otherwise (large-x regime), converging to double-
    precision accuracy within 30–100 iterations for all practical values.
    """
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _igamma_series(a, x)
    return 1.0 - _igamma_cf(a, x)


def _igamma_series(a: float, x: float) -> float:
    """Series representation of P(a, x) for x < a+1.

    P(a, x) = e^{-x} x^a / Γ(a) * Σ_{n=0}^∞ x^n / (a (a+1) ... (a+n))
    """
    if x <= 0.0:
        return 0.0
    log_g = _lgamma(a)
    term = 1.0 / a  # n = 0 term: x^0 / a
    s = term
    for n in range(1, 200):
        term *= x / (a + n)
        s += term
        if abs(term) < 1e-15 * abs(s):
            break
    return s * math.exp(-x + a * math.log(x) - log_g)


def _igamma_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x) = Γ(a, x) / Γ(a) via CF.

    Uses Lentz's method on the continued fraction (Abramowitz & Stegun 6.5.31)::

                 1
        CF = ---------
              (1-a)
             x + -----
                   1
                 1
               + ---
                 (2-a)
                x + ---
                    1
                 2
               + ---
                 ...

    which evaluates to Γ(a, x) / (x^a * e^{-x})  so that
    Q(a, x) = x^a * e^{-x} / Γ(a) * CF.
    """
    log_factor = a * math.log(x) - x - _lgamma(a)
    if log_factor < -700.0:
        return 0.0
    factor = math.exp(log_factor)

    tiny = 1e-300
    f = tiny
    C = f
    D = 0.0

    for i in range(1, 200):
        if i == 1:
            a_i = 1.0
            b_i = x if x != 0.0 else tiny
        elif i % 2 == 0:
            # even i: a = k - a, b = 1  (k = i/2)
            k = i // 2
            a_i = k - a
            b_i = 1.0
        else:
            # odd i: a = k, b = x  (k = i//2)
            k = i // 2
            a_i = float(k)
            b_i = x

        D = b_i + a_i * D
        if abs(D) < tiny:
            D = tiny
        C = b_i + a_i / C
        if abs(C) < tiny:
            C = tiny
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-14:
            break

    return factor * f


# ── chi-squared CDF / p-value ──────────────────────────────────────────


def chi_squared_pvalue(x2: float, dof: int) -> float:
    """Survival function: P(χ²(dof) > x2).  Returns 1.0 for x2 <= 0."""
    if dof < 1:
        return 1.0
    if x2 <= 0.0:
        return 1.0
    a = dof / 2.0
    return 1.0 - _reg_lower_incomplete_gamma(a, x2 / 2.0)


def chi_squared_critical_value(dof: int, alpha: float = 0.05) -> float:
    """Inverse survival function: x2 such that P(χ²(dof) > x2) = alpha.

    Binary search on ``chi_squared_pvalue``.  Uses a Wilson–Hilferty
    approximation to bracket the root for fast convergence.
    """
    if dof < 1:
        return 0.0
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0

    # Wilson–Hilferty approximation for the initial guess:
    # χ²_α ≈ dof * (1 - 2/(9*dof) + z_α * sqrt(2/(9*dof)))^3
    # where z_α is the normal quantile.  Crude but good enough for bracketing.
    import math

    # Approximate normal quantile for alpha (coarse)
    z_map = {
        0.995: 2.576,
        0.99: 2.326,
        0.975: 1.960,
        0.95: 1.645,
        0.90: 1.282,
        0.80: 0.842,
        0.50: 0.0,
        0.20: -0.842,
        0.10: -1.282,
        0.05: -1.645,
        0.025: -1.960,
        0.01: -2.326,
        0.005: -2.576,
    }
    # Find the two closest alphas and interpolate z
    sorted_alphas = sorted(z_map.keys())
    z_low, z_high = -2.576, 2.576
    for sa in sorted_alphas:
        if sa >= alpha:
            z_high = z_map[sa]
            break
        z_low = z_map[sa]
    alpha_frac = 0.0
    for i in range(len(sorted_alphas) - 1):
        if sorted_alphas[i] <= alpha <= sorted_alphas[i + 1]:
            span = sorted_alphas[i + 1] - sorted_alphas[i]
            if span > 0:
                alpha_frac = (alpha - sorted_alphas[i]) / span
            z_low = z_map[sorted_alphas[i]]
            z_high = z_map[sorted_alphas[i + 1]]
            break
    z_alpha = z_low + (z_high - z_low) * alpha_frac

    # Wilson–Hilferty approximation
    if dof >= 2:
        x_wh = dof * (1.0 - 2.0 / (9.0 * dof) + z_alpha * math.sqrt(2.0 / (9.0 * dof))) ** 3
    else:
        x_wh = dof  # fallback

    # Ensure positive
    lo = max(1e-10, x_wh * 0.1)
    hi = max(1e-5, x_wh * 10.0)

    # Expand bracket if needed
    p_lo = chi_squared_pvalue(lo, dof)
    p_hi = chi_squared_pvalue(hi, dof)
    for _ in range(20):
        if p_lo >= alpha >= p_hi:
            break
        if p_lo < alpha:
            lo /= 2.0
            p_lo = chi_squared_pvalue(lo, dof)
        if p_hi > alpha:
            hi *= 2.0
            p_hi = chi_squared_pvalue(hi, dof)

    # Binary search
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p_mid = chi_squared_pvalue(mid, dof)
        if p_mid > alpha:
            lo = mid  # need larger x2
        else:
            hi = mid
        if (hi - lo) / max(1.0, mid) < 1e-12:
            break

    return (lo + hi) / 2.0


# ── test functions ─────────────────────────────────────────────────────


def chi_squared_goodness_of_fit(
    observed: list[float],
    expected: list[float] | None = None,
    n_params: int = 0,
) -> tuple[float, float, int]:
    """Chi-squared goodness-of-fit test.

    Args:
        observed: Observed frequencies (length K).
        expected: Expected frequencies, or ``None`` for uniform.
        n_params: Number of parameters estimated from the data (reduces dof).

    Returns:
        (chi2_stat, p_value, degrees_of_freedom).

    Raises:
        ValueError: On length mismatch, negative counts, or zero total.
    """
    _validate_counts(observed, "observed")
    k = len(observed)
    n = sum(observed)

    if expected is None:
        expected = [n / k] * k
    else:
        _validate_counts(expected, "expected")
        if len(expected) != k:
            raise ValueError(
                f"observed ({k}) and expected ({len(expected)}) must have the same length"
            )

    dof = k - 1 - n_params
    if dof < 1:
        dof = 1

    chi2 = sum(
        (o - e) * (o - e) / e if e > 0 else 0.0 for o, e in zip(observed, expected, strict=True)
    )
    p = chi_squared_pvalue(chi2, dof)
    return chi2, p, dof


def chi_squared_test(
    table: list[list[float]],
) -> tuple[float, float, int]:
    """Chi-squared test of independence / homogeneity for an RxC table.

    ``chi_squared_homogeneity`` and ``chi_squared_independence`` are aliases
    — the computation is identical (expected = row_total * col_total / grand_total).

    Args:
        table: R rows × C columns of observed frequencies.

    Returns:
        (chi2_stat, p_value, degrees_of_freedom) where dof = (R-1)*(C-1).

    Raises:
        ValueError: On empty table, negative counts, or zero grand total.
    """
    return _chi_squared_table(table)


chi_squared_homogeneity = chi_squared_test
chi_squared_independence = chi_squared_test


def _chi_squared_table(table: list[list[float]]) -> tuple[float, float, int]:
    """Internal implementation for RxC contingency tables."""
    if not table or not table[0]:
        raise ValueError("table must have at least one row and one column")

    rows = len(table)
    cols = len(table[0])

    # Validate and compute marginals
    row_totals = [0.0] * rows
    col_totals = [0.0] * cols

    for r in range(rows):
        for c in range(cols):
            v = table[r][c]
            if v < 0:
                raise ValueError(f"negative count at [{r}][{c}]: {v}")
            row_totals[r] += v
            col_totals[c] += v

    grand_total = sum(row_totals)
    if grand_total <= 0:
        raise ValueError("grand total must be positive")

    dof = (rows - 1) * (cols - 1)
    if dof < 1:
        dof = 1

    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            exp = row_totals[r] * col_totals[c] / grand_total
            if exp <= 0:
                continue
            obs = table[r][c]
            chi2 += (obs - exp) * (obs - exp) / exp

    p = chi_squared_pvalue(chi2, dof)
    return chi2, p, dof


# ── effect size ────────────────────────────────────────────────────────


def cramers_v(
    chi2: float,
    n: int,
    num_rows: int,
    num_cols: int,
) -> float:
    """Cramér's V effect size for a chi-squared test.

    V = sqrt(χ² / (n * min(r-1, c-1))).

    Conventional interpretation: 0.1 = small, 0.3 = medium, 0.5 = large.
    """
    k = min(num_rows - 1, num_cols - 1)
    if k <= 0 or n <= 0 or chi2 <= 0:
        return 0.0
    return math.sqrt(chi2 / (n * k))


# ── utility ────────────────────────────────────────────────────────────


def _validate_counts(counts: list[float], name: str) -> None:
    """Validate a frequency vector."""
    if not counts:
        raise ValueError(f"{name} must not be empty")
    for i, v in enumerate(counts):
        if v < 0:
            raise ValueError(f"negative {name} at index {i}: {v}")


# ── descriptive table builder ──────────────────────────────────────────


class ContingencyTable:
    """Build and analyse an RxC contingency table.

    Accumulates counts by ``add()``, then runs ``chi_squared()``.

    Example::

        ct = ContingencyTable()
        ct.add("op_a", "success", 45)
        ct.add("op_a", "failure", 12)
        ct.add("op_b", "success", 30)
        ct.add("op_b", "failure", 40)
        chi2, p, dof = ct.chi_squared()
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], float] = defaultdict(float)
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []
        self._frozen = False

    def add(self, row_label: str, col_label: str, count: float = 1.0) -> None:
        """Increment *count* for the given row x column cell."""
        if self._frozen:
            raise RuntimeError("ContingencyTable is frozen after chi_squared()")
        self._data[(row_label, col_label)] += count
        if row_label not in self._row_labels:
            self._row_labels.append(row_label)
        if col_label not in self._col_labels:
            self._col_labels.append(col_label)

    def _build(self) -> tuple[list[list[float]], dict[str, int], dict[str, int]]:
        """Build the observed matrix and label→index maps."""
        row_idx = {lbl: i for i, lbl in enumerate(self._row_labels)}
        col_idx = {lbl: j for j, lbl in enumerate(self._col_labels)}
        r, c = len(row_idx), len(col_idx)
        if r == 0 or c == 0:
            raise ValueError("ContingencyTable has no data")
        mat: list[list[float]] = [[0.0] * c for _ in range(r)]
        for (rl, cl), v in self._data.items():
            mat[row_idx[rl]][col_idx[cl]] = v
        return mat, row_idx, col_idx

    @property
    def row_marginals(self) -> dict[str, float]:
        """Row totals keyed by row label."""
        mat, ridx, _ = self._build()
        return {lbl: sum(mat[i]) for lbl, i in ridx.items()}

    @property
    def col_marginals(self) -> dict[str, float]:
        """Column totals keyed by column label."""
        mat, _, cidx = self._build()
        c_totals = [sum(mat[r][c] for r in range(len(mat))) for c in range(len(cidx))]
        return {lbl: c_totals[j] for lbl, j in cidx.items()}

    @property
    def grand_total(self) -> float:
        """Sum of all cells."""
        return sum(self._data.values())

    @property
    def observed(self) -> list[list[float]]:
        """Observed frequency matrix (R × C)."""
        mat, _, _ = self._build()
        return mat

    @property
    def expected(self) -> list[list[float]]:
        """Expected frequency matrix under H₀."""
        mat, _, _ = self._build()
        rows, cols = len(mat), len(mat[0])
        r_tot = [sum(mat[r]) for r in range(rows)]
        c_tot = [sum(mat[r][c] for r in range(rows)) for c in range(cols)]
        gt = sum(r_tot)
        exp = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if gt > 0:
                    exp[r][c] = r_tot[r] * c_tot[c] / gt
        return exp

    @property
    def residuals(self) -> list[list[float]]:
        """Raw residuals (observed - expected)."""
        obs, exp = self.observed, self.expected
        return [[obs[r][c] - exp[r][c] for c in range(len(obs[0]))] for r in range(len(obs))]

    @property
    def std_residuals(self) -> list[list[float]]:
        """Standardised (Pearson) residuals."""
        obs, exp = self.observed, self.expected
        res = self.residuals
        return [
            [res[r][c] / math.sqrt(exp[r][c]) if exp[r][c] > 0 else 0.0 for c in range(len(obs[0]))]
            for r in range(len(obs))
        ]

    def cramers_v(self) -> float:
        """Cramér's V effect size for the current table."""
        mat, _, _ = self._build()
        chi2, _, _ = _chi_squared_table(mat)
        n = sum(sum(r) for r in mat)
        return cramers_v(chi2, n, len(mat), len(mat[0]))

    def chi_squared(self) -> tuple[float, float, int]:
        """Run the chi-squared test of independence.

        Freezes the table (no further ``add()`` calls allowed).
        Returns (chi2, p_value, dof).
        """
        self._frozen = True
        mat, _, _ = self._build()
        return _chi_squared_table(mat)
