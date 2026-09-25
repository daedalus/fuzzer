"""Dirichlet-Multinomial helpers: digamma and empirical-Bayes concentration.

A symmetric Dirichlet(α) prior over K categories turns raw counts into the
posterior predictive ``(n_k + α) / (N + K·α)`` — the smoothing used by the
Markov chain and the CEM byte model. ``dm_alpha`` learns α from the counts
themselves (maximum marginal likelihood) instead of hand-picking it:

    peaked rows  (one byte per context)  →  α → 0    trust the counts
    flat rows    (every byte distinct)   →  α → ∞    fall back to uniform
"""

from __future__ import annotations

import enum
import logging
import math
from collections.abc import Iterable, Sequence

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

log = logging.getLogger(__name__)

# Bumped when the to_dict() layout changes; older payloads load as fresh.
_STATE_VERSION = 1


class AlphaMode(enum.Enum):
    """How a Dirichlet smoothing α is chosen."""

    FIXED = "fixed"  # hand-set constant
    LEARNED = "learned"  # dm_alpha MLE refit from the counts


# Shift x up to this before the asymptotic series (error < 1e-15 beyond it).
_DIGAMMA_ASYMPTOTIC_MIN = 10.0

# Clamp for the learned α: MLE diverges to 0 / ∞ on degenerate data.
_MIN_ALPHA = 1e-6
_MAX_ALPHA = 1e4

# Dir(1,…,1): uniform prior mass per token before any win.
_TOKEN_PRIOR = 1.0

# Rows with fewer observations carry no information about α.
_MIN_ROW_TOTAL = 2

# Bisection steps on log α: ln(1e10) / 2^40 ≈ 2e-11 relative resolution.
_BISECT_STEPS = 40


def digamma(x: float) -> float:
    """ψ(x) = d/dx ln Γ(x) for x > 0 (recurrence + asymptotic series).

    Example: ``digamma(1.0) == -0.5772156649015329`` (−Euler γ).
    """
    if not x > 0.0:
        raise ValueError(f"digamma needs x > 0, got {x}")

    # Recurrence ψ(x) = ψ(x+1) − 1/x lifts x into the series' accurate range
    acc = 0.0
    while x < _DIGAMMA_ASYMPTOTIC_MIN:
        acc -= 1.0 / x
        x += 1.0

    # ψ(x) ≈ ln x − 1/2x − Σ B_2n / (2n·x^2n)
    inv2 = 1.0 / (x * x)
    series = inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
    return acc + math.log(x) - 0.5 / x - series


def _digamma_arr(x: np.ndarray) -> np.ndarray:
    """Vectorised :func:`digamma` (same recurrence + series), x > 0."""
    x = x.astype(np.float64, copy=True)
    acc = np.zeros_like(x)
    for _ in range(int(_DIGAMMA_ASYMPTOTIC_MIN)):
        low = x < _DIGAMMA_ASYMPTOTIC_MIN
        if not low.any():
            break
        acc -= np.where(low, 1.0 / x, 0.0)
        x = np.where(low, x + 1.0, x)

    inv2 = 1.0 / (x * x)
    series = inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
    return acc + np.log(x) - 0.5 / x - series


def _histograms(rows: Iterable[Iterable[int]]) -> tuple[np.ndarray, ...]:
    """Distinct non-zero cell counts and row totals, with multiplicities.

    The score only depends on how many cells/rows share a value, so each
    bisection step costs O(distinct values), not O(cells).
    Example: rows [[3, 1], [3], [1]] → cells {1: 1, 3: 2}, totals {3: 1, 4: 1}
    (the single-observation row [1] is dropped).
    """
    lens: list[int] = []
    flat: list[int] = []
    for row in rows:
        before = len(flat)
        flat.extend(row)
        lens.append(len(flat) - before)

    vals = np.clip(np.asarray(flat, dtype=np.int64), 0, None)
    sizes = np.asarray(lens, dtype=np.int64)
    sizes = sizes[sizes > 0]
    starts = np.cumsum(sizes) - sizes
    totals = np.add.reduceat(vals, starts) if starts.size else vals[:0]

    # A row with < 2 observations has a flat likelihood in α: drop it
    informative = totals >= _MIN_ROW_TOTAL
    vals = vals[np.repeat(informative, sizes)]
    totals = totals[informative]

    cell_v, cell_m = np.unique(vals[vals > 0], return_counts=True)
    tot_v, tot_m = np.unique(totals, return_counts=True)
    return cell_v, cell_m, tot_v, tot_m


def _score(hist: tuple[np.ndarray, ...], k: int, a: float) -> float:
    """d/dα of the log marginal likelihood (Minka's numerator − denominator)."""
    cell_v, cell_m, tot_v, tot_m = hist
    num = cell_m @ (_digamma_arr(cell_v + a) - digamma(a))
    den = tot_m @ (_digamma_arr(tot_v + k * a) - digamma(k * a))
    return float(num - k * den)


def dm_alpha(rows: Iterable[Iterable[int]], k: int, alpha: float = 1.0) -> float:
    """MLE of the symmetric Dirichlet-Multinomial concentration α.

    Solves the score equation (Minka 2000) over rows c and cells k:

        Σ_c Σ_k [ψ(n_ck + α) − ψ(α)] = K · Σ_c [ψ(N_c + Kα) − ψ(Kα)]

    by bisection on log α. Minka's fixed point needs thousands of steps
    when started far from the optimum; bisection is start-independent.

    Args:
        rows: Per-context counts (zeros allowed, ignored).
        k: Number of categories (256 for bytes).
        alpha: Returned unchanged when there is nothing to fit.

    Returns:
        Learned α, clamped to [1e-6, 1e4].
    """
    hist = _histograms(rows)
    if k < 2 or not hist[0].size:
        return alpha

    # Degenerate data: the likelihood peaks at a boundary
    if _score(hist, k, _MIN_ALPHA) <= 0.0:
        return _MIN_ALPHA
    if _score(hist, k, _MAX_ALPHA) >= 0.0:
        return _MAX_ALPHA

    lo, hi = math.log(_MIN_ALPHA), math.log(_MAX_ALPHA)
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if _score(hist, k, math.exp(mid)) > 0.0:
            lo = mid
            continue
        hi = mid
    return math.exp(0.5 * (lo + hi))


def _token(t) -> bytes:
    """Validate a persisted token key (``bytes(int)`` would zero-fill)."""
    if not isinstance(t, bytes):
        raise TypeError(f"token {t!r} is not bytes")
    return t


class DirichletPicker:
    """Thompson sampling over dictionary tokens with a Dirichlet posterior.

    Per round: p ~ Dir(prior + wins), then n token indices ~ Cat(p). One p
    per round commits the round to one hypothesis; a fresh p per index
    would collapse to the posterior predictive (plain frequency weighting).

        draw(tokens, n) ──► mutate/exec ──► reward(n_used) | clear()

    Wins are keyed by token bytes, so they survive the fuzzer truncating
    or appending to its dictionary between rounds.
    """

    def __init__(self, rng: RandPool, prior: float = _TOKEN_PRIOR) -> None:
        self._rng = rng
        self._prior = prior
        self.wins: dict[bytes, int] = {}
        self._pending: list[int] = []
        self._drawn_from: Sequence[bytes] = ()

        # Cached α vector; rebuilt when the dictionary or the wins change
        self._alpha: np.ndarray | None = None
        self._alpha_src: Sequence[bytes] | None = None
        self._alpha_len = -1

    def draw(self, tokens: Sequence[bytes], n: int) -> list[int]:
        """Sample one p on the simplex, then *n* token indices from it."""
        self._pending = []
        self._drawn_from = tokens
        k = len(tokens)
        if not k or n <= 0:
            return []

        p = self._rng.dirichlet(self._alphas(tokens))
        self._pending = self._rng.categorical(p, n)
        return self._pending

    def reward(self, n_used: int) -> None:
        """Credit each distinct token among the first *n_used* drawn."""
        used = {self._drawn_from[i] for i in self._pending[:n_used]}
        for token in used:
            self.wins[token] = self.wins.get(token, 0) + 1
        if used:
            self._alpha = None
        self.clear()

    def clear(self) -> None:
        """Drop the pending round without credit (no coverage gain)."""
        self._pending = []
        self._drawn_from = ()

    def to_dict(self) -> dict:
        """Per-token wins for ``--resume``; the pending round is not kept."""
        return {"version": _STATE_VERSION, "wins": dict(self.wins)}

    def from_dict(self, data) -> None:
        """Restore :meth:`to_dict` output; a malformed payload is ignored whole."""
        if not data:
            return
        try:
            if data.get("version") != _STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            wins = {_token(t): int(n) for t, n in data["wins"].items()}
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("dict-thompson state unreadable, starting fresh: %s", e)
            return

        self.wins = wins
        self._alpha = None

    def _alphas(self, tokens: Sequence[bytes]) -> np.ndarray:
        fresh = tokens is self._alpha_src and len(tokens) == self._alpha_len
        if self._alpha is not None and fresh:
            return self._alpha

        wins = self.wins
        self._alpha = np.fromiter(
            (self._prior + wins.get(t, 0) for t in tokens), dtype=np.float64, count=len(tokens)
        )
        self._alpha_src = tokens
        self._alpha_len = len(tokens)
        return self._alpha
