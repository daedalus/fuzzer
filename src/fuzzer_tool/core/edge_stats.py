"""Edge-coverage distribution statistics.

    global edge hits ──► hit_profile() ──► HitProfile  top-k share, quantiles, decades
    coverage timeline ─► plateau()     ──► Plateau     two-segment least-squares break
    owner counts, m ───► rarefy()      ──► E[edges]    exact, at t of m seeds
    seed incidence ────► boot_ci()     ──► (lo, hi)    seed resample, percentile CI

Rarefaction answers "how many edges would t seeds have found" without
Chao2's extrapolation. Chao2 reads Q1/Q2, which coverage-guided admission
inflates (every new edge starts with one owner); rarefaction only
interpolates inside the observed corpus.

References: Colwell et al. (2012) "Models and estimators linking
individual-based and sample-based rarefaction"; Efron (1979) bootstrap.
"""

import enum
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

TOP_K = (1, 10, 100)
QUANTILES = (10, 50, 90, 99)
DECADES = 10  # 10^0 .. 10^9 covers uint32 hit counters
MIN_SEGMENT = 3  # points per side of the plateau break
KEXEC = 1000  # rates are edges per 1000 execs
# Tail at or below this share of the head rate is a plateau, above it a
# slowdown. Scale-free: fuzzgoat 13.65 -> 0.64 (5%) plateaus, FFmpeg 5k
# 5171 -> 1004 edges/kexec (19%) is still growing.
PLATEAU_RATIO = 0.1
BOOT_RESAMPLES = 200
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

_POW10 = 10.0 ** np.arange(DECADES)


@dataclass(frozen=True)
class HitProfile:
    """How concentrated hit volume is across edges."""

    edges: int
    total: int
    top_share: dict[int, float]  # {k: share of hits on the k hottest edges}
    quantiles: dict[int, float]  # {percentile: hits}
    decades: tuple[int, ...]  # edges with hits in [10^d, 10^(d+1))


class CurveShape(enum.Enum):
    """Coverage curve after the break, by tail / head rate."""

    PLATEAU = "Plateau"
    SLOWDOWN = "Slowdown"
    SPEEDUP = "Speedup"


@dataclass(frozen=True)
class Plateau:
    """Best two-segment fit of the coverage curve."""

    exec_at: int  # first exec of the tail segment
    head_rate: float  # edges / kexec before the break
    tail_rate: float  # edges / kexec after it
    tail_edges: int  # edges gained from the break to the end

    @property
    def shape(self) -> CurveShape:
        # A flat curve (0 -> 0) is a plateau.
        if self.tail_rate <= PLATEAU_RATIO * max(self.head_rate, 0.0):
            return CurveShape.PLATEAU
        if self.tail_rate < self.head_rate:
            return CurveShape.SLOWDOWN
        return CurveShape.SPEEDUP


def hit_profile(hits: Iterable[int]) -> HitProfile | None:
    """Concentration of *hits* (per-edge hit counts); None when all zero."""
    arr = np.fromiter(hits, dtype=np.float64)
    arr = arr[arr > 0]
    if not arr.size:
        return None

    # Shares from the descending cumulative sum.
    cum = np.cumsum(np.sort(arr)[::-1])
    total = cum[-1]
    n = arr.size
    top = {k: float(cum[min(k, n) - 1] / total) for k in TOP_K}

    # Decade by comparison, not floor(log10): log10(1000) may land on 2.999...
    dec = np.searchsorted(_POW10, arr, side="right") - 1
    decades = np.bincount(dec, minlength=DECADES)
    qs = np.percentile(arr, QUANTILES)
    return HitProfile(
        edges=n,
        total=int(total),
        top_share=top,
        quantiles={q: float(v) for q, v in zip(QUANTILES, qs, strict=True)},
        decades=tuple(int(c) for c in decades),
    )


def _segment_sse(sx, sy, sxx, sxy, syy, n):
    """Least-squares residual and slope per segment from running sums."""
    vx = sxx - sx * sx / n
    cxy = sxy - sx * sy / n
    vy = syy - sy * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(vx > 0, cxy / vx, 0.0)
    return vy - slope * cxy, slope


def plateau(timeline: Sequence[tuple[int, int]]) -> Plateau | None:
    """Split *timeline* ``[(execs, edges), ...]`` where two lines fit best.

    O(n): prefix sums give every split's SSE at once. None with fewer than
    ``2 * MIN_SEGMENT`` points or no exec spread.
    """
    n = len(timeline)
    if n < 2 * MIN_SEGMENT:
        return None
    pts = np.asarray(timeline, dtype=np.float64)[:, :2]
    x_raw, y_raw = pts[:, 0], pts[:, 1]
    if np.ptp(x_raw) == 0:
        return None

    # Center before squaring: exec counts near 1e9 cancel in raw sums.
    x = x_raw - x_raw.mean()
    y = y_raw - y_raw.mean()
    cols = np.stack([np.ones(n), x, y, x * x, x * y, y * y])
    pre = np.concatenate([np.zeros((6, 1)), np.cumsum(cols, axis=1)], axis=1)

    # Split b: head = [0, b), tail = [b, n).
    b = np.arange(MIN_SEGMENT, n - MIN_SEGMENT + 1)
    head = pre[:, b]
    tail = pre[:, -1:] - head
    sse_h, slope_h = _segment_sse(head[1], head[2], head[3], head[4], head[5], head[0])
    sse_t, slope_t = _segment_sse(tail[1], tail[2], tail[3], tail[4], tail[5], tail[0])
    best = int(np.argmin(sse_h + sse_t))

    brk = int(b[best])
    return Plateau(
        exec_at=int(x_raw[brk]),
        head_rate=float(slope_h[best] * KEXEC),
        tail_rate=float(slope_t[best] * KEXEC),
        tail_edges=int(y_raw[-1] - y_raw[brk]),
    )


def rarefy(owners: Iterable[int], m: int, ts: Iterable[int]) -> np.ndarray:
    """Expected distinct edges in a random t-of-*m* seed subset, per t in *ts*.

    E[S(t)] = sum_e 1 - C(m - q_e, t) / C(m, t), q_e = seeds owning edge e.
    Exact (no sampling); log-factorial table, grouped by distinct q.
    Unowned edges (q <= 0) are dropped; q is clipped to m, t to [0, m].
    """
    q = np.fromiter(owners, dtype=np.int64)
    q = np.minimum(q[q > 0], max(m, 1))
    t = np.clip(np.fromiter(ts, dtype=np.int64), 0, max(m, 0))
    if not q.size or m <= 0:
        return np.zeros(t.size)

    # log k! for k in [0, m]; log C(a, t) = lf[a] - lf[t] - lf[a - t].
    lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m + 1)))])
    uq, mult = np.unique(q, return_counts=True)
    a = (m - uq)[:, None]
    tt = t[None, :]
    with np.errstate(invalid="ignore"):
        log_ratio = lf[a] - lf[np.maximum(a - tt, 0)] - lf[m] + lf[m - tt]
    miss = np.where(a >= tt, np.exp(log_ratio), 0.0)
    return (mult[:, None] * (1.0 - miss)).sum(axis=0)


def kept_fraction(owners: np.ndarray) -> float:
    """Share of edges a resample still owns."""
    return float((owners > 0).mean())


def boot_ci(
    incidence: Sequence[Iterable[int]],
    stat: Callable[[np.ndarray], float],
    resamples: int,
    rng,
) -> tuple[float, float] | None:
    """Percentile CI of ``stat(owners)`` over seed resamples with replacement.

    *owners* holds, per observed edge, its owner count in the resample (0 =
    lost). *rng* is a RandPool (``randrange_list``). None with < 2 seeds,
    no edges or no resamples.
    """
    m = len(incidence)
    if m < 2 or resamples < 1:
        return None
    lens = np.fromiter((len(s) for s in incidence), dtype=np.int64, count=m)
    if not lens.sum():
        return None

    # Flat incidence: seed_of[i] owns edge_of[i] (dense edge index).
    flat = np.fromiter((e for s in incidence for e in s), dtype=np.int64, count=int(lens.sum()))
    _, edge_of = np.unique(flat, return_inverse=True)
    n_edges = int(edge_of.max()) + 1
    seed_of = np.repeat(np.arange(m), lens)

    reps = np.empty(resamples)
    for r in range(resamples):
        w = np.bincount(rng.randrange_list(m, m), minlength=m)
        reps[r] = stat(np.bincount(edge_of, weights=w[seed_of], minlength=n_edges))
    low, high = np.percentile(reps, [CI_LOW_PCT, CI_HIGH_PCT])
    return float(low), float(high)
