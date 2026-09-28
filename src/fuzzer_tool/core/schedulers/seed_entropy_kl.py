"""Entropy-KL seed strategy: pick seeds whose bytes diverge from the corpus.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §2.

Score a seed by the relative entropy of its own byte distribution against
the pooled distribution of the corpus it sits in::

    score(s) = KL(P_s || Q) = sum_b P_s(b) * log2(P_s(b) / Q(b))

Not the same criterion as a gap between two entropy *numbers*: two seeds
uniform over two different byte pairs carry identical entropy and opposite
divergence, so a scalar comparison ranks them equal and this does not.

KL is evaluated in its cross-entropy form, which is what makes the whole
corpus one matrix-vector product::

    KL = sum_b p log2 p  -  sum_b p log2 q  =  (-H_s) + P_s . (-log2 Q)

``-H_s`` is a static per-seed number, so only the dot product moves when the
pool does. The per-seed distributions live in one slab that grows
amortized and fills evictions from the tail, so a corpus admission costs a
matvec and not a rebuild: measured on a 2000-seed corpus, the pick after an
admission is 1.0 ms against 6.6 ms when the slab was re-stacked each time,
and a pick that changes nothing is 0.33 ms. A prune is the expensive
direction -- each evicted seed unfolds from the pool at ~50 us -- but it
runs once per prune, not once per pick.

The pool is owned here rather than read off ``Fuzzer._corpus_entropy``:
that tracker is rebuilt once per ``load_corpus()`` and never sees a seed
discovered mid-campaign, so scoring against it would measure every seed
found during the run against a distribution that predates it -- and it has
no way to drop a seed the corpus pruned. This one folds on admission and
unfolds on eviction, so ``Q`` is exactly the live corpus.

Length calibration. Plug-in KL from an n-byte sample is biased upward by
about (K - 1) / (2n) nats, so a 16-byte seed drawn from the pool's own
distribution (true KL 0) out-scored a 4 KiB seed that truly diverged, and
selection -- proportional to score -- over-picked short seeds. Measured on
corpora whose seeds all come from one distribution (6 runs, lengths
16-4096): Spearman(score, length) = -0.99 and seeds <= 64 B drew 2.25x
their uniform share of the selection weight. :meth:`scores` now returns
``clip((KL - E0(n)) / sd0(n), 0, Z_CAP)``: how many null standard deviations
the seed sits above what a seed of the same sample length would show if it
were drawn from the pool. Same corpora: Spearman +0.01, short-seed share
0.99x, and the AUC for telling a diverging seed from a matching one 0.73 ->
0.98. :meth:`raw_scores` keeps the plug-in KL in bits.

Subtracting E0 alone is not enough: the mean is fixed but short seeds are
far noisier, so weight proportional to the clipped excess still gave short
seeds 2.3-3.0x their share. Dividing by the null spread is what evens it
out. ``E0`` is exact (:func:`null_kl_bits`; the chi-square expansion
overshoots 5x below n ~ K); ``sd0`` is Monte-Carlo
(:func:`null_kl_sd_bits`). Limits: the pool contains the seed being scored
(mild downward bias for a seed that is a large share of it -- see
``seed_entropy_loo``), and Z_CAP is a heuristic, not a derived constant.

``Q`` is smoothed (``byte_entropy.POOL_SMOOTHING``) so a byte value the pool
has never seen keeps positive probability. Every seed passed to
:meth:`scores` is folded before it is scored, so its support is already in
the pool and the floor only matters at the margin -- it is there because an
unsmoothed ``log2(p/0)`` is undefined, not because it tunes anything.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fuzzer_tool.core.byte_entropy import (
    ENTROPY_SAMPLE_CAP,
    CumulativeByteEntropy,
    byte_histogram,
    entropy_bits_from_counts,
)

#: Floor added to every weight so a seed matching the pool exactly (KL 0)
#: is rare rather than unreachable. Matches seed_kruskal_count.
MIN_WEIGHT = 1e-6

#: The Monte-Carlo null spread is rebuilt once the pool has drifted this far
#: (L1 distance between the pool distribution it was built from and the
#: current one). The spread moves proportionally with the drift, so a 2 %
#: drift is a ~2 % error on a z-score; a swap in a large corpus never trips
#: this, an admission into a tiny one always does.
NULL_STALE_L1 = 0.02

#: Excess KL is expressed in null standard deviations and clipped here: a
#: 4 KiB seed 100 sigma out is no more worth picking than one 8 sigma out,
#: and without a clip it would starve every other seed of selection weight.
Z_CAP = 8.0

#: Draws and seed for the Monte-Carlo null spread. A private, fixed-seed
#: generator: this fixes a calibration table, it never drives a scheduling
#: decision, so it stays reproducible and off the campaign's RandPool.
NULL_MC_DRAWS = 200
NULL_MC_SEED = 0x4B4C

#: Floor on the null spread, in bits, so a degenerate pool (one byte value)
#: whose draws never vary does not divide by zero.
NULL_MIN_SD = 1e-6

_LN2 = math.log(2.0)


def null_kl_bits(q: Any, n: int) -> float:
    """Exact E[KL(P_hat_n || q)] in bits for n i.i.d. draws from ``q``.

    With counts c_b ~ Binomial(n, q_b) the size-biased identity
    E[c f(c)] = n q E[f(Y + 1)], Y ~ Binomial(n - 1, q), turns
    E[sum_b (c_b / n) ln(c_b / (n q_b))] into::

        sum_b q_b * (E[ln(1 + Y_b)] - ln q_b)  -  ln n

    Bins with equal ``q`` share one pmf row, so a pool over few distinct
    byte values is cheap. n = 1 gives H(q); 0 for n < 1.
    """
    if n < 1:
        return 0.0
    probs = np.clip(np.asarray(q, dtype=np.float64), 1e-300, 1.0 - 1e-12)
    uniq, mult = np.unique(probs, return_counts=True)
    m = n - 1
    # Only k within ~12 sigma of the largest bin mean carries any pmf mass;
    # the rest underflows, so the (bins x k) matrix stops there rather than
    # at n. Cuts the n = 4096 grid point from 4096 columns to a few hundred.
    mu_max = m * float(uniq[-1])
    kmax = min(m, int(mu_max + 12.0 * math.sqrt(mu_max) + 20.0))
    k = np.arange(kmax + 1, dtype=np.float64)
    lf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, m + 1, dtype=np.float64)))))
    log_choose = lf[m] - lf[: kmax + 1] - lf[m - kmax : m + 1][::-1]
    log_pmf = (
        log_choose[None, :]
        + k[None, :] * np.log(uniq)[:, None]
        + (m - k)[None, :] * np.log1p(-uniq)[:, None]
    )
    e_log1p = np.exp(log_pmf) @ np.log1p(k)
    nats = float(np.dot(mult * uniq, e_log1p - np.log(uniq))) - math.log(n)
    return max(nats / _LN2, 0.0)


def null_kl_sd_bits(q: Any, grid: Any, draws: int = NULL_MC_DRAWS, seed: int = NULL_MC_SEED) -> Any:
    """Std-dev in bits of KL(P_hat_n || q) under the null, for each n in ``grid``.

    The mean is exact (:func:`null_kl_bits`); the spread has no comparably
    cheap closed form in the sparse regime, so it is measured on ``draws``
    multinomial samples per n from a fixed-seed generator.
    """
    probs = np.asarray(q, dtype=np.float64)
    probs = probs / probs.sum()
    gen = np.random.Generator(np.random.PCG64(seed))
    out = []
    for n in np.asarray(grid, dtype=np.int64):
        freq = gen.multinomial(int(n), probs, size=draws).astype(np.float64) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(freq > 0, freq * np.log2(freq / probs), 0.0)
        out.append(float(terms.sum(axis=1).std(ddof=1)))
    return np.asarray(out)


#: Rows the distribution slab starts with; it doubles from there.
SLAB_INITIAL_ROWS = 256
STATE_VERSION = 1

_COUNTERS = ("scored", "selected")


def _valid_state(data: Any) -> bool:
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return False

    return all(type(data.get(key, 0)) is int and data.get(key, 0) >= 0 for key in _COUNTERS)


def _null_grid(cap: int) -> np.ndarray:
    """Powers of two up to ``cap``, plus ``cap`` itself."""
    grid = [1 << i for i in range(max(cap, 1).bit_length()) if (1 << i) <= cap]
    if grid[-1] != cap:
        grid.append(max(cap, 1))
    return np.asarray(grid, dtype=np.float64)


class EntropyKLSeedStrategy:
    """Elo-arbitrated ``entropy_kl`` seed arm (``--entropy-kl``)."""

    def __init__(self, rng: Any, cap: int = ENTROPY_SAMPLE_CAP) -> None:
        self._rng = rng
        self._cap = cap
        self._pool = CumulativeByteEntropy()
        # Slab of per-seed byte distributions, one row each, parallel to
        # _keys; _index maps a seed to its row.
        self._probs = np.zeros((SLAB_INITIAL_ROWS, 256), dtype=np.float64)
        self._neg_ent = np.zeros(SLAB_INITIAL_ROWS, dtype=np.float64)
        # Sample length (capped) behind each row: the n in E0(n).
        self._n = np.zeros(SLAB_INITIAL_ROWS, dtype=np.int64)
        self._keys: list[bytes] = []
        self._index: dict[bytes, int] = {}
        self._kl: dict[bytes, float] = {}
        self._raw_kl: dict[bytes, float] = {}
        # E0(n) sampled on a power-of-two grid, in the pool it was built for.
        self._null_q: np.ndarray | None = None
        self._null_log_n = np.log(_null_grid(cap))
        self._null_log_v = np.zeros_like(self._null_log_n)
        self._null_log_sd = np.zeros_like(self._null_log_n)
        # Bumped on every fold/unfold; _kl is stale while it disagrees with
        # _scored_at, which is cheaper than diffing 256 pooled bins.
        self._pool_version = 0
        self._scored_at = -1
        self._scored = 0
        self._selected = 0

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Length-calibrated divergence of every seed from the pooled distribution.

        ``clip((KL - E0(n)) / sd0(n), 0, Z_CAP)``: how many null standard
        deviations a seed's KL sits above what a seed of the same sample
        length would show if it were drawn from the pool. Dimensionless.
        """
        self._sync(seeds)
        if self._scored_at != self._pool_version:
            self._refresh()
        return [self._kl.get(s, 0.0) for s in seeds]

    def raw_scores(self, seeds: list[bytes]) -> list[float]:
        """Uncalibrated plug-in KL (biased upward for short seeds)."""
        self.scores(seeds)
        return [self._raw_kl.get(s, 0.0) for s in seeds]

    def miller_madow_scores(self, seeds: list[bytes]) -> list[float]:
        """Plug-in KL minus the Miller-Madow bias, ``(K_hat - 1) / (2 n ln 2)`` bits.

        ``K_hat`` is the number of distinct byte values in the seed's sample.
        Only a first-order correction: in the sparse regime (n < K) most bins
        are empty, ``K_hat`` is far below the K the bias actually scales
        with, and it under-corrects -- kept as the cheap baseline that
        :meth:`scores` is measured against, not as the scheduling score.
        """
        raw = np.asarray(self.raw_scores(seeds), dtype=np.float64)
        rows = [self._index.get(seed, -1) for seed in seeds]
        out = np.zeros(len(seeds), dtype=np.float64)
        for i, row in enumerate(rows):
            n = int(self._n[row]) if row >= 0 else 0
            if n < 1:
                continue
            distinct = int(np.count_nonzero(self._probs[row]))
            out[i] = max(raw[i] - (distinct - 1) / (2.0 * n * _LN2), 0.0)
        return out.tolist()

    def _null_bits(self, n: int) -> float:
        """E0(n) from the cached curve, log-log interpolated; 0 for n < 1."""
        if n < 1:
            return 0.0
        return float(np.exp(np.interp(math.log(n), self._null_log_n, self._null_log_v)))

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw proportional to ``KL + MIN_WEIGHT``; None on an empty corpus."""
        if not seeds:
            return None

        weights = [w + MIN_WEIGHT for w in self.scores(seeds)]
        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    def _sync(self, seeds: list[bytes]) -> None:
        """Make the pool the live corpus: fold admissions, unfold evictions."""
        live = set(seeds)
        for seed in live.difference(self._index):
            self._fold(seed)

        for seed in [s for s in self._keys if s not in live]:
            self._unfold(seed)

    def _fold(self, seed: bytes) -> None:
        counts, total = byte_histogram(seed, self._cap)
        row = np.asarray(counts, dtype=np.float64)
        if total:
            row /= total

        used = len(self._keys)
        self._reserve(used + 1)
        self._probs[used] = row
        self._neg_ent[used] = -entropy_bits_from_counts(counts, total)
        self._n[used] = total
        self._index[seed] = used
        self._keys.append(seed)
        self._scored += 1
        if not total:
            return  # an empty seed is a zero row and contributes no bytes

        self._pool.add(seed, self._cap)
        self._pool_version += 1

    def _unfold(self, seed: bytes) -> None:
        # Fill the hole from the tail rather than shifting the slab down.
        row, last = self._index.pop(seed), len(self._keys) - 1
        if row != last:
            self._probs[row] = self._probs[last]
            self._neg_ent[row] = self._neg_ent[last]
            self._n[row] = self._n[last]
            self._keys[row] = self._keys[last]
            self._index[self._keys[row]] = row

        self._keys.pop()
        self._kl.pop(seed, None)
        self._raw_kl.pop(seed, None)
        if not seed[: self._cap]:
            return

        self._pool.remove(seed, self._cap)
        self._pool_version += 1

    def _reserve(self, size: int) -> None:
        """Grow the slab to hold ``size`` rows, doubling to stay amortized."""
        capacity = self._probs.shape[0]
        if size <= capacity:
            return

        used = len(self._keys)
        capacity = max(size, capacity * 2)
        probs = np.zeros((capacity, 256), dtype=np.float64)
        probs[:used] = self._probs[:used]
        neg_ent = np.zeros(capacity, dtype=np.float64)
        neg_ent[:used] = self._neg_ent[:used]
        lengths = np.zeros(capacity, dtype=np.int64)
        lengths[:used] = self._n[:used]
        self._probs, self._neg_ent, self._n = probs, neg_ent, lengths

    def _refresh(self) -> None:
        """Recompute every seed's KL against the current pool, in one matvec."""
        self._scored_at = self._pool_version
        used = len(self._keys)
        if not used:
            self._kl = {}
            self._raw_kl = {}
            return
        q = np.asarray(self._pool.freq_dist(), dtype=np.float64)
        self._refresh_null(q)
        neg_log_q = -np.log2(q)
        # KL >= 0 analytically; the subtraction can land a hair below zero.
        raw = np.maximum(self._neg_ent[:used] + self._probs[:used] @ neg_log_q, 0.0)
        lengths = self._n[:used]
        null = np.where(
            lengths > 0,
            np.exp(np.interp(np.log(np.maximum(lengths, 1)), self._null_log_n, self._null_log_v)),
            0.0,
        )
        log_len = np.log(np.maximum(lengths, 1))
        spread = np.exp(np.interp(log_len, self._null_log_n, self._null_log_sd))
        z = np.clip((raw - null) / spread, 0.0, Z_CAP)
        self._raw_kl = dict(zip(self._keys, raw.tolist(), strict=True))
        self._kl = dict(zip(self._keys, z.tolist(), strict=True))

    def _refresh_null(self, q: np.ndarray) -> None:
        """Rebuild the null curves for pool distribution ``q``.

        The mean E0 is exact and cheap (1-17 ms on the grid) and is measured
        against a spread of only a few thousandths of a bit for long seeds,
        so a 2 % pool drift already moves a z-score by ~0.3: it is rebuilt on
        every pool change. The Monte-Carlo spread costs 15-40 ms but moves
        proportionally with the drift, so it is reused until the pool has
        moved :data:`NULL_STALE_L1` from the one it was built for.
        """
        grid = np.exp(self._null_log_n).round()
        vals = [null_kl_bits(q, int(n)) for n in grid]
        self._null_log_v = np.log(np.maximum(vals, 1e-12))
        ref = self._null_q
        if ref is not None and float(np.abs(q - ref).sum()) <= NULL_STALE_L1:
            return
        self._null_log_sd = np.log(np.maximum(null_kl_sd_bits(q, grid), NULL_MIN_SD))
        self._null_q = q

    def stats(self) -> dict[str, Any]:
        live = self._kl.values()
        return {
            "scored": self._scored,
            "selected": self._selected,
            "pooled": len(self._keys),
            "pool_bytes": len(self._pool),
            "mean_kl": math.fsum(live) / len(live) if live else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Counters only; scores are a pure function of the corpus bytes."""
        return {
            "version": STATE_VERSION,
            "scored": self._scored,
            "selected": self._selected,
        }

    @classmethod
    def from_dict(cls, data: Any, rng: Any, cap: int = ENTROPY_SAMPLE_CAP) -> EntropyKLSeedStrategy:
        """Restore counters; a malformed or unversioned payload is ignored whole."""
        out = cls(rng, cap)
        if not _valid_state(data):
            return out

        out._scored = data.get("scored", 0)
        out._selected = data.get("selected", 0)
        return out
