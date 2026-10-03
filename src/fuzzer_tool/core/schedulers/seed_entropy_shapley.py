"""Permutation-sampled Shapley for pooled corpus byte-entropy.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §5,
second step. The characteristic function is ``v(S) = H(pooled bytes of S)``;
a seed's Shapley value is its marginal ``v(S + s) - v(S)`` averaged over
random orderings. Unlike leave-one-out it prices near-duplicate cliques
correctly (two copies of a pattern each get half the pattern's credit rather
than 0 each).

One permutation costs one cumulative sum over the permuted ``n x 256`` count
slab plus one ``log2`` pass: O(n * 256), no Python loop over seeds. Each
permutation is also played in reverse (antithetic pair), which cancels most of
the order-dependent variance for submodular games.

Pure functions only; wiring is deliberately not done (the doc gates it on a
measured LOO-vs-Shapley disagreement, see ``tools/entropy_shapley_vs_loo.py``).
"""

from __future__ import annotations

import numpy as np


def _prefix_entropy(cum: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Shannon bits of each cumulative histogram row; 0 where the total is 0."""
    safe = np.maximum(totals, 1).astype(np.float64)
    xlogx = (cum * np.log2(np.maximum(cum, 1))).sum(axis=1)
    return np.where(totals > 0, np.log2(safe) - xlogx / safe, 0.0)


def _marginals(counts: np.ndarray, sizes: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Marginal entropy gain of each seed (indexed by seed) for one ordering."""
    cum = np.cumsum(counts[order], axis=0, dtype=np.float64)
    totals = np.cumsum(sizes[order])
    h = _prefix_entropy(cum, totals)
    gain = np.diff(h, prepend=0.0)
    out = np.empty_like(gain)
    out[order] = gain
    return out


def shapley_entropy(
    counts: np.ndarray,
    sizes: np.ndarray,
    n_perm: int = 64,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo Shapley values of pooled byte entropy.

    *counts* is ``n x 256`` per-seed byte histograms, *sizes* the per-seed
    totals. Runs ``n_perm`` random orderings, each paired with its reverse.
    Returns ``(phi, stderr)`` in bits, both length ``n``. Efficiency holds
    exactly: ``phi.sum() == H(all seeds pooled)``.
    """
    n = counts.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)

    rng = rng or np.random.default_rng()
    counts = np.asarray(counts)
    sizes = np.asarray(sizes)

    # Each antithetic pair is one independent sample (its two halves are
    # negatively correlated), so the stderr is taken over pair means.
    pair_means = np.empty((n_perm, n))
    for k in range(n_perm):
        order = rng.permutation(n)
        pair_means[k] = 0.5 * (
            _marginals(counts, sizes, order) + _marginals(counts, sizes, order[::-1])
        )

    phi = pair_means.mean(axis=0)
    stderr = pair_means.std(axis=0, ddof=1) / np.sqrt(n_perm) if n_perm > 1 else np.zeros(n)
    return phi, stderr


def loo_entropy(counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Leave-one-out ``H(pool) - H(pool without s)`` (reference for the tool)."""
    pool = counts.sum(axis=0, dtype=np.float64)[None, :]
    total = np.array([sizes.sum()])
    h_all = _prefix_entropy(pool, total)[0]
    rest = pool - counts
    return h_all - _prefix_entropy(rest, total[0] - sizes)
