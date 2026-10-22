"""SeedDPPScheduler: sequential determinantal point process over coverage.

Plan: ``docs/handover/handover_dpp_seed_selector_2026-10-22.md``.

Kernel ``L = diag(q) · Φ̂Φ̂ᵀ · diag(q)``: ``Φ̂`` is each seed's edge set
hashed into ``HASH_DIM`` buckets and L2-normalised (cosine similarity), and
``q`` is rare-edge quality ``(Σ 1/freq(e))^θ``. Given the last ≤ K picks W::

    P(i | W) ∝ det(L_{W∪i}) / det(L_W) = d_i²      (Schur complement)

``d²`` is updated in place by fast greedy-MAP Cholesky (Chen et al.,
NeurIPS 2018): O(N·(D+K)) per pick, with no N×N matrix. A seed that
duplicates a recent pick drops to ``d² = 0``::

    A = B = {1, 2}, C = {3, 4}     d² = [1, 1, 2]
    pick A                         d² = [0, 0, 2]   ->  C next, never B

This is sequential volume sampling, not exact k-DPP marginals. One
``rng.random()`` per pick (Hard Rule 39).
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.cadence import phase_of
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

#: Φ width 2^HASH_BITS. N×256 float32 is 10 MB at 10k seeds.
HASH_BITS = 8
HASH_DIM = 1 << HASH_BITS
#: Picks conditioned on before the window restarts.
WINDOW = 8
#: Executions between kernel rebuilds (Tang's cadence).
REFIT_INTERVAL = 2000
#: θ in q = (Σ 1/freq)^θ.
QUALITY_EXP = 0.5
#: Total residual at or below this: window exhausted.
RESIDUAL_EPS = 1e-9
#: Fibonacci hashing: edge id -> top bits of a 64-bit product.
_EDGE_MULT = np.uint64(0x9E3779B97F4A7C15)
_BUCKET_SHIFT = np.uint64(64 - HASH_BITS)
_MASK64 = (1 << 64) - 1
#: Edge ids below this count by bincount (≤ 32 MB); larger ids by sort.
_BINCOUNT_MAX = 1 << 22


def _edge_array(edges: set[int]) -> np.ndarray:
    """Edge ids as int64; ids outside int64 wrap mod 2^64."""
    try:
        return np.fromiter(edges, np.int64, len(edges))
    except OverflowError:
        return np.fromiter((e & _MASK64 for e in edges), np.uint64, len(edges)).view(np.int64)


def _unit_row(ids: np.ndarray) -> np.ndarray:
    """Hashed incidence row, L2-normalised; zeros for an edgeless seed."""
    buckets = (ids.view(np.uint64) * _EDGE_MULT) >> _BUCKET_SHIFT
    row = np.bincount(buckets.astype(np.int64), minlength=HASH_DIM).astype(np.float32)
    norm = float(np.sqrt(row @ row))
    return row / norm if norm > 0 else row


def _edge_freq(edges: np.ndarray) -> np.ndarray:
    """Per incidence: how many seeds hit that edge.

    bincount when ids are small (SHM / slot ids), else sort-based;
    ``np.unique(return_inverse=True)`` took 4 s at 20M incidences.
    """
    if int(edges.min()) >= 0 and int(edges.max()) < _BINCOUNT_MAX:
        return np.bincount(edges)[edges]
    uniq, counts = np.unique(edges, return_counts=True)
    return counts[np.searchsorted(uniq, edges)]


def _cdf_index(weights: np.ndarray, u: float) -> int:
    """Inverse-CDF index; zero-weight entries are never returned."""
    cum = np.cumsum(weights)
    i = int(np.searchsorted(cum, u * cum[-1], side="right"))
    if i < len(weights) and weights[i] > 0:
        return i
    return int(np.flatnonzero(weights)[-1])  # float round-off past the end


class SeedDPPScheduler(ArmCounts):
    """Sequential DPP seed sampler over hashed edge-set features.

    Args:
        rng: A ``RandPool`` (Hard Rule 16).
        window: K, picks conditioned on before the window restarts.
        refit_interval: Executions between kernel rebuilds.
    """

    #: Seed arm; selection uses the kernel, not Beta priors.
    supports_priors = False

    def __init__(self, rng, window: int = WINDOW, refit_interval: int = REFIT_INTERVAL) -> None:
        if rng is None:
            raise ValueError("SeedDPPScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng
        self._window = max(1, int(window))
        self._refit_interval = max(1, int(refit_interval))
        self._last_refit_exec = -(1 << 60)
        self.refits = 0
        # key -> (edge ids, unit hashed row); live keys only.
        self._edge_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._phi = np.zeros((0, HASH_DIM), np.float32)  # q-scaled rows
        self._row: dict[str, int] = {}
        self._base = np.zeros(0)  # diag(L) = q²
        self._d2 = np.zeros(0)  # residual given the window
        self._chol: list[np.ndarray] = []  # Cholesky rows, ≤ window
        self._idx_of: tuple[list[str] | None, np.ndarray] = (None, np.zeros(0, np.int64))

    # ------------------------------------------------------------------ fit

    def maybe_refit(self, edge_tracker, exec_count: int) -> bool:
        """Refit if ``refit_interval`` executions passed; first call refits at once."""
        if exec_count - self._last_refit_exec < self._refit_interval:
            return False

        # Phase-shift later refits off the shared multiples of 2000 (see Tang).
        first = self._last_refit_exec < 0
        self._last_refit_exec = exec_count + (
            phase_of("dpp.refit", self._refit_interval) if first else 0
        )
        return self.refit(edge_tracker)

    def refit(self, edge_tracker) -> bool:
        """Rebuild Φ and q from ``edge_tracker.seed_edges``; resets the window.

        Returns False, keeping the previous kernel, when no seed has an edge.
        """
        seed_edges = getattr(edge_tracker, "seed_edges", None)
        if not seed_edges:
            return False
        keys = sorted(seed_edges)
        entries = self._edge_entries(seed_edges, keys)
        lengths = np.fromiter((len(ids) for ids, _row in entries), np.int64, len(keys))
        if int(lengths.sum()) == 0:
            return False

        # Quality: rare edges count more. freq(e) = seeds hitting e; per-seed
        # sums via prefix sums over the flat incidence list.
        inv_freq = 1.0 / _edge_freq(np.concatenate([ids for ids, _row in entries]))
        prefix = np.concatenate(([0.0], np.cumsum(inv_freq)))
        ends = np.cumsum(lengths)
        quality = (prefix[ends] - prefix[ends - lengths]) ** QUALITY_EXP

        phi = np.stack([row for _ids, row in entries]) * quality[:, None].astype(np.float32)
        self._load(keys, phi)
        self.refits += 1
        return True

    def _edge_entries(self, seed_edges, keys: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-seed (ids, unit row), rebuilt only when a seed's set changed size.

        Sets only grow between refits, so length is the change check; the
        set conversion and hashing were most of a 20M-incidence refit.
        """
        cache = self._edge_cache
        fresh: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k in keys:
            edges = seed_edges[k]
            entry = cache.get(k)
            if entry is None or len(entry[0]) != len(edges):
                ids = _edge_array(edges)
                entry = (ids, _unit_row(ids))
            fresh[k] = entry
        self._edge_cache = fresh  # departed seeds drop out
        return [fresh[k] for k in keys]

    def _load(self, keys: list[str], phi: np.ndarray) -> None:
        """Install q-scaled features *phi* (one row per key) and reset."""
        self._phi = np.ascontiguousarray(phi, dtype=np.float32)
        self._row = {k: i for i, k in enumerate(keys)}
        self._idx_of = (None, np.zeros(0, np.int64))
        wide = self._phi.astype(np.float64)
        self._base = (wide * wide).sum(axis=1)
        self._reset()

    def _reset(self) -> None:
        self._chol = []
        self._d2 = self._base.copy()

    # --------------------------------------------------------------- select

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        self._trim(seed_ids)
        if len(seed_ids) == 1:
            return seed_ids[0]

        n = len(seed_ids)
        if not self._row:
            return seed_ids[min(int(self._rng.random() * n), n - 1)]

        idx = self._rows_of(seed_ids)
        weights = self._weights(idx)
        u = self._rng.random()
        if weights.sum() <= RESIDUAL_EPS:
            return seed_ids[min(int(u * n), n - 1)]  # no seed has coverage mass

        pick = _cdf_index(weights, u)
        if idx[pick] >= 0:
            self._condition(int(idx[pick]))
        return seed_ids[pick]

    def _rows_of(self, seed_ids: list[str]) -> np.ndarray:
        """Kernel row per key (-1: newer than the refit).

        Cached on list identity: seed_picker reuses one key list until the
        corpus changes (as DeficitRR relies on). 0.5 ms per pick at 10k seeds.
        """
        src, idx = self._idx_of
        if src is seed_ids and len(idx) == len(seed_ids):
            return idx
        idx = np.fromiter((self._row.get(k, -1) for k in seed_ids), np.int64, len(seed_ids))
        self._idx_of = (seed_ids, idx)
        return idx

    def _weights(self, idx: np.ndarray) -> np.ndarray:
        """Residual per candidate; keys newer than the refit take the live max.

        Restarts the window when every live known seed is conditioned out.
        """
        known = idx >= 0
        weights = np.zeros(len(idx))
        weights[known] = self._d2[idx[known]]
        if self._chol and weights[known].sum() <= RESIDUAL_EPS:
            self._reset()
            weights[known] = self._d2[idx[known]]

        novel = ~known
        if novel.any():
            weights[novel] = weights[known].max() if known.any() else 1.0
        return weights

    def _condition(self, j: int) -> None:
        """Add row j to the window: one incremental Cholesky step.

        ``e = (L_j − Σ c_t[j]·c_t) / √d²_j``, then ``d² −= e²``.
        """
        if len(self._chol) >= self._window:
            self._reset()
        dj = self._d2[j]
        if dj <= RESIDUAL_EPS:
            return

        e = (self._phi @ self._phi[j]).astype(np.float64)
        for c in self._chol:
            e -= c[j] * c
        e /= np.sqrt(dj)

        self._d2 -= e * e
        np.maximum(self._d2, 0.0, out=self._d2)
        self._d2[j] = 0.0
        self._chol.append(e)
