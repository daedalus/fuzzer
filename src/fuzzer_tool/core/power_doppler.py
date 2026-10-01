"""Power Doppler seed energy: how much of a seed's coverage its mutants move.

Ultrasound power Doppler fires N pulses at a pixel, filters out the strong
static tissue echo, and integrates what is left: energy of moving
scatterers, direction-blind, no frequency estimate, sensitive to slow flow.
Here a seed is the probe, its mutants are the pulses, edges are the pixels:

    mutants of seed s (slow time)  ->  X[n, e] = log2(1 + hits_e)
         |
    mean removal       static clutter: edges every mutant hits alike
         |
    SVD, drop coherent flash: an early reject moves the whole path at once
         |
    CFAR chi^2 test    flow: residual power above the ensemble noise floor
         |
    power = sum PD_e   -> energy in [0, 1] -> SeedScorer 'doppler'

Seeds whose mutants steer many edges locally get energy; seeds whose
mutants change nothing, or only fail parsing en bloc, do not.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping

import numpy as np

from fuzzer_tool.core.chi_squared import chi_squared_critical_value
from fuzzer_tool.core.lru import LRUCache

# Mutants per ensemble (pulses per Doppler frame).
DEFAULT_ENSEMBLE = 32
# Open ensembles held at once: each is an ENSEMBLE x edges float32 matrix.
DEFAULT_MAX_SEEDS = 64
# Edge columns per ensemble; 64 x 32 x 2048 x 4 B bounds the matrices at 16 MiB.
DEFAULT_MAX_EDGES = 2048
# Closed-ensemble scores kept (two floats and a frozenset each).
DEFAULT_MAX_SCORES = 4096
# CFAR false-alarm probability per edge.
FALSE_ALARM = 1e-3
# A component spread over at least this share of the seed's edges is a flash.
COHERENT_FRACTION = 0.5
# ...and over at least this many edges: one toggling edge is never a flash.
MIN_COHERENT_EDGES = 4
# Ordered-statistic CFAR: noise = this quantile of per-edge residual variance.
NOISE_QUANTILE = 0.5
# Per-sample variance floor (log2 units) for fully deterministic targets.
NOISE_FLOOR = 1e-3
# Singular values below this fraction of the largest are numerical zero.
SV_EPS = 1e-9
# First column allocation; grows by doubling up to max_edges.
INITIAL_COLS = 64

_MIN_ENSEMBLE = 3  # mean removal + one clutter rank still leaves a dof


def _components(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-component projections P = U^T y and energies s^2 (descending).

    Samples << edges, so the n x n Gram eigenproblem replaces a full SVD:
    same U and s^2, O(n^2 m) through one matmul.
    """
    w, u = np.linalg.eigh(y @ y.T)
    p = u[:, ::-1].T @ y
    return p, np.maximum(w[::-1], 0.0)


def _clutter_mask(p: np.ndarray, energy: np.ndarray, m: int) -> np.ndarray:
    """Mask of coherent components: participation ratio spans the path.

    For unit v = P_i / s_i, PR = 1 / sum(v^4) counts the edges it moves
    (one edge -> 1, k equal edges -> k); here PR = s^4 / sum(P_i^4).
    """
    live = energy > energy[0] * SV_EPS**2
    p2 = p * p
    pr = energy**2 / np.maximum((p2 * p2).sum(axis=1), np.finfo(float).tiny)
    wide = pr >= max(COHERENT_FRACTION * m, MIN_COHERENT_EDGES)
    return live & wide


@functools.cache
def _cfar_gain(dof: int) -> float:
    """chi^2 critical value at FALSE_ALARM; dof <= ensemble, so few distinct."""
    return chi_squared_critical_value(dof, FALSE_ALARM)


def doppler_power(x: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Clutter-filtered flow power of one ensemble.

    Args:
        x: Samples x edges matrix of log hit counts.

    Returns:
        (power, flow mask per edge, clutter rank removed).
    """
    n, m = x.shape
    none = np.zeros(m, dtype=bool)
    if n < 2 or m == 0:
        return 0.0, none, 0

    # Order-0 wall filter: drop the DC (static path).
    y = x - x.mean(axis=0)
    if not y.any():
        return 0.0, none, 0

    # SVD wall filter: drop spatially coherent motion (flash).
    p, energy = _components(y)
    clutter = _clutter_mask(p, energy, m)
    rank = int(clutter.sum())
    dof = n - 1 - rank
    if dof < 1:
        return 0.0, none, rank

    # Residual power per edge: sum over kept components of (s_i v_ie)^2.
    pk = p[~clutter]
    pd = (pk * pk).sum(axis=0)

    # CFAR: under H0 (noise only) pd / sigma^2 ~ chi^2(dof).
    sigma2 = max(float(np.quantile(pd / dof, NOISE_QUANTILE)), NOISE_FLOOR)
    flow = pd > sigma2 * _cfar_gain(dof)
    return float(pd[flow].sum()) / dof, flow, rank


class _Ensemble:
    """One seed's open slow-time matrix with an edge-id -> column map."""

    __slots__ = ("x", "n", "m", "cap", "ids", "skeys", "sslots", "last")

    def __init__(self, length: int, cap: int) -> None:
        cols = min(INITIAL_COLS, cap)
        self.x = np.zeros((length, cols), dtype=np.float32)
        self.n = 0
        self.m = 0
        self.cap = cap
        self.ids = np.zeros(cols, dtype=np.int64)  # column -> edge id
        self.skeys = np.zeros(0, dtype=np.int64)  # sorted edge ids
        self.sslots = np.zeros(0, dtype=np.int64)  # column of skeys[i]
        # Previous sample's (ids, columns, kept): mutants mostly replay the
        # seed's path in the same SHM order, so the lookup is usually reused.
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _grow(self, need: int) -> None:
        cols = self.x.shape[1]
        if need <= cols:
            return

        cols = min(max(need, cols * 2), self.cap)
        x = np.zeros((self.x.shape[0], cols), dtype=np.float32)
        x[:, : self.m] = self.x[:, : self.m]
        ids = np.zeros(cols, dtype=np.int64)
        ids[: self.m] = self.ids[: self.m]
        self.x, self.ids = x, ids

    def columns(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Columns for *ids* (unique), adding new edges while room lasts.

        Returns (columns of kept ids, kept mask over *ids*).
        """
        last = self.last
        if last is not None and np.array_equal(last[0], ids):
            return last[1], last[2]

        cols, kept = self._lookup(ids)
        self.last = (ids, cols, kept)
        return cols, kept

    def _lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uncached :meth:`columns`; sorted needles keep searchsorted cache-friendly."""
        k = self.skeys.size
        order = np.argsort(ids)
        pos = np.empty(ids.size, dtype=np.int64)
        pos[order] = np.searchsorted(self.skeys, ids[order])
        pos = np.minimum(pos, max(k - 1, 0))
        kept = self.skeys[pos] == ids if k else np.zeros(ids.size, dtype=bool)
        slots = np.zeros(ids.size, dtype=np.int64)
        slots[kept] = self.sslots[pos[kept]]

        # New edges: absent from earlier samples, i.e. zero there already.
        fresh = np.flatnonzero(~kept)[: self.cap - self.m]
        if fresh.size:
            new = np.arange(self.m, self.m + fresh.size)
            self._grow(self.m + fresh.size)
            self.ids[new] = ids[fresh]
            self.m += fresh.size
            slots[fresh] = new
            kept[fresh] = True
            keys = np.concatenate([self.skeys, ids[fresh]])
            order = np.argsort(keys, kind="stable")
            self.skeys = keys[order]
            self.sslots = np.concatenate([self.sslots, new])[order]
        return slots[kept], kept


class PowerDoppler:
    """Per-seed power Doppler over successive mutant executions.

    Args:
        ensemble: Mutants per closed ensemble.
        max_seeds: Open ensembles kept (LRU).
        max_edges: Edge columns per ensemble; extra edges are dropped.
        max_scores: Closed-ensemble scores kept (LRU).
    """

    def __init__(
        self,
        ensemble: int = DEFAULT_ENSEMBLE,
        max_seeds: int = DEFAULT_MAX_SEEDS,
        max_edges: int = DEFAULT_MAX_EDGES,
        max_scores: int = DEFAULT_MAX_SCORES,
    ) -> None:
        if ensemble < _MIN_ENSEMBLE:
            raise ValueError(f"ensemble must be >= {_MIN_ENSEMBLE}, got {ensemble}")
        if max_edges < 1:
            raise ValueError(f"max_edges must be >= 1, got {max_edges}")
        self._ensemble = ensemble
        self._max_edges = max_edges
        self._open: LRUCache = LRUCache(max_seeds)
        self._scores: LRUCache = LRUCache(max_scores, on_evict=self._evicted)
        self._max_power = 0.0
        self._max_stale = False
        self._closed = 0
        self._dropped = 0
        self._samples = 0

    def _evicted(self, _key) -> None:
        self._max_stale = True

    def observe(self, seed_key: str, hits: Mapping[int, int]) -> None:
        """Add one mutant execution of *seed_key*: ``{edge_id: hit count}``."""
        n = len(hits)
        if not n:
            return

        ids = np.fromiter(hits.keys(), dtype=np.int64, count=n)
        counts = np.fromiter(hits.values(), dtype=np.int64, count=n)
        ens = self._open.get(seed_key)
        if ens is None:
            ens = _Ensemble(self._ensemble, self._max_edges)
            self._open[seed_key] = ens

        cols, kept = ens.columns(ids)
        self._dropped += int(ids.size - cols.size)
        ens.x[ens.n, cols] = np.log2(1.0 + counts[kept])
        ens.n += 1
        self._samples += 1
        if ens.n < self._ensemble:
            return

        # Frame complete: score it, start the next one fresh.
        del self._open[seed_key]
        self._close(seed_key, ens)

    def _close(self, seed_key: str, ens: _Ensemble) -> None:
        power, flow, _ = doppler_power(ens.x[:, : ens.m].astype(np.float64))
        old = self._scores.get(seed_key)
        if old is not None and old[0] >= self._max_power:
            self._max_stale = True
        self._scores[seed_key] = (power, frozenset(ens.ids[: ens.m][flow].tolist()))
        self._max_power = max(self._max_power, power)
        self._closed += 1

    def _peak(self) -> float:
        if self._max_stale:
            self._max_power = max((p for p, _ in self._scores.values()), default=0.0)
            self._max_stale = False
        return self._max_power

    def energy(self, seed_key: str) -> float:
        """Seed's flow power normalized to [0, 1] (log scale); 0 if unscored."""
        score = self._scores.get(seed_key)
        peak = self._peak()
        if score is None or peak <= 0.0:
            return 0.0
        return math.log1p(score[0]) / math.log1p(peak)

    def flow_edges(self, seed_key: str) -> frozenset[int]:
        """Input-sensitive edges of the seed's last closed ensemble."""
        score = self._scores.get(seed_key)
        return frozenset() if score is None else score[1]

    def stats(self) -> dict[str, float]:
        """Diagnostics for reports."""
        return {
            "samples": self._samples,
            "open": len(self._open),
            "scored": len(self._scores),
            "ensembles": self._closed,
            "dropped_edges": self._dropped,
            "max_power": self._peak(),
        }
