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
# Closed-ensemble scores kept (power and flow-edge ids each).
DEFAULT_MAX_SCORES = 4096
# Flow-edge ids kept across all scores: 8 B each bounds them at 8 MiB.
DEFAULT_MAX_FLOW_IDS = 1 << 20
# Open frame untouched for this many full frames of samples is abandoned
# (initial horizon; widened to REVISIT_MARGIN x any gap a dropped seed proves).
STALE_FRAMES = 4
# Horizon headroom over the slowest observed revisit gap.
REVISIT_MARGIN = 2
# Dropped keys remembered (key -> last-touch tick); must outlast a corpus
# cycle's worth of drops. ~150 B each bounds it near 5 MiB.
DEFAULT_MAX_DROPPED = 1 << 15
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

    __slots__ = ("x", "n", "m", "cap", "ids", "skeys", "sslots", "last", "touched")

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
        self.touched = 0  # PowerDoppler tick of the last sample

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
        max_seeds: Open ensembles kept. When full, new seeds wait for a
            slot rather than evict partial frames: LRU eviction never closed
            a frame once more than max_seeds seeds were picked in turn.
        max_edges: Edge columns per ensemble; extra edges are dropped.
        max_scores: Closed-ensemble scores kept (LRU).
        max_flow_ids: Flow-edge ids kept across all scores (LRU).
        max_dropped: Dropped-frame keys remembered (LRU) to measure revisit gaps.
    """

    def __init__(
        self,
        ensemble: int = DEFAULT_ENSEMBLE,
        max_seeds: int = DEFAULT_MAX_SEEDS,
        max_edges: int = DEFAULT_MAX_EDGES,
        max_scores: int = DEFAULT_MAX_SCORES,
        max_flow_ids: int = DEFAULT_MAX_FLOW_IDS,
        max_dropped: int = DEFAULT_MAX_DROPPED,
    ) -> None:
        if ensemble < _MIN_ENSEMBLE:
            raise ValueError(f"ensemble must be >= {_MIN_ENSEMBLE}, got {ensemble}")
        if max_seeds < 1:
            raise ValueError(f"max_seeds must be >= 1, got {max_seeds}")
        if max_edges < 1:
            raise ValueError(f"max_edges must be >= 1, got {max_edges}")
        if max_scores < 1:
            raise ValueError(f"max_scores must be >= 1, got {max_scores}")
        if max_flow_ids < 0:
            raise ValueError(f"max_flow_ids must be >= 0, got {max_flow_ids}")
        if max_dropped < 1:
            raise ValueError(f"max_dropped must be >= 1, got {max_dropped}")
        self._ensemble = ensemble
        self._max_seeds = max_seeds
        self._max_edges = max_edges
        self._max_scores = max_scores
        self._max_flow_ids = max_flow_ids
        self._stale_after = max_seeds * ensemble * STALE_FRAMES
        # Dropped-as-abandoned key -> its frame's last-touch tick.
        self._dropped_keys: LRUCache = LRUCache(max_dropped)
        # Insertion order is recency order; capacity is enforced by hand
        # (_admit, _trim) so evicted frames can be scored / ids uncounted.
        self._open: dict[str, _Ensemble] = {}
        self._scores: LRUCache = LRUCache(max_scores + 1)
        self._flow_ids = 0
        self._max_power = 0.0
        self._max_stale = False
        self._closed = 0
        self._dropped = 0
        self._refused = 0
        self._samples = 0
        self._ticks = 0

    def observe(self, seed_key: str, hits: Mapping[int, int]) -> None:
        """Add one mutant execution of *seed_key*: ``{edge_id: hit count}``."""
        n = len(hits)
        if not n:
            return

        self._ticks += 1
        ens = self._open.pop(seed_key, None)
        if ens is None:
            self._revisit(seed_key)
            ens = self._admit()
        if ens is None:
            self._refused += 1
            return

        # Re-insert: most recent last.
        self._open[seed_key] = ens
        ens.touched = self._ticks
        ids = np.fromiter(hits.keys(), dtype=np.int64, count=n)
        counts = np.fromiter(hits.values(), dtype=np.int64, count=n)
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

    def _revisit(self, seed_key: str) -> None:
        """A dropped seed came back: it was slow, not gone. Widen the horizon.

        A fixed horizon thrashes once the corpus cycle outlasts it, e.g. 13
        seeds in turn vs a 12-tick horizon: every frame is dropped one tick
        before its seed returns. The horizon rises to a margin over the
        measured gap, never per return: 64 returns of a 200-tick cycle give
        400, where doubling per key gave 2^64.
        """
        touched = self._dropped_keys.pop(seed_key, None)
        if touched is None:
            return

        gap = self._ticks - touched
        self._stale_after = max(self._stale_after, REVISIT_MARGIN * gap)

    def _admit(self) -> _Ensemble | None:
        """Fresh frame if a slot is free or the oldest frame is abandoned.

        Waiting instead of evicting keeps open frames progressing whatever
        the corpus size, e.g. 200 seeds picked in turn into 64 slots: the
        first 64 fill and close, then the next 64 get their slots.
        """
        if len(self._open) >= self._max_seeds:
            key = next(iter(self._open))
            old = self._open[key]
            if self._ticks - old.touched < self._stale_after:
                return None

            # Abandoned: score what it has rather than throw it away.
            del self._open[key]
            self._dropped_keys[key] = old.touched
            if old.n >= _MIN_ENSEMBLE:
                self._close(key, old)
        return _Ensemble(self._ensemble, self._max_edges)

    def _close(self, seed_key: str, ens: _Ensemble) -> None:
        power, flow, _ = doppler_power(ens.x[: ens.n, : ens.m].astype(np.float64))
        old = self._scores.pop(seed_key, None)
        if old is not None:
            self._drop(old)
        flow_ids = np.sort(ens.ids[: ens.m][flow])
        self._scores[seed_key] = (power, flow_ids)
        self._flow_ids += flow_ids.size
        self._max_power = max(self._max_power, power)
        self._closed += 1
        self._trim()

    def _drop(self, score: tuple[float, np.ndarray]) -> None:
        """Forget *score*'s ids; the peak is recomputed if it was the max."""
        self._flow_ids -= score[1].size
        if score[0] >= self._max_power:
            self._max_stale = True

    def _trim(self) -> None:
        """Evict LRU scores past the count or flow-id budget.

        Flow ids live in compact int64 arrays (8 B per id; a frozenset of
        ints costs ~70 B), and their total is capped as well as the count.
        """
        scores = self._scores
        while len(scores) > self._max_scores or self._flow_ids > self._max_flow_ids:
            _, score = scores.popitem(last=False)
            self._drop(score)

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
        return frozenset() if score is None else frozenset(score[1].tolist())

    def stats(self) -> dict[str, float]:
        """Diagnostics for reports."""
        return {
            "samples": self._samples,
            "open": len(self._open),
            "scored": len(self._scores),
            "ensembles": self._closed,
            "dropped_edges": self._dropped,
            "refused": self._refused,
            "flow_ids": self._flow_ids,
            "stale_after": self._stale_after,
            "max_power": self._peak(),
        }
