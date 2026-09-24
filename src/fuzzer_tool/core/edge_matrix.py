"""The seed x edge-class matrix that ``seed_residual`` and ``op_credit`` both read.

Design and evidence: ``docs/handover/handover_edge_id_axis_2026-09-18.md``, section
"A scheduler built on all of this" (P2-3, P3-3, P3-4). What this module adds on top of
``core/scheduler_substrate.py`` (which owns coverage trust, ``effective_edges`` and
``EdgeCanonicalizer``, and is reused here rather than restated):

* :class:`MatrixSubstrate` -- one refit cadence, one canonical edge space, and one gate
  (``coverage_trust``, fed the edge-id stability probe) for both arms.
* :class:`MatrixFold` -- the per-seed quantities a scorer needs, computed on classes:
  ``total`` (hit volume), ``degree`` (classes touched) and ``mass`` (``sum 1/owners``
  over those classes: incidence, never volume, because rho(owners, total) = +0.957 is
  exactly how the two were confused once, F6).
* Rank helpers with no scipy (Hard Rule 51), including the partial rank correlation the
  falsification monitor needs.

Nothing here draws random numbers, and nothing here uses a statistic the handover
excludes: no id-axis quantity (F3, F12), no low-rank seed score, no l2 mass, no GF(2)
minimiser (F8).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer, coverage_trust

log = logging.getLogger(__name__)

#: Nonzero (seed, edge) cells above which the fold is skipped: refit and fold are
#: linear in them, and a refit must not stall the campaign loop. ~80 ms here,
#: the cost of the old dense fold at its 2e6-cell bound (90 ms).
NNZ_BUDGET = 200_000
MIN_SEEDS = 3
DEFAULT_REFIT_INTERVAL = 2000
#: Snapshots kept for the saturation signal.
SATURATION_WINDOW = 8
#: Raw-edge growth over the window at or below which the edge count reads as "flat".
FLAT_EDGE_GROWTH = 0.01
#: Fraction of the window's starting 2^H whose loss saturates the signal. The shape
#: (2^H falling while the edge count is flat) is the handover's; this scale is not
#: calibrated, so consumers only widen exploration with it, never stop on it.
SATURATION_SCALE = 0.25


def average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based ranks with ties given their average rank."""
    _uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    return (first + (counts - 1) / 2.0)[inverse]


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; 0.0 when either side is constant or there are under 3 points."""
    if len(x) < 3 or len(x) != len(y):
        return 0.0
    rx, ry = average_ranks(np.asarray(x, float)), average_ranks(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def partial_rank_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of *x* and *y* controlling for *z*.

    ``(r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))``. This is the statistic that
    read +0.006 (Wilcoxon p = 1.0, ten matrices) for Tang's score against total hits:
    a derived seed score that is a row sum in disguise. 0.0 when *z* explains either
    side completely (denominator zero), which is the same verdict.
    """
    rxy, rxz, ryz = rank_corr(x, y), rank_corr(x, z), rank_corr(y, z)
    den = math.sqrt(max(0.0, (1 - rxz * rxz) * (1 - ryz * ryz)))
    return (rxy - rxz * ryz) / den if den > 1e-12 else 0.0


def residualize_ranks(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of ``rank(y)`` after ordinary least squares on ``rank(x)``.

    This is "regress out rank(total hits)" made the estimator rather than the audit:
    whatever a score carries beyond volume is what is left. All zeros when *x* is
    constant (nothing to regress on) and when *y* is a perfect monotone of *x*.
    """
    ry, rx = average_ranks(np.asarray(y, float)), average_ranks(np.asarray(x, float))
    rxc = rx - rx.mean()
    var = float((rxc * rxc).sum())
    slope = float((rxc * (ry - ry.mean())).sum() / var) if var > 0 else 0.0
    return ry - ry.mean() - slope * rxc


@dataclass
class MatrixFold:
    """One fold of the tracker's seed x edge matrix onto canonical classes.

    All arrays are index-aligned with ``seed_keys``.
    """

    seed_keys: list[str]
    seed_classes: list[np.ndarray]
    class_owners: dict[int, int]
    total: np.ndarray
    degree: np.ndarray
    mass: np.ndarray
    n_edges: int
    n_classes: int

    @property
    def n_seeds(self) -> int:
        return len(self.seed_keys)


def profiles_from_tracker(tracker) -> dict[str, dict[int, int]]:
    """Seed -> {edge: count}. A seed with edges but no counts reads as 1 per edge."""
    counts = getattr(tracker, "seed_hit_counts", {}) or {}
    out: dict[str, dict[int, int]] = {}
    for key, edges in (getattr(tracker, "seed_edges", {}) or {}).items():
        hc = counts.get(key)
        out[key] = dict(hc) if hc else {e: 1 for e in edges}
    return out


def build_fold(
    profiles: Mapping[str, Mapping[int, int]],
    canon: EdgeCanonicalizer,
    derived: frozenset[int] | set[int] = frozenset(),
    cell_budget: int = NNZ_BUDGET,
) -> tuple[MatrixFold | None, str]:
    """Refit *canon* on *profiles* and fold them; ``(None, reason)`` when it cannot.

    Args:
        profiles: Seed key -> {edge id: hit count}.
        canon: Canonicalizer to refit. Classes are rebuilt wholesale every call, never
            merged incrementally, so a class that one new input splits is split here.
        derived: Edges determined by others (independent-coordinate mask). They
            carry no mass and earn no credit. Nothing populates this: P1-2 found most
            count relations are not node laws (edge-id handover F17), so only a
            graph-derived mask may fill it.
        cell_budget: Bound on nonzero (seed, edge) cells. Was seeds x edges; the
            name is kept so positional callers do not break.
    """
    keys = list(profiles)
    if len(keys) < MIN_SEEDS:
        return None, f"fewer than {MIN_SEEDS} seeds"
    edges = {e for hc in profiles.values() for e in hc}
    if not edges:
        return None, "empty matrix"
    nnz = sum(len(hc) for hc in profiles.values())
    if nnz > cell_budget:
        return None, f"{nnz} nonzeros over budget {cell_budget}"

    canon.refit(profiles)
    per_seed, owners, mass = _fold_classes(profiles, canon, derived)
    total = np.array([float(sum(hc.values())) for hc in profiles.values()])
    return (
        MatrixFold(
            seed_keys=keys,
            seed_classes=per_seed,
            class_owners=owners,
            total=total,
            degree=np.array([len(c) for c in per_seed], dtype=np.float64),
            mass=mass,
            n_edges=len(edges),
            n_classes=len(owners),
        ),
        "",
    )


def _fold_classes(profiles, canon: EdgeCanonicalizer, derived):
    """Per-seed sorted classes, class owner counts and 1/owners mass, over all cells at once.

    One (seed, class) pair per seed and class however many member edges the seed
    hits, so a chain counts once: unique the pairs, then owners is a bincount
    over classes and mass a weighted bincount over seeds.
    """
    rows = list(profiles.values())
    lens = [len(hc) for hc in rows]
    edges = np.fromiter(itertools.chain.from_iterable(rows), dtype=np.int64, count=sum(lens))
    seed = np.repeat(np.arange(len(rows)), lens)
    if derived:
        keep = ~np.isin(edges, np.fromiter(derived, dtype=np.int64, count=len(derived)))
        edges, seed = edges[keep], seed[keep]

    # Dense class index, then one int key per (seed, class): 1-D unique is a sort,
    # where unique over rows is a structured-array sort several times slower.
    labels, cls_idx = np.unique(canon.class_array(edges), return_inverse=True)
    pair = np.unique(seed * len(labels) + cls_idx.ravel())
    seed, cls_idx = np.divmod(pair, max(len(labels), 1))
    owner_n = np.bincount(cls_idx, minlength=len(labels))
    mass = np.bincount(seed, weights=1.0 / owner_n[cls_idx], minlength=len(rows))

    # pairs are sorted by (seed, class), so each seed's classes are a sorted run.
    per_seed = np.split(labels[cls_idx], np.searchsorted(seed, np.arange(1, len(rows))))
    owners = dict(zip(labels.tolist(), owner_n.tolist(), strict=True))
    return per_seed, owners, mass


class MatrixSubstrate:
    """Fold, refit cadence and preflight gate, shared by both arms.

    Args:
        target: Target binary path (for the F11 instrumentation check), or None.
        use_coverage: False disables the trust check, as in ``coverage_trust``.
        ptrace: True when coverage comes from breakpoints (no build-time instrumentation).
        refit_interval: Minimum executions between refits.
    """

    def __init__(
        self,
        target: str | None = None,
        use_coverage: bool = True,
        ptrace: bool = False,
        refit_interval: int = DEFAULT_REFIT_INTERVAL,
    ):
        self._target = target
        self._use_coverage = use_coverage
        self._ptrace = ptrace
        self.refit_interval = max(1, int(refit_interval))
        self.canon = EdgeCanonicalizer()
        self.derived: frozenset[int] = frozenset()
        self.fold: MatrixFold | None = None
        self.version = 0
        self.skip_reason = "not fitted"
        self.stability: float | None = None
        self._last_exec = -(1 << 60)
        self._stamp: tuple[int, int] | None = None
        self._history: deque[tuple[int, float]] = deque(maxlen=SATURATION_WINDOW)
        self._trusted, self._reason = self._decide()

    # ── Preflight gate: one decision point, ``coverage_trust`` ────────────
    def _decide(self) -> tuple[bool, str | None]:
        return coverage_trust(
            self._target,
            use_coverage=self._use_coverage,
            ptrace=self._ptrace,
            id_stability=self.stability,
        )

    def set_stability(self, jaccard: float | None) -> None:
        """Feed the edge-id stability probe (F1); ``None`` = not measured."""
        self.stability = jaccard
        self._trusted, self._reason = self._decide()
        if not self._trusted:
            log.warning("matrix arms abstaining: %s", self._reason)

    @property
    def trusted(self) -> bool:
        return self._trusted

    @property
    def distrust_reason(self) -> str | None:
        return self._reason

    def gate_state(self) -> str:
        if not self._trusted:
            return "closed"
        return "unverified" if self.stability is None else "open"

    # ── Refit ─────────────────────────────────────────────────────────────
    def maybe_refit(self, tracker, exec_count: int, force: bool = False) -> bool:
        """Rebuild the fold if the cadence allows and the tracker has grown."""
        if not force and exec_count - self._last_exec < self.refit_interval:
            return False
        stamp = (
            len(getattr(tracker, "seed_edges", {})),
            len(getattr(tracker, "cumulative_edges", ())),
        )
        if not force and stamp == self._stamp:
            return False
        self._last_exec, self._stamp = exec_count, stamp
        fold, reason = build_fold(profiles_from_tracker(tracker), self.canon, self.derived)
        self.skip_reason = reason
        if fold is None:
            return False
        self.fold = fold
        self.version += 1
        eff = getattr(tracker, "effective_edges", None)
        self._history.append((stamp[1], float(eff()) if callable(eff) else 0.0))
        return True

    def saturation_signal(self) -> float:
        """In [0, 1]: ``2^H`` falling while the raw edge count is flat (F5 / P2-1)."""
        if len(self._history) < 4:
            return 0.0
        (n0, e0), (n1, e1) = self._history[0], self._history[-1]
        if n0 <= 0 or e0 <= 0 or (n1 - n0) / n0 > FLAT_EDGE_GROWTH:
            return 0.0
        return float(np.clip((e0 - e1) / e0 / SATURATION_SCALE, 0.0, 1.0))

    # ── Credit ────────────────────────────────────────────────────────────
    def class_credit(self, edges: Iterable[int]) -> int:
        """Distinct canonical classes among *edges*, derived edges excluded.

        A function of the current partition, not an accumulator: when a class splits
        the same edge set maps onto more classes and the credit rises, and nothing
        stored can go stale. An edge the fold has not seen is its own class.
        """
        return len({self.canon.class_of(e) for e in edges if e not in self.derived})
