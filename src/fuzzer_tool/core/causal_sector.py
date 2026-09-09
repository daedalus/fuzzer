"""Causal-sector graph from transfer entropy (Time-Causal Structure analogue).

Paper (Du): the Time-Causal Structure Postulate selects an objective
antecedent–consequent orientation sector. RO leaves that sector; RD stays
inside it. This module maintains a lightweight directed graph of
bias-corrected transfer-entropy edges and a stability predicate so later
wiring can soft-gate RO-class operators and score causal alignment.

Pure primitive: no scheduler, CLI, or EdgeTracker coupling. Feed it TE
observations (from TransferEntropy.edge_to_edge_flow or pairwise TE calls)
via ``observe_flow`` / ``observe_pair``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class DirectedTEEdge:
    """One directed TE observation between two node ids."""

    source: int
    target: int
    te: float
    weight: float = 1.0  # observation weight / multiplicity
    last_seen: float = field(default_factory=time.monotonic)


@dataclass
class SectorSnapshot:
    """Read-only view of the current causal sector."""

    n_nodes: int
    n_edges: int
    stable: bool
    top_edges: list[tuple[int, int, float]]  # (src, tgt, te) by te desc
    asymmetry_score: float  # mean |T_ab - T_ba| over observed pairs
    observation_windows: int


class CausalSectorGraph:
    """Decaying directed graph of transfer-entropy edges + stability.

    Nodes are opaque integer ids (edge ids, operator ids, …). Edges store
    the latest bias-corrected TE and an exponentially decayed weight.

    Stability: the sign pattern of the top asymmetries has not flipped for
    ``stability_windows`` consecutive observe batches, and the mean absolute
    asymmetry stays above ``min_asymmetry``.
    """

    def __init__(
        self,
        *,
        max_nodes: int = 256,
        max_edges: int = 1024,
        decay: float = 0.95,
        min_te: float = 1e-6,
        min_asymmetry: float = 1e-4,
        stability_windows: int = 3,
        top_k_asymmetry: int = 32,
    ):
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        if max_nodes < 1 or max_edges < 1:
            raise ValueError("max_nodes and max_edges must be >= 1")
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.decay = decay
        self.min_te = min_te
        self.min_asymmetry = min_asymmetry
        self.stability_windows = stability_windows
        self.top_k_asymmetry = top_k_asymmetry

        # (src, tgt) -> DirectedTEEdge
        self._edges: dict[tuple[int, int], DirectedTEEdge] = {}
        self._node_mass: dict[int, float] = defaultdict(float)
        self._windows = 0
        self._stable_streak = 0
        self._last_sign_fingerprint: frozenset[tuple[int, int]] | None = None
        self._stable = False

    def clear(self) -> None:
        self._edges.clear()
        self._node_mass.clear()
        self._windows = 0
        self._stable_streak = 0
        self._last_sign_fingerprint = None
        self._stable = False

    def observe_pair(self, source: int, target: int, te: float, *, weight: float = 1.0) -> None:
        """Record one directed TE value (already bias-corrected by caller)."""
        if source == target:
            return
        te_f = float(te)
        if te_f < self.min_te:
            return
        key = (int(source), int(target))
        now = time.monotonic()
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = DirectedTEEdge(
                source=key[0], target=key[1], te=te_f, weight=weight, last_seen=now
            )
        else:
            # EMA blend toward new observation
            w = existing.weight * self.decay + weight
            te_blend = (existing.te * existing.weight * self.decay + te_f * weight) / w
            existing.te = te_blend
            existing.weight = w
            existing.last_seen = now
        self._node_mass[key[0]] += weight
        self._node_mass[key[1]] += weight
        self._enforce_caps()

    def observe_flow(self, flow: dict[tuple[int, int], float], *, weight: float = 1.0) -> None:
        """Record a batch of directed TE edges (e.g. edge_to_edge_flow output)."""
        for (src, tgt), te in flow.items():
            self.observe_pair(src, tgt, te, weight=weight)
        self.end_window()

    def end_window(self) -> None:
        """Close an observation window and update stability."""
        self._decay_all()
        self._windows += 1
        fp = self._sign_fingerprint()
        asym = self.mean_asymmetry()
        if (
            fp is not None
            and fp == self._last_sign_fingerprint
            and asym >= self.min_asymmetry
            and len(fp) > 0
        ):
            self._stable_streak += 1
        else:
            self._stable_streak = 1 if (fp is not None and len(fp) > 0 and asym >= self.min_asymmetry) else 0
        self._last_sign_fingerprint = fp
        self._stable = self._stable_streak >= self.stability_windows

    def _decay_all(self) -> None:
        dead: list[tuple[int, int]] = []
        for key, edge in self._edges.items():
            edge.weight *= self.decay
            if edge.weight < 1e-9 or edge.te < self.min_te:
                dead.append(key)
        for key in dead:
            del self._edges[key]
        # soft decay node mass
        for n in list(self._node_mass):
            self._node_mass[n] *= self.decay
            if self._node_mass[n] < 1e-12:
                del self._node_mass[n]

    def _enforce_caps(self) -> None:
        if len(self._node_mass) > self.max_nodes:
            # Drop lowest-mass nodes and their incident edges
            ranked = sorted(self._node_mass.items(), key=lambda kv: kv[1])
            drop = {n for n, _ in ranked[: len(ranked) - self.max_nodes]}
            for n in drop:
                del self._node_mass[n]
            self._edges = {
                k: e for k, e in self._edges.items() if e.source not in drop and e.target not in drop
            }
        if len(self._edges) > self.max_edges:
            ranked_e = sorted(self._edges.items(), key=lambda kv: kv[1].te * kv[1].weight)
            keep = dict(ranked_e[len(ranked_e) - self.max_edges :])
            self._edges = keep

    def _sign_fingerprint(self) -> frozenset[tuple[int, int]] | None:
        """Set of directed pairs where T_ab > T_ba by at least min_asymmetry."""
        pairs: set[tuple[int, int]] = set()
        seen_undirected: set[frozenset[int]] = set()
        # Evaluate top-weight edges first
        ranked = sorted(self._edges.values(), key=lambda e: -e.te * e.weight)
        for edge in ranked[: self.top_k_asymmetry * 2]:
            a, b = edge.source, edge.target
            und = frozenset((a, b))
            if und in seen_undirected:
                continue
            seen_undirected.add(und)
            te_ab = self._te(a, b)
            te_ba = self._te(b, a)
            if te_ab - te_ba >= self.min_asymmetry:
                pairs.add((a, b))
            elif te_ba - te_ab >= self.min_asymmetry:
                pairs.add((b, a))
            if len(pairs) >= self.top_k_asymmetry:
                break
        return frozenset(pairs) if pairs else None

    def _te(self, src: int, tgt: int) -> float:
        e = self._edges.get((src, tgt))
        return e.te if e is not None else 0.0

    def mean_asymmetry(self) -> float:
        if not self._edges:
            return 0.0
        vals: list[float] = []
        seen: set[frozenset[int]] = set()
        for (a, b), edge in self._edges.items():
            und = frozenset((a, b))
            if und in seen:
                continue
            seen.add(und)
            vals.append(abs(edge.te - self._te(b, a)))
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @property
    def stable(self) -> bool:
        return self._stable

    @property
    def observation_windows(self) -> int:
        return self._windows

    def te(self, source: int, target: int) -> float:
        return self._te(int(source), int(target))

    def successors(self, source: int, *, min_te: float | None = None) -> list[tuple[int, float]]:
        thr = self.min_te if min_te is None else min_te
        out = [
            (e.target, e.te)
            for e in self._edges.values()
            if e.source == int(source) and e.te >= thr
        ]
        out.sort(key=lambda t: -t[1])
        return out

    def predecessors(self, target: int, *, min_te: float | None = None) -> list[tuple[int, float]]:
        thr = self.min_te if min_te is None else min_te
        out = [
            (e.source, e.te)
            for e in self._edges.values()
            if e.target == int(target) and e.te >= thr
        ]
        out.sort(key=lambda t: -t[1])
        return out

    def aligns(self, source: int, target: int) -> bool:
        """True if source→target is the preferred orientation when both directions exist."""
        te_ab = self._te(int(source), int(target))
        te_ba = self._te(int(target), int(source))
        if te_ab < self.min_te and te_ba < self.min_te:
            return False
        return te_ab >= te_ba

    def top_edges(self, k: int = 16) -> list[tuple[int, int, float]]:
        ranked = sorted(self._edges.values(), key=lambda e: -e.te)
        return [(e.source, e.target, e.te) for e in ranked[:k]]

    def snapshot(self) -> SectorSnapshot:
        return SectorSnapshot(
            n_nodes=len(self._node_mass),
            n_edges=len(self._edges),
            stable=self._stable,
            top_edges=self.top_edges(16),
            asymmetry_score=self.mean_asymmetry(),
            observation_windows=self._windows,
        )

    def __len__(self) -> int:
        return len(self._edges)
