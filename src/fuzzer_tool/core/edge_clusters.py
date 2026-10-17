"""Edge clusters: edges grouped by the seeds that reach them.

    seed -> edges ──► EdgeCanonicalizer on 0/1 profiles ──► classes
                        every seed hits all of a class or none of it
    classes ──► leader clustering, Jaccard >= CLUSTER_JACCARD ──► clusters
                        near-duplicates: a few seeds tell the members apart
    seeds x clusters ──► near_misses(seed)
                        rare classes of a cluster the seed is on but lacks
    clusters ──► medoid (centroid: max mean Jaccard to the members)
    seed ──► gaps(seed)  members it lacks of clusters whose medoid it holds,
                         nearest the medoid first (pos_centroid's targets)

A class is the unit rarity should count. FFmpeg 5k: 2,536 singleton edges
were 75 singleton classes, one block per keystone seed, so a per-edge rare
count scored a seed owning one 1,000-edge block 1,000x one owning a single
rare branch. Leader clustering (members must reach the threshold against
their leader, not a neighbour) avoids single-linkage chaining: on FFmpeg,
single linkage at Jaccard 0.5 fused 13,171 of 13,406 edges into one cluster.
"""

from collections.abc import Iterable, Mapping

import numpy as np

from fuzzer_tool.core.edge_matrix import DEFAULT_REFIT_INTERVAL, RefitCadence
from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer

RARE_OWNERS = 3  # same bound as seed_picker.RARE_EDGE_OWNERS
CLUSTER_JACCARD = 0.75
# Refit time ~ K^2 x seeds: FFmpeg 10k (3,230 classes, 581 seeds) 1.5 s; at
# 2,000 the cap skipped clustering there and left every seed without gaps.
MAX_CLUSTER_CLASSES = 8_000
LEAD_BLOCK = 512  # Jaccard rows per block: memory LEAD_BLOCK x K float32
MAX_INCIDENCE_CELLS = 4_000_000  # dense seeds x classes float32 matrix (16 MB)
GAP_LIMIT = 8  # gaps returned per seed
REPORT_NNZ_BUDGET = 500_000  # seed-edge cells; same bound as the summary's edge classes

_EMPTY = np.empty(0, dtype=np.int64)


def count_distinct(cls: np.ndarray, mask: np.ndarray) -> int:
    """Distinct classes under *mask*; unclassified edges (cls < 0) count one each.

    Edges found after the last refit have no class yet and stay per-edge.
    """
    known = mask & (cls >= 0)
    return int(np.unique(cls[known]).size + np.count_nonzero(mask & (cls < 0)))


def fit_tracker(tracker) -> "EdgeClusters | None":
    """Fresh fit of *tracker*'s seed edges for reporting; None if empty or over budget."""
    seeds = getattr(tracker, "seed_edges", None)
    if not seeds or sum(map(len, seeds.values())) > REPORT_NNZ_BUDGET:
        return None
    clusters = EdgeClusters()
    clusters.refit(seeds)
    return clusters if clusters.stats()["edges"] else None


class EdgeClusters:
    """Incidence classes and near-duplicate clusters, refit wholesale."""

    def __init__(
        self, refit_interval: int = DEFAULT_REFIT_INTERVAL, jaccard: float = CLUSTER_JACCARD
    ) -> None:
        if not 0.0 < jaccard <= 1.0:
            raise ValueError(f"cluster Jaccard threshold must be in (0, 1], got {jaccard}")
        self.jaccard = jaccard  # higher -> more, tighter clusters (more centroids)
        self.cadence = RefitCadence(interval=refit_interval)
        self.version = 0
        self._reset()

    def _reset(self) -> None:
        self._canon = EdgeCanonicalizer()
        self._heads = _EMPTY  # class head edge ids, ascending
        self._owners = _EMPTY  # seeds per class
        self._sizes = _EMPTY  # edges per class
        self._leader = _EMPTY  # class index of each class's cluster leader
        self._medoid = _EMPTY  # class index of each class's cluster medoid
        self._to_medoid = np.empty(0)  # Jaccard of each class to its medoid
        self._near: dict[object, int] = {}
        self._singleton_edges = 0

    # ── fitting ────────────────────────────────────────────────────────

    def maybe_refit(self, tracker, exec_count: int) -> bool:
        """Refit from *tracker* on the shared substrate cadence; True if refit."""
        n_seeds, n_edges = len(tracker.seed_edges), len(tracker.cumulative_edges)
        go, _ = self.cadence.due(n_seeds, n_edges, exec_count)
        if not go:
            return False
        self.refit(tracker.seed_edges)
        self.cadence.commit(n_seeds, n_edges, exec_count)
        return True

    def refit(self, seed_edges: Mapping[object, Iterable[int]]) -> None:
        """Rebuild classes, clusters and near-misses from seed -> edges."""
        self._reset()
        self.version += 1
        keys = [k for k in seed_edges if seed_edges[k]]
        if not keys:
            return

        # Flat (seed, edge) cells; classes from 0/1 profiles.
        lens = np.fromiter((len(seed_edges[k]) for k in keys), dtype=np.int64, count=len(keys))
        flat = np.fromiter(
            (e for k in keys for e in seed_edges[k]), dtype=np.int64, count=int(lens.sum())
        )
        seed = np.repeat(np.arange(len(keys)), lens)
        self._canon.refit({k: dict.fromkeys(seed_edges[k], 1) for k in keys})
        heads, cls = np.unique(self._canon.class_array(flat), return_inverse=True)
        n_cls = heads.size

        # One (seed, class) pair per seed holding the class: owners per class.
        pair = np.unique(seed * n_cls + cls)
        p_seed, p_cls = np.divmod(pair, n_cls)
        owners = np.bincount(p_cls, minlength=n_cls)
        _, first = np.unique(flat, return_index=True)
        edge_cls = cls[first]

        self._heads, self._owners = heads, owners
        self._sizes = np.bincount(edge_cls, minlength=n_cls)
        self._singleton_edges = int(np.count_nonzero(owners[edge_cls] == 1))
        self._leader = self._medoid = np.arange(n_cls)
        self._to_medoid = np.ones(n_cls)
        if n_cls > MAX_CLUSTER_CLASSES or len(keys) * n_cls > MAX_INCIDENCE_CELLS:
            return

        # float32 matmul runs on BLAS (int32 does not); counts < 2^24 stay exact.
        inc = np.zeros((len(keys), n_cls), dtype=np.float32)
        inc[p_seed, p_cls] = 1.0
        self._leader = self._lead(inc)
        self._medoids(inc)
        self._near = self._near_misses(keys, inc > 0)

    def _jaccard_rows(self, inc: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Jaccard of classes *rows* against classes *cols*."""
        inter = inc[:, rows].T @ inc[:, cols]
        owners = self._owners
        return inter / (owners[rows, None] + owners[None, cols] - inter)

    def _medoids(self, inc: np.ndarray) -> None:
        """Per cluster, the member with the highest mean Jaccard to the others.

        Ties go to the widest class, then the lowest head: the leader's order.
        """
        for lead in np.unique(self._leader).tolist():
            mem = np.flatnonzero(self._leader == lead)
            if mem.size == 1:
                continue
            jac = self._jaccard_rows(inc, mem, mem)
            mean = (jac.sum(axis=1) - 1.0) / (mem.size - 1)
            pick = np.lexsort((self._heads[mem], -self._owners[mem], -mean))[0]
            self._medoid[mem] = mem[pick]
            self._to_medoid[mem] = jac[pick]

    def _lead(self, inc: np.ndarray) -> np.ndarray:
        """Leader clustering: widest class first; join the most similar leader.

        Jaccard rows are computed LEAD_BLOCK classes at a time in visiting
        order, so memory is LEAD_BLOCK x K, not K x K.
        """
        n_cls = self._owners.size
        order = np.lexsort((self._heads, -self._owners))
        leader = np.empty(n_cls, dtype=np.int64)
        leaders: list[int] = []
        for start in range(0, n_cls, LEAD_BLOCK):
            block = order[start : start + LEAD_BLOCK]
            jac = self._jaccard_rows(inc, block, np.arange(n_cls))
            for row, c in enumerate(block.tolist()):
                if leaders:
                    sims = jac[row, leaders]
                    best = int(np.argmax(sims))
                    if sims[best] >= self.jaccard:
                        leader[c] = leaders[best]
                        continue
                leaders.append(c)
                leader[c] = c
        return leader

    def _near_misses(self, keys: list, inc: np.ndarray) -> dict[object, int]:
        """Per seed: rare classes of clusters it touches but does not hit."""
        rare = np.flatnonzero(self._owners <= RARE_OWNERS)
        if not rare.size:
            return {}
        touched = np.zeros_like(inc)
        s_idx, c_idx = np.nonzero(inc)
        touched[s_idx, self._leader[c_idx]] = True
        near = (touched[:, self._leader[rare]] & ~inc[:, rare]).sum(axis=1)
        return {keys[i]: int(n) for i, n in enumerate(near.tolist()) if n}

    # ── lookups ────────────────────────────────────────────────────────

    def class_index(self, edges: np.ndarray) -> np.ndarray:
        """Class index per edge id; -1 for edges the last refit did not see."""
        if not self._heads.size:
            return np.full(len(edges), -1, dtype=np.int64)
        heads = self._canon.class_array(np.asarray(edges, dtype=np.int64))
        at = np.searchsorted(self._heads, heads).clip(max=self._heads.size - 1)
        return np.where(self._heads[at] == heads, at, -1)

    def _seen(self, edges: Iterable[int]) -> np.ndarray:
        idx = self.class_index(np.fromiter(edges, dtype=np.int64))
        return idx[idx >= 0]

    def class_of(self, edge: int) -> int:
        """Head edge id of *edge*'s class (the edge itself if unseen)."""
        return self._canon.class_of(edge)

    def cluster_of(self, edge: int) -> int:
        """Head edge id of *edge*'s cluster leader; -1 if unseen."""
        idx = self._seen([edge])
        return int(self._heads[self._leader[idx[0]]]) if idx.size else -1

    def rare_classes(self, edges: Iterable[int], max_owners: int = RARE_OWNERS) -> int:
        """Distinct classes among *edges* held by at most *max_owners* seeds."""
        idx = np.unique(self._seen(edges))
        return int(np.count_nonzero(self._owners[idx] <= max_owners))

    def unique_fraction(self, edges: Iterable[int]) -> float:
        """Share of the seed's classes no other seed holds."""
        idx = np.unique(self._seen(edges))
        return float(np.count_nonzero(self._owners[idx] == 1) / idx.size) if idx.size else 0.0

    def medoid_of(self, edge: int) -> int:
        """Head edge id of *edge*'s cluster medoid; -1 if unseen."""
        idx = self._seen([edge])
        return int(self._heads[self._medoid[idx[0]]]) if idx.size else -1

    def gaps(self, edges: Iterable[int], limit: int = GAP_LIMIT) -> list[int]:
        """Head edges of member classes *edges* lack, in clusters whose medoid they hold.

        Nearest the medoid (highest Jaccard) first: the siblings most seeds
        already reach from the same centre.
        """
        idx = self._seen(edges)
        if not idx.size:
            return []
        held = np.zeros(self._heads.size, dtype=bool)
        held[idx] = True
        cand = np.flatnonzero(held[self._medoid] & ~held)
        order = cand[np.lexsort((self._heads[cand], -self._to_medoid[cand]))]
        return self._heads[order[:limit]].tolist()

    def near_misses(self, seed_key: object) -> int:
        return self._near.get(seed_key, 0)

    def near_miss_seeds(self) -> int:
        return len(self._near)

    def class_owner_counts(self) -> np.ndarray:
        return self._owners.copy()

    def stats(self) -> dict:
        return {
            "edges": int(self._sizes.sum()),
            "classes": int(self._heads.size),
            "clusters": int(np.unique(self._leader).size),
            "singleton_edges": self._singleton_edges,
            "singleton_classes": int(np.count_nonzero(self._owners == 1)),
            "largest_class": int(self._sizes.max(initial=0)),
        }
