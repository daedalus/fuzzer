"""Connected components of the seed Gram matrix B = A.A^T (``--edge-components``).

Seeds i, j are adjacent when they share an edge. Edges held by more than
*strip_frac* of the seeds (start-up code, libc paths) join everything, so
they are dropped first. What remains shows how many independent code
regions the corpus reaches, and how many seeds reach nothing else.

    a={1,10} b={1,10,11} c={1,20} f={}      strip_frac 0.5: edge 1 stripped
    components: {a,b} {c} {f}
"""

import numpy as np

STRIP_FRAC = 0.1
TITLE = "Edge Components"


def _incidence(edges: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Keys, and the seed index and edge id of every incidence."""
    keys = list(edges)
    sizes = np.fromiter((edges[k].size for k in keys), dtype=np.int64, count=len(keys))
    seed_idx = np.repeat(np.arange(len(keys), dtype=np.int64), sizes)
    if not sizes.sum():
        return keys, seed_idx, np.zeros(0, dtype=np.int64)
    edge_ids = np.concatenate([edges[k] for k in keys]).astype(np.int64)
    return keys, seed_idx, edge_ids


def _propagate(
    seed_idx: np.ndarray, edge_dense: np.ndarray, n_seeds: int, n_edges: int
) -> np.ndarray:
    """Min-label propagation over the bipartite seed-edge graph."""
    lab = np.arange(n_seeds, dtype=np.int64)
    while True:
        edge_lab = np.full(n_edges, n_seeds, dtype=np.int64)
        np.minimum.at(edge_lab, edge_dense, lab[seed_idx])
        new = lab.copy()
        np.minimum.at(new, seed_idx, edge_lab[edge_dense])
        if np.array_equal(new, lab):
            return lab
        lab = new


def components(
    edges: dict[str, np.ndarray], strip_frac: float = STRIP_FRAC
) -> tuple[dict[str, int], int]:
    """Component label per seed key, and the number of edges stripped.

    Seeds left with no edge after stripping get a label of their own.
    """
    if not 0.0 < strip_frac <= 1.0:
        raise ValueError("strip_frac must be in (0, 1]")
    keys, seed_idx, edge_ids = _incidence(edges)
    if not keys:
        return {}, 0

    uniq, dense, held = np.unique(edge_ids, return_inverse=True, return_counts=True)
    drop = held > strip_frac * len(keys)
    keep = ~drop[dense]
    lab = _propagate(seed_idx[keep], dense[keep], len(keys), len(uniq))
    return dict(zip(keys, lab.tolist(), strict=True)), int(drop.sum())


def components_report(edges: dict[str, np.ndarray], strip_frac: float = STRIP_FRAC) -> str:
    """Report text for ``minimize -c --edge-components``."""
    lines = ["", f"--- {TITLE} (strip edges held by > {strip_frac:.0%} of seeds) ---"]
    labels, stripped = components(edges, strip_frac)
    if not labels:
        lines.append("  No seeds with edge data.")
        return "\n".join(lines)

    sizes = np.unique(np.fromiter(labels.values(), dtype=np.int64), return_counts=True)[1]
    has_edge = {k for k, e in edges.items() if e.size}
    singles = int((sizes == 1).sum())
    lines += [
        f"  {'Seeds:':<16s}{len(labels)} ({len(labels) - len(has_edge)} without edges)",
        f"  {'Stripped edges:':<16s}{stripped}",
        f"  {'Components:':<16s}{sizes.size}",
        f"  {'Largest:':<16s}{sizes.max()} seeds ({sizes.max() / len(labels):.1%})",
        f"  {'Isolated seeds:':<16s}{singles}",
    ]
    return "\n".join(lines)
