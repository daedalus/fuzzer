"""SQLite sink for per-seed edge ids, keyed by edge.

    edges                         edge_seeds                     seeds
    +---------------+----------+  +---------+---------+  +-------------+---------+
    | edge_id (PK)  | n_seeds  |  | edge_id | seed_id |  | seed_id(PK) | n_edges |
    +---------------+----------+  +---------+---------+  +-------------+---------+
    | 3             | 2        |--| 3       | s1      |--| s1          | 1       |
    |               |          |  | 3       | s2      |  | s2          | 1       |
                                                         | s3 (empty)  | 0       |

Built from ``{seed_id: uint32 edge ids}`` in one vectorized pass. ``seeds``
lists every seed, edgeless ones too, so a reader recovers the full dict.
"""

import os
import sqlite3
from pathlib import Path

import numpy as np

_SCHEMA = (
    "CREATE TABLE edges (edge_id INTEGER PRIMARY KEY, n_seeds INTEGER NOT NULL)",
    "CREATE TABLE edge_seeds ("
    "edge_id INTEGER NOT NULL REFERENCES edges(edge_id), "
    "seed_id TEXT NOT NULL, "
    "PRIMARY KEY (edge_id, seed_id)) WITHOUT ROWID",
    "CREATE TABLE seeds (seed_id TEXT PRIMARY KEY, n_edges INTEGER NOT NULL) WITHOUT ROWID",
)


def write_edges_sqlite(path: str | Path, edges: dict[str, np.ndarray]) -> None:
    """Write the edge-keyed inverse of *edges* to *path*, replacing any file there.

    Built in ``<path>.tmp`` and renamed: no half-written DB on crash.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)

    seeds = list(edges)
    ids, owner = _flatten(edges)

    # Edge-sorted rows: sequential inserts into both B-trees.
    order = np.argsort(ids, kind="stable")
    ids, owner = ids[order], owner[order]
    uniq, counts = np.unique(ids, return_counts=True)

    con = sqlite3.connect(tmp)
    try:
        for stmt in _SCHEMA:
            con.execute(stmt)
        con.executemany(
            "INSERT INTO edges VALUES (?, ?)", zip(uniq.tolist(), counts.tolist(), strict=True)
        )
        con.executemany(
            "INSERT INTO edge_seeds VALUES (?, ?)",
            zip(ids.tolist(), (seeds[i] for i in owner.tolist()), strict=True),
        )
        con.executemany("INSERT INTO seeds VALUES (?, ?)", ((s, edges[s].size) for s in seeds))
        con.commit()
    finally:
        con.close()
    os.replace(tmp, out)


def read_edges_sqlite(path: str | Path) -> dict[str, np.ndarray]:
    """``{seed_id: sorted uint32 edge ids}`` back from a DB this module wrote.

    Raises:
        sqlite3.DatabaseError: *path* is not such a DB.
    """
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        seeds = [s for (s,) in con.execute("SELECT seed_id FROM seeds")]
        rows = con.execute("SELECT seed_id, edge_id FROM edge_seeds").fetchall()
    finally:
        con.close()

    # Group edge ids per seed in one sort: seed index major, edge id minor.
    index = {s: i for i, s in enumerate(seeds)}
    owner = np.fromiter((index[s] for s, _ in rows), dtype=np.int64, count=len(rows))
    ids = np.fromiter((e for _, e in rows), dtype=np.uint32, count=len(rows))
    order = np.lexsort((ids, owner))
    ids, owner = ids[order], owner[order]
    bounds = np.searchsorted(owner, np.arange(len(seeds) + 1))
    return {s: ids[bounds[i] : bounds[i + 1]].copy() for i, s in enumerate(seeds)}


def _flatten(edges: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(all edge ids as int64, seed index per id), e.g. {a:[1,2], b:[2]} -> [1,2,2], [0,0,1]."""
    arrays = list(edges.values())
    if not arrays:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    ids = np.concatenate(arrays).astype(np.int64)
    sizes = np.fromiter((a.size for a in arrays), dtype=np.int64, count=len(arrays))
    owner = np.repeat(np.arange(len(arrays), dtype=np.int64), sizes)
    return ids, owner
