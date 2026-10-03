#!/usr/bin/env python3
"""Measure LOO-vs-Shapley disagreement on pooled byte entropy (handover §5).

The handover says: build permutation Shapley only if LOO's near-duplicate
misprice is large enough to matter for minimization. This answers that on
(a) synthetic corpora with planted near-duplicate cliques of growing size, and
(b) any real corpus directory given with --corpus.

Reports, per scenario: Spearman rho(LOO, Shapley); overlap of the bottom-m
"retire these" sets (m = 25% of the corpus); the same overlap between two
independent Shapley runs (the Monte-Carlo noise floor: LOO disagreement only
counts above it); and the fraction of clique members LOO prices below 10% of
their Shapley credit. ``clique=1`` has no duplicates (baseline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, byte_histogram
from fuzzer_tool.core.schedulers.seed_entropy_shapley import loo_entropy, shapley_entropy


def _slab(seeds: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    rows = [byte_histogram(s, ENTROPY_SAMPLE_CAP) for s in seeds]
    counts = np.stack([np.asarray(c, dtype=np.int64) for c, _ in rows])
    return counts, np.array([t for _, t in rows], dtype=np.int64)


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def planted(
    rng: np.random.Generator, n_unique: int, clique: int, n_cliques: int
) -> tuple[list[bytes], np.ndarray]:
    """Unique random seeds plus cliques of byte-permuted copies (same histogram)."""
    seeds = [bytes(rng.integers(0, 256, 256, dtype=np.uint8)) for _ in range(n_unique)]
    in_clique = np.zeros(n_unique + clique * n_cliques, dtype=bool)
    for c in range(n_cliques):
        base = np.frombuffer(
            bytes(rng.integers(0, 64 + 32 * c, 256, dtype=np.uint8)), dtype=np.uint8
        )
        for _ in range(clique):
            seeds.append(bytes(rng.permutation(base)))
            in_clique[len(seeds) - 1] = True
    return seeds, in_clique


def report(
    name: str, seeds: list[bytes], in_clique: np.ndarray | None, n_perm: int, seed: int
) -> None:
    counts, sizes = _slab(seeds)
    loo = loo_entropy(counts, sizes)
    phi, se = shapley_entropy(counts, sizes, n_perm=n_perm, rng=np.random.default_rng(seed))
    phi2, _ = shapley_entropy(counts, sizes, n_perm=n_perm, rng=np.random.default_rng(seed + 1))
    m = max(1, len(seeds) // 4)
    floor = (
        len(
            set(np.argsort(phi, kind="stable")[:m].tolist())
            & set(np.argsort(phi2, kind="stable")[:m].tolist())
        )
        / m
    )
    bottom_loo = set(np.argsort(loo, kind="stable")[:m].tolist())
    bottom_phi = set(np.argsort(phi, kind="stable")[:m].tolist())
    line = (
        f"{name:28s} n={len(seeds):4d}  rho={_spearman(loo, phi):6.3f}  "
        f"bottom{m}-overlap={len(bottom_loo & bottom_phi) / m:5.2f}  "
        f"noise-floor-overlap={floor:5.2f}  max-se={se.max():.4f}"
    )
    if in_clique is not None and in_clique.any():
        mis = (np.abs(loo[in_clique]) < 0.1 * np.maximum(phi[in_clique], 1e-12)).mean()
        line += f"  clique-misprice={mis:5.2f}"
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, help="real corpus dir (flat files)")
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for clique in (1, 2, 4, 8, 16):
        seeds, mask = planted(rng, 100, clique, 5)
        report(f"planted clique={clique}", seeds, mask, args.perms, args.seed)

    if args.corpus:
        files = sorted(p for p in args.corpus.rglob("*") if p.is_file())[: args.max_files]
        report(
            f"real:{args.corpus.name}", [p.read_bytes() for p in files], None, args.perms, args.seed
        )


if __name__ == "__main__":
    main()
