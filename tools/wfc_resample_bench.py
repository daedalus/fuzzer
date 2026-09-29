#!/usr/bin/env python3
"""Benchmark wfc_reorder_chunks resample="restart" vs "mt" (swap-based Moser-Tardos).

See docs/handover/handover_moser_tardos_2026-09-28.md. "LLL" there is the
Lovasz Local Lemma. Reports, per format and per table density, the mean
illegal adjacencies per output (strict mode, under the learned table) and mean
ms/call. Density = fraction of ordered kind pairs the table observes; the
symmetric local lemma condition p*e*(d+1)<=1 with d=2 needs p<=~0.123, i.e.
density >= ~0.88.

Usage: PYTHONPATH=src:tests python tools/wfc_resample_bench.py [--n 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from test_wfc_chunks import _FORMATS_UNDER_TEST, _train  # noqa: E402

from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from fuzzer_tool.core.wfc import AdjacencyTable  # noqa: E402
from fuzzer_tool.core.wfc_chunks import (  # noqa: E402
    WfcChunkTableStore,
    _illegal_adjacencies,
    wfc_reorder_chunks,
)


def thin(table_pairs, kinds, density: float, rng, keep) -> AdjacencyTable:
    """Rebuild a table with only a *density* fraction of the observed pairs
    (pairs in *keep* are always retained so the input stays realisable)."""
    pairs = [p for p in table_pairs]
    rng.shuffle(pairs)
    n_keep = max(len(keep), int(round(density * len(pairs))))
    t = AdjacencyTable()
    for a, b in keep:
        t.add_forward(a, b)
    for a, b in pairs[:n_keep]:
        t.add_forward(a, b)
    return t


def synthetic(n_calls: int) -> None:
    """Larger, sparser problems than the real-format fixtures: K kinds, N
    chunks (multiset random), table = a random fraction of ordered pairs plus
    a Hamiltonian-ish backbone so a legal order exists."""
    from fuzzer_tool.core.wfc_chunks import ChunkFormat

    fmt = ChunkFormat(
        name="syn",
        parse=lambda d: [d[i : i + 2] for i in range(0, len(d), 2)],
        serialize=lambda cs: b"".join(cs),
        kind=lambda c: c[:1],
        pin_first=False,
        pin_last=False,
    )
    print(f"\nsynthetic (kinds x chunks)  {'dens':>5s} {'mode':8s} {'illegal/out':>11s} {'zero%':>6s} {'ms/call':>8s}")
    for K, N in ((8, 30), (16, 80)):
        kinds = [bytes([65 + i]) for i in range(K)]
        rng0 = RandPool(seed=5)
        chunks = [kinds[rng0.randint(0, K - 1)] + b"1" for _ in range(N)]
        for d in (0.9, 0.5, 0.25):
            t = AdjacencyTable()
            for i in range(K):
                t.add_forward(kinds[i], kinds[(i + 1) % K])
                t.add_forward(kinds[i], kinds[i])
            allp = [(a, b) for a in kinds for b in kinds if a != b]
            rng0.shuffle(allp)
            for a, b in allp[: int(d * len(allp))]:
                t.add_forward(a, b)
            for mode in ("restart", "mt", "hybrid"):
                rng = RandPool(seed=42)
                ill = zero = 0
                t0 = time.perf_counter()
                for _ in range(n_calls):
                    out = wfc_reorder_chunks(fmt, chunks, t, rng, mode="strict", resample=mode)
                    k = _illegal_adjacencies(fmt, fmt.parse(out), t)
                    ill += k
                    zero += k == 0
                ms = (time.perf_counter() - t0) / n_calls * 1000
                print(f"  K={K:<3d} N={N:<4d}            {d:5.2f} {mode:8s} {ill / n_calls:11.3f} {100 * zero / n_calls:6.1f} {ms:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--synthetic", action="store_true", help="also run the large/sparse synthetic sweep")
    ap.add_argument("--densities", default="1.0,0.9,0.7,0.5")
    args = ap.parse_args()
    densities = [float(x) for x in args.densities.split(",")]

    print(f"{'format':9s} {'dens':>5s} {'mode':8s} {'illegal/out':>11s} {'zero%':>6s} {'ms/call':>8s}")
    for name, data, _tp, fmt in _FORMATS_UNDER_TEST:
        store = WfcChunkTableStore()
        _train(store, fmt, data)
        full = store.table_for(name)
        chunks = fmt.parse(data)
        kinds = list(dict.fromkeys(fmt.kind(c) for c in chunks))
        pairs = [
            (a, b) for a in kinds for b in kinds if a != b and full.compatible(a, b, "right")
        ]
        own = {
            (fmt.kind(a), fmt.kind(b))
            for a, b in zip(chunks, chunks[1:])
            if full.compatible(fmt.kind(a), fmt.kind(b), "right")
        }
        for d in densities:
            table = thin(pairs, kinds, d, RandPool(seed=7), own)
            for mode in ("restart", "mt", "hybrid"):
                rng = RandPool(seed=42)
                ill = zero = 0
                t0 = time.perf_counter()
                for _ in range(args.n):
                    out = wfc_reorder_chunks(fmt, chunks, table, rng, mode="strict", resample=mode)
                    parsed = fmt.parse(out)
                    if parsed is None:
                        continue
                    k = _illegal_adjacencies(fmt, parsed, table)
                    ill += k
                    zero += k == 0
                ms = (time.perf_counter() - t0) / args.n * 1000
                print(
                    f"{name:9s} {d:5.2f} {mode:8s} {ill / args.n:11.3f} {100 * zero / args.n:6.1f} {ms:8.2f}"
                )
    if args.synthetic:
        synthetic(max(10, args.n // 10))


if __name__ == "__main__":
    main()
