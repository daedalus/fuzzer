"""E4 calibration: crash-cluster vs fuzzgoat ground truth (adjusted Rand index).

    tools/benchmark.py crashcluster BIN [BIN ...] [--per-bug 15] [--budget 200]

Builds per-bug crashing JSON inputs (one planted fuzzgoat bug each, random
noise around it), reduces each, and scores stack signature, threshold
linkage and spectral linkage against the bug that was planted. Control
(Hard Rule 46): ARI(truth, truth) == 1 and a shuffle ~ 0 before any row.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from fuzzer_tool.core.crash_graph import (
    cluster_kernel,
    cluster_spectral,
    kernel_matrix,
    wl_features,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.crash_cluster import CoverageReducer, TraceProbe

# One trigger per planted bug; noise never contains an object, an empty
# array, or a string of length < 2 (each of those is itself a trigger).
TRIGGERS = {"uaf_empty_array": "[]", "object_len": '{"":0}', "empty_str": '""', "null_1byte": '"A"'}


def _noise_scalar(rp):
    kind = rp.randint(0, 3)
    if kind == 0:
        return str(rp.randint(-(10**6), 10**6))
    if kind == 1:
        return rp.choice(["true", "false", "null", "1.5e3", "-0.25"])
    n = rp.randint(2, 12)
    return '"' + "".join(rp.choice("abcdefgh xyz") for _ in range(n)) + '"'


def _doc(rp, trigger):
    """Nested non-empty arrays of scalars, trigger at a random slot."""
    depth = rp.randint(0, 4)
    inner = [_noise_scalar(rp) for _ in range(rp.randint(0, 5))]
    inner.insert(rp.randint(0, len(inner)), trigger)
    doc = "[" + ",".join(inner) + "]"
    for _ in range(depth):
        items = [_noise_scalar(rp) for _ in range(rp.randint(0, 4))]
        items.insert(rp.randint(0, len(items)), doc)
        sep = rp.choice([",", ", ", " ,\n"])
        doc = "[" + sep.join(items) + "]"
    return doc.encode()


def ari(a, b):
    """Adjusted Rand index (Hubert-Arabie), in-house (Hard Rule 51)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ct = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(ct, (ai, bi), 1)

    def c2(x):
        return (x * (x - 1) / 2).sum()

    n = len(a)
    sij, sa, sb = c2(ct), c2(ct.sum(1)), c2(ct.sum(0))
    exp = sa * sb / (n * (n - 1) / 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)


def labels_of(groups, n):
    lab = np.empty(n, dtype=int)
    for g, members in enumerate(groups):
        lab[members] = g
    return lab


def _control(rp) -> None:
    """ARI must score truth 1.0 and a shuffle near 0, or the rows mean nothing."""
    truth = np.repeat(np.arange(len(TRIGGERS)), 50)
    shuffled = truth.copy()
    rp.shuffle(shuffled)
    assert ari(truth, truth) == 1.0
    assert abs(ari(truth, np.array(shuffled))) < 0.1


def _score(b: str, inputs: list[bytes], truth_build: list[str], budget: int) -> None:
    t0 = time.time()
    probe = TraceProbe(b, timeout=2.0)
    red = CoverageReducer(probe, RandPool(7), budget)
    feats, sigs, truth, sizes = [], [], [], []
    try:
        for data, lab in zip(inputs, truth_build, strict=True):
            tr = probe(data)
            if tr.sig is None:
                continue
            _, tr2 = red.reduce(data, tr)
            feats.append(wl_features(tr2.graph))
            sigs.append(tr.sig)
            truth.append(lab)
            sizes.append((tr.graph.size, tr2.graph.size))
    finally:
        probe.close()

    n = len(truth)
    t = np.array(truth)
    sim = kernel_matrix(feats)
    stack = labels_of([[i for i, s in enumerate(sigs) if s == u] for u in dict.fromkeys(sigs)], n)
    before, after = np.median([s[0] for s in sizes]), np.median([s[1] for s in sizes])

    print(f"\n== {Path(b).name}: {n} crashes, {len(set(truth))} bugs, {time.time() - t0:.0f}s")
    print(f"  edges before/after reduction (median): {before:.0f} -> {after:.0f}")
    print(f"  stack signature : {len(set(sigs)):3d} clusters  ARI {ari(t, stack):.3f}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        g = cluster_kernel(sim, thr)
        print(f"  threshold {thr:.1f}   : {len(g):3d} clusters  ARI {ari(t, labels_of(g, n)):.3f}")
    g = cluster_spectral(sim)
    print(f"  spectral        : {len(g):3d} clusters  ARI {ari(t, labels_of(g, n)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "bins", nargs="+", help="fuzzgoat builds (ASAN; -D__AFL_TRACE_FIRES=1 for arcs)"
    )
    ap.add_argument("--per-bug", type=int, default=15, help="crashing inputs per planted bug")
    ap.add_argument("--budget", type=int, default=200, help="reduction execs per crash")
    ap.add_argument("--seed", type=int, default=2026, help="input generator seed")
    args = ap.parse_args()

    rp = RandPool(args.seed)
    _control(rp)
    inputs, truth_build = [], []
    for name, trig in TRIGGERS.items():
        for _ in range(args.per_bug):
            inputs.append(_doc(rp, trig))
            truth_build.append(name)

    for b in args.bins:
        _score(b, inputs, truth_build, args.budget)


if __name__ == "__main__":
    main()
