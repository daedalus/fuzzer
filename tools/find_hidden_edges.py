#!/usr/bin/env python3
"""Find real (static) branches the corpus never split, using KatzChannel's
existing ICFG + node-bitmap plumbing. No afl_shim.c changes: __AFL_DISTANCE_MODE
is already on by default in the compiled shim, and DistanceTableShm /
NodeBitmapShm are pure-Python SHM segments the shim already knows to probe.

For every ICFG node with out-degree >= 2, if exactly one successor was ever
visited across the corpus and at least one sibling never was, that sibling
is a "hidden edge": reachable in principle, never taken by any seed.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.katz_channel import KatzChannel

TARGET = sys.argv[1] if len(sys.argv) > 1 else "targets/fuzzgoat_read"
CORPUS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/corpus_fg"
MAP_SIZE = 1 << 16


def run_one(target, cov, ch, path, timeout=5.0):
    import contextlib
    import subprocess

    cov.reset_edge_map()
    env = dict(
        os.environ,
        __AFL_SHM_ID=str(cov.shm_id),
        AFL_MAP_SIZE=str(cov.num_entries),
    )
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run([str(target), str(path)], env=env, capture_output=True, timeout=timeout)
    bits = ch.sample()
    return bits


def main():
    disable_aslr()

    ch = KatzChannel.build(TARGET, debug=True)
    if ch is None:
        print("KatzChannel.build failed -- target not trace-pc viable?")
        return 1
    print(f"ICFG: {ch.icfg.n_nodes} nodes, {len(ch.node_of)} probe-mapped")

    if not ch.upload():
        print("SHM upload failed")
        return 1

    inputs = sorted(Path(CORPUS).glob("*"))
    inputs = [p for p in inputs if p.is_file()]
    print(f"corpus: {len(inputs)} inputs")

    cov = ShmCoverage(size=MAP_SIZE)
    ever_visited = np.zeros(ch.n_nodes, dtype=bool)
    try:
        # warm-up, discarded (matches edge_matrix_analysis.py's regime)
        run_one(TARGET, cov, ch, inputs[0])
        for p in inputs:
            bits = run_one(TARGET, cov, ch, p)
            if bits is not None:
                ever_visited |= bits
    finally:
        cov.cleanup()
        ch.cleanup()

    print(f"nodes ever visited: {int(ever_visited.sum())}/{ch.n_nodes}")

    icfg = ch.icfg
    # branch_src/branch_dst (icfg.py) are real intraprocedural branch/
    # fallthrough edges only -- NOT icfg.src/dst, which also carries
    # caller->callee call edges. A call to a rarely/never-"visited" helper
    # (e.g. __sanitizer_cov_trace_pc itself, which can never show as
    # visited -- it IS the probe, not a probed site) would otherwise look
    # exactly like an unreached branch sibling.
    succs = {}
    for s, d in zip(icfg.branch_src.tolist(), icfg.branch_dst.tolist()):
        succs.setdefault(s, []).append(d)

    hidden = []
    for s, dsts in succs.items():
        if len(dsts) < 2:
            continue
        if not ever_visited[s]:
            continue  # the branch point itself was never reached
        visited = [d for d in dsts if ever_visited[d]]
        unvisited = [d for d in dsts if not ever_visited[d]]
        if visited and unvisited:
            hidden.append((s, visited, unvisited))

    print(f"\nreal branches with an un-split sibling: {len(hidden)}")
    for s, visited, unvisited in hidden[:30]:
        fn = icfg.node_funcs[s]
        s_addr = icfg.node_addrs[s]
        print(f"  {fn}+0x{s_addr:x}: taken={[hex(icfg.node_addrs[d]) for d in visited]} "
              f"NEVER-TAKEN={[hex(icfg.node_addrs[d]) for d in unvisited]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
