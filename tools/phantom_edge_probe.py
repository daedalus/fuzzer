#!/usr/bin/env python3
"""Measure how many scheduler-facing edges are phantom ids (handover F2).

F2: the first execution against a table whose contents differ reports ids
that never reappear.  The fuzzer runs every input against a table left
dirty by the previous one, so this is not a first-execution-only effect.
This tool quantifies what that does to the signals schedulers read.

Per input, the *steady state* is the id set three reruns agree on, after a
discarded warm-up.  The *production* run is one persistent table, one
discarded warm-up, inputs in order -- the fuzzer's own arrangement.  A
phantom is an id in the production run absent from that input's steady
state.

Reported, per input order:

* executions carrying phantom ids
* "new edge" successes, and how many exist only because of phantoms
* ownership of the rare edges (owner == 1, owner <= RARE_EDGE_OWNERS)
* seed weights through the real ``SeedPicker._weight_edge_penalties``

Usage::

    python3 tools/phantom_edge_probe.py --target targets/fuzzgoat_read \\
        --corpus ~/fuzzing/corpus/json [--shuffles 2]

Run ``disable_aslr`` first (done here); with ASLR on, F1 swamps F2.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_matrix_analysis import _run_one  # noqa: E402

from fuzzer_tool.adapters.process import disable_aslr  # noqa: E402
from fuzzer_tool.adapters.shm import ShmCoverage  # noqa: E402
from fuzzer_tool.core.edge_tracker import EdgeTracker  # noqa: E402
from fuzzer_tool.services.seed_picker import RARE_EDGE_OWNERS, SeedPicker  # noqa: E402

STEADY_RERUNS = 3
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAP_SIZE = 65536
DEFAULT_SHUFFLES = 2
DEFAULT_SEED = 3


def _ids(target: Path, cov: ShmCoverage, path: Path, timeout: float) -> frozenset[int]:
    return frozenset(_run_one(target, cov, path, timeout)[0].tolist())


def steady_state(target: Path, path: Path, timeout: float) -> frozenset[int]:
    """Id set of *path* on a warmed table; asserts the reruns agree."""
    cov = ShmCoverage(size=DEFAULT_MAP_SIZE)
    try:
        _run_one(target, cov, path, timeout)
        runs = [_ids(target, cov, path, timeout) for _ in range(STEADY_RERUNS)]
    finally:
        cov.cleanup()
    if len(set(runs)) != 1:
        raise SystemExit(f"{path.name}: reruns disagree; target is not deterministic")
    return runs[0]


def production(target: Path, files: list[Path], order: list[int], timeout: float):
    """One persistent table, one discarded warm-up, inputs in *order*."""
    cov = ShmCoverage(size=DEFAULT_MAP_SIZE)
    try:
        _run_one(target, cov, files[order[0]], timeout)
        return [_ids(target, cov, files[i], timeout) for i in order]
    finally:
        cov.cleanup()


def success_events(obs, steady) -> dict[str, int]:
    """Executions that add a new id, observed vs steady, in arrival order."""
    seen_o: set[int] = set()
    seen_s: set[int] = set()
    out = Counter()
    for o, s in zip(obs, steady, strict=True):
        new_o, new_s = o - seen_o, s - seen_s
        out["observed"] += bool(new_o)
        out["steady"] += bool(new_s)
        out["phantom_only"] += bool(new_o) and not new_s
        out["carry_phantoms"] += bool(o - s)
        seen_o |= o
        seen_s |= s
    out["phantom_ids"] = len(seen_o - seen_s)
    out["ids"] = len(seen_o)
    return dict(out)


def rare_ownership(obs, steady) -> dict[str, int]:
    """How many of the rarest edges exist only as phantoms."""
    own_o = Counter(e for o in obs for e in o)
    own_s = Counter(e for s in steady for e in s)
    phantom = set(own_o) - set(own_s)
    return {
        "singletons": sum(v == 1 for v in own_o.values()),
        "singletons_phantom": sum(own_o[e] == 1 for e in phantom),
        "rare": sum(v <= RARE_EDGE_OWNERS for v in own_o.values()),
        "rare_phantom": sum(own_o[e] <= RARE_EDGE_OWNERS for e in phantom),
    }


def _weights(sets) -> list[float]:
    """Per-seed weights through the production rarity code path."""
    tracker = EdgeTracker(map_size=DEFAULT_MAP_SIZE)
    for i, s in enumerate(sets):
        tracker.record_edges(f"s{i}", set(s))
    fake = SimpleNamespace(_edge_tracker=tracker, _recent_seed_edges=None, _rng=None)
    picker = SeedPicker(fake)
    return [picker._weight_edge_penalties(f"s{i}", 1.0, 0, fake) for i in range(len(sets))]


def weight_shift(obs, steady) -> dict[str, float]:
    """Seeds whose weight the phantoms raise, and by how much at most."""
    w_o, w_s = _weights(obs), _weights(steady)
    up = [o / s for o, s in zip(w_o, w_s, strict=True) if o > s + 1e-9]
    return {"boosted": len(up), "max_boost": max(up, default=1.0)}


def _report(name: str, obs, steady) -> None:
    n = len(obs)
    ev, rare, wt = success_events(obs, steady), rare_ownership(obs, steady), weight_shift(obs, steady)
    print(f"[{name}]")
    print(f"  execs carrying phantom ids   {ev['carry_phantoms']}/{n}")
    print(f"  new-edge successes           observed {ev['observed']}  steady {ev['steady']}  phantom-only {ev['phantom_only']}")
    print(f"  phantom ids in cumulative    {ev['phantom_ids']} of {ev['ids']}")
    print(f"  singleton edges              {rare['singletons']}  phantom {rare['singletons_phantom']}")
    print(f"  owner<={RARE_EDGE_OWNERS} edges              {rare['rare']}  phantom {rare['rare_phantom']}")
    print(f"  seeds boosted by phantoms    {wt['boosted']}/{n}  max boost x{wt['max_boost']:.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--shuffles", type=int, default=DEFAULT_SHUFFLES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    disable_aslr()
    files = sorted(p for p in args.corpus.expanduser().iterdir() if p.is_file())
    steady = [steady_state(args.target, p, args.timeout) for p in files]
    rng = np.random.default_rng(args.seed)
    orders = {"corpus order": list(range(len(files)))}
    for k in range(args.shuffles):
        orders[f"shuffle {k}"] = rng.permutation(len(files)).tolist()

    for name, order in orders.items():
        obs = production(args.target, files, order, args.timeout)
        _report(name, obs, [steady[i] for i in order])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
