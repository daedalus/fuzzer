#!/usr/bin/env python3
"""Per-edge hit spectrum of a corpus replay: the real weights for sim_funnel_probe.

The shim looks an edge up on every execution of it, so an edge's weight on the
hot path is its summed per-exec count. Replays the corpus oldest first (mtime
= discovery order), sums ShmCoverage counts per edge, and records first-seen
order. Reports:

    s_fit     Zipf rank-frequency exponent from core/zipf.fit_zipf (1/(alpha-1))
    s_disc    -slope of log(total) vs log(first-seen rank): the model
              sim_funnel_probe --mode hot assumes (w_r = r^-s, r = discovery)
    corr      Spearman(first-seen rank, total); < 0 means hot edges come first

Usage:
    python tools/edge_hit_spectrum.py TARGET -d CORPUS -o spectrum.pkl
    python tools/sim_funnel_probe.py --mode real --spectrum spectrum.pkl
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.edge_pickle import EdgePickleWriter, read_edges_pickle
from fuzzer_tool.adapters.process import disable_aslr, run_target_file
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import estimate_map_size
from fuzzer_tool.core.rank_stats import rank_corr
from fuzzer_tool.core.zipf import fit_zipf
from fuzzer_tool.services.minimize import _discover_corpus_files

IDS_KEY = "ids"
TOTALS_KEY = "totals"
MIN_EDGES = 2
PROGRESS_EVERY = 500


class Ctx(Enum):
    """Shim call-site context ($__AFL_CTX_SENSITIVE)."""

    ON = "1"
    OFF = "0"


class Spectrum:
    """Summed hit counts per edge, kept in first-seen order."""

    def __init__(self) -> None:
        self._totals: dict[int, int] = {}

    def add(self, counts: dict[int, int]) -> None:
        """Fold one exec's {edge_id: count}; dict order keeps first-seen order."""
        totals = self._totals
        for eid, n in counts.items():
            totals[eid] = totals.get(eid, 0) + n

    def ids(self) -> np.ndarray:
        return np.fromiter(self._totals, dtype=np.uint32, count=len(self._totals))

    def totals(self) -> np.ndarray:
        return np.fromiter(self._totals.values(), dtype=np.uint64, count=len(self._totals))

    def _ranked(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self._totals) < MIN_EDGES:
            raise ValueError(f"need >= {MIN_EDGES} edges, have {len(self._totals)}")
        totals = self.totals().astype(np.float64)
        return np.arange(1, totals.size + 1, dtype=np.float64), totals

    def discovery_slope(self) -> float:
        """s in total_r ~ r^-s, r = first-seen rank (least squares in log-log)."""
        ranks, totals = self._ranked()
        slope = np.polyfit(np.log(ranks), np.log(totals), 1)[0]
        return float(-slope)

    def hot_first_corr(self) -> float:
        ranks, totals = self._ranked()
        return rank_corr(ranks, totals)

    def save(self, path: Path) -> None:
        writer = EdgePickleWriter(path)
        writer.open()
        writer.write([(IDS_KEY, self.ids()), (TOTALS_KEY, self.totals())])
        writer.close()

    @classmethod
    def load(cls, path: Path) -> Spectrum:
        data = read_edges_pickle(path)
        sp = cls()
        sp._totals = dict(zip(data[IDS_KEY].tolist(), data[TOTALS_KEY].tolist(), strict=True))
        return sp


def collect(
    inputs: Iterable[bytes],
    run: Callable[[bytes], object],
    counts: Callable[[], dict[int, int]],
) -> Spectrum:
    """Run each input, then fold the edge counts it left behind."""
    sp = Spectrum()
    for k, data in enumerate(inputs, 1):
        run(data)
        sp.add(counts())
        if k % PROGRESS_EVERY == 0:
            print(f"\r[*] {k} inputs, {len(sp.ids())} edges", end="", flush=True)
    return sp


def _corpus(corpus: Path) -> list[bytes]:
    """Corpus bytes, oldest first: mtime is the fuzzer's discovery order."""
    files = sorted(_discover_corpus_files(corpus), key=lambda p: p.stat().st_mtime_ns)
    return [p.read_bytes() for p in files]


def replay(target: str, corpus: Path, timeout: float, ctx: Ctx) -> Spectrum:
    """Spawn *target* per input on one SHM segment; file-mode input."""
    disable_aslr()  # caller-context ids hash return addresses
    shm = ShmCoverage(estimate_map_size(target))
    env = os.environ.copy()
    env.update(
        {
            "__AFL_SHM_ID": shm.env_id,
            "AFL_MAP_SIZE": str(shm.num_entries),
            "__AFL_CTX_SENSITIVE": ctx.value,
        }
    )
    tmp = tempfile.mkdtemp(prefix="ehs_")

    def run(data: bytes) -> object:
        shm.reset_edge_map()
        return run_target_file(target, data, timeout, tmp, [], env=env)

    try:
        return collect(_corpus(corpus), run, shm.get_edge_counts)
    finally:
        shm.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)


def report(sp: Spectrum) -> None:
    totals = sp.totals()
    fit = fit_zipf(totals.tolist())
    print(f"\nedges {totals.size}  touches {int(totals.sum())}")
    print(f"s_fit  {fit.s:.3f}  (alpha {fit.alpha:.3f}, xmin {fit.xmin}, law {fit.law.name})")
    print(f"s_disc {sp.discovery_slope():.3f}")
    print(f"corr   {sp.hot_first_corr():+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target")
    ap.add_argument("-d", "--corpus", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path, help="spectrum pickle")
    ap.add_argument("-t", "--timeout", type=float, default=1.0)
    ap.add_argument("--ctx", choices=[c.name.lower() for c in Ctx], default=Ctx.OFF.name.lower())
    args = ap.parse_args()

    sp = replay(args.target, args.corpus, args.timeout, Ctx[args.ctx.upper()])
    sp.save(args.out)
    report(sp)


if __name__ == "__main__":
    main()
