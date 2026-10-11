#!/usr/bin/env python3
"""Which recombination operators solve the integer windows of targets/arith_gates.c?

Operator in isolation, not a fuzzer A/B: a two-seed corpus (one seed below every
window, one past them) is grown by applying ONE operator to random corpus pairs.
A child is admitted when it fires a distance-band edge no corpus seed has fired
or solves a new gate (the edges the real target fires). Scoring is ground truth from the plain
build of the target, never fuzzer feedback.

Arms:
  arith_mid  weighted arithmetic mean of the pair
  geo_mid    geometric mean of the pair
  power_mid  harmonic or quadratic mean of the pair (declines close pairs)
  bit_cross  bit-level uniform crossover
  splice     prefix of a + suffix of b
  crossover  two-point segment swap
  byte_rand  one random byte replaced (ignores the donor)
  identity   returns a (control: must solve nothing, else the oracle is broken)

Usage:
    python tools/arith_mid_experiment.py --execs 10000 --seeds 20
"""

from __future__ import annotations

import argparse
import ctypes
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from fuzzer_tool.core.mutations.generic import (
    arith_mid,
    bit_cross,
    crossover,
    geo_mid,
    power_mid,
    splice,
)
from fuzzer_tool.core.rand_pool import RandPool

REPO = Path(__file__).resolve().parent.parent
GATES = 4
SEED_LEN = 16
SOLVED_ALL = (1 << GATES) - 1
LOW_SEED = b"K" + bytes(SEED_LEN - 1)
HIGH_SEED = b"K" + b"\xff" * (SEED_LEN - 1)


def _byte_rand(a: bytes, _b: bytes, rng) -> bytes:
    out = bytearray(a)
    out[rng.randrange(len(a))] = rng.randbytes(1)[0]
    return bytes(out)


ARMS = {
    "arith_mid": arith_mid,
    "geo_mid": geo_mid,
    "power_mid": power_mid,
    "bit_cross": bit_cross,
    "splice": splice,
    "crossover": crossover,
    "byte_rand": _byte_rand,
    "identity": lambda a, _b, _rng: a,
}


class Oracle:
    """ctypes view of the plain (no shim, no coverage) build of arith_gates.c."""

    def __init__(self, lib: Path):
        self._lib = ctypes.CDLL(str(lib))
        self._lib.fuzz_gates.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.fuzz_gates.restype = ctypes.c_int
        self._lib.gates_edges.restype = ctypes.POINTER(ctypes.c_uint64)

    def run(self, buf: bytes) -> tuple[int, tuple[int, ...]]:
        """(solved mask, band-edge mask per gate)."""
        mask = self._lib.fuzz_gates(buf, len(buf))
        edges = self._lib.gates_edges()
        return mask, tuple(edges[g] for g in range(GATES))


def _novel(edges: tuple[int, ...], seen: list[int]) -> bool:
    return any(e & ~s for e, s in zip(edges, seen, strict=True))


def run_one(oracle: Oracle, op, seed: int, execs: int) -> tuple[int, int, int | None]:
    """(solved mask, corpus size, execs to solve every gate or None)."""
    rng = RandPool(seed=seed)
    corpus = [LOW_SEED, HIGH_SEED]
    mask, seen = 0, [0] * GATES
    for s in corpus:
        m, e = oracle.run(s)
        mask |= m
        seen = [x | y for x, y in zip(seen, e, strict=True)]
    for n in range(1, execs + 1):
        i = rng.randrange(len(corpus))
        j = (i + 1 + rng.randrange(len(corpus) - 1)) % len(corpus)
        child = op(corpus[i], corpus[j], rng)
        m, e = oracle.run(child)
        if m & ~mask or _novel(e, seen):
            corpus.append(child)
            mask |= m
            seen = [x | y for x, y in zip(seen, e, strict=True)]
        if mask == SOLVED_ALL:
            return mask, len(corpus), n
    return mask, len(corpus), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="arith_gates_"))
    try:
        lib = tmp / "arith_gates_ref.so"
        subprocess.run(
            [
                "clang",
                "-O2",
                "-shared",
                "-fPIC",
                "-DGATES_PLAIN",
                "-o",
                str(lib),
                str(REPO / "targets" / "arith_gates.c"),
            ],
            check=True,
        )
        oracle = Oracle(lib)
        print(f"{'arm':10s} {'all':>5s} {'G1':>4s} {'G2':>4s} {'G3':>4s}  median-execs-to-all")
        for arm in a.arms:
            runs = [run_one(oracle, ARMS[arm], s, a.execs) for s in range(1, a.seeds + 1)]
            solved = [sum(m >> g & 1 for m, _c, _n in runs) for g in (1, 2, 3)]
            done = [n for _m, _c, n in runs if n is not None]
            med = f"{statistics.median(done):.0f}" if done else f">{a.execs}"
            print(
                f"{arm:10s} {len(done):>2d}/{a.seeds:<2d} "
                f"{solved[0]:>4d} {solved[1]:>4d} {solved[2]:>4d}  {med}",
                flush=True,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
