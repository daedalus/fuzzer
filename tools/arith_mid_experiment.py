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
  line_blend extended line recombination a + t*(b-a), t in [-1/2, 3/2]
  field_minmax per-field min or max of the pair
  bit_cross  bit-level uniform crossover
  splice     prefix of a + suffix of b
  crossover  two-point segment swap
  byte_rand  one random byte replaced (ignores the donor)
  identity   returns a (control: must solve nothing, else the oracle is broken)

Three-seed arms (diff_evo, field_median, nm_reflect, opposition) need a third parent,
so their corpus starts as [LOW, MID, HIGH] (MID = 0x80 fill) and the other two seeds
are drawn at random; they are not comparable one-to-one with the two-seed arms.

--ext scores the extended gates instead (inputs of EXT_LEN = 28 bytes):
  G4  window PAST both seeds (gradient below only): needs t > 1 extrapolation
  G5  per-field combination (f1 high AND f2 low, each rewarded alone)
G6 (modular affine) is a cmplog gate and is not scored here. Every result prints the
budget (execs per run, runs per arm, mode) on its header line; quote it with the numbers.

Usage:
    python tools/arith_mid_experiment.py --execs 10000 --seeds 20
    python tools/arith_mid_experiment.py --ext --execs 10000 --seeds 20
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
    diff_evo,
    field_median,
    field_minmax,
    geo_mid,
    line_blend,
    nm_reflect,
    opposition,
    power_mid,
    splice,
)
from fuzzer_tool.core.rand_pool import RandPool

REPO = Path(__file__).resolve().parent.parent
GATES = 4
EXT_GATES = 6  # G0..G5 are scored through the plain library; G6 needs cmplog
SEED_LEN = 16
SOLVED_ALL = (1 << GATES) - 1
EXT_LEN = 28
EXT_SOLVED_ALL = (1 << 4) | (1 << 5)
LOW_SEED = b"K" + bytes(SEED_LEN - 1)
HIGH_SEED = b"K" + b"\xff" * (SEED_LEN - 1)
MID_SEED = b"K" + b"\x80" * (SEED_LEN - 1)


def _byte_rand(a: bytes, _b: bytes, rng) -> bytes:
    out = bytearray(a)
    out[rng.randrange(len(a))] = rng.randbytes(1)[0]
    return bytes(out)


def _ext_seed(g4: int, f1: int, f2: int) -> bytes:
    b = bytearray(b"K" + bytes(EXT_LEN - 1))
    b[11:15] = g4.to_bytes(4, "big")
    b[15:17] = f1.to_bytes(2, "big")
    b[17:19] = f2.to_bytes(2, "big")
    return bytes(b)


# Both G4 values sit below the window at 0x50000000; edge B (f2 low) is owned by the
# first seed, edge A (f1 high) by the second, so G5 needs a per-field combination.
EXT_LOW = _ext_seed(0x20000000, 0x0000, 0x0000)
EXT_MID = _ext_seed(0x28000000, 0x8000, 0x8000)
EXT_HIGH = _ext_seed(0x30000000, 0xFFFF, 0xFFFF)

ARMS = {
    "arith_mid": arith_mid,
    "line_blend": line_blend,
    "field_minmax": field_minmax,
    "geo_mid": geo_mid,
    "power_mid": power_mid,
    "bit_cross": bit_cross,
    "splice": splice,
    "crossover": crossover,
    "byte_rand": _byte_rand,
    "identity": lambda a, _b, _rng: a,
}

ARMS3 = {
    "diff_evo": diff_evo,
    "field_median": field_median,
    "nm_reflect": nm_reflect,
    "opposition": opposition,
}


class Oracle:
    """ctypes view of the plain (no shim, no coverage) build of arith_gates.c."""

    def __init__(self, lib: Path, n_gates: int = GATES):
        self._n = n_gates
        self._lib = ctypes.CDLL(str(lib))
        self._lib.fuzz_gates.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.fuzz_gates.restype = ctypes.c_int
        self._lib.gates_edges.restype = ctypes.POINTER(ctypes.c_uint64)

    def run(self, buf: bytes) -> tuple[int, tuple[int, ...]]:
        """(solved mask, band-edge mask per gate)."""
        mask = self._lib.fuzz_gates(buf, len(buf))
        edges = self._lib.gates_edges()
        return mask, tuple(edges[g] for g in range(self._n))


def _novel(edges: tuple[int, ...], seen: list[int]) -> bool:
    return any(e & ~s for e, s in zip(edges, seen, strict=True))


def run_one(
    oracle: Oracle, op, seed: int, execs: int, three: bool = False, ext: bool = False
):
    """(solved mask, corpus size, execs to solve every gate or None)."""
    rng = RandPool(seed=seed)
    low, mid, high = (EXT_LOW, EXT_MID, EXT_HIGH) if ext else (LOW_SEED, MID_SEED, HIGH_SEED)
    n_gates = EXT_GATES if ext else GATES
    target = EXT_SOLVED_ALL if ext else SOLVED_ALL
    corpus = [low, mid, high] if three else [low, high]
    mask, seen = 0, [0] * n_gates
    for s in corpus:
        m, e = oracle.run(s)
        mask |= m
        seen = [x | y for x, y in zip(seen, e, strict=True)]
    for n in range(1, execs + 1):
        i = rng.randrange(len(corpus))
        j = (i + 1 + rng.randrange(len(corpus) - 1)) % len(corpus)
        if three:
            k = rng.choice([x for x in range(len(corpus)) if x not in (i, j)])
            child = op(corpus[i], corpus[j], corpus[k], rng)
        else:
            child = op(corpus[i], corpus[j], rng)
        m, e = oracle.run(child)
        if m & ~mask or _novel(e, seen):
            corpus.append(child)
            mask |= m
            seen = [x | y for x, y in zip(seen, e, strict=True)]
        if mask & target == target:
            return mask, len(corpus), n
    return mask, len(corpus), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=[*ARMS, *ARMS3])
    ap.add_argument("--ext", action="store_true", help="score G4/G5 (28-byte inputs)")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="arith_gates_"))
    try:
        lib = tmp / "arith_gates_ref.so"
        subprocess.run(
            [
                "clang" if shutil.which("clang") else "gcc",
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
        oracle = Oracle(lib, EXT_GATES if a.ext else GATES)
        gate_ids = (4, 5) if a.ext else (1, 2, 3)
        mode = "ext (G4,G5)" if a.ext else "base (G1-G3)"
        print(f"# budget: {a.execs} execs/run, {a.seeds} runs/arm, mode={mode}")
        cols = "".join(f" {'G' + str(g):>4s}" for g in gate_ids)
        print(f"{'arm':10s} {'all':>5s}{cols}  median-execs-to-all")
        for arm in a.arms:
            three = arm in ARMS3
            op = ARMS3[arm] if three else ARMS[arm]
            runs = [run_one(oracle, op, s, a.execs, three, a.ext) for s in range(1, a.seeds + 1)]
            solved = [sum(m >> g & 1 for m, _c, _n in runs) for g in gate_ids]
            done = [n for _m, _c, n in runs if n is not None]
            med = f"{statistics.median(done):.0f}" if done else f">{a.execs}"
            print(
                f"{arm:10s} {len(done):>2d}/{a.seeds:<2d} "
                f"{''.join(f' {n:>4d}' for n in solved)}  {med}",
                flush=True,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
