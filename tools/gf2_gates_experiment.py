#!/usr/bin/env python3
"""Which strategies open the GF(2)-affine gates of targets/gf2_gates.c?

Each arm gets a fixed exec budget per seed; every oracle call counts as one exec.
Scoring is ground truth from the plain build of the target, never fuzzer feedback.
Corpus arms admit a child only when it solves a gate no corpus seed solved (a solved
gate is the only edge these gates fire, so there is no distance gradient).

Arms:
  random        uniform random 11-byte input
  byte_rand     one random byte replaced, from a random corpus seed
  bit_flip      one random bit flipped
  havoc         1-4 stacked byte/bit edits
  baseline      alternates havoc and compare-substitution (operand -> target)
  probe_affine  D1 as specified: classify the solved bit of each gate field
                (core/degree_probe), solve_affine when AFFINE
  operand_map   learn each gate's operand as an affine map from unit vectors
                (the value cmplog logs), solve for the target; n+2 execs per gate
  identity      always the same seed (control: must solve nothing)

Usage:
    python tools/gf2_gates_experiment.py --execs 20000 --seeds 20
"""

from __future__ import annotations

import argparse
import ctypes
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from fuzzer_tool.core.degree_probe import Degree, classify, solve_affine
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.xor_map_solver import IncrementalXorMapSolver

REPO = Path(__file__).resolve().parent.parent
GATES = 4
INPUT_LEN = 11
SOLVED_ALL = (1 << GATES) - 1
BASE = bytes(INPUT_LEN)
MAX_STACK = 4
SUBST_MAX_WIDTH = 4

# gate -> (offset, field bits, byte order): where each gate reads its field.
FIELDS = (
    (0, 8, "big"),
    (1, 16, "big"),
    (3, 32, "big"),
    (7, 32, "little"),
)


class Oracle:
    """ctypes view of the plain (no shim, no coverage) build of gf2_gates.c."""

    def __init__(self, lib: Path):
        self._lib = ctypes.CDLL(str(lib))
        self._lib.fuzz_gates.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.fuzz_gates.restype = ctypes.c_int
        self._lib.gates_operand.restype = ctypes.POINTER(ctypes.c_uint32)
        self._lib.gates_target.restype = ctypes.POINTER(ctypes.c_uint32)
        self.execs = 0

    def run(self, buf: bytes) -> int:
        """Solved-gate mask; counts one exec."""
        self.execs += 1
        return self._lib.fuzz_gates(buf, len(buf))

    def operand(self, g: int) -> int:
        """Operand of gate g on the last run (the value cmplog would log)."""
        return self._lib.gates_operand()[g]

    def target(self, g: int) -> int:
        return self._lib.gates_target()[g]


def put_field(g: int, x: int) -> bytes:
    """BASE with gate g's field set to x."""
    off, bits, order = FIELDS[g]
    out = bytearray(BASE)
    out[off : off + bits // 8] = x.to_bytes(bits // 8, order)
    return bytes(out)


def solve_system(masks: list[int], rhs: int) -> int | None:
    """x with parity(masks[j] & x) == bit j of rhs for every j, else None.

    Free variables are 0.
    """
    pivots: dict[int, tuple[int, int]] = {}
    for j, m in enumerate(masks):
        r = rhs >> j & 1
        for p, (pm, pr) in pivots.items():
            if m >> p & 1:
                m ^= pm
                r ^= pr
        if m == 0:
            if r:
                return None
            continue
        low = (m & -m).bit_length() - 1
        for p, (pm, pr) in list(pivots.items()):
            if pm >> low & 1:
                pivots[p] = (pm ^ m, pr ^ r)
        pivots[low] = (m, r)
    return sum(r << p for p, (_m, r) in pivots.items())


# ── Corpus arms ─────────────────────────────────────────────────────────────


def _byte_rand(buf: bytes, rng) -> bytes:
    out = bytearray(buf)
    out[rng.randrange(len(out))] = rng.randbytes(1)[0]
    return bytes(out)


def _bit_flip(buf: bytes, rng) -> bytes:
    out = bytearray(buf)
    out[rng.randrange(len(out))] ^= 1 << rng.randrange(8)
    return bytes(out)


def _havoc(buf: bytes, rng) -> bytes:
    for _ in range(1 + rng.randrange(MAX_STACK)):
        buf = _byte_rand(buf, rng) if rng.randrange(2) else _bit_flip(buf, rng)
    return buf


def _subst(oracle: Oracle, buf: bytes, rng) -> bytes:
    # Compare substitution: replace the logged operand with the target where it occurs.
    g = rng.randrange(GATES)
    oracle.run(buf)
    op, tgt = oracle.operand(g), oracle.target(g)
    width = max(1, (tgt.bit_length() + 7) // 8)
    if width > SUBST_MAX_WIDTH:
        return buf
    for order in ("big", "little"):
        pat = op.to_bytes(width, order)
        at = buf.find(pat)
        if at >= 0:
            return buf[:at] + tgt.to_bytes(width, order) + buf[at + width :]
    return buf


def _corpus_run(mutate: Callable, oracle: Oracle, rng, execs: int) -> int:
    corpus = [BASE]
    mask = oracle.run(BASE)
    while oracle.execs < execs and mask != SOLVED_ALL:
        child = mutate(oracle, corpus[rng.randrange(len(corpus))], rng)
        m = oracle.run(child)
        if m & ~mask:
            corpus.append(child)
            mask |= m
    return mask


def _arm_random(oracle: Oracle, rng, execs: int) -> int:
    mask = 0
    while oracle.execs < execs and mask != SOLVED_ALL:
        mask |= oracle.run(rng.randbytes(INPUT_LEN))
    return mask


def _arm_identity(oracle: Oracle, _rng, execs: int) -> int:
    mask = oracle.run(BASE)
    oracle.execs = execs
    return mask


# ── Probe arms ──────────────────────────────────────────────────────────────


def _arm_probe_affine(oracle: Oracle, rng, execs: int) -> int:
    mask = oracle.run(BASE)
    for g in range(GATES):
        if mask >> g & 1 or oracle.execs >= execs:
            continue
        bits = FIELDS[g][1]

        def f(x: int, g=g) -> int:
            return oracle.run(put_field(g, x)) >> g & 1

        if classify(f, bits, rng) is not Degree.AFFINE:
            continue
        x = solve_affine(f, bits, 1)
        if x is not None:
            mask |= oracle.run(put_field(g, x))
    return mask


def _arm_operand_map(oracle: Oracle, _rng, execs: int) -> int:
    mask = oracle.run(BASE)
    for g in range(GATES):
        if mask >> g & 1 or oracle.execs >= execs:
            continue
        bits = FIELDS[g][1]
        oracle.run(put_field(g, 0))
        const = oracle.operand(g)
        solver = IncrementalXorMapSolver(bits)
        for i in range(bits):
            oracle.run(put_field(g, 1 << i))
            solver.add_pair(1 << i, oracle.operand(g) ^ const)
        sol, _sat = solver.solve()
        if sol is None:
            continue
        x = solve_system(
            [sum(1 << i for i in idx) for idx in sol], oracle.target(g) ^ const
        )
        if x is not None:
            mask |= oracle.run(put_field(g, x))
    return mask


ARMS: dict[str, Callable] = {
    "random": _arm_random,
    "byte_rand": lambda o, r, e: _corpus_run(lambda _o, b, rg: _byte_rand(b, rg), o, r, e),
    "bit_flip": lambda o, r, e: _corpus_run(lambda _o, b, rg: _bit_flip(b, rg), o, r, e),
    "havoc": lambda o, r, e: _corpus_run(lambda _o, b, rg: _havoc(b, rg), o, r, e),
    "baseline": lambda o, r, e: _corpus_run(
        lambda oo, b, rg: _subst(oo, b, rg) if rg.randrange(2) else _havoc(b, rg), o, r, e
    ),
    "probe_affine": _arm_probe_affine,
    "operand_map": _arm_operand_map,
    "identity": _arm_identity,
}


def run_one(oracle: Oracle, arm: str, seed: int, execs: int) -> tuple[int, int | None]:
    """(solved mask, execs used when every gate was solved, else None)."""
    oracle.execs = 0
    mask = ARMS[arm](oracle, RandPool(seed=seed), execs)
    return mask, (oracle.execs if mask == SOLVED_ALL else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="gf2_gates_"))
    try:
        lib = tmp / "gf2_gates_ref.so"
        subprocess.run(
            [
                "cc",
                "-O2",
                "-shared",
                "-fPIC",
                "-DGATES_PLAIN",
                "-o",
                str(lib),
                str(REPO / "targets" / "gf2_gates.c"),
            ],
            check=True,
        )
        oracle = Oracle(lib)
        print(
            f"{'arm':13s} {'all':>5s} {'G0':>4s} {'G1':>4s} {'G2':>4s} {'G3':>4s}  median-execs-to-all"
        )
        for arm in a.arms:
            runs = [run_one(oracle, arm, s, a.execs) for s in range(1, a.seeds + 1)]
            solved = [sum(m >> g & 1 for m, _n in runs) for g in range(GATES)]
            done = [n for _m, n in runs if n is not None]
            med = f"{statistics.median(done):.0f}" if done else f">{a.execs}"
            print(
                f"{arm:13s} {len(done):>2d}/{a.seeds:<2d} "
                + " ".join(f"{s:>4d}" for s in solved)
                + f"  {med}",
                flush=True,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
