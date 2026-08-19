#!/usr/bin/env python3
"""Collect real-coverage-diff samples for the item 4 sensitivity sweep.

Runs a corpus of JPEG seeds through jpeg_read in-process, applies cheap
deterministic mutations at selected offsets, and logs the resulting
(region_idx, diff_bits) pairs to a TSV using the same sparse format as
the round-9/10 sweep data.

This bypasses the fuzzer's mutation/selection machinery so we can target
large seeds and specific byte ranges directly, avoiding the minimizer
shrinkage that makes the live-fuzzer path impractical on low-edge targets.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import sys
from pathlib import Path

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.randomness import profile_buffer

# Reproduce the estimator's constants so the replay script can share them.

SEED = 42
MAP_SIZE = 65536  # must match the shim's AFL_MAP_SIZE


def _setup_shm() -> ShmCoverage:
    shm = ShmCoverage(size=MAP_SIZE)
    os.environ["__AFL_SHM_ID"] = shm.env_id
    return shm


def _load_lib(target: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(target))
    lib.fuzz_jpeg.restype = ctypes.c_int
    lib.fuzz_jpeg.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
    return lib


def _run_jpeg(lib: ctypes.CDLL, data: bytes, shm: ShmCoverage) -> set[int]:
    shm.reset_edge_map()
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    lib.fuzz_jpeg(buf, len(data))
    return shm.get_edge_ids()


def _mutate(data: bytes, offset: int, rng: random.Random) -> bytes:
    if not data:
        return bytes([rng.randint(0, 255)])
    ba = bytearray(data)
    if offset >= len(ba):
        offset = len(ba) - 1
    op = rng.choice(["inc", "flip", "set"])
    if op == "inc":
        ba[offset] = (ba[offset] + rng.randint(1, 255)) % 256
    elif op == "flip":
        ba[offset] ^= 0xFF
    else:
        ba[offset] = rng.randint(0, 255)
    return bytes(ba)


def _region_for_offset(data: bytes, offset: int) -> int | None:
    profiles = profile_buffer(data)
    for i, p in enumerate(profiles):
        if p.offset <= offset < p.offset + p.length:
            return i
    return None


def collect_samples(
    corpus: Path,
    target: Path,
    samples: int,
    mutations_per_seed: int,
    seed: int = 42,
) -> list[str]:
    rng = random.Random(seed)
    shm = _setup_shm()
    lib = _load_lib(target)

    seeds = sorted(p.read_bytes() for p in corpus.glob("*.jpg") if p.is_file())
    if not seeds:
        seeds = sorted(p.read_bytes() for p in corpus.glob("id_*") if p.is_file())
    if not seeds:
        raise ValueError(f"no seeds found under {corpus}")

    rng.shuffle(seeds)
    seeds = seeds[: min(samples, len(seeds))]

    lines: list[str] = []
    for data in seeds:
        if len(data) < 512:
            # Too short for region profiling; skip.
            continue
        baseline = _run_jpeg(lib, data, shm)
        if not baseline:
            continue
        for _ in range(mutations_per_seed):
            offset = rng.randrange(len(data))
            mutated = _mutate(data, offset, rng)
            mutant = _run_jpeg(lib, mutated, shm)
            diff = baseline ^ mutant
            if not diff:
                lines.append("0\t\n")
                continue
            region = _region_for_offset(data, offset)
            if region is None:
                continue
            bits = sorted(e % MAP_SIZE for e in diff)
            lines.append(f"{region}\t{','.join(map(str, bits))}\n")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect jpeg_read real-corpus diff samples")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--target", type=Path, default=Path("targets/jpeg_read.so"))
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--mutations-per-seed", type=int, default=5)
    p.add_argument("--output", default="-")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    lines = collect_samples(
        corpus=args.corpus,
        target=args.target,
        samples=args.samples,
        mutations_per_seed=args.mutations_per_seed,
        seed=args.seed,
    )

    text = "".join(lines)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
