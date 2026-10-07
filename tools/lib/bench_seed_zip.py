"""A/B: loose seed files vs --zip-seed-corpus, write and read.

Arms (interleaved per repetition, so drift hits all arms alike):
  files      save_to_corpus -> seeds/<hh>/id_<h>          (default layout)
  files_ctl  same arm again: the noise floor (Hard Rule 46)
  zip        save_to_corpus -> seeds.zip, defaults (block grows with archive)
  zip_fixed  fixed BLOCK_SEEDS block (growth off): directory rewrite O(N^2)
  zip_b1     block of 1: one directory rewrite per seed (why blocks exist)

Write = N saves + final flush. Read = load_corpus warm, and cold after
drop_caches when writable (root). Seeds are mutated slices of real files
(ELF, text) plus incompressible noise, sized like a fuzz corpus.

    python tools/lib/bench_seed_zip.py --n 1000 10000 --reps 5 --out ~/bench_seed_zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import time
from pathlib import Path

from fuzzer_tool.adapters import seed_zip
from fuzzer_tool.adapters.filesystem import load_corpus, save_to_corpus
from fuzzer_tool.adapters.seed_zip import ZipMode
from fuzzer_tool.core.rand_pool import RandPool

_SOURCES = ("/usr/bin/python3", "/usr/bin/bash", "/usr/share/common-licenses/GPL-3")
_SIZES = (32, 128, 512, 2048, 8192, 32768)  # skewed small, as corpora are
_MUTATIONS = 4
_DROP = Path("/proc/sys/vm/drop_caches")


def _material() -> bytes:
    return b"".join(Path(p).read_bytes()[: 4 << 20] for p in _SOURCES if Path(p).is_file())


def make_seeds(n: int, seed: int) -> list[bytes]:
    """Mutated slices of real files; every 5th seed is pure noise."""
    rng = RandPool(seed)
    mat = _material()
    out: list[bytes] = []
    while len(out) < n:
        size = rng.choice(_SIZES) + rng.randint(0, 31)
        if len(out) % 5 == 4:
            out.append(rng.randbytes(size))
            continue
        off = rng.randint(0, len(mat) - size - 1)
        buf = bytearray(mat[off : off + size])
        for _ in range(_MUTATIONS):
            buf[rng.randint(0, size - 1)] = rng.randint(0, 255)
        out.append(bytes(buf))
    return out


def _du_kib(path: Path) -> tuple[int, int]:
    """(allocated KiB, apparent KiB) under *path*."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    alloc = sum(os.stat(p).st_blocks * 512 for p in files)
    return alloc // 1024, sum(p.stat().st_size for p in files) // 1024


def _cold() -> bool:
    try:
        os.sync()
        _DROP.write_text("3\n")
        return True
    except OSError:
        return False


_ZIP_ARMS = {
    "zip": (seed_zip.BLOCK_SEEDS, seed_zip.BLOCK_GROWTH),
    "zip_fixed": (seed_zip.BLOCK_SEEDS, 0),
    "zip_b1": (1, 0),
}


def _arm_mode(arm: str) -> tuple[ZipMode, int, int]:
    if arm in _ZIP_ARMS:
        return ZipMode.ON, *_ZIP_ARMS[arm]
    return ZipMode.OFF, seed_zip.BLOCK_SEEDS, seed_zip.BLOCK_GROWTH


def run_arm(arm: str, seeds: list[bytes], root: Path) -> dict:
    corpus = root / arm
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    mode, block, growth = _arm_mode(arm)

    store = seed_zip.configure(corpus, mode, block_seeds=block, growth=growth)
    seen: set[str] = set()
    t0 = time.perf_counter()
    for s in seeds:
        save_to_corpus(s, corpus, seen)
    if store is not None:
        store.flush()
    write = time.perf_counter() - t0

    seed_zip.configure(corpus, mode)  # fresh store: no in-memory index reuse
    t0 = time.perf_counter()
    got, _, _ = load_corpus(corpus, add_default=False)
    warm = time.perf_counter() - t0
    assert len(got) == len(set(seeds)), (arm, len(got))

    cold = float("nan")
    if _cold():
        seed_zip.configure(corpus, mode)
        t0 = time.perf_counter()
        load_corpus(corpus, add_default=False)
        cold = time.perf_counter() - t0
    seed_zip.configure(corpus, ZipMode.OFF)

    target = corpus / seed_zip.ZIP_NAME if mode is ZipMode.ON else corpus / "seeds"
    alloc, apparent = _du_kib(target)
    return {"write": write, "warm": warm, "cold": cold, "alloc": alloc, "apparent": apparent}


def _fmt(xs: list[float]) -> str:
    med = statistics.median(xs)
    return f"{med * 1e3:9.1f} ms [{min(xs) * 1e3:.1f}-{max(xs) * 1e3:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--arms", nargs="+", default=["files", "files_ctl", "zip", "zip_fixed", "zip_b1"]
    )
    ap.add_argument("--b1-max", type=int, default=10000, help="skip zip_b1 above this N")
    ap.add_argument("--out", type=Path, default=Path.home() / "bench_seed_zip")
    a = ap.parse_args()

    for n in a.n:
        seeds = make_seeds(n, seed=0xC0FFEE + n)
        raw = sum(map(len, seeds)) // 1024
        arms = [x for x in a.arms if not (x == "zip_b1" and n > a.b1_max)]
        res: dict[str, list[dict]] = {x: [] for x in arms}
        for _ in range(a.reps):
            for arm in arms:
                res[arm].append(run_arm(arm, seeds, a.out / str(n)))

        print(f"\nN={n}  raw={raw} KiB  reps={a.reps}  (median [min-max])")
        print(f"{'arm':10} {'write':>28} {'read warm':>28} {'read cold':>28}  alloc/apparent KiB")
        for arm in arms:
            r = res[arm]
            cols = [_fmt([x[k] for x in r]) for k in ("write", "warm", "cold")]
            print(
                f"{arm:10} {cols[0]:>28} {cols[1]:>28} {cols[2]:>28}  {r[0]['alloc']}/{r[0]['apparent']}"
            )
        shutil.rmtree(a.out / str(n), ignore_errors=True)


if __name__ == "__main__":
    main()
