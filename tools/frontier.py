#!/usr/bin/env python3
"""Coverage frontier: branch conditions a corpus takes one way only.

Runs every corpus file through a clang source-coverage build
(``tools/build_targets.sh --frontier`` -> ``<name>_cov``) and reports::

    both-ways    condition seen true and false        explored
    frontier     one side only (missing side listed)  the next edges
    never        condition not reached

Each frontier condition gets a source hint (ALLOC / LIMIT / SETTINGS / OPEN):
the first three are usually infeasible from input bytes alone.

With seed state in CORPUS (``state.pkl.gz``), the per-seed number of frontier
conditions reached is rank-correlated with child rate, next to a shuffled
control (Hard Rule 46).

Usage:
    tools/frontier.py ~/fuzzing/builds/fuzzgoat_read_cov ~/corpus_goat [--top 20]
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.frontier_cov import (  # noqa: F401  (re-exported for tests)
    STUB_NAME,
    Branch,
    Coverage,
    Run,
    Side,
    frontier_of,
)
from fuzzer_tool.core.rank_stats import rank_corr

SEED_SAMPLE = 200  # inputs exported one by one (per-seed reach)
SAMPLE_SEED = 5
SKIP_SUFFIXES = (".pkl", ".gz", ".json")


class Kind(Enum):
    """Source hint for why a side may be unreachable."""

    ALLOC = "alloc"  # allocation-failure check
    LIMIT = "limit"  # size / memory limit
    SETTINGS = "settings"  # configuration the harness fixes
    OPEN = "open"  # input-reachable as far as the line tells


_HINTS = (
    (Kind.ALLOC, re.compile(r"alloc")),
    (Kind.LIMIT, re.compile(r"max_memory|_max\b|uint_max|ulong_max")),
    (Kind.SETTINGS, re.compile(r"settings")),
)


def classify(line: str) -> Kind:
    """Source hint for one condition's line."""
    for kind, pat in _HINTS:
        if pat.search(line):
            return kind
    return Kind.OPEN


# ── report ──────────────────────────────────────────────────────────────────


def _corpus_files(corpus: Path) -> list[Path]:
    return sorted(
        p for p in corpus.rglob("*") if p.is_file() and not p.name.endswith(SKIP_SUFFIXES)
    )


def _source_line(sources: dict[str, str], b: Branch, cache: dict[str, list[str]]) -> str:
    path = sources.get(b[0])
    if path is None:
        return ""
    if path not in cache:
        cache[path] = Path(path).read_text(errors="replace").splitlines()
    lines = cache[path]
    return lines[b[1] - 1].strip() if 0 < b[1] <= len(lines) else ""


def _print_frontier(cov: Coverage, n_inputs: int, runs: list[Run], merged, front, top: int) -> None:
    both = sum(1 for t, f in merged.values() if t and f)
    never = sum(1 for t, f in merged.values() if not t and not f)
    print(
        f"inputs {n_inputs} | conditions {len(merged)}: both-ways {both}, frontier {len(front)}, never {never}"
    )

    # Sampled seeds reaching each frontier condition (either side counted).
    reach = dict.fromkeys(front, 0)
    for r in runs:
        for b in front:
            reach[b] += sum(r.branches.get(b, (0, 0))) > 0

    cache: dict[str, list[str]] = {}
    kinds = {b: classify(_source_line(cov.sources, b, cache)) for b in front}
    print("by hint: " + ", ".join(f"{k.value} {sum(v is k for v in kinds.values())}" for k in Kind))
    print(f"frontier (file:line:col  missing  hint  seeds of {len(runs)} sampled  source):")
    for b in sorted(front, key=lambda b: (kinds[b] is not Kind.OPEN, -reach[b], b))[:top]:
        line = _source_line(cov.sources, b, cache)[:60]
        print(
            f"  {b[0]}:{b[1]}:{b[2]}  {front[b].value:5s}  {kinds[b].value:8s}  {reach[b]:5d}  {line}"
        )


def _seed_meta(corpus: Path) -> dict:
    """seed_meta from the corpus state, {} when there is none."""
    from fuzzer_tool.core.state_store import STATE_FILENAME, StateStore

    if not (corpus / STATE_FILENAME).exists():
        return {}
    store = StateStore(str(corpus))
    store.load()
    return (store.get("corpus") or {}).get("seed_meta") or {}


def _sample(files: list[Path], meta: dict, n: int) -> list[Path]:
    """Up to n inputs for per-seed export: fuzzed seeds first (they carry a child rate)."""
    from fuzzer_tool.core.hashing import key16
    from fuzzer_tool.core.rand_pool import RandPool

    if len(files) <= n:
        return files

    rng = RandPool(SAMPLE_SEED)
    fuzzed = [f for f in files if (meta.get(key16(f.read_bytes())) or {}).get("fuzz_count")]
    taken = set(fuzzed)
    rest = [f for f in files if f not in taken]
    picked = rng.sample(fuzzed, min(n, len(fuzzed)))
    return picked + rng.sample(rest, n - len(picked))


def _print_seed_reach(meta: dict, runs: list[Run], front) -> None:
    """Frontier reach vs child rate over the sampled seeds, with a shuffled control."""
    from fuzzer_tool.core.hashing import key16
    from fuzzer_tool.core.rand_pool import RandPool

    rows = []
    for r in runs:
        m = meta.get(key16(r.path.read_bytes()))
        if not m or not m.get("fuzz_count"):
            continue
        hit = sum(sum(r.branches.get(b, (0, 0))) > 0 for b in front)
        rows.append((m["child_count"] / m["fuzz_count"], hit))
    if len(rows) < 3:
        print("seed reach: fewer than 3 fuzzed seeds with state; skipped")
        return

    rate, hit = (np.array(c, float) for c in zip(*rows, strict=True))
    shuffled = rate[np.array(RandPool(SAMPLE_SEED).sample(range(len(rate)), len(rate)))]
    print(
        f"seed reach ({len(rows)} seeds): Spearman child-rate ~ frontier reached {rank_corr(rate, hit):+.2f}"
        f" | control (shuffled) {rank_corr(shuffled, hit):+.2f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("binary", type=Path, help="<name>_cov from build_targets.sh --frontier")
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--top", type=int, default=20, help="frontier rows to list")
    ap.add_argument("--workers", type=int, default=4, help="parallel runs / exports")
    ap.add_argument(
        "--seed-sample",
        type=int,
        default=SEED_SAMPLE,
        help="inputs exported per seed (4 s each on FFmpeg)",
    )
    args = ap.parse_args(argv)

    corpus = args.corpus.expanduser()
    files = _corpus_files(corpus)
    meta = _seed_meta(corpus)
    with tempfile.TemporaryDirectory(prefix="frontier_") as work:
        cov = Coverage(args.binary.expanduser(), Path(work), args.workers)
        merged = cov.profile(files)
        runs = cov.run(_sample(files, meta, args.seed_sample))

    front = frontier_of(merged)
    _print_frontier(cov, len(files), runs, merged, front, args.top)
    if meta:
        _print_seed_reach(meta, runs, front)
    return 0


if __name__ == "__main__":
    sys.exit(main())
