#!/usr/bin/env python3
"""Single entry point for every benchmark harness.

    tools/benchmark.py list
    tools/benchmark.py <name> [args...]     # args forwarded verbatim

Harnesses live in tools/lib/ and keep their own flags; this only routes. `<name> --help` shows
the harness's help. Exit code is the harness's.

    benchmark.py paired run ...
         |
         v  exec (same pid, same exit code)
    python tools/lib/bench_paired.py run ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent / "lib"

# name -> (script under tools/lib/, one-line purpose)
_REGISTRY: dict[str, tuple[str, str]] = {
    "smoke": ("bench.sh", "one run per named config (baseline/enhanced/optimal/qea); smoke test"),
    "sweep": ("bench_sweep.sh", "feature combination sweep at -n 1k"),
    "paired": ("bench_paired.py", "paired A/B over a locked (target, seed) matrix, McNemar"),
    "replicated": ("bench_replicated.py", "replicated, time-paired A/B across two source trees"),
    "noise": ("noise_probe.py", "same-arm replicate spread of a paired cell"),
    "diff": ("bench_diff.py", "differential analysis of two bench logs"),
    "cache": ("bench_cache.py", "profile cache: cold vs cached startup"),
    "lineage": ("lineage_benchmark.py", "--lineage on vs off, fixed exec budget"),
    "randpool": ("bench_randpool.py", "RandPool vs random on the hotpath call mix"),
    "havoc": ("bench_havoc_subop.py", "adaptive havoc sub-op sampler vs uniform"),
    "huffman": ("bench_huffman_scheduler.py", "Huffman/Fenwick vs batched-CDF seed sampler"),
}

BENCHMARKS: dict[str, str] = {name: script for name, (script, _) in _REGISTRY.items()}

_SHELL_SUFFIX = ".sh"


def _usage() -> str:
    width = max(map(len, _REGISTRY))
    rows = [f"  {name:<{width}}  {desc}" for name, (_, desc) in _REGISTRY.items()]
    return "usage: benchmark.py {list | <name> [args...]}\n\n" + "\n".join(rows)


def _argv(script: Path, args: list[str]) -> list[str]:
    """Interpreter command for *script*: bash for .sh, this python otherwise."""
    if script.suffix == _SHELL_SUFFIX:
        return ["bash", str(script), *args]

    return [sys.executable, str(script), *args]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0 if argv else 2

    name, rest = argv[0], argv[1:]
    if name == "list":
        print(_usage())
        return 0

    script = BENCHMARKS.get(name)
    if script is None:
        print(f"benchmark.py: unknown benchmark '{name}'\n\n{_usage()}", file=sys.stderr)
        return 2

    # exec, not subprocess: signals, stdin and the exit code stay the harness's.
    cmd = _argv(LIB / script, rest)
    os.execvp(cmd[0], cmd)
    return 1  # unreachable: execvp raises on failure


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
