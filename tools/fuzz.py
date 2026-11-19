#!/usr/bin/env python3
"""Robustness self-fuzzer for fuzzer-tool's own parsers.

Throws malformed and pathological inputs at fuzzer-tool entry points and
reports every uncaught exception, broken output contract or slow call:

* ``elf``     -- ELF parsers (bytes + path entry points), AFL dictionaries
* ``state``   -- corrupt ``--resume`` state files and persisted sections
                 (markov, edge tracker, format learner)
* ``learner`` -- format-learner transition streams at type-valid extremes
* ``ops``     -- every armed op mutator on degenerate buffers
* ``mutate``  -- seeded mutation campaign over all of the above, plus
                 ``regress`` mutators regenerating past bugs with random
                 parameters and metamorphic oracles

Ported from impactguard ``tools/fuzz.py``.

Usage::

    python tools/fuzz.py                       # all suites
    python tools/fuzz.py elf state             # selected suites
    python tools/fuzz.py mutate --iterations 20000 --seed 1234
    python tools/fuzz.py mutate --kinds regress --iterations 500
    python tools/fuzz.py --replay ~/fuzzing/selffuzz-crashes/state__0123456789ab.pkl

Exit status is 1 when any failure was found.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# Sibling ``lib/fuzzkit`` importable as a script and when loaded by path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from fuzzkit.campaign import KIND_WEIGHTS, fuzz_mutate, replay  # noqa: E402
from fuzzkit.corpus import CRASH_DIR  # noqa: E402
from fuzzkit.fixed import fuzz_elf, fuzz_learner, fuzz_ops, fuzz_state  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from fuzzkit.corpus import Failure

SEED_BITS = 32

SUITES: dict[str, Callable[[Path, argparse.Namespace], list[Failure]]] = {
    "elf": fuzz_elf,
    "state": fuzz_state,
    "learner": fuzz_learner,
    "ops": fuzz_ops,
    "mutate": fuzz_mutate,
}


def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "suites", nargs="*", default=list(SUITES), help=f"{', '.join(SUITES)} (default: all)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show library logs")

    mut = parser.add_argument_group("mutate suite")
    mut.add_argument("--iterations", type=int, default=2000, help="cases (default: 2000)")
    mut.add_argument("--seed", type=int, default=None, help="campaign seed (default: random)")
    mut.add_argument("--time-budget", type=float, default=0, help="seconds, 0 = no limit")
    mut.add_argument("--crash-dir", default=str(CRASH_DIR), help=f"default: {CRASH_DIR}")
    mut.add_argument(
        "--kinds", default=",".join(KIND_WEIGHTS), help="comma list of mutation families"
    )
    mut.add_argument("--replay", type=Path, metavar="FILE", help="re-run one crash file")
    return parser


def _silence_libraries() -> None:
    """Hide library logs and warnings (unless --verbose)."""
    import logging

    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")


def _run_suites(args: argparse.Namespace) -> list[Failure]:
    """Each suite in its own temp dir; print its failures."""
    total: list[Failure] = []
    with tempfile.TemporaryDirectory(prefix="selffuzz_") as tmp:
        for name in args.suites:
            workdir = Path(tmp) / name
            workdir.mkdir()
            started = time.monotonic()
            fails = SUITES[name](workdir, args)

            print(f"[{name}] {len(fails)} failure(s) in {time.monotonic() - started:.1f}s")
            for label, detail in fails:
                print(f"  {label}: {detail}")
            total.extend(fails)
    return total


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.replay is not None:
        return replay(args.replay.expanduser())

    unknown = [s for s in args.suites if s not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}")
    bad_kinds = [k for k in args.kinds.split(",") if k and k not in KIND_WEIGHTS]
    if bad_kinds:
        parser.error(f"unknown --kinds: {', '.join(bad_kinds)}")

    # Seed printing is the reproduction handle; SystemRandom only picks it.
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1 << SEED_BITS)
    if not args.verbose:
        _silence_libraries()

    total = _run_suites(args)
    print(f"TOTAL: {len(total)} failure(s)")
    return 1 if total else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
