"""Seeded ``mutate`` campaign and crash replay.

    campaign seed ──► RandPool ──► per-case seed ──► kind builder ──► _Case
                                                                        │
             crash_dir/<target>__<hash>.<ext>  ◄── save ◄── finding ◄── run

Each case draws its own sub-seed, so building a case never shifts the
draws of the next one: ``case_labels`` replays generation without running.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.rand_pool import RandPool

from . import corpus, targets
from .mutators import mutate_bytes, mutate_obj, save_crash
from .regress import REGRESSION_MUTATORS, run_regression

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

KIND_WEIGHTS = {"elf": 3, "dict": 1, "state": 3, "learner": 2, "ops": 4, "regress": 2}
REGRESS_PREFIX = "regress."
SUBSEED_RANGE = 1 << 32
LEARNER_ROWS_MAX = 200
# Adversarial-but-type-valid transition fields.
_OFFSETS = (None, -1, 0, 1 << 40)
_WIDTHS = (0, 1, -3, 1 << 20)
_LEARNER_OPS = ("bit_flip", "splice", "havoc")
_LEARNER_PREFIXES = (b"\x89PNG", b"\xff\xd8", b"PK\x03\x04", b"")


@dataclass
class _Case:
    """One generated input: a target and its bytes, or a regress mutator."""

    target: str
    data: bytes
    seed: int

    def label(self) -> str:
        return f"{self.target}#{hash_data(self.data)[:8]}"


# ── case builders (one per kind) ─────────────────────────────────────────────


def _build_elf(rng: RandPool) -> tuple[str, bytes]:
    target = "elf_path" if rng.random() < 0.2 else "elf_bytes"
    return target, mutate_bytes(rng, rng.choice(corpus.elf_seeds()))


def _build_dict(rng: RandPool) -> tuple[str, bytes]:
    return "dict", mutate_bytes(rng, rng.choice(corpus.text_seeds()))


def _build_state(rng: RandPool) -> tuple[str, bytes]:
    """Whole-file corruption or one section with a structural mutation."""
    sections = corpus.section_seeds()
    target = rng.choice(("state", *sections))
    if target == "state":
        return target, mutate_bytes(rng, corpus.state_seed())
    return target, pickle.dumps(mutate_obj(rng, sections[target]))


def _rand_row(rng: RandPool) -> tuple:
    data = rng.choice(_LEARNER_PREFIXES) + rng.randbytes(rng.randint(0, 32))
    before = rng.randint(-5, 100)
    offset = rng.choice(_OFFSETS) if rng.random() < 0.3 else rng.randint(0, len(data) + 8)
    width = rng.choice(_WIDTHS) if rng.random() < 0.2 else rng.randint(1, 8)
    return (
        data,
        rng.choice(_LEARNER_OPS),
        offset,
        width,
        before,
        before + rng.randint(-3, 3),
        {rng.randint(0, 64) for _ in range(rng.randint(0, 3))},
        {rng.randint(0, 64) for _ in range(rng.randint(0, 2))},
    )


def _build_learner(rng: RandPool) -> tuple[str, bytes]:
    rows = [_rand_row(rng) for _ in range(rng.randint(1, LEARNER_ROWS_MAX))]
    return "learner", pickle.dumps(rows)


def _build_ops(rng: RandPool) -> tuple[str, bytes]:
    pool = [*corpus.elf_seeds()[1:], *corpus.text_seeds(), b"\x89PNG\r\n\x1a\n", b""]
    data = mutate_bytes(rng, rng.choice(pool))
    # Draw from the full table, not `op_names`: availability reads registry
    # caches, which would make generation depend on earlier runs.
    return targets.OP_PREFIX + rng.choice(targets.all_ops()), data


def _build_regress(rng: RandPool) -> tuple[str, bytes]:
    return REGRESS_PREFIX + rng.choice(sorted(REGRESSION_MUTATORS)), b""


_BUILDERS: dict[str, Callable[[RandPool], tuple[str, bytes]]] = {
    "elf": _build_elf,
    "dict": _build_dict,
    "state": _build_state,
    "learner": _build_learner,
    "ops": _build_ops,
    "regress": _build_regress,
}


# ── campaign ─────────────────────────────────────────────────────────────────


def parse_kinds(spec: str) -> list[str]:
    """Validate a comma list of kinds; ValueError names the bad ones."""
    kinds = [k for k in spec.split(",") if k]
    bad = [k for k in kinds if k not in KIND_WEIGHTS]
    if bad:
        raise ValueError(f"unknown kinds: {', '.join(bad)}")
    return kinds


def _cases(opts: argparse.Namespace):
    """Yield cases until the iteration count or the time budget runs out."""
    kinds = parse_kinds(opts.kinds)
    weights = [KIND_WEIGHTS[k] for k in kinds]
    rng = RandPool(opts.seed)
    deadline = time.monotonic() + opts.time_budget if opts.time_budget else None

    for _ in range(opts.iterations):
        if deadline is not None and time.monotonic() > deadline:
            return

        seed = rng.randrange(SUBSEED_RANGE)
        sub = RandPool(seed)
        target, data = _BUILDERS[rng.weighted_choice(kinds, weights)](sub)
        yield _Case(target, data, seed)


def case_labels(opts: argparse.Namespace, _workdir: Path) -> list[str]:
    """Generated case labels without running them (determinism checks)."""
    return [c.label() for c in _cases(opts)]


def _run_case(case: _Case, workdir: Path) -> tuple[str | None, str | None]:
    """``(dedup key, detail)`` for one case."""
    if not case.target.startswith(REGRESS_PREFIX):
        return targets.execute(case.target, case.data)

    name = case.target.removeprefix(REGRESS_PREFIX)
    detail = run_regression(name, RandPool(case.seed), workdir)
    if detail is None:
        return None, None
    return f"{case.target}:{detail.split(':')[0]}", f"{detail} (case seed {case.seed})"


def fuzz_mutate(workdir: Path, opts: argparse.Namespace) -> list[corpus.Failure]:
    """Run the campaign; one failure (and crash file) per distinct key."""
    workdir.mkdir(parents=True, exist_ok=True)
    crash_dir = Path(opts.crash_dir).expanduser()
    seen: set[str] = set()
    fails: list[corpus.Failure] = []
    print(f"[mutate] seed {opts.seed}")

    for case in _cases(opts):
        key, detail = _run_case(case, workdir)
        if key is None or key in seen:
            continue

        seen.add(key)
        where = ""
        target = targets.resolve(case.target)
        if target is not None:
            where = f" -> {save_crash(crash_dir, case.target, target.ext, case.data)}"
        fails.append((case.target, f"{detail}{where}"))

    return fails


def replay(path: Path) -> int:
    """Re-run one crash file: 0 clean, 1 still failing, 2 unknown target."""
    target = targets.resolve(path.name.split("__")[0])
    if target is None:
        print(f"unknown target for {path.name}")
        return 2

    detail = targets.run_target(target.name, path.read_bytes())
    print(f"{path.name}: {detail or 'clean'}")
    return 1 if detail else 0
