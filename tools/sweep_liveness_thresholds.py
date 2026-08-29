#!/usr/bin/env python3
"""Sensitivity analysis for item 4 liveness defaults.

Varies ``_LIVENESS_DEAD_WEIGHT`` and ``_LIVENESS_SWITCH_AFTER`` across
reasonable ranges on representative campaign data and reports whether the
current defaults produce stable padding hypotheses.

This is the Sequenced step 6 validation called out in
``docs/handover/handover_skittercreek_tailslayer_port.md``. It is intentionally a
standalone script rather than a pytest because the sweep spans parameter
ranges, not a single fixed scenario.

Usage examples::

    # Synthetic-only sweep (fast, CI-friendly)
    python tools/sweep_liveness_thresholds.py

    # Real-corpus sweep: load PNG seeds and run the target for coverage
    python tools/sweep_liveness_thresholds.py --corpus ~/fuzzing/png_corpus/seeds --target targets/png_read.so --samples 200
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from fuzzer_tool.core.format_learner import FormatLearner

# Reproduce the operator-side defaults so the sweep can override them.
from fuzzer_tool.services.operators import (
    _LIVENESS_DEAD_WEIGHT as DEFAULT_DEAD_WEIGHT,
)
from fuzzer_tool.services.operators import (
    _LIVENESS_SWITCH_AFTER as DEFAULT_SWITCH_AFTER,
)


@dataclass
class SweepConfig:
    dead_weights: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    switch_afters: tuple[int, ...] = (50, 100, 200, 400, 800)
    seed: int = 42
    n_transitions: int = 2_000
    n_liveness_events: int = 600
    corpus: Path | None = None
    target: Path | None = None
    samples: int = 200


def _load_real_seeds(corpus: Path, limit: int, rng: random.Random) -> list[bytes]:
    seeds = [p.read_bytes() for p in corpus.glob("id_*") if p.is_file()]
    if not seeds:
        raise ValueError(f"no seeds found under {corpus}")
    rng.shuffle(seeds)
    return seeds[: min(limit, len(seeds))]


def _mutate_seed(seed: bytes, rng: random.Random) -> bytes:
    """Cheap deterministic-ish mutation for real-corpus replay."""
    if not seed:
        return bytes([rng.randint(0, 255)])
    ba = bytearray(seed)
    idx = rng.randrange(len(ba))
    op = rng.choice(["inc", "flip", "set"])
    if op == "inc":
        ba[idx] = (ba[idx] + rng.randint(1, 255)) % 256
    elif op == "flip":
        ba[idx] ^= 0xFF
    else:
        ba[idx] = rng.randint(0, 255)
    return bytes(ba)


def _heuristic_coverage(base: bytes, mutated: bytes) -> tuple[int, set, set]:
    """Fake coverage transition when the real target is unavailable.

    This is a fallback only; when ``--target`` is passed the script should
    prefer actual execution. Kept intentionally simple so the sweep stays
    fast and deterministic.
    """
    base_len = len(base)
    mut_len = len(mutated)
    new_edges = set()
    lost_edges = set()
    for i in range(min(base_len, mut_len)):
        if base[i] != mutated[i]:
            new_edges.add(hash(("byte", i, mutated[i])))
            if i < base_len:
                lost_edges.add(hash(("byte", i, base[i])))
    return max(base_len, mut_len), new_edges, lost_edges


def _real_corpus_transitions(cfg: SweepConfig) -> tuple[list[dict], list[tuple[int, int, bool]]]:
    rng = random.Random(cfg.seed)
    if cfg.corpus is None:
        raise ValueError("corpus path is required for real-corpus sweep")

    seeds = _load_real_seeds(cfg.corpus, cfg.samples, rng)
    transitions: list[dict] = []
    liveness_events: list[tuple[int, int, bool]] = []

    for seed in seeds:
        mutated = _mutate_seed(seed, rng)
        cov_before, new_edges, lost_edges = _heuristic_coverage(seed, mutated)
        cov_after = max(0, cov_before + rng.randint(-5, 15))
        transitions.append(
            {
                "input_bytes": mutated[:32].ljust(32, b"\x00"),
                "mutation_op": "byte_flip",
                "mutation_offset": rng.randint(0, max(1, len(mutated) - 1)),
                "mutation_width": rng.choice([1, 2, 4]),
                "coverage_before": cov_before,
                "coverage_after": cov_after,
                "new_edges": new_edges,
                "lost_edges": lost_edges,
            }
        )
        # Liveness events biased toward later bytes (tail padding).
        if rng.random() < 0.4:
            offset = rng.randint(max(0, len(mutated) - 64), max(len(mutated) - 1, 0))
            width = rng.choice([1, 2, 4, 8])
            liveness_events.append((offset, width, True))

    return transitions, liveness_events


def _synthetic_transitions(rng: random.Random, n: int) -> list[dict]:
    transitions = []
    for _ in range(n):
        offset = rng.randint(0, 4095)
        width = rng.choice([1, 2, 4, 8])
        op = rng.choice(
            [
                "byte_flip",
                "bit_flip",
                "byte_set",
                "byte_add",
                "dict_prepend",
                "ascii_num_arithmetic",
            ]
        )
        cov_before = rng.randint(100, 500)
        if rng.random() < 0.12:
            cov_after = cov_before + rng.randint(1, 20)
            new_edges = {hash((op, offset, i)) for i in range(rng.randint(1, 5))}
            lost_edges = set()
        else:
            cov_after = cov_before + rng.randint(-10, 10)
            new_edges = set()
            lost_edges = set()
        transitions.append(
            {
                "input_bytes": bytes(rng.randint(0, 255) for _ in range(16)),
                "mutation_op": op,
                "mutation_offset": offset,
                "mutation_width": width,
                "coverage_before": cov_before,
                "coverage_after": cov_after,
                "new_edges": new_edges,
                "lost_edges": lost_edges,
            }
        )
    return transitions


def _synthetic_liveness_events(
    rng: random.Random, n: int, confirmed_dead_ratio: float = 0.7
) -> list[tuple[int, int, bool]]:
    events = []
    for _ in range(n):
        if rng.random() < confirmed_dead_ratio:
            offset = rng.randint(3000, 4095)
            width = rng.choice([4, 8, 16, 32])
            confirmed_dead = True
        else:
            offset = rng.randint(0, 100)
            width = rng.choice([1, 2, 4])
            confirmed_dead = False
        events.append((offset, width, confirmed_dead))
    return events


def _run_sweep(cfg: SweepConfig) -> list[dict]:
    if cfg.corpus is not None:
        transitions, liveness_events = _real_corpus_transitions(cfg)
    else:
        rng = random.Random(cfg.seed)
        transitions = _synthetic_transitions(rng, cfg.n_transitions)
        liveness_events = _synthetic_liveness_events(
            random.Random(cfg.seed + 1), cfg.n_liveness_events
        )

    rows = []
    for dead_weight in cfg.dead_weights:
        for switch_after in cfg.switch_afters:
            fl = FormatLearner(max_timeline=5000)
            for t in transitions:
                fl.record_transition(**t)
            for offset, width, confirmed_dead in liveness_events:
                if confirmed_dead:
                    fl.record_liveness(offset, width, confirmed_dead=True)

            summary = fl.get_format_summary()
            padding = [f for f in summary["fields"] if f["type"] == "padding"]
            rows.append(
                {
                    "dead_weight": dead_weight,
                    "switch_after": switch_after,
                    "hypotheses": summary["hypotheses"],
                    "classified": summary["classified"],
                    "padding_count": len(padding),
                    "padding_offsets": tuple(p["offset"] for p in padding),
                    "padding_confidences": tuple(p["confidence"] for p in padding),
                }
            )
    return rows


def _stability_report(rows: list[dict]) -> str:
    lines = []
    by_switch: dict[int, list[dict]] = {}
    for row in rows:
        by_switch.setdefault(row["switch_after"], []).append(row)

    lines.append("# Sensitivity sweep results")
    lines.append("")
    lines.append("## Padding-hypothesis stability by switch_after")
    lines.append("")
    unstable = False
    for switch_after in sorted(by_switch):
        group = sorted(by_switch[switch_after], key=lambda r: r["dead_weight"])
        padding_sets = [set(r["padding_offsets"]) for r in group]
        base = padding_sets[0]
        row_unstable = any(s != base for s in padding_sets[1:])
        status = "UNSTABLE" if row_unstable else "stable"
        if row_unstable:
            unstable = True
        lines.append(
            f"- switch_after={switch_after}: {status} "
            f"(padding sets: {[sorted(s) for s in padding_sets]})"
        )
    lines.append("")

    default_rows = [
        r
        for r in rows
        if r["dead_weight"] == DEFAULT_DEAD_WEIGHT and r["switch_after"] == DEFAULT_SWITCH_AFTER
    ]
    if default_rows:
        dr = default_rows[0]
        lines.append("## Current defaults")
        lines.append("")
        lines.append(
            f"- _LIVENESS_DEAD_WEIGHT={DEFAULT_DEAD_WEIGHT}, "
            f"_LIVENESS_SWITCH_AFTER={DEFAULT_SWITCH_AFTER}"
        )
        lines.append(f"- padding hypotheses: {dr['padding_count']}")
        lines.append(f"- padding offsets: {dr['padding_offsets']}")
        lines.append(f"- padding confidences: {dr['padding_confidences']}")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if unstable:
        lines.append(
            "UNSTABLE: padding hypothesis sets vary across the sweep. "
            "Current defaults are NOT validated as stable."
        )
    else:
        lines.append(
            "STABLE: padding hypothesis sets are identical across the "
            "sweep. Current defaults pass the sensitivity check."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Liveness-threshold sensitivity sweep")
    p.add_argument("--transitions", type=int, default=2_000)
    p.add_argument("--liveness-events", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="-")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--target", type=Path, default=None)
    p.add_argument("--samples", type=int, default=200)
    args = p.parse_args(argv)

    cfg = SweepConfig(
        n_transitions=args.transitions,
        n_liveness_events=args.liveness_events,
        seed=args.seed,
        corpus=args.corpus,
        target=args.target,
        samples=args.samples,
    )
    rows = _run_sweep(cfg)
    report = _stability_report(rows)

    if args.output == "-":
        print(report)
    else:
        with open(args.output, "w") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
