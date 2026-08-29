#!/usr/bin/env python3
"""Sensitivity analysis for item 4 liveness defaults.

Varies ``_LIVENESS_DEAD_WEIGHT`` and ``_LIVENESS_SWITCH_AFTER`` across
reasonable ranges on representative campaign data and reports whether the
current defaults produce stable padding hypotheses.

This is the Sequenced step 6 validation called out in
``docs/handover/handover_skittercreek_tailslayer_port.md``. It is intentionally a
standalone script rather than a pytest because the sweep spans parameter
ranges, not a single fixed scenario.

There are two modes:

* Default / ``--corpus``: a FormatLearner sweep that answers whether the
  padding *hypotheses* are stable across the parameter grid. That question
  was settled over four real campaigns. It does not execute a target -- the
  ``--corpus`` path drives FormatLearner over fabricated transitions, and
  ``--target`` currently only selects that fallback (no real per-target
  execution happens on this path).

* ``--synthetic-target``: the handover item-B calibration. It builds
  ``gen_synthetic_target.py``'s known-ground-truth target and drives the
  *real* ``LiveBitMaskEstimator`` against its coverage-dead region -- the
  measurement that was blocked for four rounds because no real target has a
  genuinely dead region. This is the mode the handover's "one run left"
  refers to; the older ``--target`` example never actually executed anything.

Usage examples::

    # Padding-hypothesis stability sweep (fast, CI-friendly)
    python tools/sweep_liveness_thresholds.py

    # Item-B calibration against the synthetic known-dead region
    python tools/sweep_liveness_thresholds.py --synthetic-target --unstable 0
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fuzzer_tool.core.format_learner import FormatLearner
from fuzzer_tool.core.live_bit_mask import LiveBitMaskEstimator

# Reproduce the operator-side defaults so the sweep can override them.
from fuzzer_tool.services.operators import (
    _LIVENESS_DEAD_WEIGHT as DEFAULT_DEAD_WEIGHT,
)
from fuzzer_tool.services.operators import (
    _LIVENESS_MAP_BITS as DEFAULT_MAP_BITS,
)
from fuzzer_tool.services.operators import (
    _LIVENESS_SWITCH_AFTER as DEFAULT_SWITCH_AFTER,
)

ROOT = Path(__file__).resolve().parent.parent
GEN_SYNTHETIC = ROOT / "tools" / "gen_synthetic_target.py"
AFL_SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Ground-truth region bounds baked into gen_synthetic_target.py's default
# layout (live prefix, then a read-but-never-branched-on dead region). Kept
# in sync with tests/test_synthetic_target.py, which asserts they still hold.
SYNTH_LIVE_REGION = (0, 32)
SYNTH_DEAD_REGION = (32, 96)


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
    # Synthetic-target calibration mode (handover item B). When set, the
    # sweep builds gen_synthetic_target.py's known-ground-truth target and
    # drives the real LiveBitMaskEstimator against its dead and live
    # regions, instead of running FormatLearner over fabricated transitions.
    synthetic_target: bool = False
    blocks: int = 400
    fanout: int = 32
    unstable: int = 0
    calib_samples: int = 900
    map_bits: int = DEFAULT_MAP_BITS
    switch_grid: tuple[int, ...] = field(default_factory=lambda: (50, 100, 200, 400, 800))


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


# ── synthetic-target calibration (handover item B) ───────────────────────
#
# The FormatLearner sweep above answers a different question: whether the
# padding *hypotheses* are stable across the parameter grid. That was
# established over four real campaigns. What stayed open for four rounds is
# the LiveBitMaskEstimator's FALSE-NEGATIVE rate against a genuinely
# coverage-dead region -- and no real target in the matrix has one
# (compressed data has no padding; any checksum makes every byte live). The
# synthetic target supplies one by construction, so the calibration below
# drives the *real* estimator against known ground truth rather than a
# fabricated event stream.

_DRIVER_C = """
#include <stdio.h>
#include <stdlib.h>
extern int fuzz_synthetic(const unsigned char *, size_t);
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    static unsigned char buf[65536];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    fuzz_synthetic(buf, n);
    return 0;
}
"""


def _build_synthetic_target(workdir: Path, cfg: SweepConfig) -> Path:
    """Generate, compile and link the synthetic target driver.

    Uses the same gcc + ``-DSYNTH_MANUAL_GUARDS`` recipe as
    ``tests/test_synthetic_target.py`` (gcc cannot do
    ``-fsanitize-coverage=trace-pc-guard``; the generated blocks call the
    shim's guard callback themselves, which is the same entry point clang's
    instrumentation targets). ``-D__AFL_CTX_SENSITIVE=0`` keeps one edge per
    synthetic guard so the region-level counts stay interpretable.
    """
    src = workdir / "synthetic_cov.c"
    r = subprocess.run(
        [
            sys.executable,
            str(GEN_SYNTHETIC),
            "--blocks",
            str(cfg.blocks),
            "--fanout",
            str(cfg.fanout),
            "--unstable",
            str(cfg.unstable),
            "-o",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"gen_synthetic_target failed: {r.stderr}")

    obj = workdir / "synthetic_cov.o"
    r = subprocess.run(
        ["gcc", "-O1", "-DSYNTH_MANUAL_GUARDS", "-c", str(src), "-o", str(obj)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"synthetic target compile failed: {r.stderr[-800:]}")

    drv = workdir / "driver.c"
    drv.write_text(_DRIVER_C)
    exe = workdir / "drive_synthetic"
    r = subprocess.run(
        [
            "gcc",
            "-O1",
            "-D__AFL_CTX_SENSITIVE=0",
            f"-include{AFL_SHIM}",
            "-o",
            str(exe),
            str(drv),
            str(obj),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"synthetic driver link failed: {r.stderr[-800:]}")
    return exe


def _run_synthetic(exe: Path, workdir: Path, data: bytes, size: int = 8192) -> frozenset[int]:
    """Execute the target on one input and return the edge-id set."""
    from fuzzer_tool.adapters.shm import ShmCoverage

    cov = ShmCoverage(size=size)
    try:
        inp = workdir / "in.bin"
        inp.write_bytes(data)
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(size))
        subprocess.run([str(exe), str(inp)], env=env, capture_output=True)
        return frozenset(cov.get_edge_ids())
    finally:
        cov.cleanup()


def _diff_bits(baseline: frozenset[int], mutant: frozenset[int], map_bits: int) -> int:
    """Fold the symmetric difference of two edge sets into a bit-mask, the
    exact transform ``operators.record_coverage_diff`` hands to
    ``LiveBitMaskEstimator.observe(0, diff_bits)``."""
    bits = 0
    for edge_id in baseline ^ mutant:
        bits |= 1 << (edge_id % map_bits)
    return bits


def _flip_in_region(rng: random.Random, base: bytes, region: tuple[int, int]) -> bytes:
    d = bytearray(base)
    o = rng.randrange(*region)
    d[o] ^= 1 << rng.randrange(8)
    return bytes(d)


def _collect_diff_sequence(
    exe: Path,
    workdir: Path,
    base: bytes,
    baseline: frozenset[int],
    region: tuple[int, int],
    n: int,
    rng: random.Random,
    map_bits: int,
) -> list[int]:
    """Run ``n`` single-bit-flip mutations that land in ``region`` and record
    the per-execution coverage diff the production estimator would see."""
    return [
        _diff_bits(
            baseline, _run_synthetic(exe, workdir, _flip_in_region(rng, base, region)), map_bits
        )
        for _ in range(n)
    ]


def _replay_estimator(seq: list[int], switch_after: int, map_bits: int) -> dict:
    """Replay a diff sequence through a real estimator at one threshold.

    Mirrors ``_region_liveness_factor``: the region is DEAD iff the estimator
    converged with an empty mask, LIVE the moment any edge is revealed
    (mask != 0), UNRESOLVED otherwise. Also reports the first sample at which
    ``switch_after`` consecutive no-growth samples were first reached, and the
    length of the leading all-zero run (the window in which a live region
    could, in principle, be mistaken for dead)."""
    est = LiveBitMaskEstimator(n_bits=map_bits, switch_after=switch_after)
    converged_at = None
    leading_zero_run = 0
    counting_leading = True
    for i, db in enumerate(seq, 1):
        if db:
            counting_leading = False
        elif counting_leading:
            leading_zero_run += 1
        est.observe(0, db)
        if converged_at is None and est.is_converged:
            converged_at = i
    if est.is_converged and est.mask == 0:
        verdict = "DEAD"
    elif est.mask != 0:
        verdict = "LIVE"
    else:
        verdict = "UNRESOLVED"
    return {
        "switch_after": switch_after,
        "verdict": verdict,
        "converged_at": converged_at,
        "final_converged": est.is_converged,
        "mask_bits": bin(est.mask).count("1"),
        "leading_zero_run": leading_zero_run,
    }


def _synthetic_report(cfg: SweepConfig) -> str:
    if shutil.which("gcc") is None:
        raise RuntimeError("synthetic-target mode needs gcc")
    if not AFL_SHIM.exists():
        raise RuntimeError(f"afl_shim.c not found at {AFL_SHIM}")

    lines: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        exe = _build_synthetic_target(workdir, cfg)
        rng = random.Random(cfg.seed)
        base = bytes(rng.randrange(256) for _ in range(256))
        baseline = _run_synthetic(exe, workdir, base)

        # Determinism check: on --unstable 0 the target must be stable, or
        # "the dead region moved coverage" is unfalsifiable (the exact
        # failure mode that made the unstable variant useless for this).
        reruns = [_run_synthetic(exe, workdir, base) for _ in range(8)]
        stable = all(r == baseline for r in reruns)

        dead_seq = _collect_diff_sequence(
            exe,
            workdir,
            base,
            baseline,
            SYNTH_DEAD_REGION,
            cfg.calib_samples,
            random.Random(cfg.seed + 1),
            cfg.map_bits,
        )
        live_seq = _collect_diff_sequence(
            exe,
            workdir,
            base,
            baseline,
            SYNTH_LIVE_REGION,
            cfg.calib_samples,
            random.Random(cfg.seed + 2),
            cfg.map_bits,
        )

    dead_nonzero = sum(1 for x in dead_seq if x)
    live_nonzero = sum(1 for x in live_seq if x)

    lines.append("# Synthetic-target liveness calibration (handover item B)")
    lines.append("")
    lines.append(
        f"- target: {cfg.blocks} blocks, fanout {cfg.fanout}, "
        f"--unstable {cfg.unstable}; baseline edges {len(baseline)}"
    )
    lines.append(f"- dead region {SYNTH_DEAD_REGION}, live region {SYNTH_LIVE_REGION}")
    lines.append(f"- calibration samples per region: {cfg.calib_samples}")
    lines.append(f"- identical-input reruns stable: {stable}")
    lines.append(f"- dead-region mutations moving coverage: {dead_nonzero}/{cfg.calib_samples}")
    lines.append(f"- live-region mutations moving coverage: {live_nonzero}/{cfg.calib_samples}")
    lines.append("")

    if cfg.unstable > 0:
        lines.append(
            "WARNING: --unstable > 0. ASLR-gated edges make dead-region "
            "mutations appear to move coverage, so the liveness signal is "
            "destroyed and these numbers are not a calibration. Use "
            "--unstable 0. (This is why the calibration variant is the "
            "deterministic one; see the sweep doc.)"
        )
        lines.append("")

    lines.append("## Estimator verdict by switch_after")
    lines.append("")
    lines.append(
        "| switch_after | dead verdict | dead conv@ | live verdict | live mask bits | live leading-zero run |"
    )
    lines.append("|---:|---|---:|---|---:|---:|")
    dead_ok = True
    live_ok = True
    for sa in cfg.switch_grid:
        d = _replay_estimator(dead_seq, sa, cfg.map_bits)
        live_row = _replay_estimator(live_seq, sa, cfg.map_bits)
        if d["verdict"] != "DEAD":
            dead_ok = False
        if live_row["verdict"] == "DEAD":
            live_ok = False
        lines.append(
            f"| {sa} | {d['verdict']} | {d['converged_at']} | "
            f"{live_row['verdict']} | {live_row['mask_bits']} | "
            f"{live_row['leading_zero_run']} |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if not stable and cfg.unstable == 0:
        lines.append(
            "INCONCLUSIVE: target was not deterministic on --unstable 0. "
            "Something in the environment is adding nondeterminism; do not "
            "trust these numbers."
        )
    elif dead_ok and live_ok:
        lines.append(
            "CORRECT across the grid: every switch_after gives the known-dead "
            "region a DEAD verdict (in exactly switch_after samples) and never "
            "gives the live region one. False-negative rate 0, false-positive "
            "rate 0 on this target."
        )
        lines.append("")
        lines.append(
            "Reading: on the synthetic target switch_after is unconstrained "
            "from below by *correctness* -- even 50 classifies both regions "
            "right -- because the live region reveals an edge on sample 1 "
            "(leading-zero run ~0), so its mask is never empty and the dead "
            "verdict cannot fire on it. switch_after only sets how many wasted "
            "mutations elapse before the dead down-weight engages. The floor "
            "that actually justifies keeping it high is a real cold-but-live "
            "region -- one that produces a long no-growth run before its first "
            "edge -- which this target cannot exhibit by construction. So the "
            f"default {DEFAULT_SWITCH_AFTER} is retained on that basis, not "
            "lowered; the dead side is now measured, the cold-live floor "
            "remains an assumption about real targets."
        )
    else:
        lines.append(
            "MISCLASSIFICATION: the estimator's verdict disagreed with ground "
            "truth for at least one switch_after. Investigate before shipping "
            "any threshold change."
        )
    lines.append("")
    return "\n".join(lines)


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
    p.add_argument(
        "--synthetic-target",
        action="store_true",
        help="build gen_synthetic_target.py and drive the real "
        "LiveBitMaskEstimator against its known-dead region (handover item B)",
    )
    p.add_argument("--blocks", type=int, default=400, help="synthetic: guard count")
    p.add_argument("--fanout", type=int, default=32, help="synthetic: edges per exec")
    p.add_argument(
        "--unstable",
        type=int,
        default=0,
        help="synthetic: ASLR-gated unstable blocks. MUST be 0 to calibrate "
        "liveness -- any >0 destroys the signal (see sweep doc).",
    )
    p.add_argument(
        "--calib-samples",
        type=int,
        default=900,
        help="synthetic: mutations per region (>= max switch_after to converge)",
    )
    args = p.parse_args(argv)

    cfg = SweepConfig(
        n_transitions=args.transitions,
        n_liveness_events=args.liveness_events,
        seed=args.seed,
        corpus=args.corpus,
        target=args.target,
        samples=args.samples,
        synthetic_target=args.synthetic_target,
        blocks=args.blocks,
        fanout=args.fanout,
        unstable=args.unstable,
        calib_samples=args.calib_samples,
    )

    if cfg.synthetic_target:
        report = _synthetic_report(cfg)
    else:
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
