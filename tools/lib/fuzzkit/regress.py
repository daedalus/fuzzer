"""Past fuzzer-tool bugs regenerated with random parameters.

Each mutator rebuilds the input class of one fixed bug and checks an
oracle that the bug violated, so a regression in any variant, not just the
pinned unit-test case, is caught.

| mutator      | bug (regression test)                                         |
|--------------|---------------------------------------------------------------|
| none_offset  | backtest TypeError on delocalised ops                         |
|              |   (test_regression_format_learner_none_offset.py)             |
| mg_flood     | unbounded signature counters under a prefix flood             |
|              |   (test_regression_format_learner_misra_gries.py)             |
| wide_length  | int->str crash on multi-KB `length` fields                    |
|              |   (_length_candidates; docs/TODO.md 2026-09-24)               |
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fuzzer_tool.core.analyzers import analyzer_format_learner as afl
from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    FormatLearner,
    TimelineEntry,
)
from fuzzer_tool.core.format_seed_generator import generate_seeds

from .corpus import err

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fuzzer_tool.core.rand_pool import RandPool

# Mirror of `_predict_effect`'s confidence gate; the oracle re-derives it.
PREDICT_CONFIDENCE = 0.3
MAX_HYPOTHESES = 6
MAX_ENTRIES = 64
FLOOD_FACTOR = 4
SIG_BYTES = 2
WIDE_MIN = 1 << 10
WIDE_MAX = 1 << 16
SEED_LEN_MAX = 512
_OPS = ("bit_flip", "splice", "havoc", "dict_insert")


def _flood_cap() -> int:
    """Signature-counter bound the flood oracle enforces."""
    return afl.MAX_TRACKED_SIGNATURES


def _rand_cluster(rng: RandPool) -> FormatCluster:
    """Cluster with random confident/unconfident hypotheses."""
    c = FormatCluster(signature="regress")
    for _ in range(rng.randint(1, MAX_HYPOTHESES)):
        c._add_hypothesis(
            FieldHypothesis(
                offset=rng.randint(0, 64),
                width=rng.randint(1, 16),
                field_type="data",
                confidence=rng.random(),
                sensitive_ops={rng.choice(_OPS): 1},
            )
        )
    return c


def _rand_entry(rng: RandPool) -> TimelineEntry:
    before = rng.randint(0, 100)
    return TimelineEntry(
        input_hash="0" * 16,
        mutation_op=rng.choice(_OPS),
        mutation_offset=None if rng.random() < 0.5 else rng.randint(0, 80),
        mutation_width=rng.randint(1, 8),
        coverage_before=before,
        coverage_after=before + rng.randint(-2, 2),
        new_edges=set(),
        lost_edges=set(),
    )


def _reg_none_offset(rng: RandPool, _workdir: Path) -> str | None:
    """Metamorphic: a None offset predicts from op sensitivity alone."""
    c = _rand_cluster(rng)
    c.timeline.extend(_rand_entry(rng) for _ in range(rng.randint(1, MAX_ENTRIES)))

    for entry in c.timeline:
        if entry.mutation_offset is not None:
            continue

        by_op = any(
            entry.mutation_op in h.sensitive_ops and h.confidence > PREDICT_CONFIDENCE
            for h in c.hypotheses
        )
        got = c._predict_effect(entry)
        if got is not (True if by_op else None):
            return f"None offset predicted {got}, op-only says {by_op}"

    c.backtest()
    return None


def _reg_mg_flood(rng: RandPool, _workdir: Path) -> str | None:
    """Bounded memory under a flood of distinct format signatures."""
    fl = FormatLearner()
    cap = _flood_cap()
    for _ in range(FLOOD_FACTOR * afl.MAX_TRACKED_SIGNATURES):
        data = rng.randbytes(SIG_BYTES) + b"payload"
        fl.record_transition(data, "op", 0, 1, 0, rng.randint(0, 1), set(), set())

        if len(fl._signature_counts) > cap:
            return f"{len(fl._signature_counts)} tracked signatures > cap {cap}"
        if not set(fl._pending) <= set(fl._signature_counts):
            return "pending buffer for an untracked signature"
        if any(len(v) > fl.promote_threshold for v in fl._pending.values()):
            return "pending buffer past promote_threshold"
    return None


def _reg_wide_length(rng: RandPool, _workdir: Path) -> str | None:
    """Seed generation survives a `length` field kilobytes wide."""
    base = rng.randbytes(rng.randint(1, SEED_LEN_MAX))
    field = {
        "offset": rng.randint(0, len(base)),
        "width": rng.randint(WIDE_MIN, WIDE_MAX),
        "type": "length",
        "confidence": 1.0,
        "observations": 50,
        "controlled_edges": 10,
    }
    for seed in generate_seeds([field], base, n_seeds=8, rng=rng):
        if not isinstance(seed.data, bytes):
            return f"seed is {type(seed.data).__name__}"
    return None


REGRESSION_MUTATORS: dict[str, Callable[[RandPool, Path], str | None]] = {
    "none_offset": _reg_none_offset,
    "mg_flood": _reg_mg_flood,
    "wide_length": _reg_wide_length,
}


def run_regression(name: str, rng: RandPool, workdir: Path) -> str | None:
    """Failure detail for one regenerated case, or None."""
    try:
        return REGRESSION_MUTATORS[name](rng, workdir)
    except Exception as exc:
        return err(exc)
