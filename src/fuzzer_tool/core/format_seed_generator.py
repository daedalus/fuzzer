"""Seed generator driven by FormatLearner's inferred format fields.

``FormatLearner`` (see ``core/analyzers/analyzer_format_learner.py``) turns
mutation/coverage observations into a table of field hypotheses:

    Offset  Width  Type  Conf  Obs  Edges  Sensitive ops

This module turns that table back into *seeds*: instead of mutating blindly,
it targets each classified field with a strategy suited to its inferred
type, so new seeds exercise the boundaries the model says actually move
coverage rather than wasting budget on bytes already known to be dead
(``padding``) or ones that must stay fixed to keep the input parseable
(``magic``).

Field-type strategy:

  magic    — never touched (correlated with parser acceptance; corrupting
             it wholesale tends to just hit an early bail-out).
  length   — swept through width-appropriate boundary values (0, 1,
             max-for-width, current±1, and a value derived from the seed's
             own size) in both endiannesses, since the model doesn't know
             which one the format uses.
  crc      — cheap, algorithm-agnostic stress values (zero, all-ones,
             single-bit-flip) since the learner doesn't know the checksum
             polynomial and can't recompute a valid one.
  data     — reuses whichever mutation operators the learner already
             observed moving coverage at that offset (``sensitive_ops``),
             approximated locally so this module stays dependency-free of
             the live ``OperatorEngine``; falls back to the same
             AFL-style INTERESTING tables used elsewhere in the fuzzer.
  unknown  — same fallback as ``data``, but only the byte-granularity
             subset (less evidence to justify wider edits).
  padding  — skipped by default (the model already says it's inert).

Fields are visited in descending order of
``confidence * (1 + controlled_edges) * log1p(observations)`` — the same
"how much do we trust this, and how much does it matter" signal
``suggest_discriminating_mutation`` uses — so a capped seed budget spends
itself on the fields most likely to matter first.

Stdlib-only, no dependency on a live Fuzzer/OperatorEngine instance, so it
works equally from inside a running fuzzer (fed a live ``FormatLearner``)
and offline (fed a JSON dump of ``FormatLearner.get_state()``); see
``tools/gen_format_seeds.py`` for the offline CLI.

Two entry points cover the two situations a caller is in:

  cold_start_seed()   — no base seed to mutate yet (fresh corpus, or the
                         live fuzzer's cold-start path in
                         ``SeedPicker._format_aware_seed``). Builds one
                         seed from nothing, filling each confident field
                         with its ``most_common_value`` (the per-position
                         byte histogram ``FormatLearner._track_values``
                         maintains), occasionally swapped for a
                         type-specific stress default. Originally lived
                         as ``SeedPicker._format_learner_seed``; moved
                         here so both call sites share one implementation.
  generate_seeds()     — there *is* a base seed (a real corpus member, or
                         cold_start_seed()'s own output), and the goal is
                         many field-targeted variants of it: boundary
                         sweeps for length fields, stress fills for crc,
                         replayed sensitive_ops for data/unknown.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import field as dc_field

from fuzzer_tool.core.mutations.generic import (
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
)

# Field types that are deliberately never targeted for direct mutation.
_SKIPPED_TYPES = frozenset({"magic", "padding"})

# Cheap, algorithm-agnostic stress values for crc-classified fields: we
# don't know the polynomial, so recomputing a "valid" checksum isn't
# possible, but a checksum field is exactly the byte range most parsers
# either verify strictly (early reject — interesting) or skip entirely
# (silent corruption downstream — also interesting).
_CRC_STRESS = (0x00, 0xFF, 0x01, 0x80)


# ── Cold-start seed (previously SeedPicker._format_learner_seed) ───────────
#
# Confidence bar a field must clear before cold_start_seed() will trust its
# value enough to emit it. Same threshold the live picker used.
SEED_FIELD_CONFIDENCE = 0.5

# Occasional type-specific stress default to swap in instead of a field's
# most_common_value (30% of the time, per field) — keeps the cold-start
# seed from being *purely* a replay of whatever was already observed.
SEED_TYPE_DEFAULTS: dict[str, list[int]] = {
    "magic": [],  # never overridden: use the learned value or nothing
    "length": [0, 1, 255, 256, 65535],
    "crc": [0],
    "flags": [0, 0xFF, 1],
    "padding": [0],
    "data": [],  # use learned value or leave zero
    "unknown": [],  # use learned value or leave zero
}


# Local approximations of the operators FormatLearner reports as
# "sensitive_ops" for a field, keyed by the same op names services/operators.py
# registers under. Each takes (buf: bytearray, offset: int, width: int,
# rng: random.Random) and mutates buf in place. Only ops cheap and safe to
# approximate without the live OperatorEngine are covered; anything else
# falls back to _fallback_byte_ops.
def _op_bit_flip(buf: bytearray, offset: int, width: int, rng: random.Random) -> None:
    bit = rng.randrange(8 * max(width, 1))
    idx = offset + bit // 8
    if 0 <= idx < len(buf):
        buf[idx] ^= 1 << (bit % 8)


def _op_byte_flip(buf: bytearray, offset: int, width: int, rng: random.Random) -> None:
    for i in range(offset, min(offset + width, len(buf))):
        buf[i] ^= 0xFF


def _op_xor_byte(buf: bytearray, offset: int, width: int, rng: random.Random) -> None:
    if 0 <= offset < len(buf):
        buf[offset] ^= rng.randrange(1, 256)


def _op_arith(buf: bytearray, offset: int, width: int, rng: random.Random) -> None:
    if 0 <= offset < len(buf):
        delta = rng.choice([-35, -8, -1, 1, 8, 35])
        buf[offset] = (buf[offset] + delta) & 0xFF


_SENSITIVE_OP_TABLE = {
    "bit_flip": _op_bit_flip,
    "byte_flip": _op_byte_flip,
    "byte_flip_2": _op_byte_flip,
    "byte_flip_4": _op_byte_flip,
    "xor_byte": _op_xor_byte,
    "havoc_arith": _op_arith,
    "arith8": _op_arith,
}


def _fallback_byte_ops(buf: bytearray, offset: int, width: int, rng: random.Random) -> None:
    """Generic byte-granularity stress when no sensitive-op mapping applies."""
    n = min(width, len(buf) - offset)
    if n <= 0:
        return
    table = INTERESTING_8
    val = rng.choice(table) & 0xFF
    for i in range(n):
        buf[offset + i] = val if i == 0 else buf[offset + i]


@dataclass
class GeneratedSeed:
    """One synthesized seed plus provenance for naming/debugging."""

    data: bytes
    offset: int
    width: int
    field_type: str
    strategy: str
    score: float = 0.0

    def label(self) -> str:
        return f"off{self.offset}_w{self.width}_{self.field_type}_{self.strategy}"


@dataclass
class _FieldSpec:
    offset: int
    width: int
    field_type: str
    confidence: float
    observations: int
    controlled_edges: int
    sensitive_ops: dict = dc_field(default_factory=dict)
    # Most frequent byte value seen across this field's positions, per
    # FormatLearner's value_counts histogram — None if never tracked.
    most_common_value: int | None = None

    def score(self) -> float:
        return (
            max(self.confidence, 0.0)
            * (1 + self.controlled_edges)
            * math.log1p(max(self.observations, 0))
        )


def _most_common_value_from_counts(value_counts: dict) -> int | None:
    """Aggregate a FieldHypothesis.value_counts histogram (relative
    position -> byte value -> count) into a single most-frequent byte,
    the same reduction FormatLearner.get_format_summary() performs."""
    if not value_counts:
        return None
    totals: dict[int, int] = {}
    for pos_counts in value_counts.values():
        for byte_val, count in pos_counts.items():
            totals[byte_val] = totals.get(byte_val, 0) + count
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _coerce_fields(fields) -> list[_FieldSpec]:
    """Accept FieldHypothesis objects, get_format_summary() dicts, or
    get_state()['hypotheses'] dicts — whichever the caller has handy."""
    specs = []
    for f in fields:
        if isinstance(f, dict):
            edges = f.get("controlled_edges", 0)
            edges = len(edges) if isinstance(edges, list | set) else int(edges)
            mcv = f.get("most_common_value")
            if mcv is None and "value_counts" in f:
                mcv = _most_common_value_from_counts(f["value_counts"])
            specs.append(
                _FieldSpec(
                    offset=f["offset"],
                    width=f["width"],
                    field_type=f.get("type") or f.get("field_type") or "unknown",
                    confidence=float(f.get("confidence", 0.0)),
                    observations=int(f.get("observations", 0)),
                    controlled_edges=edges,
                    sensitive_ops=dict(f.get("sensitive_ops", {})),
                    most_common_value=mcv,
                )
            )
        else:
            specs.append(
                _FieldSpec(
                    offset=f.offset,
                    width=f.width,
                    field_type=f.field_type,
                    confidence=f.confidence,
                    observations=f.observations,
                    controlled_edges=len(f.controlled_edges),
                    sensitive_ops=dict(f.sensitive_ops),
                    most_common_value=_most_common_value_from_counts(
                        getattr(f, "value_counts", None)
                    ),
                )
            )
    return specs


def cold_start_seed(
    fields,
    max_len: int = 0,
    rng: random.Random | None = None,
    confidence_threshold: float = SEED_FIELD_CONFIDENCE,
) -> bytes | None:
    """Build one seed from nothing, for when there's no base seed to mutate.

    ``fields`` may be ``FormatLearner.hypotheses``, the ``fields`` list
    from ``get_format_summary()``, or ``get_state()["hypotheses"]``.

    Only fields with ``confidence >= confidence_threshold`` and a recorded
    ``most_common_value`` are trusted; everything else is left at zero.
    Each trusted field is filled with its most-common byte across its
    width, with a 30% per-field chance of swapping in a type-specific
    stress default (``SEED_TYPE_DEFAULTS``) instead — the same behavior
    previously implemented as ``SeedPicker._format_learner_seed``.

    Returns ``None`` when no field clears the confidence bar (nothing
    reliable enough to build a seed from).
    """
    specs = sorted(_coerce_fields(fields), key=lambda s: s.offset)
    learned = [
        s
        for s in specs
        if s.confidence >= confidence_threshold and s.most_common_value is not None
    ]
    if not learned:
        return None

    rng = rng or random.Random()
    seed_len = max(s.offset + s.width for s in learned)
    if max_len > 0:
        seed_len = min(seed_len, max_len)

    seed = bytearray(seed_len)
    for spec in learned:
        if spec.offset >= seed_len:
            continue
        defaults = SEED_TYPE_DEFAULTS.get(spec.field_type, [])
        value_byte = spec.most_common_value
        if defaults and rng.random() < 0.3:
            value_byte = rng.choice(defaults) & 0xFF
        end = min(spec.offset + spec.width, seed_len)
        for i in range(spec.offset, end):
            seed[i] = value_byte & 0xFF
    return bytes(seed)


def _length_candidates(
    spec: _FieldSpec, seed_len: int, rng: random.Random
) -> list[tuple[bytes, str]]:
    """Boundary-ish values for a length-classified field, width bytes wide,
    tried in both endiannesses since the model doesn't record which one
    the format actually uses."""
    w = max(spec.width, 1)
    max_val = (1 << (8 * w)) - 1
    payload_after = max(seed_len - (spec.offset + w), 0)
    raw_values = {
        0,
        1,
        max_val,
        max_val // 2,
        payload_after,
        max(payload_after - 1, 0),
        min(payload_after + 1, max_val),
        min(payload_after * 2, max_val),
    }
    if w == 1:
        raw_values.update(v & 0xFF for v in INTERESTING_8)
    elif w == 2:
        raw_values.update(v & 0xFFFF for v in INTERESTING_16)
    elif w >= 4:
        raw_values.update(v & 0xFFFFFFFF for v in INTERESTING_32)

    out = []
    for v in sorted(raw_values):
        v &= max_val
        out.append((v.to_bytes(w, "little"), f"len_le_{v}"))
        if w > 1:
            out.append((v.to_bytes(w, "big"), f"len_be_{v}"))
    return out


def _crc_candidates(spec: _FieldSpec) -> list[tuple[bytes, str]]:
    w = max(spec.width, 1)
    out = []
    for stress in _CRC_STRESS:
        out.append((bytes([stress]) * w, f"crc_fill_{stress:#04x}"))
    return out


class FormatSeedGenerator:
    """Synthesizes candidate seeds from a FormatLearner's field hypotheses."""

    def __init__(self, fields, rng: random.Random | None = None):
        """``fields`` may be ``FormatLearner.hypotheses``, the ``fields``
        list from ``get_format_summary()``, or ``get_state()["hypotheses"]``."""
        self.set_fields(fields)
        self.rng = rng or random.Random()
        # Per-generation stats for the report's format-learning section.
        self.generator_stats: dict[str, int | float] = {
            "total_seeds_generated": 0,
            "generation_attempt_count": 0,
            "successful_attempts": 0,
            "last_generated_seed_offset": -1,
            "last_generated_seed_width": 0,
            "last_generated_field_type": "",
            "last_generated_strategy": "",
            "field_confidences": [],
            "field_types_used": {},
            "strategies_used": {},
        }

    def set_fields(self, fields) -> None:
        """Replace the field table, keeping stats (the learner keeps refining it)."""
        self.fields = sorted(_coerce_fields(fields), key=lambda s: s.score(), reverse=True)

    def generate(self, base_seed: bytes, n_seeds: int = 32) -> list[GeneratedSeed]:
        """Produce up to ``n_seeds`` field-targeted variants of ``base_seed``.

        Isolates one field's mutation per output seed (matching the
        learner's own discriminating-mutation philosophy: a change whose
        effect can't be attributed to a single field is not useful
        evidence), cycling through fields by score until the budget is
        spent or every field/strategy combination has been used once.
        """
        if n_seeds <= 0 or not base_seed:
            return []

        candidates = [f for f in self.fields if f.field_type not in _SKIPPED_TYPES]
        if not candidates:
            self.generator_stats["generation_attempt_count"] += 1
            return []

        self.generator_stats["generation_attempt_count"] += 1

        # Pre-build the (bytes-patch, strategy-name) options for each field.
        per_field_options: list[tuple[_FieldSpec, list[tuple[bytes, str]]]] = []
        for spec in candidates:
            if spec.width <= 0 or spec.offset < 0 or spec.offset >= len(base_seed):
                continue
            opts = self._options_for(spec, base_seed)
            if opts:
                per_field_options.append((spec, opts))

        out: list[GeneratedSeed] = []
        cursors = [0] * len(per_field_options)
        exhausted = [False] * len(per_field_options)
        while len(out) < n_seeds and not all(exhausted):
            progressed = False
            for i, (spec, opts) in enumerate(per_field_options):
                if len(out) >= n_seeds:
                    break
                if exhausted[i]:
                    continue
                patch, strategy = opts[cursors[i]]
                cursors[i] += 1
                if cursors[i] >= len(opts):
                    exhausted[i] = True
                progressed = True

                buf = bytearray(base_seed)
                end = min(spec.offset + len(patch), len(buf))
                buf[spec.offset : end] = patch[: end - spec.offset]
                out.append(
                    GeneratedSeed(
                        data=bytes(buf),
                        offset=spec.offset,
                        width=spec.width,
                        field_type=spec.field_type,
                        strategy=strategy,
                        score=spec.score(),
                    )
                )
                # Update per-field stats
                self.generator_stats["total_seeds_generated"] += 1
                self.generator_stats["field_confidences"].append(spec.confidence)
                self.generator_stats["field_types_used"][spec.field_type] = (
                    self.generator_stats["field_types_used"].get(spec.field_type, 0) + 1
                )
                self.generator_stats["strategies_used"][strategy] = (
                    self.generator_stats["strategies_used"].get(strategy, 0) + 1
                )
                self.generator_stats["last_generated_seed_offset"] = spec.offset
                self.generator_stats["last_generated_seed_width"] = spec.width
                self.generator_stats["last_generated_field_type"] = spec.field_type
                self.generator_stats["last_generated_strategy"] = strategy
            if not progressed:
                break
        self.generator_stats["successful_attempts"] += len(out)
        return out

    def _options_for(self, spec: _FieldSpec, base_seed: bytes) -> list[tuple[bytes, str]]:
        if spec.field_type == "length":
            return _length_candidates(spec, len(base_seed), self.rng)
        if spec.field_type == "crc":
            return _crc_candidates(spec)
        # data / unknown: replay whichever real operators the learner saw
        # move coverage here, approximated locally; a few variants each so
        # a single field still yields more than one candidate seed.
        out: list[tuple[bytes, str]] = []
        ops = sorted(spec.sensitive_ops, key=spec.sensitive_ops.get, reverse=True)
        mapped = [op for op in ops if op in _SENSITIVE_OP_TABLE]
        chosen_ops = mapped[:3] if mapped else ["_fallback"]
        variants_per_op = 3 if spec.field_type == "data" else 1
        for op in chosen_ops:
            fn = _SENSITIVE_OP_TABLE.get(op, _fallback_byte_ops)
            for _ in range(variants_per_op):
                buf = bytearray(base_seed[spec.offset : spec.offset + spec.width])
                fn(buf, 0, spec.width, self.rng)
                out.append((bytes(buf), op if op != "_fallback" else "byte_stress"))
        return out


def generate_seeds(
    fields, base_seed: bytes, n_seeds: int = 32, rng: random.Random | None = None
) -> list[GeneratedSeed]:
    """Convenience wrapper: ``FormatSeedGenerator(fields, rng).generate(...)``."""
    return FormatSeedGenerator(fields, rng=rng).generate(base_seed, n_seeds)
