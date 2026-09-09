"""RO vs RD classification primitives (no wiring).

Paper (Du): reversal of objective temporal orientation (RO) is distinct from
dynamical-history reversal (RD) inside a fixed orientation. At the formula
level they share the same map R, but Π_RD = Π_RO = RΠ does not imply RD = RO.

This module only provides:

- OrientationClass enum (RO / RD / NEUTRAL)
- tagging helpers for operator names / records
- invertibility classification for common mutator families
- a pure lineage RD-reverse helper that builds a reversed operator sequence
  when every step is invertible

It does **not** register operators, change schedulers, touch lineage storage,
or call the target. Wiring is a later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrientationClass(str, Enum):
    """Sector role of a mutation relative to objective temporal orientation."""

    RO = "ro"
    """Orientation-reversing: path-negation, opposite-branch / SMT path-negation."""

    RD = "rd"
    """History-reversing inside a fixed orientation (ordinary mutators + lineage reverse)."""

    NEUTRAL = "neutral"
    """No temporal-orientation commitment (e.g. pure dictionary insert with no path intent)."""


# Operator-name substrings / exact names treated as RO-class by default.
# Kept deliberately narrow; extend when wiring path-negation / SMT entry points.
_RO_NAME_MARKERS: tuple[str, ...] = (
    "path_negat",
    "path-negat",
    "pathnegat",
    "negate_path",
    "branch_invert",
    "invert_branch",
    "opposite_branch",
    "smt_path_neg",
    "path_negation",
)

# Mutator families that are bitwise or arithmetically invertible on a fixed buffer.
_INVERTIBLE_EXACT: frozenset[str] = frozenset(
    {
        "bit_flip",
        "bit_flip_1",
        "bit_flip_2",
        "bit_flip_4",
        "bit_flip_8",
        "bit_flip_16",
        "bit_flip_32",
        "byte_flip",
        "byte_flip_1",
        "span_invert",
        "arith_8",
        "arith_16",
        "arith_32",
        "arith_64",
        "interesting_8",
        "interesting_16",
        "interesting_32",
    }
)

_INVERTIBLE_PREFIXES: tuple[str, ...] = (
    "bit_flip",
    "byte_flip",
    "arith_",
    "interesting_",
    "bit_rotate",
    "bit_shift",
)


def classify_operator_name(name: str | None) -> OrientationClass:
    """Classify an operator by its registry / display name.

    RO markers win over RD defaults. Unknown names are RD (ordinary mutators
    stay inside the orientation sector).
    """
    if not name:
        return OrientationClass.NEUTRAL
    lower = name.lower().strip()
    for marker in _RO_NAME_MARKERS:
        if marker in lower:
            return OrientationClass.RO
    return OrientationClass.RD


def is_invertible_operator(name: str | None) -> bool:
    """Return True if a single application is expected to be self-inverse or invertible.

    Used by the RD lineage reverse helper. Non-invertible steps cause reverse
    to refuse rather than invent a silent approximation.
    """
    if not name:
        return False
    lower = name.lower().strip()
    if lower in _INVERTIBLE_EXACT:
        return True
    return any(lower.startswith(p) for p in _INVERTIBLE_PREFIXES)


@dataclass(frozen=True)
class MutationStep:
    """One recorded mutation step in a lineage (minimal fields for RD reverse)."""

    operator: str
    # Opaque site / argument payload the mutator recorded (offsets, masks, …).
    site: Any = None
    # Optional pre/post buffer digests for verification; not required to reverse.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def orientation(self) -> OrientationClass:
        return classify_operator_name(self.operator)

    @property
    def invertible(self) -> bool:
        return is_invertible_operator(self.operator)


@dataclass(frozen=True)
class LineageRecord:
    """Ordered mutation steps from an ancestor seed to a derived seed."""

    steps: tuple[MutationStep, ...]

    def orientation_profile(self) -> dict[OrientationClass, int]:
        profile = {c: 0 for c in OrientationClass}
        for step in self.steps:
            profile[step.orientation] += 1
        return profile

    def all_invertible(self) -> bool:
        return all(step.invertible for step in self.steps)

    def has_ro(self) -> bool:
        return any(step.orientation is OrientationClass.RO for step in self.steps)


@dataclass(frozen=True)
class RDReverseResult:
    """Outcome of attempting an RD (history) reverse of a lineage."""

    ok: bool
    reversed_steps: tuple[MutationStep, ...] = ()
    reason: str = ""


def rd_reverse_lineage(lineage: LineageRecord) -> RDReverseResult:
    """Build the RD-reversed operator sequence (orientation held fixed).

    Paper RD: keep objective temporal orientation fixed and reverse history
    content. Here that means reverse the order of steps and keep each
    invertible operator as its own inverse (bit/byte flips and many arith
    interesting-value overwrites are self-inverse on the same site).

    If any step is non-invertible, refuse with ``ok=False`` rather than
    inventing an approximation (handover Phase 2 policy).
    """
    if not lineage.steps:
        return RDReverseResult(ok=True, reversed_steps=(), reason="empty lineage")

    if lineage.has_ro():
        return RDReverseResult(
            ok=False,
            reason="lineage contains RO-class steps; RD reverse is intra-sector only",
        )

    non_inv = [s.operator for s in lineage.steps if not s.invertible]
    if non_inv:
        return RDReverseResult(
            ok=False,
            reason=f"non-invertible operators: {', '.join(dict.fromkeys(non_inv))}",
        )

    # Self-inverse steps: reverse order, keep operator + site.
    reversed_steps = tuple(
        MutationStep(operator=s.operator, site=s.site, meta=dict(s.meta))
        for s in reversed(lineage.steps)
    )
    return RDReverseResult(ok=True, reversed_steps=reversed_steps, reason="ok")


def tag_operator_record(name: str, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a shallow copy of ``record`` with orientation class attached.

    Pure helper for later wiring; does not mutate the input dict.
    """
    out = dict(record) if record else {}
    out["orientation_class"] = classify_operator_name(name).value
    out["invertible"] = is_invertible_operator(name)
    out["operator"] = name
    return out
