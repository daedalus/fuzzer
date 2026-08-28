"""Validity as a second fitness channel (Zest, ISSTA'19).

Coverage alone cannot distinguish "reached new code" from "reached new
code inside the parser's error path". Zest asks the harness instead: an
input is *valid* when the target accepted it, *invalid* when the target
rejected it, and the two are scored separately -- coverage reached on
valid runs gets its own map, so an input that is valid and covers
something no valid input covered before is worth saving even when its
total coverage is not new. That is the signal that gets a campaign past
a syntax check and into the semantic stages behind it.

The harness convention is one exit code, chosen by the user
(``--reject-code``). Nothing is inferred without it: a target that does
not report rejection yields UNKNOWN for every execution and the channel
stays inert.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum

log = logging.getLogger(__name__)

# Seed-selection multiplier for a seed the target accepted. A boost, not a
# gate: an invalid seed is still the shortest path to some branches, so
# validity ranks the corpus and never excludes from it.
VALID_SEED_BONUS = 1.5

# Exit statuses a rejection code may take. Negative codes are signals in
# this fuzzer's runner convention (-1 timeout, -2 infrastructure, -N fatal
# signal), and 0 is how a harness reports acceptance -- either as a
# rejection code would turn crashes or successes into parser verdicts.
_MIN_REJECT_CODE = 1
_MAX_REJECT_CODE = 255


class Validity(Enum):
    """What one execution said about the input's acceptability."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class ValidityChannel:
    """Tracks parser verdicts and the coverage reached under a valid one."""

    # The valid map is a set of edge ids, so it grows with the target rather
    # than with the campaign -- but a hashed edge space has no fixed ceiling
    # and a runaway target could still fill memory. Same order as the edge
    # tracker's own caps.
    MAX_VALID_EDGES = 200_000

    def __init__(self, reject_code: int | None):
        if reject_code is not None and not (_MIN_REJECT_CODE <= reject_code <= _MAX_REJECT_CODE):
            raise ValueError(
                f"reject code must be {_MIN_REJECT_CODE}-{_MAX_REJECT_CODE}, "
                f"got {reject_code}: 0 means accepted and negatives are signals"
            )

        self.reject_code = reject_code
        self.valid_edges: set[int] = set()
        self.valid_count = 0
        self.invalid_count = 0

    @property
    def enabled(self) -> bool:
        return self.reject_code is not None

    @property
    def valid_rate(self) -> float:
        """Share of classified runs the target accepted.

        Denominator excludes UNKNOWN: a crash is not a verdict on the
        input's syntax, and counting it as invalid would make the rate
        track crash frequency instead of parser acceptance.
        """
        total = self.valid_count + self.invalid_count
        if total == 0:
            return 0.0
        return self.valid_count / total

    def classify(self, returncode: int) -> Validity:
        """Map one exit status to a parser verdict."""
        if not self.enabled:
            return Validity.UNKNOWN
        if returncode == self.reject_code:
            return Validity.INVALID
        if returncode == 0:
            return Validity.VALID
        return Validity.UNKNOWN

    def record(self, validity: Validity, edges: Iterable[int] | None) -> bool:
        """Fold one execution in; report whether valid coverage grew.

        Edges from invalid and unknown runs are deliberately not banked.
        Banking them would let an error-path execution claim coverage that
        the first genuinely valid input to reach it should be credited
        with -- the channel would then answer the same question as the
        main coverage map, one execution later.
        """
        if validity is Validity.VALID:
            self.valid_count += 1
        elif validity is Validity.INVALID:
            self.invalid_count += 1

        if validity is not Validity.VALID or not edges:
            return False

        fresh = {e for e in edges if e not in self.valid_edges}
        if not fresh:
            return False

        if len(self.valid_edges) >= self.MAX_VALID_EDGES:
            log.debug("valid edge map full at %d entries", len(self.valid_edges))
            return False

        self.valid_edges.update(fresh)
        return True
