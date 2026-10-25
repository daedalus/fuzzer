"""Run-mode and feature-switch enums (Hard Rule 31: no boolean parameters).

RunMode.FULL   every enabled subsystem observes every round
RunMode.LITE   --lightweight: execute, classify, scan coverage, record
               edges, then crash / admit / drop. No op credit, no side
               signals, no learners -- execs/s over search quality.
"""

from enum import Enum


class RunMode(Enum):
    """How much per-round bookkeeping a fuzz round does."""

    FULL = "full"
    LITE = "lite"


class Switch(Enum):
    """On/off for a subsystem that is on by default."""

    ON = "on"
    OFF = "off"


class InputPlacement(Enum):
    """Where direct in-process modes put the input bytes.

    EXACT  exact-size heap copy; ASAN redzones catch overreads.
    GUARD  flush against a PROT_NONE page (``adapters/guard_arena.py``);
           catches overreads without ASAN, labelled ``guard-overread``.
    """

    EXACT = "exact"
    GUARD = "guard"
