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


class StateRestore(Enum):
    """What direct in-process modes reset before each exec (``--snapshot-restore``).

    OFF      nothing: target globals carry over (``--no-snapshot-restore``).
    GLOBALS  default: the target .so's .data/.bss as captured after load
             (``adapters/global_snapshot.py``). Heap is not restored.
             Refused for targets linking a shared registry (libgobject).
    GLOBALS_UNSAFE  GLOBALS without that refusal: reproduces restore
             artifacts (GStreamer clipping-meta SIGSEGV). Not for fuzzing.
    """

    OFF = "off"
    GLOBALS = "globals"
    GLOBALS_UNSAFE = "globals-unsafe"


class CrashReward(Enum):
    """Which crashes pay the operators that produced them (``--novel-crash-reward``).

    ANY    every crash, repeats included (default).
    NOVEL  only a crash that would be saved: repeats of a known bug do not.
    """

    ANY = "any"
    NOVEL = "novel"


class InputChannel(Enum):
    """How a spawned target receives the input (``-F/--file-mode``).

    STDIN  bytes piped to stdin.
    FILE   bytes written to a temp file, path passed via ``{file}``.
    """

    STDIN = "stdin"
    FILE = "file"
