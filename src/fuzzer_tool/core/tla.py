"""Markers that tools/tla_extract.py turns into a TLA+ model.

Every decorator only attaches its spec to the object and returns it, so the
runtime cost is zero. The extractor reads the spec back, walks the class's
AST and emits a TLA+ module whose I/O outcomes come from tools/tla/Syscalls.tla.

    @tla.machine(state={"_ready": tla.Kind.BOOL, "_proc": tla.Kind.PROC}, ...)
    class Runner:
        @tla.entry(tla.Phase.SETUP)
        def start(self): ...

        @tla.entry(requires="self._ready")
        def run_one(self, data): ...
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

T = TypeVar("T")

MACHINE_ATTR = "__tla_machine__"
ENTRY_ATTR = "__tla_entry__"
OPAQUE_ATTR = "__tla_opaque__"


class Kind(Enum):
    """Abstract type of a modeled field."""

    BOOL = "bool"
    PROC = "proc"  # subprocess.Popen handle, or None
    READER = "reader"  # PipeReader over a proc's stdout, or None


class Op(Enum):
    """Syscalls.tla operator a module-level call is modeled by."""

    SPAWN = "Spawn"  # subprocess.Popen(...) -> fresh live proc
    READER = "NewReader"  # PipeReader(fd) -> reader over fd's proc
    CLOSE_STREAMS = "CloseStreams"  # close a proc's pipes
    MAYBE = "Maybe"  # returns a value or None


class Phase(Enum):
    """When the caller invokes an entry."""

    SETUP = "setup"  # once, from Init; only successful (truthy) runs are kept
    CALL = "call"  # any number of times after setup


@dataclass(frozen=True)
class MachineSpec:
    """Model boundary of one class."""

    state: dict[str, Kind]
    invariants: dict[str, str] = field(default_factory=dict)
    model: dict[str, Op] = field(default_factory=dict)
    ignore: tuple[str, ...] = ()
    replies: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EntrySpec:
    """One caller-visible method."""

    phase: Phase
    requires: str | None


def machine(
    *,
    state: dict[str, Kind],
    invariants: dict[str, str] | None = None,
    model: dict[str, Op] | None = None,
    ignore: tuple[str, ...] = (),
    replies: dict[str, str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Mark a class as a state machine.

    Args:
        state: modeled fields and their kinds; every other field is abstracted away.
        invariants: name -> Python expression over ``self`` checked between calls.
        model: dotted call text -> Syscalls.tla operator.
        ignore: dotted call-text prefixes with no modeled effect (logging, timers).
        replies: request word -> reply word, e.g. ``{"RUN": "RC"}``; requests
            in this map are queued until a readline consumes their reply.
    """
    spec = MachineSpec(
        dict(state), dict(invariants or {}), dict(model or {}), ignore, dict(replies or {})
    )

    def mark(cls: type[T]) -> type[T]:
        setattr(cls, MACHINE_ATTR, spec)
        return cls

    return mark


def entry(phase: Phase = Phase.CALL, requires: str | None = None) -> Callable[[T], T]:
    """Mark a method the caller invokes; *requires* is its caller-side guard."""
    spec = EntrySpec(phase, requires)

    def mark(fn: T) -> T:
        setattr(fn, ENTRY_ATTR, spec)
        return fn

    return mark


def opaque(*returns: object) -> Callable[[T], T]:
    """Model a method as a black box returning one of *returns*, without effects."""

    def mark(fn: T) -> T:
        setattr(fn, OPAQUE_ATTR, tuple(returns))
        return fn

    return mark
