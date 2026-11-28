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
    FILE = "file"  # open file object (TemporaryFile), or None
    PROCSET = "procset"  # list of procs this object will reap (appended to)
    PID = "pid"  # raw pid from os.fork(), or None; signaled with os.kill
    HANDLER = "handler"  # signal disposition or handler callable; "unset" until assigned
    LOCK = "lock"  # threading.Lock: makes the class a thread model (tools/tla_threads.py)
    COND = "cond"  # threading.Condition over a LOCK field (or its own lock)
    QUEUE = "queue"  # deque used as a work queue: modeled as its length (<= MaxQueue)
    THREAD = "thread"  # threading.Thread(target=self.<method>), or None
    SEGMENT = "segment"  # SysV shm segment (its id or attach address), or None / -1


class Op(Enum):
    """Syscalls.tla operator a module-level call is modeled by."""

    SPAWN = "Spawn"  # subprocess.Popen(...) -> fresh live proc
    READER = "NewReader"  # PipeReader(fd) -> reader over fd's proc
    CLOSE_STREAMS = "CloseStreams"  # close a proc's pipes
    MAYBE = "Maybe"  # returns a value or None
    OPEN_FILE = "FileOpen"  # tempfile.TemporaryFile() -> fresh open file
    KILLPG = "KillPg"  # os.killpg(proc.pid, sig)
    WRAP = "Wrap"  # record constructor holding a proc, e.g. _Sender(proc, ...) -> proc
    MAY_FAIL = "MayFail"  # work that returns or raises RuntimeError (e.g. a measurement)
    SHM_GET = "ShmGet"  # libc_shm.shmget(size) -> fresh segment, or None
    SHM_AT = "ShmAt"  # libc_shm.shmat(seg) -> attached address, or None
    SHM_DT = "ShmDt"  # libc_shm.shmdt(addr) -> bool
    SHM_RMID = "ShmRmid"  # libc_shm.shmctl_rmid(seg) -> bool
    PTRACE = "Ptrace"  # libc.ptrace(req, pid, ...): PTRACE_CONT/DETACH resume a stopped tracee


class Init(Enum):
    """Where the model's initial state comes from."""

    RUN = "run"  # compile __init__
    ASSUMED = "assumed"  # skip __init__; fields start at kind defaults plus `assume`


class Phase(Enum):
    """When the caller invokes an entry."""

    SETUP = "setup"  # once, from Init after __init__; a failure is checked by SetupCleanup
    CALL = "call"  # any number of times after setup


@dataclass(frozen=True)
class MachineSpec:
    """Model boundary of one class."""

    state: dict[str, Kind]
    invariants: dict[str, str] = field(default_factory=dict)
    model: dict[str, Op] = field(default_factory=dict)
    ignore: tuple[str, ...] = ()
    replies: dict[str, str] = field(default_factory=dict)
    init: Init = Init.RUN
    assume: dict[str, object] = field(default_factory=dict)
    init_args: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EntrySpec:
    """One caller-visible method."""

    phase: Phase
    requires: str | None
    raises: tuple[str, ...] = ()


def machine(
    *,
    state: dict[str, Kind],
    invariants: dict[str, str] | None = None,
    model: dict[str, Op] | None = None,
    ignore: tuple[str, ...] = (),
    replies: dict[str, str] | None = None,
    init: Init = Init.RUN,
    assume: dict[str, object] | None = None,
    init_args: dict[str, object] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Mark a class as a state machine.

    Args:
        state: modeled fields and their kinds; every other field is abstracted away.
        invariants: name -> Python expression over ``self`` checked between calls.
        model: dotted call text -> Syscalls.tla operator.
        ignore: dotted call-text prefixes with no modeled effect (logging, timers).
        replies: request word -> reply word, e.g. ``{"RUN": "RC"}``; requests
            in this map are queued until a readline consumes their reply.
        init: Init.ASSUMED skips an __init__ too heavy to model; the state after it
            is declared in *assume* (field -> bool/str/None), the rest at defaults.
        assume: initial field values for Init.ASSUMED.
        init_args: __init__ arguments to pin (enum member, bool, str or None), e.g. the
            production mode; unpinned arguments stay unmodeled.
    """
    spec = MachineSpec(
        dict(state),
        dict(invariants or {}),
        dict(model or {}),
        ignore,
        dict(replies or {}),
        init,
        dict(assume or {}),
        dict(init_args or {}),
    )

    def mark(cls: type[T]) -> type[T]:
        setattr(cls, MACHINE_ATTR, spec)
        return cls

    return mark


def entry(
    phase: Phase = Phase.CALL, requires: str | None = None, raises: tuple[str, ...] = ()
) -> Callable[[T], T]:
    """Mark a method the caller invokes.

    Args:
        phase: when the caller invokes it.
        requires: caller-side guard, a Python expression over ``self``.
        raises: exception classes the caller catches; escaping them is not NoUncaught.
    """
    spec = EntrySpec(phase, requires, raises)

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
