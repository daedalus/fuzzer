"""SIGPROF stack sampler for ``--profile-hotpath --profile-mode sample``.

cProfile instruments every call, so per-exec savings of 1-10 us sit below
its own overhead, and on 3.13 it bills other threads' waits to the
caller (DEEP_DIVE §ffmpeg lightweight overhead). This samples instead:

    ITIMER_PROF (process CPU time) ──tick──> SIGPROF
        └─ handler (main thread): for each thread whose CPU clock moved
              └─ Counter[(thread:Name, outer, ..., leaf)] += 1

Ticks follow process CPU time; a thread is sampled only when its own CPU
clock moved since the last tick, so threads parked in waits draw none. The handler
runs between bytecodes in the main thread; while the main thread is in C
(a direct_lite target call) the tick is delivered on return, billed to
the Python frame that made the call. SA_RESTART keeps most of the
target's own syscalls (read, write, wait) from failing with EINTR;
poll/select/nanosleep return EINTR regardless (signal(7)). ``fork`` does not inherit the timer;
``exec`` resets the handler.

Output is collapsed stacks (``a;b;c 12``), the input format of
flamegraph.pl, speedscope and inferno.
"""

from __future__ import annotations

import collections
import os
import signal
import sys
import threading
import time
from types import CodeType, FrameType

DEFAULT_INTERVAL_S = 0.005  # 200 Hz of CPU time
MAX_DEPTH = 64
MAX_STACKS = 50_000  # distinct stacks kept; Hard Rule 54
OVERFLOW = ("<overflow>",)
_ROOT = "thread:"


class StackSampler:
    """Count sampled Python stacks per thread, bounded in depth and table size."""

    def __init__(
        self,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_depth: int = MAX_DEPTH,
        max_stacks: int = MAX_STACKS,
    ) -> None:
        self._interval = interval_s
        self._max_depth = max_depth
        self._max_stacks = max_stacks
        self._stacks: collections.Counter[tuple[str, ...]] = collections.Counter()
        self._labels: dict[CodeType, str] = {}
        self._names: dict[int, str] = {}
        self._cpu: dict[int, float] = {}  # per-thread CPU seconds at last tick
        self._prev_handler: object = None
        self._running = False

    @property
    def samples(self) -> int:
        """Ticks recorded (one per thread per tick)."""
        return sum(self._stacks.values())

    # ── Timer ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Install the SIGPROF handler and arm the CPU-time timer."""
        if self._running:
            raise RuntimeError("StackSampler already running")

        self._prev_handler = signal.signal(signal.SIGPROF, self._on_tick)
        signal.siginterrupt(signal.SIGPROF, False)  # SA_RESTART
        signal.setitimer(signal.ITIMER_PROF, self._interval, self._interval)
        self._running = True

    def stop(self) -> None:
        """Disarm the timer and restore the previous handler. Idempotent."""
        if not self._running:
            return

        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, self._prev_handler)
        self._running = False

    def _on_tick(self, _signum: int, frame: FrameType | None) -> None:
        frames = sys._current_frames()

        # The main thread's snapshot is this handler; *frame* is what it interrupted.
        if frame is not None:
            frames[threading.main_thread().ident] = frame

        self._record({i: f for i, f in frames.items() if self._ran(i, frames)})

    def _ran(self, ident: int, live: dict[int, FrameType]) -> bool:
        """True when thread *ident* used CPU since the previous tick.

        Ticks follow process CPU, but every thread has a frame: without this
        a thread parked in a wait would draw a sample per tick. A thread's
        first tick sets its baseline and is not sampled.
        """
        try:
            now = time.clock_gettime(time.pthread_getcpuclockid(ident))
        except OSError:
            return False  # thread exited after the frame snapshot: nothing ran

        # Drop baselines of exited threads: the table stays thread-count sized.
        if len(self._cpu) > len(live):
            self._cpu = {i: t for i, t in self._cpu.items() if i in live}

        prev = self._cpu.get(ident)
        self._cpu[ident] = now
        return prev is not None and now > prev

    # ── Recording ──────────────────────────────────────────────────────

    def _record(self, frames: dict[int, FrameType]) -> None:
        """Add one sample per thread: root label, then frames outer -> leaf."""
        for ident, frame in frames.items():
            stack = self._walk(frame)
            stack.append(self._thread_label(ident))
            stack.reverse()
            key = tuple(stack)

            # Past the cap, new shapes share one bucket: memory stays bounded.
            if key not in self._stacks and len(self._stacks) >= self._max_stacks:
                key = OVERFLOW
            self._stacks[key] += 1

    def _walk(self, frame: FrameType | None) -> list[str]:
        """Leaf-first labels of at most ``max_depth`` frames."""
        out: list[str] = []
        labels = self._labels
        depth = self._max_depth
        while frame is not None and len(out) < depth:
            code = frame.f_code
            label = labels.get(code)
            if label is None:
                label = labels[code] = _code_label(code)
            out.append(label)
            frame = frame.f_back
        return out

    def _thread_label(self, ident: int) -> str:
        name = self._names.get(ident)
        if name is None:
            # Refresh on an unseen ident only; enumerate() per tick is wasted work.
            self._names = {t.ident: t.name for t in threading.enumerate() if t.ident}
            name = self._names.get(ident, str(ident))
        return _ROOT + name

    # ── Reports ────────────────────────────────────────────────────────

    def top_self(self, n: int) -> list[tuple[str, int]]:
        """Leaf frames by sample count (self time)."""
        c: collections.Counter[str] = collections.Counter()
        for stack, count in self._stacks.items():
            c[stack[-1]] += count
        return [(k, v) for k, v in c.most_common() if not k.startswith(_ROOT)][:n]

    def top_total(self, n: int) -> list[tuple[str, int]]:
        """Frames by samples they appear in (inclusive time); recursion counted once."""
        c: collections.Counter[str] = collections.Counter()
        for stack, count in self._stacks.items():
            for label in set(stack):
                c[label] += count
        return [(k, v) for k, v in c.most_common() if not k.startswith(_ROOT)][:n]

    def write_collapsed(self, path: str | os.PathLike) -> None:
        """One ``frame;frame;leaf count`` line per distinct stack."""
        with open(path, "w", encoding="utf-8") as f:
            for stack, count in self._stacks.most_common():
                f.write(f"{';'.join(stack)} {count}\n")


def _code_label(code: CodeType) -> str:
    """``qualname (file.py:line)``; ``;`` and spaces kept out of the name."""
    name = code.co_qualname.replace(";", ":").replace(" ", "_")
    return f"{name} ({os.path.basename(code.co_filename)}:{code.co_firstlineno})"
