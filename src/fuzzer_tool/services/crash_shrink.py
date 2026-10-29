"""Shrink novel crashes at maintenance ticks (``--crash-tmin-execs``).

```
save_crash (novel) ──enqueue──► CrashShrinker queue (≤ MAX_PENDING)
maintenance tick ──► tick(): one crash at a time, within budget_ms
                       reproduce ─► Reducer(ChunkPass) ─► <base>.min
                       oracle: run_fn(cand) -> sig_fn(rc, stderr) == pinned sig
```

The ``.bin`` is never touched; ``<base>.min`` is written only for a strictly
smaller input that reproduced the same signature.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from fuzzer_tool.adapters.filesystem import save_min_crash
from fuzzer_tool.core.reducer import ChunkPass, Oracle, Phase, Reducer, Verdict

# Crashes waiting for a tick (Hard Rule 54); later ones are dropped.
MAX_PENDING = 64
DEFAULT_BUDGET_MS = 200.0

RunFn = Callable[[bytes], tuple[int, str]]
SigFn = Callable[[int, str], str | None]


class CrashShrinker:
    """Budgeted, signature-pinned reduction of saved crashes.

    Args:
        run_fn: Runs one input the campaign's way: ``(rc, stderr)``.
        sig_fn: Signature of a run, None when it did not crash.
        max_execs: Target runs per crash, reproduction included.
        budget_ms: Wall time per tick; the crash in progress finishes its
            current run, the rest wait for the next tick.
    """

    def __init__(
        self,
        run_fn: RunFn,
        sig_fn: SigFn,
        max_execs: int,
        budget_ms: float = DEFAULT_BUDGET_MS,
    ) -> None:
        self._run = run_fn
        self._sig = sig_fn
        self._max_execs = max_execs
        self._budget_s = budget_ms / 1000
        self._queue: deque[tuple[Path, str]] = deque()
        self.shrunk = 0

    @property
    def pending(self) -> int:
        return len(self._queue)

    def enqueue(self, crash: Path, signature: str) -> None:
        """Queue *crash* (a ``.bin``) for shrinking under *signature*."""
        if len(self._queue) < MAX_PENDING:
            self._queue.append((crash, signature))

    def tick(self) -> None:
        """Shrink queued crashes until the tick budget is spent (one at least)."""
        deadline = time.monotonic() + self._budget_s
        while self._queue:
            crash, sig = self._queue.popleft()
            self._shrink(crash, sig, deadline)
            if time.monotonic() >= deadline:
                return

    def _shrink(self, crash: Path, sig: str, deadline: float) -> None:
        if not crash.is_file():
            return
        data = crash.read_bytes()

        runs = 0

        def test(cand: bytes) -> Verdict:
            # Out of execs or time: FAIL without running, the reducer unwinds.
            nonlocal runs
            if runs >= self._max_execs or (runs and time.monotonic() >= deadline):
                return Verdict.FAIL
            runs += 1
            return Verdict.PASS if self._sig(*self._run(cand)) == sig else Verdict.FAIL

        # Reproduce first: a flaky crash gives nothing to pin.
        if test(data) is not Verdict.PASS:
            return

        smallest = Reducer(Oracle(test), [(Phase.MAIN, ChunkPass())]).run(data)
        if len(smallest) < len(data):
            save_min_crash(crash, smallest)
            self.shrunk += 1
