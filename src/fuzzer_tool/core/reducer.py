"""Test-case reducer: stateful passes driven to a fixpoint (ported from C-Reduce).

C-Reduce's pass manager, reduced to its core::

    FIRST once ─► MAIN until a sweep stops shrinking ─► LAST once

    per pass:  st = new(cur)
               loop: (cand, st) = transform(cur, st)   # None → pass done
                     oracle(cand) PASS → cur = cand     # keep st: same spot
                     else              → st = advance(cur, st)

Unlike restart-on-success ddmin, a success never rewinds the cursor, and
the fixpoint loop restores 1-minimality the cursor skips. Candidates are
memoized by content hash, and a candidate showing a *different* bug
(``Verdict.ALSO``) is handed to a callback instead of being dropped.
"""

import hashlib
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Protocol

from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.mutations.seq_mutate import split_msgs

# 16-byte keys: 64K entries ≈ a few MB, bounded (Hard Rule 54).
DEFAULT_CACHE_CAP = 1 << 16
_KEY_BYTES = 16


class Verdict(Enum):
    """Interestingness of one candidate."""

    PASS = "pass"  # same bug: keep the candidate
    FAIL = "fail"  # not interesting
    ALSO = "also"  # a different bug: report, do not keep


class Phase(Enum):
    """Pass group; C-Reduce's first/main/last pass priorities."""

    FIRST = "first"
    MAIN = "main"
    LAST = "last"


class ReducePass(Protocol):
    """C-Reduce pass interface: ``new`` / ``transform`` / ``advance``."""

    def new(self, data: bytes) -> Any: ...

    def transform(self, data: bytes, state: Any) -> tuple[bytes, Any] | None: ...

    def advance(self, data: bytes, state: Any) -> Any: ...


class ChunkPass:
    """Delete ``data[i:i+chunk]``; halve ``chunk`` after each full sweep.

    State is ``(chunk, i)``. Example on 8 bytes: chunks 4, 2, 1 at
    i = 0, chunk, 2*chunk, ... A success keeps ``i``: the bytes that slid
    into ``i`` are tried next.
    """

    def new(self, data: bytes) -> tuple[int, int]:
        return max(1, len(data) // 2), 0

    def transform(self, data: bytes, state: tuple[int, int]) -> tuple[bytes, Any] | None:
        return _cut(data, state)

    def advance(self, data: bytes, state: tuple[int, int]) -> tuple[int, int]:
        chunk, i = state
        return chunk, i + chunk


class MsgPass(ChunkPass):
    """``ChunkPass`` over whole messages (``seq_mutate.split_msgs``).

    VirtFuzz's frame minimizer as delta debugging. Byte cuts break the
    framing of a message stream; whole-message cuts keep it::

        [3|abc][3|BUG][2|xy]  ->  [3|BUG][2|xy]  ->  [3|BUG]

    State ``(chunk, i)`` counts messages. Data is re-split per candidate:
    a kept cut may change the boundaries. No boundaries: no candidates.
    """

    def new(self, data: bytes) -> tuple[int, int]:
        msgs = split_msgs(data) or ()
        return max(1, len(msgs) // 2), 0

    def transform(self, data: bytes, state: tuple[int, int]) -> tuple[bytes, Any] | None:
        msgs = split_msgs(data)
        if msgs is None:
            return None

        step = _cut(msgs, state)
        if step is None:
            return None
        kept, state = step
        return b"".join(kept), state


def _cut(seq: Any, state: tuple[int, int]) -> tuple[Any, tuple[int, int]] | None:
    """Next ``seq[:i] + seq[i+chunk:]`` at or after ``state``, or None when done."""
    chunk, i = state
    n = len(seq)
    while True:
        # Sweep done: next smaller chunk, or stop after single units.
        if i >= n:
            if chunk <= 1:
                return None
            chunk, i = chunk // 2, 0
            continue

        # Never propose the empty input.
        if i == 0 and chunk >= n:
            i = chunk
            continue

        return seq[:i] + seq[i + chunk :], (chunk, i)


class _Probe(Enum):
    """``PrefixPass`` search stage."""

    GALLOP = "gallop"  # cut 1, 2, 4, ... bytes off the end until a FAIL
    BISECT = "bisect"  # binary search between that FAIL and the shortest PASS


# PrefixPass state: (stage, longest failing length, cut, proposed length).
_PrefixState = tuple[_Probe, int, int, int]
_NO_PROBE = -1


class PrefixPass:
    """Shortest crashing prefix: gallop from the end, then bisect.

    Cuts 1, 2, 4, ... more bytes off the current end until a probe fails
    (clamped at length 1), then bisects between it and the shortest passing
    prefix. The current input is always the shortest verified prefix, so a
    non-monotone oracle still yields a crashing input. Runs:
    <= 2 * (ceil(log2 n) + 1); exactly 1 when the last byte is needed::

        n=100, need 7:  99 97 93 85 69 37 1x | 19 10 5x 7 6x  -> 12 runs
    """

    def new(self, data: bytes) -> _PrefixState:
        return _Probe.GALLOP, 0, 1, _NO_PROBE

    def transform(self, data: bytes, state: _PrefixState) -> tuple[bytes, Any] | None:
        stage, fail, cut, proposed = state
        n = len(data)

        # Gallop: the last probe passed (cur shrank to it), double the cut.
        if stage is _Probe.GALLOP and proposed == n:
            cut *= 2

        size = max(n - cut, 1) if stage is _Probe.GALLOP else (fail + n) // 2
        if size <= fail or size >= n:
            return None
        return data[:size], (stage, fail, cut, size)

    def advance(self, data: bytes, state: _PrefixState) -> _PrefixState:
        _stage, _fail, cut, proposed = state
        return _Probe.BISECT, proposed, cut, _NO_PROBE


class Prefix(Enum):
    """Shortest-prefix pre-pass for ``shrink_passes``.

    Opt-in: it wins only when the crash needs a short prefix of a long
    input (-31..-40% runs, prefix <1% of input); on mid-buffer bugs it
    costs more runs than ChunkPass's halving (median +13% over a sweep).
    """

    KEEP = "keep"  # no pre-pass
    GALLOP = "gallop"  # PrefixPass before MAIN


def shrink_passes(prefix: Prefix = Prefix.KEEP) -> list[tuple[Phase, ReducePass]]:
    """Crash-shrink passes: [shortest prefix], whole messages, byte chunks."""
    passes: list[tuple[Phase, ReducePass]] = [(Phase.MAIN, MsgPass()), (Phase.MAIN, ChunkPass())]
    if prefix is Prefix.GALLOP:
        passes.insert(0, (Phase.FIRST, PrefixPass()))
    return passes


class Oracle:
    """Memoized interestingness test.

    Args:
        test: Candidate → ``Verdict``.
        on_also: Receives each ``Verdict.ALSO`` candidate once.
        capacity: Max cached verdicts (LRU).
    """

    def __init__(
        self,
        test: Callable[[bytes], Verdict],
        on_also: Callable[[bytes], None] | None = None,
        capacity: int = DEFAULT_CACHE_CAP,
    ):
        self._test = test
        self._on_also = on_also
        self._cache: LRUCache = LRUCache(capacity)
        self.runs = 0
        self.hits = 0

    @property
    def cached(self) -> int:
        """Number of cached verdicts."""
        return len(self._cache)

    def __call__(self, data: bytes) -> bool:
        """True iff ``data`` reproduces the original bug."""
        key = hashlib.blake2b(data, digest_size=_KEY_BYTES).digest()
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit

        self.runs += 1
        verdict = self._test(data)
        if verdict is Verdict.ALSO and self._on_also is not None:
            self._on_also(data)

        ok = verdict is Verdict.PASS
        self._cache[key] = ok
        return ok


class Reducer:
    """Run ``(Phase, pass)`` pairs over an input until no pass shrinks it.

    Args:
        oracle: Interestingness test; the input passed to ``run`` must pass.
        passes: ``(Phase, ReducePass)`` pairs; order within a phase is kept.
        max_steps: Cap on accepted reductions (None = unbounded).
    """

    def __init__(
        self,
        oracle: Oracle,
        passes: Sequence[tuple[Phase, ReducePass]],
        max_steps: int | None = None,
    ):
        self._oracle = oracle
        self._passes = passes
        self._max_steps = max_steps
        self.accepted = 0

    def run(self, data: bytes) -> bytes:
        """Return the smallest interesting input found."""
        cur = self._sweep(Phase.FIRST, data)

        # MAIN to a fixpoint: a later pass can unlock an earlier one.
        while not self._spent():
            before = len(cur)
            cur = self._sweep(Phase.MAIN, cur)
            if len(cur) >= before:
                break

        return self._sweep(Phase.LAST, cur)

    def _spent(self) -> bool:
        return self._max_steps is not None and self.accepted >= self._max_steps

    def _sweep(self, phase: Phase, data: bytes) -> bytes:
        for p_phase, rpass in self._passes:
            if p_phase is phase:
                data = self._apply(rpass, data)
        return data

    def _apply(self, rpass: ReducePass, cur: bytes) -> bytes:
        state = rpass.new(cur)
        while not self._spent():
            step = rpass.transform(cur, state)
            if step is None:
                return cur

            # Only strict shrinks are tested: guarantees termination.
            cand, state = step
            if len(cand) < len(cur) and self._oracle(cand):
                cur = cand
                self.accepted += 1
                continue

            state = rpass.advance(cur, state)
        return cur
