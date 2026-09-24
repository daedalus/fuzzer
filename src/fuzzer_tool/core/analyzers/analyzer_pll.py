"""Observation layer over ``core/pll.py``: lock/unlock transitions with stall context.

Step 1 of ``docs/handover/handover_pll_2026-09-22.md``. One
``PhaseLockedLoop`` per series, bootstrapped from that series' own
``detect_periodicity`` call, stepped tick by tick; every lock flip is logged
with whether the fuzzer was in stall recovery at the time. Read-only: nothing
here feeds back into scheduling.

    exec times ──push──┐                      ┌─> PLLTransition log
    discovery Δ ─push──┤ pending ──flush──> _Track ─> stall_lift()
                       │  (array)   (stats tick)  └─> state() / summary()
                       └─ cap: flush inline

``push`` is the only hot-path call (an array append). Warm-up samples are
buffered until ``warmup``; a significant period > 2 builds the loop and
replays the buffer into it, otherwise the window is discarded and retried.

``stall_lift`` = P(stall | transition) / P(stall). 1.0 means transitions
ignore stalls; the handover asks whether it departs from 1 before anything
acts on lock state.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from fuzzer_tool.core.periodicity import detect_periodicity
from fuzzer_tool.core.pll import PhaseLockedLoop, PLLState

log = logging.getLogger(__name__)

#: Samples buffered before a bootstrap attempt. 256 holds >= 8 cycles of the
#: 15-30 sample periods the loop gains are tuned for.
WARMUP = 256
#: Smallest usable warm-up: detect_periodicity's own ``min_samples`` floor.
MIN_WARMUP = 64
#: Pending samples kept between flushes before an inline flush.
PENDING_CAP = 65536
MAX_TRANSITIONS = 256
#: Nyquist: ``PhaseLockedLoop.from_period`` refuses periods <= 2.
NYQUIST_PERIOD = 2.0
STATE_VERSION = 1
#: ``_Track`` counters persisted verbatim.
_COUNTERS = ("misses", "dropped", "ticks", "ticks_stalled", "transitions", "transitions_stalled")


class Series(Enum):
    EXEC_TIME = "exec_time"
    DISCOVERY = "discovery"


class Stall(Enum):
    NO = 0
    YES = 1


@dataclass(frozen=True)
class PLLTransition:
    series: Series
    tick: int
    exec_count: int
    locked: bool
    period: float
    stalled: bool


class _Track:
    """One loop on one series, plus its correlation counters."""

    def __init__(self, series: Series, warmup: int):
        self.series = series
        self._warmup = warmup
        self._buf = array("d")
        self.pll: PhaseLockedLoop | None = None
        self.bootstrap_period: float | None = None
        self.last: PLLState | None = None
        self.misses = 0
        self.dropped = 0
        self.ticks = 0
        self.ticks_stalled = 0
        self.transitions = 0
        self.transitions_stalled = 0

    def save(self) -> dict[str, Any]:
        out: dict[str, Any] = {c: getattr(self, c) for c in _COUNTERS}
        out["buf"] = self._buf.tolist()
        out["pll"] = self.pll.to_dict() if self.pll is not None else None
        out["bootstrap_period"] = self.bootstrap_period
        out["last"] = asdict(self.last) if self.last is not None else None
        return out

    @classmethod
    def load(cls, series: Series, warmup: int, data: Any) -> _Track:
        """Inverse of :meth:`save`; raises ValueError/TypeError/KeyError on bad shapes."""
        t = cls(series, warmup)
        for c in _COUNTERS:
            setattr(t, c, int(data[c]))
        t._buf = array("d", data["buf"])
        t.pll = PhaseLockedLoop.from_dict(data["pll"]) if data["pll"] is not None else None
        bp = data["bootstrap_period"]
        t.bootstrap_period = float(bp) if bp is not None else None
        t.last = PLLState(**data["last"]) if data["last"] is not None else None
        return t

    def feed(self, xs: array, exec_count: int, stall: Stall) -> list[PLLTransition]:
        out: list[PLLTransition] = []
        for x in xs:
            if not math.isfinite(x):
                self.dropped += 1
                continue
            if self.pll is not None:
                self._step(x, exec_count, stall, out)
                continue

            self._buf.append(x)
            if len(self._buf) >= self._warmup:
                self._bootstrap(exec_count, stall, out)
        return out

    def _bootstrap(self, exec_count: int, stall: Stall, out: list[PLLTransition]) -> None:
        buf, self._buf = self._buf, array("d")
        res = detect_periodicity(buf, min_samples=MIN_WARMUP)
        period = res.dominant_period
        if not res.significant or period is None or period <= NYQUIST_PERIOD:
            self.misses += 1
            return

        # Replay the warm-up so the loop starts converged on real data.
        self.pll = PhaseLockedLoop.from_period(period)
        self.bootstrap_period = period
        for x in buf:
            self._step(x, exec_count, stall, out)

    def _step(self, x: float, exec_count: int, stall: Stall, out: list[PLLTransition]) -> None:
        assert self.pll is not None
        was_locked = self.last.locked if self.last is not None else False
        st = self.pll.step(x)
        self.last = st
        stalled = stall is Stall.YES
        self.ticks += 1
        self.ticks_stalled += stalled
        if st.locked == was_locked:
            return

        self.transitions += 1
        self.transitions_stalled += stalled
        out.append(PLLTransition(self.series, st.tick, exec_count, st.locked, st.period, stalled))


class PLLMonitor:
    """Two tracked series, one transition log.

    Args:
        warmup: Samples per bootstrap attempt (>= ``MIN_WARMUP``).
        max_transitions: Transition log length.
    """

    def __init__(self, warmup: int = WARMUP, max_transitions: int = MAX_TRANSITIONS):
        if warmup < MIN_WARMUP:
            raise ValueError(f"warmup {warmup} must be >= {MIN_WARMUP}")
        self._warmup = warmup
        self._tracks = {s: _Track(s, warmup) for s in Series}
        self._pending = {s: array("d") for s in Series}
        self.transitions: deque[PLLTransition] = deque(maxlen=max_transitions)
        self._last_ctx = (0, Stall.NO)

    def save(self) -> dict[str, Any]:
        """Loops, counters, warm-up and pending buffers, transition log."""
        return {
            "version": STATE_VERSION,
            "tracks": {s.value: t.save() for s, t in self._tracks.items()},
            "pending": {s.value: b.tolist() for s, b in self._pending.items()},
            "transitions": [
                (t.series.value, t.tick, t.exec_count, t.locked, t.period, t.stalled)
                for t in self.transitions
            ],
            "last_ctx": (self._last_ctx[0], self._last_ctx[1].value),
        }

    def load(self, data: Any) -> None:
        """Restore :meth:`save` output; a malformed payload leaves this monitor fresh."""
        if not data:
            return
        try:
            if data["version"] != STATE_VERSION:
                raise ValueError(f"version {data['version']!r}")
            tracks = {s: _Track.load(s, self._warmup, data["tracks"][s.value]) for s in Series}
            pending = {s: array("d", data["pending"][s.value]) for s in Series}
            log_ = [PLLTransition(Series(r[0]), *r[1:]) for r in data["transitions"]]
            ctx = (int(data["last_ctx"][0]), Stall(data["last_ctx"][1]))
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            log.warning("pll state unreadable, starting fresh: %s", e)
            return

        self._tracks, self._pending, self._last_ctx = tracks, pending, ctx
        self.transitions.clear()
        self.transitions.extend(log_)

    def push(self, series: Series, x: float) -> None:
        """Queue one sample; flushes inline past ``PENDING_CAP``."""
        buf = self._pending[series]
        buf.append(x)
        if len(buf) >= PENDING_CAP:
            self._flush_one(series, *self._last_ctx)

    def flush(self, exec_count: int, stall: Stall) -> list[PLLTransition]:
        """Step every queued sample; return (and log) the transitions."""
        self._last_ctx = (exec_count, stall)
        out: list[PLLTransition] = []
        for s in Series:
            out.extend(self._flush_one(s, exec_count, stall))
        return out

    def _flush_one(self, series: Series, exec_count: int, stall: Stall) -> list[PLLTransition]:
        buf = self._pending[series]
        if not buf:
            return []

        self._pending[series] = array("d")
        out = self._tracks[series].feed(buf, exec_count, stall)
        for t in out:
            self.transitions.append(t)
            log.info(
                "pll %s %s at exec %d: period %.1f, stalled=%s",
                t.series.value,
                "LOCK" if t.locked else "UNLOCK",
                t.exec_count,
                t.period,
                t.stalled,
            )
        return out

    def pending(self, series: Series) -> int:
        return len(self._pending[series])

    def state(self, series: Series) -> PLLState | None:
        return self._tracks[series].last

    def bootstrap_period(self, series: Series) -> float | None:
        return self._tracks[series].bootstrap_period

    def stall_lift(self, series: Series) -> float | None:
        """P(stall | transition) / P(stall); None until both are measurable."""
        t = self._tracks[series]
        if not t.transitions or not t.ticks_stalled:
            return None
        return (t.transitions_stalled / t.transitions) / (t.ticks_stalled / t.ticks)

    def summary(self, series: Series) -> dict:
        t = self._tracks[series]
        return {
            "bootstrap_period": t.bootstrap_period,
            "misses": t.misses,
            "dropped": t.dropped,
            "ticks": t.ticks,
            "ticks_stalled": t.ticks_stalled,
            "transitions": t.transitions,
            "transitions_stalled": t.transitions_stalled,
            "locked": t.last.locked if t.last is not None else False,
            "period": t.last.period if t.last is not None else None,
            "coherence": t.last.coherence if t.last is not None else 0.0,
        }
