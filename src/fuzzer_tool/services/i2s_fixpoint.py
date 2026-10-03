"""--i2s-fixpoint: search each admitted input; queue what settles.

Admission runs ``core.i2s_fixpoint.solve`` on the new entry. Fixed points
and cycle members wait here; ``OperatorEngine.mutate`` hands one per round
to the normal round pipeline, off the bandit tournament (same path as the
format-seed queue), so coverage decides whether they are kept.
"""

from collections import Counter, deque

from fuzzer_tool.core.i2s_fixpoint import Fixpoint, Probe, solve

# Pending candidates; oldest dropped first. One search adds at most
# max_iters, so this holds several searches' worth.
QUEUE_MAX = 64


class I2SFixpoint:
    """Fixed-point search over one target's cmplog probe."""

    def __init__(self, probe: Probe, max_iters: int):
        self._probe = probe
        self._max_iters = max_iters
        self._queue: deque[bytes] = deque(maxlen=QUEUE_MAX)
        self._outcomes: Counter[str] = Counter()
        self._execs = 0

    def search(self, data: bytes) -> Fixpoint:
        """Search from *data*; queue every self-consistent candidate."""
        result = solve(data, self._probe, self._max_iters)
        self._execs += result.execs
        self._outcomes[result.outcome.value] += 1
        self._queue.extend(result.candidates)
        return result

    def pop(self) -> bytes | None:
        return self._queue.popleft() if self._queue else None

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict[str, int]:
        return {"execs": self._execs, **self._outcomes}
