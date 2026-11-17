"""Stalest-first bench and revive for Elo meta-scheduler strategies.

Port of soksolv ``lane.go`` ``ringOver`` (retire the engine with the most
expansions since its last improvement) and ``lane_revive.go`` (restart a
dead lane fresh instead of retiring it for good)::

    round played by s:   gain → stale[s] = 0     miss → stale[s] += 1
    every CHECK_EVERY rounds of an arena, at a pick over its pool:
      revive  benched s offered, COOLDOWN rounds on the bench
              → rating reset to the prior (fresh start), stale[s] = 0
      bench   stalest armed s, when stale[s] > max(STALE_FLOOR, STALE_MULT x median of the others)
              and >= 2 arms are armed (never empty the pool, soksolv pullzone rule)
    select_strategy sees only the armed arms.

Rounds are counted per arena (operator, seed, position, target): a seed
round neither ages nor checks the operator pool.
"""

from collections.abc import Callable

from fuzzer_tool.core.analyzers.analyzer_elo import Arena, strategy_arena

STALE_FLOOR = 2000  # rounds without a gain before any bench (~2k execs)
STALE_MULT = 4  # ... and this many times the armed pool's median stale count
COOLDOWN = 10_000  # arena rounds on the bench before a fresh revival
CHECK_EVERY = 64  # arena rounds between bench/revive passes (ifs up)


def _median(xs: list[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    return float(xs[mid]) if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


class StaleBench:
    """Per-strategy rounds since the last gain, and who sits on the bench.

    Memory: one int per strategy name ever recorded (a fixed, small set).
    """

    __slots__ = (
        "_benched",
        "_checked",
        "_cooldown",
        "_every",
        "_floor",
        "_mult",
        "_stale",
        "_ticks",
    )

    def __init__(
        self,
        floor: int = STALE_FLOOR,
        mult: float = STALE_MULT,
        cooldown: int = COOLDOWN,
        check_every: int = CHECK_EVERY,
    ):
        self._floor = floor
        self._mult = mult
        self._cooldown = cooldown
        self._every = check_every
        self._stale: dict[str, int] = {}
        self._benched: dict[str, int] = {}  # name -> arena tick it was benched at
        self._ticks: dict[Arena, int] = {}
        self._checked: dict[Arena, int] = {}

    def record(self, name: str, score: float) -> None:
        """One round played by *name*; a positive score is a gain."""
        arena = strategy_arena(name)
        self._ticks[arena] = self._ticks.get(arena, 0) + 1
        self._stale[name] = 0 if score > 0.0 else self._stale.get(name, 0) + 1

    def armed(self, pool: list[str], revive: Callable[[str], None]) -> list[str]:
        """*pool* minus the benched arms; never empty for a non-empty pool."""
        arena = strategy_arena(pool[0])
        tick = self._ticks.get(arena, 0)
        if tick - self._checked.get(arena, -self._every) >= self._every:
            self._checked[arena] = tick
            self._revive_due(pool, tick, revive)
            self._bench_stalest(pool, tick)

        benched = self._benched
        if not benched:
            return pool
        return [s for s in pool if s not in benched] or pool

    def _revive_due(self, pool: list[str], tick: int, revive: Callable[[str], None]) -> None:
        for s in pool:
            at = self._benched.get(s)
            if at is None or tick - at < self._cooldown:
                continue
            del self._benched[s]
            self._stale[s] = 0
            revive(s)

    def _bench_stalest(self, pool: list[str], tick: int) -> None:
        active = [s for s in pool if s not in self._benched]
        if len(active) < 2:
            return

        # Stalest first; judged against the median of the *other* armed arms,
        # so a 2-arm pool can still bench its outlier.
        stale = self._stale
        counts = [stale.get(s, 0) for s in active]
        worst = max(counts)
        i = counts.index(worst)
        bar = max(self._floor, self._mult * _median(counts[:i] + counts[i + 1 :]))
        if worst <= bar:
            return
        self._benched[active[i]] = tick

    def benched(self) -> set[str]:
        return set(self._benched)

    def stale(self, name: str) -> int:
        return self._stale.get(name, 0)

    def to_dict(self) -> dict:
        """State for the ``elo`` StateStore section (pickled)."""
        return {
            "stale": dict(self._stale),
            "benched": dict(self._benched),
            "ticks": {a.value: t for a, t in self._ticks.items()},
        }

    def from_dict(self, data) -> None:
        """Restore; a malformed payload leaves a fresh bench."""
        try:
            stale = {str(k): int(v) for k, v in data["stale"].items()}
            benched = {str(k): int(v) for k, v in data["benched"].items()}
            ticks = {Arena(a): int(t) for a, t in data["ticks"].items()}
        except (TypeError, KeyError, ValueError, AttributeError):
            return
        self._stale, self._benched, self._ticks = stale, benched, ticks
        self._checked = {}
