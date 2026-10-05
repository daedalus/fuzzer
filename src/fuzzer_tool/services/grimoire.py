"""--grimoire: generalize each admitted seed once, feed the three mutators.

Admission notes the edges the seed was admitted for (its novelties). On the
seed's first round the stage re-executes it with spans removed and keeps the
removals that still reach those edges (``core.grimoire.generalize``). A seed
with no noted novelty (initial corpus) falls back to its own full edge set,
which is stricter, hence sound but generalizes less.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterable

from fuzzer_tool.core.grimoire import (
    GENERALIZE_MAX_LEN,
    MAX_SEEDS,
    GrimoireBook,
    generalize,
)

# One execution's reached edge ids.
Run = Callable[[bytes], Iterable[int]]

# Noted novelties outlive the book's LRU a little (they hold the seed bytes,
# so the cap stays small).
_TABLE_MAX = MAX_SEEDS * 2
# Attempted seeds are remembered by hash, far longer than the book keeps
# them: a seed evicted from the book must not be re-generalized (up to
# max_execs executions) every time the scheduler returns to it.
_TRIED_MAX = 1 << 15


class GrimoireStage:
    """Novelty bookkeeping plus the once-per-seed generalization."""

    def __init__(self, run: Run, max_execs: int, max_len: int):
        self._run = run
        self._max_execs = max_execs
        self.book = GrimoireBook(max_len)
        self._novel: OrderedDict[bytes, frozenset[int]] = OrderedDict()
        self._tried: OrderedDict[int, None] = OrderedDict()

    @property
    def noted(self) -> int:
        return len(self._novel)

    @property
    def tried(self) -> int:
        return len(self._tried)

    def note(self, data: bytes, novel: Iterable[int]) -> None:
        """Remember the edges ``data`` was admitted for."""
        edges = frozenset(novel)
        if not edges or not data or len(data) > GENERALIZE_MAX_LEN:
            return
        self._novel[data] = edges
        while len(self._novel) > _TABLE_MAX:
            self._novel.popitem(last=False)

    def generalize(self, data: bytes) -> int:
        """Generalize ``data`` once; returns the executions spent."""
        if not data or len(data) > GENERALIZE_MAX_LEN:
            return 0
        key = hash(data)
        if key in self._tried:
            return 0
        self._tried[key] = None
        while len(self._tried) > _TRIED_MAX:
            self._tried.popitem(last=False)

        execs = 0

        def run(cand: bytes) -> frozenset[int]:
            nonlocal execs
            execs += 1
            return frozenset(self._run(cand))

        novel = self._novel.pop(data, None)
        if novel is None:
            novel = run(data)
            if not novel:
                return execs

        # A fresh closure per seed: `novel` differs for every one.
        result = generalize(data, lambda cand: novel <= run(cand), max(self._max_execs - execs, 0))
        if result is not None:
            self.book.add(data, result.items)
        return execs
