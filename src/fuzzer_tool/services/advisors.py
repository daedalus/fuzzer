"""--advisors: one zero-cost proposal per advisor per seed stage.

soksolv (advisors.go:79-237, fess_moves.go:423): each advisor proposes at
most one best move per node; advised moves cost 0, the rest cost 1. Here a
node is a seed stage (services/seed_stage.py) and a move is a mutant built
from data an existing pass already recorded for the seed:

    i2s    first redqueen match still in place  -> its replacement
    token  newest dictionary token at a redqueen offset (a cmp field)
    cond   first unsolved CondStmt operand found in the seed -> other operand

    stage begin ─> plan(seed, ctx) ─> queue (deduped, <= MAX_ADVISORS)
                   Fuzzer._advise: one fuzz_one(seed) per queued move;
                   OperatorEngine.mutate pops it off the bandit tournament.
                   SeedStage._left is untouched: advised execs are free.

Each advisor proposes its best move not yet tried (``ctx.fresh``), so a
seed revisited later gets the next one instead of a repeat.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from fuzzer_tool.core.hashing import hash64

# Advisors consulted per stage (soksolv runs a handful).
MAX_ADVISORS = 3
# Tried-move keys kept for dedup; oldest forgotten first.
SEEN_MAX = 4096
# Candidates one advisor inspects per proposal: bounds per-stage cost.
SCAN_MAX = 16


@dataclass(frozen=True, slots=True)
class AdvisorCtx:
    """What advisors read: the fuzzer's recorded state, never the fuzzer."""

    seed_meta: Mapping[bytes, dict]
    dictionary: Sequence[bytes]
    # Unsolved CondStmt view, built only when the cond advisor asks.
    conds: Callable[[], Sequence]
    # True when a candidate was not tried yet.
    fresh: Callable[[bytes], bool]
    max_len: int


class Advisor(Protocol):
    """Proposes at most one mutant of *seed*, or None."""

    name: str

    def propose(self, seed: bytes, ctx: AdvisorCtx) -> bytes | None: ...


def _overwrite(seed: bytes, off: int, payload: bytes) -> bytes:
    """*payload* written at *off*, clipped at the seed's end (length kept)."""
    end = min(len(seed), off + len(payload))
    return seed[:off] + payload[: end - off] + seed[end:]


class I2SAdvisor:
    """Redqueen: replay the seed's first still-valid input-to-state match."""

    name = "i2s"

    def propose(self, seed: bytes, ctx: AdvisorCtx) -> bytes | None:
        meta = ctx.seed_meta.get(seed)
        if not meta:
            return None

        for off, op_a, op_b in meta.get("redqueen_matches", ())[:SCAN_MAX]:
            if seed[off : off + len(op_a)] != op_a:
                continue
            cand = _overwrite(seed, off, op_b)
            if ctx.fresh(cand):
                return cand
        return None


class TokenAdvisor:
    """Newest dictionary token written at a comparison field offset."""

    name = "token"

    def propose(self, seed: bytes, ctx: AdvisorCtx) -> bytes | None:
        meta = ctx.seed_meta.get(seed)
        if not meta or not ctx.dictionary:
            return None

        # Newest tokens first: freshest cmplog finds.
        tokens = ctx.dictionary[-SCAN_MAX:][::-1]
        n = len(seed)
        for off in meta.get("redqueen_offsets", ())[:SCAN_MAX]:
            if off >= n:
                continue
            for token in tokens:
                cand = _overwrite(seed, off, token)
                if ctx.fresh(cand):
                    return cand
        return None


class CondAdvisor:
    """CondStmt: swap the first unsolved branch's operand found in the seed."""

    name = "cond"

    def propose(self, seed: bytes, ctx: AdvisorCtx) -> bytes | None:
        if not seed:
            return None

        conds = ctx.conds()
        for i in range(min(SCAN_MAX, len(conds))):
            base = conds[i].base
            needle = base.op_a[: base.width]
            idx = seed.find(needle) if needle else -1
            if idx == -1:
                continue
            cand = _overwrite(seed, idx, base.op_b[: base.width])
            if ctx.fresh(cand):
                return cand
        return None


def default_advisors() -> list[Advisor]:
    return [I2SAdvisor(), TokenAdvisor(), CondAdvisor()]


class AdvisorStage:
    """Plans, dedups and queues each stage's advised moves; keeps stats."""

    def __init__(self, advisors: Sequence[Advisor]):
        self._advisors = list(advisors)[:MAX_ADVISORS]
        self._queue: deque[bytes] = deque()
        # Insertion-ordered dict as a bounded FIFO set of tried keys.
        self._seen: dict[int, None] = {}
        self._stages = 0
        self._execs = 0
        self._proposed = {a.name: 0 for a in self._advisors}
        self._admitted = {a.name: 0 for a in self._advisors}

    def fresh(self, cand: bytes) -> bool:
        """True when *cand* was not proposed before (within SEEN_MAX)."""
        return hash64(cand) not in self._seen

    def plan(self, seed: bytes, ctx: AdvisorCtx) -> list[str]:
        """Queue each advisor's move for *seed*; names of those queued, in order."""
        self._stages += 1
        names = []
        for advisor in self._advisors:
            cand = advisor.propose(seed, ctx)
            if cand is None or cand == seed or not self.fresh(cand):
                continue
            self._remember(cand)
            self._queue.append(cand)
            self._proposed[advisor.name] += 1
            names.append(advisor.name)
        return names

    def _remember(self, cand: bytes) -> None:
        self._seen[hash64(cand)] = None
        if len(self._seen) > SEEN_MAX:
            del self._seen[next(iter(self._seen))]

    def pop(self) -> bytes | None:
        return self._queue.popleft() if self._queue else None

    def credit(self, name: str, admitted: bool) -> None:
        """One advised exec by *name* ran; *admitted* when it was kept."""
        self._execs += 1
        if admitted:
            self._admitted[name] += 1

    def seen_size(self) -> int:
        return len(self._seen)

    @property
    def stats(self) -> dict:
        """stages, execs, and per advisor (proposed, admitted)."""
        per = {n: (self._proposed[n], self._admitted[n]) for n in self._proposed}
        return {"stages": self._stages, "execs": self._execs, **per}
