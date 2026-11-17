"""Gramatron-style walk mutations over a Grammar (HexHive Gramatron, ISSTA'21).

Gramatron turns a grammar into an automaton so an input is a *walk*: one
choice per transition. Here the automaton is the leftmost-derivation stack
machine of ``core/grammar.Grammar``, and a walk is its choice sequence.

    stack [expr]  pop expr  choice 0 -> push term "+" expr     walk [0,
    stack [term "+" expr]   pop term  choice 1 -> push num            1,
    ...                                                              ...]

Replay treats each choice as a hint, taken mod the alternatives live at
that step, so ANY walk (spliced, repeated, garbage) expands to a
grammatical input. Mutations (Gramatron §4):

    random   keep walk[:i], regenerate the rest              (regen_from)
    splice   w1[:i] + w2[j:] where both exact states match   (splice_at)
    recurse  repeat a segment that returns to its own symbol state

A state is the symbol being expanded plus the pending stack. The exact form
(with depths) makes a splice reproduce the donor's tail byte for byte; the
depth-free form lets a recursive rule meet itself, which recursion needs.

Walks of inputs this module produced are cached per output (LRU, bounded),
so a corpus seed it generated can be mutated as a walk later.
"""

from __future__ import annotations

from fuzzer_tool.core.grammar import GENERATION_BYTE_CAP, Grammar
from fuzzer_tool.core.hashing import hash64
from fuzzer_tool.core.lru import LRUCache

MAX_STEPS = 4096  # choices per walk: bounds replay of any (hostile) walk
CACHE_SIZE = 4096  # remembered walks (memory bound, Hard Rule 54)
MAX_REPEAT_SPAN = 16  # repeat counts offered above lo: {0,1000} stays sane
MAX_RECURSE_COPIES = 4  # extra copies of a recursive segment

_UNKNOWN_LEAF = b"?"  # Grammar's leaf for depth-exhausted / undefined refs

# Stack symbols. ("lit", bytes) | ("ref", name, depth) | ("rep", name, lo, hi, depth)
_LIT, _REF, _REP = "lit", "ref", "rep"


class _Trace:
    """What one replay produced: choices taken, output, per-step states/offsets."""

    __slots__ = ("choices", "out", "exact", "loose", "offsets")

    def __init__(self) -> None:
        self.choices: list[int] = []
        self.out = bytearray()
        self.exact: list[int] = []  # state hash with depths (splice)
        self.loose: list[int] = []  # state hash without depths (recurse)
        self.offsets: list[int] = []  # output length before each step


class Gramatron:
    """Walk engine plus the walk cache for one grammar."""

    def __init__(self, grammar: Grammar, cache_size: int = CACHE_SIZE) -> None:
        self._g = grammar
        self.grammar = grammar
        self.cache_size = cache_size
        self._cache: LRUCache = LRUCache(cache_size)
        self._short: dict[tuple[str, int], int] = {}  # (rule, depth) -> alternative

    def __len__(self) -> int:
        return len(self._cache)

    # ── cache ──

    def remember(self, out: bytes, walk: list[int]) -> None:
        self._cache[hash64(out)] = walk

    def walk_for(self, data: bytes) -> list[int] | None:
        return self._cache.get(hash64(data))

    def donor(self, rng) -> list[int] | None:
        """A random remembered walk (splice donor); None when the cache is empty."""
        if not self._cache:
            return None
        return self._cache[rng.choice(list(self._cache.keys()))]

    # ── replay ──

    def expand(self, walk: list[int]) -> bytes:
        """Deterministic replay: hints first, then alternative 0 to finish."""
        return bytes(self._run(walk, None).out)

    def trace(self, walk: list[int]) -> tuple[list[int], list[int]]:
        """(exact state hashes, output offsets) per step of *walk*."""
        t = self._run(walk, None)
        return t.exact, t.offsets

    def random_walk(self, rng) -> tuple[list[int], bytes]:
        return self._result(self._run([], rng))

    # ── mutations ──

    def regen_from(self, walk: list[int], i: int, rng) -> tuple[list[int], bytes]:
        """Keep the first *i* choices, draw the rest at random."""
        return self._result(self._run(walk[:i], rng))

    def mutate_random(self, walk: list[int], rng) -> tuple[list[int], bytes]:
        return self.regen_from(walk, rng.randrange(len(walk) + 1), rng)

    def splice_points(self, w1: list[int], w2: list[int]) -> list[tuple[int, int]]:
        """Every (i, j) where w1's step i and w2's step j are in the exact same state."""
        s1, s2 = self._run(w1, None).exact, self._run(w2, None).exact
        at: dict[int, list[int]] = {}
        for j, st in enumerate(s2):
            at.setdefault(st, []).append(j)
        return [(i, j) for i, st in enumerate(s1) for j in at.get(st, ())]

    def splice_at(self, w1: list[int], w2: list[int], i: int, j: int) -> tuple[list[int], bytes]:
        return self._result(self._run(w1[:i] + w2[j:], None))

    def mutate_splice(self, w1: list[int], w2: list[int], rng) -> tuple[list[int], bytes] | None:
        points = self.splice_points(w1, w2)
        if not points:
            return None
        i, j = rng.choice(points)
        return self.splice_at(w1, w2, i, j)

    def mutate_recurse(self, walk: list[int], rng) -> tuple[list[int], bytes] | None:
        """Repeat walk[i:j] where step j re-enters step i's (depth-free) state."""
        seen: dict[int, int] = {}
        loops: list[tuple[int, int]] = []
        for j, st in enumerate(self._run(walk, None).loose):
            i = seen.get(st)
            if i is not None:
                loops.append((i, j))
            seen[st] = j
        if not loops:
            return None

        i, j = rng.choice(loops)
        copies = rng.randint(1, MAX_RECURSE_COPIES)
        grown = walk[:i] + walk[i:j] * (copies + 1) + walk[j:]
        return self._result(self._run(grown, rng))

    # ── the stack machine ──

    def _result(self, t: _Trace) -> tuple[list[int], bytes]:
        return t.choices, bytes(t.out)

    def _run(self, walk: list[int], rng) -> _Trace:
        t = _Trace()
        start = self._g.start_rule()
        if start is None:
            return t

        stack: list[tuple] = [(_REF, start, self._g.max_depth)]
        while stack and len(t.out) < GENERATION_BYTE_CAP:
            sym = stack.pop()
            if sym[0] == _LIT:
                t.out += sym[1]
                continue

            n = self._arity(sym)
            if n == 0:
                t.out += _UNKNOWN_LEAF
                continue

            # Past MAX_STEPS (or out of hints with no rng): finish on the
            # alternative that terminates soonest, so the output stays whole.
            k = len(t.choices)
            if k >= MAX_STEPS:
                self._push(stack, sym, self._shortest(sym))
                continue

            self._record(t, sym, stack)
            if k < len(walk):
                c = walk[k] % n
            else:
                c = rng.randrange(n) if rng is not None else self._shortest(sym)
            t.choices.append(c)
            self._push(stack, sym, c)
        return t

    def _shortest(self, sym: tuple) -> int:
        """Choice that ends soonest: fewest repeats, or the least-deep alternative."""
        if sym[0] == _REP:
            return 0
        key = (sym[1], sym[2])
        c = self._short.get(key)
        if c is None:
            c = self._short[key] = self._g.shortest_alternative(sym[1], sym[2])
        return c

    def _arity(self, sym: tuple) -> int:
        """Choices live at *sym*; 0 means the b"?" leaf."""
        if sym[0] == _REF:
            alts = self._g.alternatives(sym[1], sym[2])
            return len(alts) if alts else 0
        _, name, lo, hi, depth = sym
        top = min(self._g.repeat_upper(name, lo, hi, depth), lo + MAX_REPEAT_SPAN)
        return top - lo + 1

    def _push(self, stack: list[tuple], sym: tuple, c: int) -> None:
        if sym[0] == _REP:
            _, name, lo, _hi, depth = sym
            stack.extend([(_REF, name, depth - 1)] * (lo + c))
            return

        _, name, depth = sym
        alt = self._g.alternatives(name, depth)[c]
        for tok in reversed(alt):
            if tok[0] == "lit":
                stack.append((_LIT, tok[1]))
            elif tok[0] == "ref":
                stack.append((_REF, tok[1], depth - 1))
            else:  # ("repeat", name, lo, hi): count decided at the enclosing depth
                stack.append((_REP, tok[1], tok[2], tok[3], depth))

    @staticmethod
    def _record(t: _Trace, sym: tuple, stack: list[tuple]) -> None:
        t.offsets.append(len(t.out))
        t.exact.append(hash((sym, tuple(stack))))
        t.loose.append(hash((sym[:2], tuple(s[:2] for s in stack))))
