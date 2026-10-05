"""Grimoire: structure inference without a grammar (Blazytko et al., USENIX Sec '19).

Generalization finds which bytes of an input a target does not care about::

    "junk;if(x);more"   probe = "still reaches the novel edges?"
      -> GAP "if(x)" GAP

A GAP stands for "any bytes": removing them kept the novelties. What is left
are the tokens the path depends on. Three mutators recombine them across
inputs, with no grammar supplied:

    extend    seed + tokens of another generalized input (either side)
    recurse   a GAP of the seed replaced by another generalized input,
              repeated: "<GAP>" + "x" -> "<x>" -> "<xx>"
    replace   a token found in the seed swapped for another pooled token

Generalization is a three-stage removal search, each stage only dropping
bytes the probe confirms dead: fixed-size chunks, spans ending at a
delimiter, spans enclosed by a bracket pair. Running out of budget leaves
bytes live, so a short budget under-generalizes and never over-claims.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import compress

from fuzzer_tool.core.rand_pool import RandPool

# One element of a generalized input: literal bytes, or GAP for "anything".
GAP = None
Item = bytes | None
Items = tuple[Item, ...]

# candidate -> does it still reach every novelty of the original input.
Probe = Callable[[bytes], bool]

# Bounds (Hard Rule 54): generalization is O(len) probes, so long seeds are
# not worth it, and both tables are LRU/FIFO-capped.
GENERALIZE_MAX_LEN = 4096
MAX_SEEDS = 256
MAX_STRINGS = 1024
# One-byte tokens match everywhere; the swap would be noise.
MIN_TOKEN = 2

# Chunk sizes of the offsets stage, large to small (size 1 is the exact pass).
CHUNK_SIZES = (255, 127, 63, 31, 15, 7, 3, 1)
DELIMITERS = b".;,\n\r# "
BRACKETS = ((b"(", b")"), (b"[", b"]"), (b"{", b"}"), (b"<", b">"), (b"'", b"'"), (b'"', b'"'))
# Closers tried per opener, nearest first, tried from the farthest of them.
MAX_CLOSER_TRIES = 8

# recurse: replacements per mutation, and string-swap pick attempts.
MAX_DEPTH = 8
REPLACE_TRIES = 8
# Coin: below it a side/occurrence is the first one, at or above the other.
COIN = 0.5


@dataclass(frozen=True)
class Generalized:
    """Result of one generalization: items start and end with GAP."""

    items: Items
    execs: int


def strip(items: Items) -> bytes:
    """Concatenate the literal tokens, dropping every GAP."""
    return b"".join(i for i in items if i is not GAP)


class _Gen:
    """Removal search state: which bytes of ``data`` are still live."""

    def __init__(self, data: bytes, probe: Probe, max_execs: int):
        self.data = data
        self._probe = probe
        self._max_execs = max_execs
        self.alive = bytearray(b"\x01") * len(data)
        self.live = len(data)
        self.execs = 0

    @property
    def spent(self) -> bool:
        return self.execs >= self._max_execs

    def run(self, cand: bytes) -> bool:
        self.execs += 1
        return self._probe(cand)

    def try_drop(self, start: int, end: int) -> bool:
        """Drop the live bytes of ``[start, end)`` if the probe still passes.

        Dead-only windows and windows that would empty the input cost no
        execution: nothing to learn, and ``b""`` is never probed.
        """
        if self.spent:
            return False
        dropped = self.alive.count(1, start, end)
        if dropped == 0 or dropped == self.live:
            return False

        kept = bytearray(self.alive)
        kept[start:end] = bytes(end - start)
        if not self.run(bytes(compress(self.data, kept))):
            return False

        self.alive = kept
        self.live -= dropped
        return True


def _offsets(g: _Gen) -> None:
    """Stage 1: fixed-size chunks, large to small."""
    n = len(g.data)
    for size in CHUNK_SIZES:
        for start in range(0, n, size):
            if g.spent:
                return
            g.try_drop(start, min(start + size, n))


def _delims(g: _Gen) -> None:
    """Stage 2: from the last cut up to and including each delimiter."""
    data = g.data
    n = len(data)
    for delim in DELIMITERS:
        start = 0
        found = False
        while start < n and not g.spent:
            idx = data.find(delim, start)
            if idx == -1:
                break
            found = True
            g.try_drop(start, idx + 1)
            start = idx + 1
        # Bytes after the last delimiter form the final window.
        if found and start < n:
            g.try_drop(start, n)


def _closers(data: bytes, close: bytes, after: int) -> list[int]:
    """Up to MAX_CLOSER_TRIES nearest ``close`` positions past ``after``."""
    out: list[int] = []
    pos = data.find(close, after + 1)
    while pos != -1 and len(out) < MAX_CLOSER_TRIES:
        out.append(pos)
        pos = data.find(close, pos + 1)
    return out


def _brackets(g: _Gen) -> None:
    """Stage 3: an opener through its farthest closer that keeps the path."""
    data = g.data
    n = len(data)
    for open_, close in BRACKETS:
        i = data.find(open_)
        while i != -1 and i < n and not g.spent:
            closers = _closers(data, close, i)
            if not closers:
                break
            end = next((j for j in reversed(closers) if g.try_drop(i, j + 1)), None)
            i = data.find(open_, (end if end is not None else i) + 1)


def _items_of(g: _Gen) -> Items:
    """Live runs as tokens, every dead run (and both ends) as one GAP."""
    data = g.data
    items: list[Item] = [GAP]
    i = g.alive.find(1)
    while i != -1:
        j = g.alive.find(0, i)
        if j == -1:
            j = len(data)
        items.append(data[i:j])
        items.append(GAP)
        i = g.alive.find(1, j)
    return tuple(items)


def generalize(data: bytes, probe: Probe, max_execs: int) -> Generalized | None:
    """Generalize ``data`` against ``probe`` within ``max_execs`` probes.

    None when the input is empty or too long, the budget is zero, or the
    input itself does not pass the probe (unstable: nothing to preserve).
    """
    if not data or len(data) > GENERALIZE_MAX_LEN or max_execs < 1:
        return None

    g = _Gen(data, probe, max_execs)
    if not g.run(data):
        return None

    _offsets(g)
    _delims(g)
    _brackets(g)
    return Generalized(_items_of(g), g.execs)


class GrimoireBook:
    """Generalized inputs and the token pool the three mutators draw from.

    Keyed by the input's bytes (<= GENERALIZE_MAX_LEN each, <= MAX_SEEDS
    entries). Every mutator returns None to decline.
    """

    def __init__(self, max_len: int):
        self._max_len = max_len
        self._items: OrderedDict[bytes, Items] = OrderedDict()
        self._strings: OrderedDict[bytes, None] = OrderedDict()
        self._donors: list[Items] | None = None
        self._pool: list[bytes] | None = None

    @property
    def seeds(self) -> int:
        return len(self._items)

    @property
    def string_count(self) -> int:
        return len(self._strings)

    def items_of(self, key: bytes) -> Items | None:
        return self._items.get(key)

    def add(self, key: bytes, items: Items) -> None:
        """Store ``items`` for ``key`` and pool its tokens."""
        self._items[key] = items
        self._items.move_to_end(key)
        while len(self._items) > MAX_SEEDS:
            self._items.popitem(last=False)

        for tok in items:
            if tok is GAP or len(tok) < MIN_TOKEN:
                continue
            self._strings[tok] = None
            self._strings.move_to_end(tok)
        while len(self._strings) > MAX_STRINGS:
            self._strings.popitem(last=False)

        self._donors = None
        self._pool = None

    def _donor_list(self) -> list[Items]:
        if self._donors is None:
            self._donors = list(self._items.values())
        return self._donors

    def _string_list(self) -> list[bytes]:
        if self._pool is None:
            self._pool = list(self._strings)
        return self._pool

    def extend(self, data: bytes, rng: RandPool) -> bytes | None:
        """``data`` plus the tokens of one generalized input, either side."""
        donors = self._donor_list()
        if not donors:
            return None

        extra = strip(rng.choice(donors))
        if not extra:
            return None
        if rng.random() < COIN:
            return (extra + data)[: self._max_len]
        return (data + extra)[: self._max_len]

    def recurse(self, key: bytes, rng: RandPool) -> bytes | None:
        """Replace GAPs of ``key``'s generalization by other generalized inputs."""
        base = self._items.get(key)
        if base is None or GAP not in base:
            return None

        donors = self._donor_list()
        items = list(base)
        size = sum(len(i) for i in items if i is not GAP)
        for _ in range(rng.randint(1, MAX_DEPTH)):
            if size >= self._max_len:
                break
            gaps = [i for i, tok in enumerate(items) if tok is GAP]
            at = rng.choice(gaps)
            donor = rng.choice(donors)
            items[at : at + 1] = donor
            size += sum(len(i) for i in donor if i is not GAP)
        return strip(tuple(items))[: self._max_len]

    def replace(self, data: bytes, rng: RandPool) -> bytes | None:
        """Swap a pooled token found in ``data`` for another pooled token."""
        pool = self._string_list()
        if len(pool) < 2:
            return None

        old = None
        for _ in range(REPLACE_TRIES):
            cand = rng.choice(pool)
            if cand in data:
                old = cand
                break
        if old is None:
            return None

        new = rng.choice(pool)
        if new == old:
            return None

        count = 1 if rng.random() < COIN else -1
        return data.replace(old, new, count)[: self._max_len]
