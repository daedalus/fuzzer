"""PositionTokenScheduler: land mutations inside dictionary-token occurrences.

The fuzzer's dictionary (``Fuzzer.dictionary``: ``-x`` files, the target
profile's token channels, and cmplog operands folded in at run time) names
the byte strings the target compares against. Where a seed *contains* one of
them is where a keyword, a magic or a tag sits::

    seed      . . . . I H D R . . . . I D A T . . . . I E N D
    tokens    IHDR, IDAT, IEND
    pick      one occurrence uniformly, then one byte inside it

``cmplog`` covers the same ground from live comparison operands, but only
while cmplog runs; this arm needs only a token list, so it also works on
targets built without cmplog.

Tokens shorter than ``MIN_TOKEN_LEN`` are skipped: one-byte tokens match
everywhere and carry no position signal. Scanning is one pass of
``core/aho_corasick.TokenScanner`` over the first ``SCAN_CAP`` bytes, cached
per seed in an LRU of ``MAX_SEEDS``; at most ``MAX_MATCHES`` occurrences are
kept, evenly spaced over the seed so the tail is not dropped.

The dictionary grows (cmplog appends) and is pruned at run time. Rebuilding
the automaton on every change would put a trie build on the hot path, so the
scanner is rebuilt at most once per ``REBUILD_EVERY`` proposals, and only
when the list changed (identity or length).

Passive: ``record()`` is a no-op and nothing is persisted. Declines (``None``,
which the arena turns into a uniform offset charged to this arm) when no
token occurs inside the live buffer, and with probability ``EPSILON``.

Tracker-style arm (``PositionArena._add_trackers``): joins the pool only
while ``active()`` holds, i.e. the dictionary is non-empty.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Sequence

import xxhash

from fuzzer_tool.core.aho_corasick import TokenScanner
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

EPSILON = 0.1  # uniform escape (decline) probability
MIN_TOKEN_LEN = 2  # shorter tokens match everywhere
MAX_SEEDS = 256  # LRU bound on the per-seed match cache
MAX_MATCHES = 512  # occurrences kept per seed
SCAN_CAP = 64 * 1024  # bytes scanned per seed
REBUILD_EVERY = 1024  # proposals between scanner rebuilds


class _Matches:
    """One seed's occurrences, sorted by start; ``gen`` = scanner build."""

    __slots__ = ("gen", "starts", "widths")

    def __init__(self, gen: int, starts: array, widths: array) -> None:
        self.gen = gen
        self.starts = starts
        self.widths = widths


class PositionTokenScheduler:
    """Propose a byte inside a dictionary token found in the seed."""

    name = "token"

    def __init__(self, rng: RandPool, tokens_of: Callable[[], Sequence[bytes]]) -> None:
        self._rng = rng
        self._tokens_of = tokens_of
        self._scanner: TokenScanner | None = None
        self._sig: tuple[int, int] = (0, -1)
        self._since_build = 0
        self.builds = 0
        self._cache: OrderedDict[int, _Matches] = OrderedDict()

    def active(self) -> bool:
        """Arena gate: the dictionary holds at least one token."""
        return bool(self._tokens_of())

    # -- scanning -------------------------------------------------------------

    def _refresh(self, tokens: Sequence[bytes]) -> None:
        """Rebuild the scanner if the list changed and the window elapsed."""
        self._since_build += 1
        sig = (id(tokens), len(tokens))
        stale = sig != self._sig and self._since_build >= REBUILD_EVERY
        if self._scanner is not None and not stale:
            return

        self._scanner = TokenScanner(t for t in tokens if len(t) >= MIN_TOKEN_LEN)
        self._sig = sig
        self._since_build = 0
        self.builds += 1

    def _scan(self, data: bytes) -> _Matches:
        """All occurrences, sorted, subsampled evenly to MAX_MATCHES."""
        found = self._scanner.scan(data[:SCAN_CAP], min_len=MIN_TOKEN_LEN)
        occ = sorted((s, len(tok)) for tok, starts in found.items() for s in starts)

        n = len(occ)
        if n > MAX_MATCHES:
            occ = [occ[i * n // MAX_MATCHES] for i in range(MAX_MATCHES)]

        starts = array("I", (s for s, _ in occ))
        widths = array("H", (w for _, w in occ))
        return _Matches(self.builds, starts, widths)

    def _matches(self, data: bytes) -> _Matches:
        key = xxhash.xxh3_64_intdigest(data)
        hit = self._cache.get(key)
        if hit is not None and hit.gen == self.builds:
            self._cache.move_to_end(key)
            return hit

        entry = self._scan(data)
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_SEEDS:
            self._cache.popitem(last=False)
        return entry

    # -- introspection (tests, stats) ----------------------------------------

    def match_starts(self, data: bytes) -> list[int]:
        """Cached occurrence starts for *data* (empty before the first scan)."""
        hit = self._cache.get(xxhash.xxh3_64_intdigest(data))
        return list(hit.starts) if hit is not None else []

    def cached_seeds(self) -> int:
        return len(self._cache)

    # -- protocol -------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        tokens = self._tokens_of()
        if not tokens:
            return None

        self._refresh(tokens)
        m = self._matches(data)

        # Occurrences starting inside a shrunk buffer are a prefix.
        k = bisect_left(m.starts, buf_len)
        if k == 0:
            return None

        if self._rng.random() < EPSILON:
            return None

        i = self._rng.randint(0, k - 1)
        start = m.starts[i]
        width = min(m.widths[i], buf_len - start)
        return start + self._rng.randint(0, width - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: targets come from the dictionary, not outcomes."""
