"""Multi-pattern byte scanning for the cmplog operand sweep.

Two consumers walk the *same* cmplog operand pool looking for the offsets at
which each operand occurs inside a seed:

- :func:`fuzzer_tool.core.colorizer.CmplogColorizer.colorize_from_cmplog`
- :func:`fuzzer_tool.core.weizz_tags.build_tag_map_from_cmplog`

Both did it the obvious way — one ``bytes.find`` loop per token — which is
``O(P·n)`` in the token count ``P`` and the seed length ``n``.  With
``CMPLOG_PAIRS_MAX = 5000`` the pool holds up to 10,000 tokens.

This module provides one Aho–Corasick automaton that answers all of them in a
single ``O(n)`` pass, and a :class:`TokenScanner` front end that decides which
of the two strategies to use.

Why a dispatch and not a straight replacement
---------------------------------------------
``bytes.find`` is a C scan; the automaton is a Python loop over bytes.  Below a
few hundred tokens the C scan wins outright, and no amount of tuning changes
that — it is a constant-factor difference in the language the inner loop runs
in.  Measured here (random binary seed, mixed 2/4/8-byte operands, offsets
verified identical to the ``find`` loop at every point):

===== ====== ========= ========= ======== ========
    n tokens   find ms  AC build  AC scan  scan  x
===== ====== ========= ========= ======== ========
 4096    128      0.23      0.35     0.43    0.54
 4096    512      0.89      1.37     0.64    1.39
16384    256      1.60      0.58     1.76    0.91
16384    512      3.06      1.18     2.07    1.48
16384   2048     12.64      5.52     3.05    4.14
65536    512     14.99      1.44     7.90    1.90
65536   2048     60.74      7.57    10.36    5.86
===== ====== ========= ========= ======== ========

So the dispatch is on the quantity that drives the cost — the **unique token
count**, because the automaton's scan is ``O(n)`` independent of it — and not
on a flag or on ``n``.

``AC_MIN_TOKENS`` is set at 512 rather than at the scan-only crossover near
256.  At 256 the automaton merely draws even once warm and *loses* on the pass
that pays the build; at 512 it wins whether or not the build is amortised.  The
threshold is deliberately the point where the port cannot regress, not the
point where it first breaks even.

A note on the build cost: the automaton is cached per operand pool by
:func:`scanner_for_pairs`, so in a campaign it is built once per pool version
and reused across every mutation that consults it.

Deduplication comes free.  The ``find`` loops rescanned duplicate tokens: a
pool of 2048 pairs holds ~4089 distinct values among 4096 tokens, and the old
code scanned all 4096.  Both backends here scan the distinct set.
"""

from collections import deque
from collections.abc import Iterable, Sequence

#: Unique-token count at or above which the automaton is used.  See the module
#: docstring for the measurements behind the value.
AC_MIN_TOKENS = 512

#: Cap on trie nodes.  The trie costs roughly one node per pattern byte that is
#: not shared as a prefix, and cmplog operands are near-random binary, so they
#: share almost nothing past depth one or two: node count tracks the *summed*
#: operand length.  Memcmp operands have no width cap in the collector, so a
#: pool carrying a few long blobs would otherwise dominate the trie.  Patterns
#: are inserted shortest-first and whatever does not fit falls back to
#: ``bytes.find`` — which is exactly the right side of the trade for a long
#: pattern, since one long token costs one C scan.
AC_MAX_NODES = 64_000


class AhoCorasick:
    """Aho–Corasick automaton over a set of byte patterns.

    Reports **every** occurrence, including overlapping ones, matching the
    semantics of the ``find(token, pos); pos = idx + 1`` loops this replaces.
    """

    __slots__ = ("_fail", "_goto", "_out", "patterns")

    def __init__(self, patterns: Iterable[bytes]) -> None:
        goto: list[dict[int, int]] = [{}]
        out: list[list[bytes]] = [[]]
        kept: list[bytes] = []

        for pat in patterns:
            if not pat:
                # The empty pattern would match at every position with zero
                # length; ``bytes.find`` loops skip it, so we do too.
                continue
            state = 0
            for ch in pat:
                nxt = goto[state].get(ch)
                if nxt is None:
                    goto.append({})
                    out.append([])
                    nxt = len(goto) - 1
                    goto[state][ch] = nxt
                state = nxt
            if not out[state]:
                kept.append(pat)
            if pat not in out[state]:
                out[state].append(pat)

        fail = [0] * len(goto)
        queue: deque[int] = deque()
        for state in goto[0].values():
            fail[state] = 0
            queue.append(state)
        while queue:
            r = queue.popleft()
            for ch, state in goto[r].items():
                queue.append(state)
                f = fail[r]
                while f and ch not in goto[f]:
                    f = fail[f]
                target = goto[f].get(ch, 0)
                # A depth-one state can otherwise take itself as its own
                # failure link, which loops the scan forever.
                fail[state] = 0 if target == state else target
                # BFS order guarantees the suffix state's outputs are already
                # merged, so one concatenation per state suffices.
                if out[fail[state]]:
                    out[state] = out[state] + out[fail[state]]

        self._goto = goto
        self._fail = fail
        self._out = out
        self.patterns = kept

    @property
    def node_count(self) -> int:
        """Number of trie states, including the root."""
        return len(self._goto)

    def find_all(self, data: bytes, min_len: int = 1) -> dict[bytes, list[int]]:
        """Map every pattern occurring in *data* to its ascending start offsets.

        Args:
            data: Buffer to scan.
            min_len: Ignore patterns narrower than this.  A one-byte pattern
                matches at a position for every occurrence of that byte, so on
                a zero-padded seed it alone can produce as many offsets as the
                seed is long.  The colorizer never wanted those (it skipped
                ``len(token) < 2``); filtering here keeps one shared automaton
                without making it pay for spans its caller will discard.
        """
        goto = self._goto
        fail = self._fail
        out = self._out
        state = 0
        found: dict[bytes, list[int]] = {}
        for i, ch in enumerate(data):
            while state and ch not in goto[state]:
                state = fail[state]
            state = goto[state].get(ch, 0)
            hits = out[state]
            if hits:
                for pat in hits:
                    width = len(pat)
                    if width < min_len:
                        continue
                    start = i - width + 1
                    hit = found.get(pat)
                    if hit is None:
                        found[pat] = [start]
                    else:
                        hit.append(start)
        return found


class TokenScanner:
    """Offset lookup for a fixed token set, over whichever backend is cheaper.

    Attributes:
        backend: ``"find"``, ``"aho-corasick"`` or ``"hybrid"`` — reported for
            tests and for the stats line, not consulted by the scan itself.
    """

    __slots__ = ("_automaton", "_scan_tokens", "backend", "tokens")

    def __init__(
        self,
        tokens: Iterable[bytes],
        *,
        min_tokens: int = AC_MIN_TOKENS,
        max_nodes: int = AC_MAX_NODES,
    ) -> None:
        # dict.fromkeys rather than set(): deduplicates while keeping a
        # deterministic order, so the trie and the fallback list are the same
        # from run to run for the same pool.
        self.tokens: list[bytes] = [t for t in dict.fromkeys(tokens) if t]

        if len(self.tokens) < min_tokens:
            self._automaton = None
            self._scan_tokens = self.tokens
            self.backend = "find"
            return

        # Shortest first: short operands are both the numerous ones and the
        # ones the automaton helps most with, and a long operand costs a
        # single C scan on the fallback path.
        ordered = sorted(self.tokens, key=len)
        budget = max_nodes
        accepted: list[bytes] = []
        spare: list[bytes] = []
        for tok in ordered:
            if budget - len(tok) < 0:
                spare.append(tok)
            else:
                budget -= len(tok)
                accepted.append(tok)

        self._automaton = AhoCorasick(accepted)
        self._scan_tokens = spare
        self.backend = "hybrid" if spare else "aho-corasick"

    def scan(self, data: bytes, min_len: int = 1) -> dict[bytes, list[int]]:
        """Map every known token occurring in *data* to its start offsets.

        Tokens that do not occur are absent from the result; callers should
        read it with ``.get(token, ())``.

        Args:
            data: Buffer to scan.
            min_len: Skip tokens narrower than this.  See
                :meth:`AhoCorasick.find_all`.
        """
        if not data:
            return {}
        found: dict[bytes, list[int]] = (
            self._automaton.find_all(data, min_len) if self._automaton is not None else {}
        )
        n = len(data)
        for tok in self._scan_tokens:
            width = len(tok)
            if width > n or width < min_len:
                continue
            offsets: list[int] = []
            pos = 0
            while True:
                idx = data.find(tok, pos)
                if idx < 0:
                    break
                offsets.append(idx)
                pos = idx + 1
            if offsets:
                found[tok] = offsets
        return found


def tokens_from_pairs(pairs: Sequence[tuple[bytes, bytes]]) -> list[bytes]:
    """Flatten cmplog operand pairs into the token list both consumers scan."""
    out: list[bytes] = []
    for op_a, op_b in pairs:
        out.append(op_a)
        out.append(op_b)
    return out


# ── Scanner cache ────────────────────────────────────────────────────
#
# Keyed on the *identity* of the operand pool plus its length, and the pool
# object itself is retained.  Both halves are load-bearing:
#
# - Holding the reference is what makes the identity check exact.  CPython
#   reuses an address once the object at it is freed, so a bare ``id()`` can
#   match a pool that no longer exists; a pool that is still referenced cannot
#   have its address reused.  (Same hazard as the colorization cache key.)
# - The length is needed *because* the identity is stable: the collector grows
#   the pool in place with ``pairs.extend(...)`` and only rebinds it when
#   evicting.  So content changes show up either as a new object or as a new
#   length, and never as neither.
_cache_owner: object | None = None
_cache_len: int = -1
_cache_scanner: TokenScanner | None = None


def scanner_for_pairs(pairs: Sequence[tuple[bytes, bytes]]) -> TokenScanner:
    """Return a :class:`TokenScanner` over *pairs*, reusing the last automaton.

    Args:
        pairs: The cmplog operand pool.  Pass the collector's own sequence
            rather than a copy of it, or every call is a cache miss.

    Returns:
        A scanner covering both operands of every pair.
    """
    global _cache_owner, _cache_len, _cache_scanner
    if _cache_scanner is not None and _cache_owner is pairs and _cache_len == len(pairs):
        return _cache_scanner
    scanner = TokenScanner(tokens_from_pairs(pairs))
    _cache_owner = pairs
    _cache_len = len(pairs)
    _cache_scanner = scanner
    return scanner


def _reset_scanner_cache() -> None:
    """Drop the cached scanner.  For tests; production never needs it."""
    global _cache_owner, _cache_len, _cache_scanner
    _cache_owner = None
    _cache_len = -1
    _cache_scanner = None


__all__ = [
    "AC_MAX_NODES",
    "AC_MIN_TOKENS",
    "AhoCorasick",
    "TokenScanner",
    "scanner_for_pairs",
    "tokens_from_pairs",
]
