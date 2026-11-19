"""Suffix automaton over the corpus: repeated substrings -> dictionary tokens, zero execs.

``auto_dict`` learns tokens from byteflip traces, so only after the
deterministic stage drains a seed. Bytes repeated across the corpus are
format: magics, chunk tags, keywords. The automaton finds every repeat of
every seed in one O(n) pass, before the first exec::

    seed 1   .. 89 P N G 0d 0a 1a 0a 00 00 00 0d I H D R | w h ..
    seed 2   .. 89 P N G 0d 0a 1a 0a 00 00 00 0d I H D R | W H ..
             '--------------- repeat, then diverges ------'

A state is a set of substrings with the same end positions; ``count`` is
how many there are. A token is the longest string of a state that repeats
(count >= 2) and is right-maximal (>= 2 distinct next symbols), so a long
shared blob yields its edge, not a window per byte. Each seed ends in its own
separator symbol (256 + index): no repeat spans two seeds, and a repeat ending
at two seed ends is right-maximal.

Python build: 64 KiB in 0.39 s / 23 MB peak; 256 KiB in 1.85 s / 90 MB, so
input is capped at MAX_SAM_BYTES (Hard Rule 54).
"""

from __future__ import annotations

from array import array

import numpy as np

from fuzzer_tool.core.auto_dict import MAX_AUTO_TOKEN, MIN_AUTO_TOKEN

MAX_SAM_BYTES = 64 * 1024
MAX_SAM_TOKENS = 64
MIN_REPEATS = 2
_SEPARATOR = 256  # first separator symbol; byte values are 0..255


class SuffixAutomaton:
    """Generalized suffix automaton over several byte strings."""

    def __init__(self):
        self._next: list[dict[int, int]] = [{}]
        self._link = array("i", [-1])
        self._len = array("i", [0])
        self._end = array("i", [-1])  # one end offset in _text per state
        self._own = array("i", [0])  # prefixes (end positions) ending in this state
        self._text = bytearray()  # all seeds, one placeholder byte per separator
        self._seeds = 0
        self._counts: np.ndarray | None = None

    def add(self, data: bytes) -> None:
        """Append ``data`` and its own separator."""
        last = 0
        for c in data:
            last = self._extend(last, c)
        self._extend(last, _SEPARATOR + self._seeds)
        self._seeds += 1
        self._counts = None

    def _extend(self, last: int, c: int) -> int:
        """Online extension by symbol ``c``; returns the state of the new prefix.

        A seed repeating a known prefix reuses its state (or a clone of it):
        a fresh state there would duplicate one, with a suffix link of equal
        length, and the counts would drop positions.
        """
        nxt, link, ln = self._next, self._link, self._len
        self._text.append(c if c < _SEPARATOR else 0)
        end = len(self._text) - 1

        q = nxt[last].get(c)
        if q is not None:
            st = q if ln[last] + 1 == ln[q] else self._split(last, c, q)
            self._own[st] += 1
            return st

        cur = self._new(ln[last] + 1, end)
        self._own[cur] = 1
        p = last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
            return cur

        q = nxt[p][c]
        link[cur] = q if ln[p] + 1 == ln[q] else self._split(p, c, q)
        return cur

    def _split(self, p: int, c: int, q: int) -> int:
        """Clone ``q`` at length ``len(p) + 1``; redirect ``p``'s chain to it."""
        nxt, link = self._next, self._link
        cl = self._new(self._len[p] + 1, self._end[q])
        nxt[cl] = dict(nxt[q])
        link[cl] = link[q]
        while p != -1 and nxt[p].get(c) == q:
            nxt[p][c] = cl
            p = link[p]
        link[q] = cl
        return cl

    def _new(self, length: int, end: int) -> int:
        self._next.append({})
        self._link.append(0)
        self._len.append(length)
        self._end.append(end)
        self._own.append(0)
        return len(self._len) - 1

    def counts(self) -> np.ndarray:
        """Occurrences per state: own end positions summed up the suffix links."""
        if self._counts is not None:
            return self._counts

        ln = np.frombuffer(self._len, np.int32)
        cnt = np.frombuffer(self._own, np.int32).astype(np.int64)
        # One length level at a time, longest first: links are strictly
        # shorter, so a level's counts are final before they feed the next.
        # 3x the per-state loop (64 KiB corpus: 8 vs 25 ms).
        link = np.frombuffer(self._link, np.int32)
        order = np.argsort(-ln, kind="stable")
        order = order[order != 0]
        for level in np.split(order, np.flatnonzero(np.diff(ln[order])) + 1):
            np.add.at(cnt, link[level], cnt[level])
        self._counts = cnt
        return cnt

    def count(self, sub: bytes) -> int:
        """Occurrences of ``sub`` across all added strings."""
        s = 0
        for c in sub:
            s = self._next[s].get(c, -1)
            if s < 0:
                return 0
        return int(self.counts()[s])

    def repeats(self) -> list[tuple[bytes, int]]:
        """(token, count) per repeated right-maximal state, longest strings capped."""
        cnt = self.counts()
        ln, link, end, text = self._len, self._link, self._end, self._text
        out = []
        for s in np.flatnonzero(cnt >= MIN_REPEATS).tolist():
            size = min(ln[s], MAX_AUTO_TOKEN)
            # Shorter than the state's range: that string belongs to an ancestor.
            if size < MIN_AUTO_TOKEN or size <= ln[link[s]] or len(self._next[s]) < 2:
                continue
            tok = bytes(text[end[s] - size + 1 : end[s] + 1])
            if tok.count(tok[:1]) == len(tok):
                continue
            out.append((tok, int(cnt[s])))
        return out


def corpus_tokens(seeds, budget: int = MAX_SAM_BYTES, limit: int = MAX_SAM_TOKENS) -> list[bytes]:
    """Up to ``limit`` repeated tokens of ``seeds``, best first.

    Seeds are read in order, the last one truncated to fit ``budget`` bytes.
    Rank: count x length (more repeats and longer tokens carry more format).
    """
    sam = SuffixAutomaton()
    left = budget
    for data in seeds:
        if left <= 0:
            break
        sam.add(bytes(data[:left]))
        left -= len(data)

    ranked = sorted(sam.repeats(), key=lambda t: (-t[1] * len(t[0]), t[0]))
    return [tok for tok, _ in ranked[:limit]]
