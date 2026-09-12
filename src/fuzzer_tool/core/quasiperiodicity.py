"""Quasiperiodicity (shortest string cover) detection for corpus novelty.

Ported from OEIS A366160 (numbers whose binary expansion is not
quasiperiodic) — see
``docs/handover/handover_oeis_port_candidates_2026-09-12.md`` C2.

A string ``s`` is quasiperiodic if it is entirely covered by (possibly
*overlapping*) occurrences of some shorter string ``u`` (its cover). This is
distinct from ordinary periodicity, which requires the repeats to tile
exactly without overlap (``core/periodicity.py`` answers that question via
FFT autocorrelation). Quasiperiodicity is the more general, weaker
condition: a string can be quasiperiodic without being periodic at all.

This gives a *third*, structurally different novelty/redundancy signal
alongside the two already in the tree:

- ``core/corpus_compression.py`` (PPMD): does this seed fit the corpus's
  learned byte-level context model?
- ``core/periodicity.py`` (FFT autocorrelation): does this buffer look like
  N exact, non-overlapping repetitions of an L-byte record?
- This module: is this *specific seed* internally covered by repeated
  occurrences of some shorter substring, independent of any corpus model
  and independent of exact tiling?

A seed with a short cover is redundant in a sense neither of the other two
signals captures directly — e.g. "ababababX" (cover "ab", one leftover byte)
is not exactly periodic (doesn't tile), and may or may not stand out against
a corpus-wide PPMD model depending on what else is in the corpus, but it is
trivially quasiperiodic and therefore structurally simple.

Algorithm (Apostolico & Ehrenfeucht, 1993): every proper cover of a string
must also be a *border* of that string (a prefix that is also a suffix).
This bounds the search to the string's border chain — read off the KMP
prefix-function's failure links — rather than every possible substring
length, and each border-length candidate is validated in O(n) via a
Z-function-based interval-covering sweep. Candidates are tried shortest
first, so the first one that validates is the shortest cover, and no border
after it needs checking.

Superseded design note, recorded so it is not silently reproduced: an
earlier session identified ``core/bloom.py``'s Hamming-distance fuzzy dedup
as blind to near-duplicates with insertions/deletions and proposed a
Rabin-Karp rolling-hash content-defined-chunking (CDC) port as the fix.
Quasiperiodicity detection is an *alternative* angle on the same underlying
problem (internal redundant structure despite positional shifts), not an
addition to it — see the handover document's C2 entry for why building both
was explicitly left as an open decision rather than done here. This module
answers "how coverable is this one seed", which is the question actually
needed for a per-seed novelty *score*; CDC answers "which byte ranges are
shared across seeds", a different and complementary question for
cross-seed corpus minimization. Building CDC as well remains open.
"""

from __future__ import annotations

import hashlib
import logging

log = logging.getLogger(__name__)

# Worst-case cost is O(n^2) for a pathological string whose border chain has
# many candidates that all fail coverage before the true (or trivial) cover
# is found. In practice this collapses fast -- a string with many borders
# (e.g. long runs of one byte) has its *smallest* border succeed immediately,
# since border candidates are tried shortest-first -- but the cap exists as a
# backstop against the adversarial case, the same role PPMD_SAMPLE_BYTES
# plays in corpus_compression.py, just at a smaller size because this
# algorithm's worst case is quadratic rather than linear.
QP_SAMPLE_BYTES = 4096

# Cache entries are (digest -> float), cleared wholesale on overflow -- same
# policy and same justification as corpus_compression.py's PPMD_CACHE_MAX.
QP_CACHE_MAX = 4096


def _qp_cache_key(seed: bytes) -> str:
    """Stable digest for the sampled prefix of *seed*."""
    return hashlib.blake2b(seed[:QP_SAMPLE_BYTES], digest_size=16).hexdigest()


def prefix_function(s: bytes) -> list[int]:
    """KMP prefix (failure) function: pi[i] = length of the longest proper
    border of s[0:i+1] (a prefix of s[0:i+1] that is also its suffix)."""
    n = len(s)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def z_function(s: bytes) -> list[int]:
    """Z-array: z[i] = length of the longest common prefix of s and s[i:].

    z[0] is conventionally left as 0 (undefined/unused) rather than len(s).
    """
    n = len(s)
    z = [0] * n
    if n == 0:
        return z
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    return z


def _border_chain(pi: list[int]) -> list[int]:
    """All proper border lengths of the full string, ascending, excluding 0."""
    n = len(pi)
    if n == 0:
        return []
    borders = []
    b = pi[n - 1]
    while b > 0:
        borders.append(b)
        b = pi[b - 1]
    borders.reverse()
    return borders


def is_cover(s: bytes, c: int, z: list[int] | None = None) -> bool:
    """Does the length-``c`` prefix of *s* cover all of *s*?

    Coverage means every position of *s* falls inside some occurrence of
    ``s[0:c]``; occurrences may overlap. Uses the Z-array to find every
    occurrence of the prefix in O(n) and a greedy interval sweep to check
    the union covers ``[0, len(s))``.
    """
    n = len(s)
    if c <= 0 or c > n:
        return False
    if c == n:
        return True
    if z is None:
        z = z_function(s)
    covered_until = 0
    # Position 0 is always an occurrence of the prefix (trivially, of itself).
    for pos in (0,) + tuple(i for i in range(1, n) if z[i] >= c):
        if pos > covered_until:
            return False  # gap: nothing covers [covered_until, pos)
        covered_until = max(covered_until, pos + c)
        if covered_until >= n:
            return True
    return covered_until >= n


def shortest_cover_length(s: bytes) -> int:
    """Length of the shortest cover of *s* (proper cover if one exists, else n).

    Returns ``len(s)`` for the empty string and single-byte strings (no
    proper cover is possible below length 1) and as the "no proper cover
    found" fallback in general -- the whole string trivially covers itself.
    """
    n = len(s)
    if n <= 1:
        return n
    pi = prefix_function(s)
    borders = _border_chain(pi)
    if not borders:
        return n
    z = z_function(s)
    for c in borders:  # ascending, so the first success is the shortest
        if is_cover(s, c, z):
            return c
    return n


def is_quasiperiodic(s: bytes) -> bool:
    """Whether *s* has a proper cover shorter than itself."""
    return 0 < len(s) and shortest_cover_length(s) < len(s)


def cover_ratio(s: bytes) -> float:
    """``shortest_cover_length(s) / len(s)``, in (0, 1].

    Same direction as ``CorpusCompressor``'s PPMD ratio: low = redundant
    (short cover relative to the string), high = novel (no short internal
    cover; 1.0 means not quasiperiodic at all).
    """
    n = len(s)
    if n == 0:
        return 1.0
    return shortest_cover_length(s) / n


class QuasiperiodicityAnalyzer:
    """Per-seed cover-based novelty scoring, mirroring ``CorpusCompressor``'s
    interface (``enabled``, ``compute_seed_novelty``) so it wires into the
    same call sites the same way.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._ratios: dict[str, float] = {}

    def compute_seed_ratio(self, seed: bytes) -> float:
        """Cover ratio of the sampled prefix, memoised by digest."""
        if not self.enabled or not seed:
            return 1.0
        key = _qp_cache_key(seed)
        cached = self._ratios.get(key)
        if cached is not None:
            return cached
        sample = seed[:QP_SAMPLE_BYTES]
        try:
            ratio = cover_ratio(sample)
        except Exception:
            log.debug("quasiperiodicity: cover_ratio failed", exc_info=True)
            return 1.0
        if len(self._ratios) >= QP_CACHE_MAX:
            self._ratios.clear()
        self._ratios[key] = ratio
        return ratio

    def compute_seed_novelty(self, seed: bytes) -> float:
        """Novelty in [0, 1]: 1.0 = no short internal cover (novel/complex),
        near 0.0 = covered by a short repeated substring (redundant)."""
        ratio = self.compute_seed_ratio(seed)
        if ratio <= 0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio
