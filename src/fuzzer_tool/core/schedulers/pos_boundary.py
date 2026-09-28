"""PositionBoundaryScheduler: land mutations on field boundaries.

Content-only prior. Token starts, class changes and run edges are enriched
for interesting mutations whatever the feedback says (a length field starts
right after a delimiter, a payload starts where the byte class flips), so
this arm helps from the very first pick on a brand-new seed, like
``fibonacci`` but informed: the learners (``burn_front``, ``kl_ducb``,
``fractal``) start from zero on every seed and learn only from coverage.

Each *boundary* ``i`` (the gap between ``b[i-1]`` and ``b[i]``, ``i >= 1``)
is scored once per seed as the sum of::

    class change     cls(b[i]) != cls(b[i-1])                     +1.0
    delimiter start  b[i-1] in DELIMS and b[i] != b[i-1]          +1.0
    entropy step     |H(b[i:i+W]) - H(b[i-W:i])| / 4, W = 16      <= 1.0
    run edge         start/end of a run of >= 4 equal bytes       +0.5
    4-byte aligned   i % 4 == 0                                   +0.25

``H`` is the Shannon entropy in bits of the window (max ``log2(W) = 4``, hence
the ``/ 4``); windows must fit inside the scanned prefix, so the first and
last ``W`` bytes carry no entropy term. The delimiter term requires
``b[i] != b[i-1]``, so it fires on the byte *after* a delimiter run, not on
every byte inside one (a NUL or space padding run would otherwise fill the
whole top-K with ties; the run-edge term already marks the run's ends).

Only the ``TOP_K`` best boundaries per seed are kept (ties broken towards the
lower offset, so the table is deterministic). ``propose`` picks one with
probability proportional to its score and adds a jitter in ``{-1, 0, +1}``:
an offset at a boundary is where a field starts, and the byte before it is
often the previous field's tail. Candidates past a shrunken live buffer are
dropped, not clamped, so they do not pile onto its last byte.

Cost: one O(min(n, SCAN_CAP)) numpy pass the first time a seed is seen, then
a dict lookup; the tables live in an LRU of ``MAX_SEEDS`` seeds. Only the first
``SCAN_CAP`` bytes are scored; for a longer seed the unscored tail gets a
uniform draw with probability ``(n - SCAN_CAP) / n`` (its share of the seed).

Stateless in the learning sense: ``record()`` is a no-op and nothing is
persisted (the cache is derived). ``EPSILON`` of the draws decline (``None``,
which the arena turns into a uniform offset charged to this arm) so a wrong
prior cannot lock the arm onto a fixed subset. On high-entropy seeds
(compressed streams) boundaries are noise and the arm degrades to uniform,
which the Elo rating will show.

Note: ``core/mutations/fractal_voronoi.py`` has a ``_is_boundary`` but it is
the boundary of a hash-driven Voronoi partition of the index space, not a
byte-content detector, so there is nothing to reuse from it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bytecls import BYTE_CLASS
from fuzzer_tool.core.schedulers.pos_base import Outcome

EPSILON = 0.15  # uniform escape (decline) probability
MAX_SEEDS = 256  # LRU bound on cached per-seed tables
SCAN_CAP = 65536  # bytes scored per seed; the rest is drawn uniformly
TOP_K = 256  # boundaries kept per seed
WINDOW = 16  # entropy window, bytes
RUN_MIN = 4  # shortest run of equal bytes whose edges count
ENTROPY_MAX_BITS = 4.0  # log2(WINDOW): normaliser for the entropy step
W_CLASS = 1.0
W_DELIM = 1.0
W_RUN = 0.5
W_ALIGN = 0.25
_CHUNK = 8192  # windows per entropy batch: bounds the (chunk, W, W) temporary

#: Bytes after which a field usually starts: NUL, LF, CR, space and the
#: common structural punctuation ``, : ; = / < > " { } [ ]``.
DELIMS = frozenset(b"\x00\n\r ,:;=/<>\"{}[]")
CLASS_LUT = np.frombuffer(BYTE_CLASS, dtype=np.uint8)  # shared with pos_context
_DELIM_LUT = np.zeros(256, dtype=bool)
_DELIM_LUT[list(DELIMS)] = True


def _window_entropy(arr: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of every ``WINDOW``-byte window of *arr*.

    ``out[j] = H(arr[j:j+WINDOW])``, length ``len(arr) - WINDOW + 1``. Uses
    ``H = log2(W) - mean_j(log2(c_j))`` with ``c_j`` the number of window
    bytes equal to byte ``j`` (exact, and vectorised as a pairwise compare).
    """
    m = len(arr) - WINDOW + 1
    if m <= 0:
        return np.zeros(0, dtype=np.float64)
    wins = np.lib.stride_tricks.sliding_window_view(arr, WINDOW)
    out = np.empty(m, dtype=np.float64)
    log_w = np.log2(WINDOW)
    for lo in range(0, m, _CHUNK):
        w = wins[lo : lo + _CHUNK]
        counts = (w[:, :, None] == w[:, None, :]).sum(axis=2)
        out[lo : lo + len(w)] = log_w - np.log2(counts).mean(axis=1)
    return out


def score_boundaries(data: bytes) -> np.ndarray:
    """Per-offset boundary score of the first ``SCAN_CAP`` bytes of *data*.

    ``score[i]`` scores the gap before byte ``i``; ``score[0]`` is 0 (the
    start of the file is not a boundary between two bytes).
    """
    arr = np.frombuffer(data[:SCAN_CAP], dtype=np.uint8)
    m = len(arr)
    score = np.zeros(m, dtype=np.float64)
    if m < 2:
        return score

    cur, prev = arr[1:], arr[:-1]
    differs = cur != prev
    s = np.zeros(m - 1, dtype=np.float64)  # s[k] scores boundary i = k + 1

    # class change
    s += W_CLASS * (CLASS_LUT[cur] != CLASS_LUT[prev])
    # delimiter start: the byte after a delimiter, not inside a delimiter run
    s += W_DELIM * (_DELIM_LUT[prev] & differs)

    # run edges: starts/ends of maximal runs of >= RUN_MIN equal bytes
    starts = np.flatnonzero(np.concatenate(([True], differs)))  # run start offsets
    lengths = np.diff(np.concatenate((starts, [m])))
    long_runs = lengths >= RUN_MIN
    run_start_at = np.zeros(m + 1, dtype=bool)  # boundary i begins a long run
    run_end_at = np.zeros(m + 1, dtype=bool)  # boundary i ends a long run
    run_start_at[starts[long_runs]] = True
    run_end_at[(starts + lengths)[long_runs]] = True
    s += W_RUN * (run_start_at[1:m] | run_end_at[1:m])

    # entropy step across the boundary (needs a full window on each side)
    if m >= 2 * WINDOW:
        h = _window_entropy(arr)  # h[j] = H(arr[j:j+W])
        idx = np.arange(WINDOW, m - WINDOW + 1)  # boundaries with both windows
        step = np.abs(h[idx] - h[idx - WINDOW]) / ENTROPY_MAX_BITS
        s[idx - 1] += np.minimum(step, 1.0)

    # 4-byte alignment
    s += W_ALIGN * ((np.arange(1, m) % 4) == 0)

    score[1:] = s
    return score


def top_boundaries(data: bytes) -> tuple[list[int], list[float]]:
    """The ``TOP_K`` best-scoring offsets of *data* and their scores.

    Positive scores only; ordered by descending score, ties towards the lower
    offset (deterministic for a given seed).
    """
    score = score_boundaries(data)
    cand = np.flatnonzero(score > 0.0)
    if len(cand) == 0:
        return [], []
    order = np.lexsort((cand, -score[cand]))[:TOP_K]
    picked = cand[order]
    return [int(o) for o in picked], [float(score[o]) for o in picked]


class PositionBoundaryScheduler:
    """Propose offsets at content-derived field boundaries."""

    name = "boundary"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        # seed key -> (offsets, scores); derived, never persisted
        self._cache: OrderedDict[int, tuple[list[int], list[float]]] = OrderedDict()

    def _targets(self, data: bytes) -> tuple[list[int], list[float]]:
        key = xxhash.xxh3_64_intdigest(data)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        table = top_boundaries(data)
        self._cache[key] = table
        while len(self._cache) > MAX_SEEDS:
            self._cache.popitem(last=False)
        return table

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None
        if self._rng.random() < EPSILON:
            return None

        n = len(data)
        # The unscored tail of a long seed keeps its share of the draws.
        if n > SCAN_CAP and self._rng.random() < (n - SCAN_CAP) / n and buf_len > SCAN_CAP:
            return self._rng.randint(SCAN_CAP, buf_len - 1)

        offsets, scores = self._targets(data)
        if not offsets:
            return None
        if offsets[0] >= buf_len or max(offsets) >= buf_len:
            # Live buffer shrank below some sites: drop them rather than
            # clamping them all onto the last byte.
            keep = [i for i, o in enumerate(offsets) if o < buf_len]
            if not keep:
                return None
            offsets = [offsets[i] for i in keep]
            scores = [scores[i] for i in keep]

        idx = self._rng.weighted_choice(range(len(offsets)), scores)
        pos = offsets[idx] + self._rng.randint(-1, 1)
        return max(0, min(pos, buf_len - 1))

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: the arm is content-only and learns nothing from outcomes."""
