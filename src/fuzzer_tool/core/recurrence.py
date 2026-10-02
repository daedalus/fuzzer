"""Recurrence quantification analysis (RQA) over a categorical symbol stream.

A campaign trapped in a limit cycle repeats the same (seed, path) sequence.
RQA measures that on the recurrence plot of embedded m-grams::

    R[i, j] = 1  iff  s[i:i+m] == s[j:j+m]        (j > i)

    period-3 stream a b c a b c a b c            random stream
      . . . 1 . . 1 . .                            . . . . 1 . . .
        . . . 1 . . 1 .                              . . . . . . 1
          . . . 1 . . 1                                . 1 . . . .
    long diagonals: deterministic (DET ~ 1)      isolated points: DET ~ 0

- RR (recurrence rate): fraction of pairs (i < j) that recur.
- DET (determinism): fraction of recurrent points on diagonal lines of
  length >= l_min. A diagonal at lag p means "the stream repeated itself
  p steps later for l_min steps in a row".
- period: the lag in [1, n/2] with the highest recurrence rate.

Unlike Chao2, which reads any plateau as saturation, DET separates a
deterministic loop (high DET) from a stochastic plateau (recurring symbols,
short lines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rqa:
    """One RQA reading; all zero when the stream is too short."""

    rr: float
    det: float
    period: int


_EMPTY = Rqa(0.0, 0.0, 0)


def _embed(s: np.ndarray, embed: int, n: int) -> np.ndarray:
    """Upper-triangular m-gram recurrence matrix (n x n, lag >= 1)."""
    eq = s[:, None] == s[None, :]

    # m-grams recur iff every one of their m symbols does.
    r = eq[:n, :n].copy()
    for k in range(1, embed):
        r &= eq[k : k + n, k : k + n]
    return np.triu(r, 1)


def _line_points(r: np.ndarray, l_min: int) -> int:
    """Recurrent points lying on diagonal runs of length >= l_min."""
    n = r.shape[0]
    m = n - l_min + 1
    if m <= 0:
        return 0

    # starts[i, j]: a run of >= l_min begins at (i, j).
    starts = r[:m, :m].copy()
    for k in range(1, l_min):
        starts &= r[k : k + m, k : k + m]

    # Dilate each start back over the l_min points it covers.
    covered = np.zeros_like(r)
    for k in range(l_min):
        covered[k : k + m, k : k + m] |= starts
    return int(covered.sum())


def _period(r: np.ndarray) -> int:
    """Lag in [1, n/2] with the highest recurrence rate; 0 if none recur."""
    n = r.shape[0]
    i, j = np.nonzero(r)
    counts = np.bincount(j - i, minlength=n)
    rates = counts / np.maximum(n - np.arange(n), 1)

    half = rates[1 : n // 2 + 1]
    if half.size == 0 or half.max() <= 0.0:
        return 0
    return int(np.argmax(half)) + 1


def rqa(symbols: np.ndarray, embed: int, l_min: int) -> Rqa:
    """RR, DET and dominant period of a symbol stream.

    Args:
        symbols: 1-D integer array, oldest first.
        embed: m-gram length (embedding dimension).
        l_min: Shortest diagonal counted as deterministic.

    Returns:
        The reading; ``Rqa(0, 0, 0)`` when fewer than two m-grams exist.
    """
    s = np.asarray(symbols)
    n = len(s) - embed + 1
    if n < 2:
        return _EMPTY

    r = _embed(s, embed, n)
    points = int(r.sum())
    if points == 0:
        return _EMPTY

    pairs = n * (n - 1) // 2
    return Rqa(points / pairs, _line_points(r, l_min) / points, _period(r))
