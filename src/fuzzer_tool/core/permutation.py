"""Permutation structure: cycle type, sign, Steinhaus-Johnson-Trotter walk.

A permutation is a sequence ``perm`` with ``perm[i]`` the image of ``i``::

    perm = [1, 2, 0, 4, 3, 5]   0->1->2->0, 3->4->3, 5
    cycle_type -> [3, 2, 1]     perm_sign -> (-1)**(6 - 3) = -1

Uses: oracles for ``cycle_lock`` (single n-cycle) and ``_swap_tuple``'s
parity trap (an odd m-cycle is even, so rotations alone stay in A_n), and
SJT brute force for job-scheduling tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

_LEFT = -1


def cycle_type(perm: Sequence[int]) -> list[int]:
    """Cycle lengths of *perm*, in order of each cycle's smallest index.

    Raises:
        ValueError: *perm* is not a permutation of ``range(len(perm))``.
    """
    n = len(perm)
    if sorted(perm) != list(range(n)):
        raise ValueError(f"not a permutation of range({n})")

    seen = bytearray(n)
    lengths = []
    for start in range(n):
        if seen[start]:
            continue

        length = 0
        i = start
        while not seen[i]:
            seen[i] = 1
            length += 1
            i = perm[i]
        lengths.append(length)
    return lengths


def perm_sign(perm: Sequence[int]) -> int:
    """+1 for an even permutation, -1 for odd: ``(-1)**(n - cycles)``."""
    parity = (len(perm) - len(cycle_type(perm))) & 1
    return -1 if parity else 1


def sjt_swaps(n: int) -> Iterator[int]:
    """Yield i for each adjacent swap (i, i+1) of an SJT walk (Even's rule).

    Starting from the identity, the n! - 1 swaps visit every permutation
    once. E.g. n=3: 012 -> 021 -> 201 -> 210 -> 120 -> 102.
    """
    if n < 1:
        return

    perm = list(range(n))
    pos = list(range(n))
    dirs = [_LEFT] * n
    while True:
        # Largest mobile value: its neighbour in its direction is smaller.
        mobile = -1
        for v in range(n - 1, -1, -1):
            q = pos[v] + dirs[v]
            if 0 <= q < n and perm[q] < v:
                mobile = v
                break
        if mobile < 0:
            return

        p = pos[mobile]
        q = p + dirs[mobile]
        other = perm[q]
        perm[p], perm[q] = other, mobile
        pos[mobile], pos[other] = q, p
        yield min(p, q)

        # Values above the mover reverse direction.
        for v in range(mobile + 1, n):
            dirs[v] = -dirs[v]
