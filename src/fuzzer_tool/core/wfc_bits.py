"""Bitmask ``WaveGrid`` backend: one int per cell, optional real backtracking.

The bitmask N-Queens solver keeps its whole state in three ints and peels
candidates with ``avail & (avail - 1)``; recursion passes the state by value,
so undo is free. This module applies the same idiom to Wave Function
Collapse.

``BitWaveGrid`` is a drop-in ``WaveGrid`` subclass:

* **Bitmask propagation.** A cell's superposition is one Python int (bit ``t``
  set = tile ``t`` still possible). Arc pruning ``row &= support`` replaces the
  ``(n_tiles x n_tiles) @ (n_tiles,)`` matmul: ``support`` is the OR of one
  precomputed mask per neighbour tile, memoised on ``(direction,
  neighbour mask)``. Entropy is cached per distinct mask.
* **Same stream as ``WaveGrid``.** Every RNG draw, tie-break and float
  reduction mirrors the numpy implementation, so for a given seed and
  ``max_backtracks=0`` the output is identical (``tests/test_wfc_bits.py``).
* **Backtracking (opt-in).** ``run(max_backtracks=N)`` replaces
  restart-on-contradiction with depth-first search: observe a cell, propagate,
  and on a dead cell undo through a trail of ``(cell, old mask)`` pairs and try
  that cell's next tile, instead of throwing the whole wave away. The search is
  complete: it reports unsatisfiable only after exhausting every tile
  at every decision (modulo the AC-3 / work budgets, which still apply).

Callers keep pinning cells through ``superpositions`` before ``run()``; the
array is packed to ints on entry and unpacked on exit, so nothing else changes.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.wfc import (
    DEFAULT_AC3_BUDGET,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_WORK_BUDGET,
    AdjacencyTable,
    Tile,
    WaveGrid,
)

_DIRS = 4


def _bit_ids(mask: int) -> list[int]:
    """Set-bit positions of *mask*, ascending (the lowest-bit peel)."""
    ids: list[int] = []
    while mask:
        low = mask & -mask
        ids.append(low.bit_length() - 1)
        mask ^= low
    return ids


class BitWaveGrid(WaveGrid):
    """``WaveGrid`` with int-bitmask cells, memoised pruning, optional DFS."""

    def __init__(
        self,
        tiles: list[Tile],
        adjacency: AdjacencyTable,
        width: int,
        height: int = 1,
        seed=None,
    ):
        super().__init__(tiles, adjacency, width, height, seed)
        mat = self._adj_matrix
        assert mat is not None
        # _tiles_with[d][j] = mask of tiles t that may have tile j in direction d
        self._tiles_with: list[list[int]] = [
            [_mask_of(np.flatnonzero(mat[:, j, d])) for j in range(self.n_tiles)]
            for d in range(_DIRS)
        ]
        self._support_cache: dict[tuple[int, int], int] = {}
        self._entropy_cache: dict[int, float] = {}
        # (neighbour, direction-to-neighbour) per cell, in WaveGrid's order.
        self._nbr_dirs: list[list[tuple[int, int]]] = [
            [(n, self._dir_index(i, n)) for n in self._neighbors(i)] for i in range(self.n)
        ]
        self._bits: list[int] | None = None
        self._init_bits: list[int] = []
        self._trail: list[tuple[int, int]] | None = None
        self._nbytes = max(1, -(-self.n_tiles // 8))
        # Strict pruning also re-checks already-collapsed cells. WaveGrid skips
        # them (a pinned cell next to an incompatible pin goes unnoticed), which
        # is harmless for a best-effort sampler but breaks a complete search.
        self._strict = False

    # ── pack / unpack ───────────────────────────────────────────────

    def _pack(self) -> list[int]:
        packed = np.packbits(self.superpositions, axis=1, bitorder="little")
        return [int.from_bytes(r.tobytes(), "little") for r in packed]

    def _unpack(self) -> None:
        if self._bits is None:
            return
        raw = b"".join(m.to_bytes(self._nbytes, "little") for m in self._bits)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.n, self._nbytes)
        self.superpositions[:, :] = np.unpackbits(arr, axis=1, bitorder="little")[
            :, : self.n_tiles
        ].astype(bool)

    # ── state mutation (trail-aware) ────────────────────────────────

    def _set(self, idx: int, new: int) -> None:
        assert self._bits is not None
        if self._trail is not None:
            self._trail.append((idx, self._bits[idx]))
        self._bits[idx] = new

    def _undo(self, mark: int) -> None:
        assert self._bits is not None and self._trail is not None
        trail = self._trail
        while len(trail) > mark:
            idx, old = trail.pop()
            self._bits[idx] = old

    # ── public API ──────────────────────────────────────────────────

    def run(  # type: ignore[override]
        self,
        seed: int | None = None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        ac3_budget: int = DEFAULT_AC3_BUDGET,
        work_budget: int = DEFAULT_WORK_BUDGET,
        max_backtracks: int = 0,
    ) -> list[list[bytes | None]]:
        """Collapse the wave; see :meth:`WaveGrid.run`.

        Args:
            max_backtracks: 0 keeps WaveGrid's restart-on-contradiction. N > 0
                switches to depth-first search with at most N undone
                assignments per attempt. Exceeding it counts as a failed
                attempt (restarts, reseeded, still apply); exhausting the
                search proves the pinned wave unsatisfiable and stops at once.
                On failure the grid is left at its post-pin propagation fixpoint
                with ``contradiction`` set.
        """
        if seed is not None:
            self._rng.reseed(seed)
        self._initial = self.superpositions.copy()
        self._strict = max_backtracks > 0
        self._bits = self._pack()
        self._init_bits = list(self._bits)
        try:
            for attempt in range(max_restarts + 1):
                if attempt > 0:
                    self._reset()
                    if seed is not None:
                        self._rng.reseed(seed + attempt * 7919)
                self.contradiction = False
                self._budget_exhausted = False
                proven_unsat = False
                if max_backtracks > 0:
                    proven_unsat = self._search(ac3_budget, work_budget, max_backtracks)
                else:
                    self._run_loop(ac3_budget, work_budget)
                if not self.contradiction and not self._budget_exhausted:
                    break
                if proven_unsat:
                    break
            self._unpack()
            return super()._to_grid()
        finally:
            self._bits = None
            self._trail = None

    def _to_grid(self) -> list[list[bytes | None]]:
        self._unpack()
        return super()._to_grid()

    def _reset(self) -> None:
        if self._bits is None:
            super()._reset()
            return
        self._bits = list(self._init_bits)
        self._trail = None
        self.contradiction = False

    # ── depth-first search ──────────────────────────────────────────

    def _search(self, ac3_budget: int, work_budget: int, max_backtracks: int) -> bool:
        """DFS with trail undo. Returns True iff the wave is proven unsatisfiable."""
        self._work_used = 0
        self._trail = None
        self._propagate(ac3_budget, sources=None, work_budget=work_budget)
        if self._budget_exhausted:
            return False
        if self.contradiction:
            return True  # the caller's pins already conflict
        self._trail = []
        trail = self._trail
        stack: list[list] = []  # [cell, remaining tile ids, trail mark]
        backtracks = 0
        pick = True
        while True:
            if pick:
                idx = self._find_min_entropy()
                if not self.contradiction:
                    if idx is None:
                        return False  # every cell collapsed: solved
                    assert self._bits is not None
                    stack.append([idx, _bit_ids(self._bits[idx]), len(trail)])
            pick = False
            # Try the next tile of the deepest decision; exhausted decisions pop.
            while True:
                if self._budget_exhausted:
                    return False
                if not stack:
                    self._undo(0)
                    self.contradiction = True
                    return True
                frame = stack[-1]
                self._undo(frame[2])
                self.contradiction = False
                remaining = frame[1]
                if not remaining:
                    stack.pop()
                    continue
                if backtracks > max_backtracks:
                    self._undo(0)
                    self.contradiction = True
                    return False
                tile = self._weighted_pick(remaining)
                remaining.remove(tile)
                self._set(frame[0], 1 << tile)
                self._propagate(
                    ac3_budget, sources=self._neighbors(frame[0]), work_budget=work_budget
                )
                if self._budget_exhausted:
                    return False
                if not self.contradiction:
                    pick = True
                    break
                backtracks += 1

    # ── collapse primitives (bitmask versions of the WaveGrid methods) ─

    def _count(self, idx: int) -> int:
        assert self._bits is not None
        return self._bits[idx].bit_count()

    def _find_min_entropy(self) -> int | None:
        bits = self._bits
        assert bits is not None
        counts = [m.bit_count() for m in bits]
        dead = next((i for i, c in enumerate(counts) if c == 0), None)
        limit = dead if dead is not None else self.n
        open_cells = [i for i in range(limit) if counts[i] > 1]
        noise = self._rng.random_sequential(len(open_cells))
        if dead is not None:
            self.contradiction = True
            return None
        if not open_cells:
            return None
        cache = self._entropy_cache
        ent = []
        for i in open_cells:
            m = bits[i]
            e = cache.get(m)
            if e is None:
                e = cache[m] = self._mask_entropy(m)
            ent.append(e)
        return open_cells[int(np.argmin(np.array(ent) + np.asarray(noise) * 1e-9))]

    def _mask_entropy(self, mask: int) -> float:
        # Same reduction as WaveGrid._entropy over the ascending tile ids.
        w = self._weights[_bit_ids(mask)]
        total = w.sum()
        if total <= 0:
            return 0.0
        p = w / total
        return float(-(p * np.log2(p)).sum())

    def _weighted_pick(self, possible: list[int]) -> int:
        """One weighted draw from *possible* (WaveGrid._observe's exact rule)."""
        pos = np.asarray(possible)
        weights = self._weights[pos]
        total = weights.sum()
        if total <= 0:
            return int(self._rng.choice(pos))
        r = self._rng.random() * total
        cumulative = 0.0
        chosen = possible[-1]
        for tid, w in zip(possible, weights.tolist(), strict=True):
            cumulative += w
            if r <= cumulative:
                chosen = tid
                break
        return int(chosen)

    def _observe(self, idx: int) -> None:
        assert self._bits is not None
        possible = _bit_ids(self._bits[idx])
        if not possible:
            self.contradiction = True
            return
        self._set(idx, 1 << self._weighted_pick(possible))

    def _prune_step(self, idx: int, work_budget: int) -> bool | None:
        if self._budget_exhausted:
            return None
        assert self._bits is not None
        m = self._bits[idx]
        if m & (m - 1) == 0 and not (self._strict and m):
            return False
        self._work_used += 1
        if self._work_used > work_budget:
            self._budget_exhausted = True
            return None
        return self._prune_cell(idx)

    def _support(self, d: int, nbr: int) -> int:
        """Tiles that may sit next to *some* tile of *nbr* in direction *d*."""
        key = (d, nbr)
        s = self._support_cache.get(key)
        if s is None:
            s = 0
            table = self._tiles_with[d]
            m = nbr
            while m:
                low = m & -m
                s |= table[low.bit_length() - 1]
                m ^= low
            self._support_cache[key] = s
        return s

    def _prune_cell(self, idx: int) -> bool | None:
        bits = self._bits
        assert bits is not None
        old = row = bits[idx]
        for nidx, d in self._nbr_dirs[idx]:
            row &= self._support(d, bits[nidx])
            if not row:
                self._set(idx, 0)
                self.contradiction = True
                return None
        if row != old:
            self._set(idx, row)
            return True
        return False

    def _fallback_greedy(self) -> None:
        assert self._bits is not None
        for i in range(self.n):
            m = self._bits[i]
            if m & (m - 1) and not self.contradiction:
                self._observe(i)


def _mask_of(ids: np.ndarray) -> int:
    m = 0
    for i in ids.tolist():
        m |= 1 << i
    return m
