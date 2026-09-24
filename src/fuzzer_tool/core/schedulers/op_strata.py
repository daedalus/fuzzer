"""OpStrataScheduler: Thompson over (op, stratum) cells with partial pooling (§3.4).

Design: ``docs/handover/handover_strata_schedulers_2026-09-19.md``.

    p_op      = (S_op + 1) / (N_op + 2)                  # pooled across strata
    theta     ~ Beta(K*p_op + s_cell, K*(1 - p_op) + f_cell)
    pick      = argmax theta over the offered ops

A cold cell starts at the operator's pooled mean with ``K`` pseudo-counts;
``K`` observations in the cell outweigh the pool. Example, K=8, op pooled
3/10 successes, cell 1 success in 1 pull: p = 4/12,
theta ~ Beta(8/3 + 1, 16/3 + 0).

Stratum is set by the fuzzer per round: the ``strata`` seed arm's phi, else
the parent seed's rarest family (``EdgeLedger.rarest_family``). ``None``
reads the pool only and updates no cell.

Reward: ``weight`` clamped to [0, 1] on success, else 0; fractional rewards
split into ``s += r, f += 1 - r``. Non-finite or negative weights count as
failures. ``supports_priors = False``. Elo-only, absent from
``_FALLBACK_PRECEDENCE``. Not A/B validated.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

#: Pool pseudo-count: cell evidence equal to K pulls outweighs the pool.
POOL_K = 8.0
_INIT_CAP = 64


def _reward(success: bool, weight: float) -> float:
    if not success or not math.isfinite(weight) or weight <= 0.0:
        return 0.0
    return min(weight, 1.0)


class OpStrataScheduler:
    """Stratified Thompson operator scheduler.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        k: Pool pseudo-count.
    """

    supports_priors = False  # The pooled posterior is the prior.

    def __init__(self, rng: RandPool | None = None, k: float = POOL_K):
        if rng is None:
            raise ValueError("OpStrataScheduler requires a RandPool (Hard Rule 16)")
        self._rng = rng
        self._k = k
        self._idx: dict[str, int] = {}
        self._cap = _INIT_CAP
        self._s_pool = np.zeros(self._cap)
        self._n_pool = np.zeros(self._cap)
        self._cells: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._stratum: int | None = None
        self._last_key: tuple[str, ...] = ()
        self._last_idx = np.zeros(0, dtype=np.intp)
        self._pulls = 0

    # ── Arms ────────────────────────────────────────────────────────────
    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *name*; priors ignored (``supports_priors = False``)."""
        self._index(name)

    def _index(self, op: str) -> int:
        i = self._idx.get(op)
        if i is not None:
            return i

        i = self._idx[op] = len(self._idx)
        if i >= self._cap:
            self._grow()
        return i

    def _grow(self) -> None:
        cap = self._cap * 2
        pad = cap - self._cap
        self._s_pool = np.concatenate([self._s_pool, np.zeros(pad)])
        self._n_pool = np.concatenate([self._n_pool, np.zeros(pad)])
        for st, (s, f) in self._cells.items():
            self._cells[st] = (
                np.concatenate([s, np.zeros(pad)]),
                np.concatenate([f, np.zeros(pad)]),
            )
        self._cap = cap

    def _indices(self, ops: list[str]) -> np.ndarray:
        key = tuple(ops)
        if key != self._last_key:
            self._last_idx = np.fromiter(
                (self._index(op) for op in ops), dtype=np.intp, count=len(ops)
            )
            self._last_key = key
        return self._last_idx

    # ── Scheduling ──────────────────────────────────────────────────────
    def set_stratum(self, stratum: int | None) -> None:
        self._stratum = stratum

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        idx = self._indices(ops)
        p = (self._s_pool[idx] + 1.0) / (self._n_pool[idx] + 2.0)
        a = self._k * p
        b = self._k - a
        cell = self._cells.get(self._stratum) if self._stratum is not None else None
        if cell is not None:
            a = a + cell[0][idx]
            b = b + cell[1][idx]
        draws = self._rng.betavariate_array(a, b)
        return ops[int(np.argmax(draws))]

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        r = _reward(success, weight)
        i = self._index(op)
        self._pulls += 1
        self._s_pool[i] += r
        self._n_pool[i] += 1.0
        if self._stratum is None:
            return

        cell = self._cells.get(self._stratum)
        if cell is None:
            cell = self._cells[self._stratum] = (np.zeros(self._cap), np.zeros(self._cap))
        cell[0][i] += r
        cell[1][i] += 1.0 - r

    def cell(self, op: str, stratum: int | None) -> tuple[float, float]:
        """(successes, failures) of one cell; stratum None -> pooled."""
        i = self._idx.get(op)
        if i is None:
            return (0.0, 0.0)
        if stratum is None:
            s = float(self._s_pool[i])
            return (s, float(self._n_pool[i]) - s)
        c = self._cells.get(stratum)
        return (0.0, 0.0) if c is None else (float(c[0][i]), float(c[1][i]))

    def bandit_stats(self) -> dict:
        return {
            "strata_pulls": self._pulls,
            "strata_cells": len(self._cells),
            "strata_k": self._k,
            "operators_tracked": len(self._idx),
        }
