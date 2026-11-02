"""SeedGramScheduler: seed picks weighted by the top eigenvector of B = A Aᵀ.

``A`` is the seed x edge incidence with the shared parser core stripped
(``core_stripped`` weighting, ``core/gram_matrixfree.py``). The dominant
eigenvector ``v`` of ``B`` is the eigenvector centrality of each seed: how
much a seed sits in the corpus's main mode of overlap. It is computed
matrix-free (power iteration over sparse products), so no S x S array exists
at any corpus size::

    peripheral   tickets = FLOOR + (1 - v / max v)    favour seeds off the main mode
    central      tickets = FLOOR + v / max v          favour seeds inside it

``peripheral`` is the default: the hypothesis is that seeds with low loading
on the dominant mode reach coverage the bulk of the corpus does not. The
handover (``docs/handover/handover_coverage_gram_matrix_2026-10-26.md`` §8)
found every Gram readout tracking coverage size on fuzzgoat and png_read, so
this arm may be no better than uniform; it is off by default and is not in
``--hail-mary`` until it wins its A/B.

Refits run off the pick path (``maybe_refit``, like DPP). A seed newer than
the last refit takes the largest live ticket count. One ``rng.random()`` per
pick (Hard Rule 16).
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.cadence import phase_of
from fuzzer_tool.core.gram_matrixfree import Incidence, compact_columns, top_eigenpair
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts
from fuzzer_tool.core.schedulers.seed_dpp import _cdf_index, _edge_array

#: Executions between eigenvector refits.
REFIT_INTERVAL = 2000
#: Ticket floor, so no seed is ever starved (the loading is a prior, not a verdict).
FLOOR = 0.1
MODES = ("peripheral", "central")


class SeedGramScheduler(ArmCounts):
    """Eigenvector-centrality seed sampler (matrix-free).

    Args:
        rng: A ``RandPool`` (Hard Rule 16).
        mode: ``"peripheral"`` or ``"central"``.
        refit_interval: Executions between refits.
    """

    #: Seed arm; selection reads the eigenvector, not Beta priors.
    supports_priors = False

    def __init__(self, rng, mode: str = "peripheral", refit_interval: int = REFIT_INTERVAL) -> None:
        if rng is None:
            raise ValueError("SeedGramScheduler requires a RandPool (Hard Rule 16)")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        super().__init__()
        self._rng = rng
        self._mode = mode
        self._refit_interval = max(1, int(refit_interval))
        self._last_refit_exec = -(1 << 60)
        self.refits = 0
        self._loading: dict[str, float] = {}  # key -> v / max v in [0, 1]

    def maybe_refit(self, edge_tracker, exec_count: int) -> bool:
        """Refit if ``refit_interval`` executions passed; first call refits at once."""
        if exec_count - self._last_refit_exec < self._refit_interval:
            return False
        first = self._last_refit_exec < 0
        self._last_refit_exec = exec_count + (
            phase_of("gram.refit", self._refit_interval) if first else 0
        )
        return self.refit(edge_tracker)

    def refit(self, edge_tracker) -> bool:
        """Recompute loadings from ``edge_tracker.seed_edges``.

        Returns False, keeping the previous loadings, when no seed has an edge.
        """
        seed_edges = getattr(edge_tracker, "seed_edges", None)
        if not seed_edges:
            return False
        keys = sorted(seed_edges)
        rows = [_edge_array(seed_edges[k]) for k in keys]
        if not any(len(r) for r in rows):
            return False
        rows, n_cols = compact_columns(rows)
        inc = Incidence(rows, n_edges=n_cols, weighting="core_stripped")
        lam, v, _ = top_eigenpair(inc)
        top = float(v.max()) if lam > 0.0 else 0.0
        scaled = v / top if top > 0.0 else np.zeros(len(keys))
        self._loading = dict(zip(keys, scaled.tolist(), strict=True))
        self.refits += 1
        return True

    def _ticket(self, key: str) -> float:
        v = self._loading.get(key)
        if v is None:
            return -1.0  # newer than the refit; resolved against the live max
        return FLOOR + (1.0 - v if self._mode == "peripheral" else v)

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        self._trim(seed_ids)
        if len(seed_ids) == 1:
            return seed_ids[0]
        n = len(seed_ids)
        u = self._rng.random()
        if not self._loading:
            return seed_ids[min(int(u * n), n - 1)]
        tickets = np.fromiter((self._ticket(k) for k in seed_ids), np.float64, n)
        known = tickets >= 0.0
        tickets[~known] = tickets[known].max() if known.any() else 1.0
        return seed_ids[_cdf_index(tickets, u)]
