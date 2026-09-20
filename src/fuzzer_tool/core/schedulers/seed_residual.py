"""``seed_residual``: score what hit volume does not explain (P3-4).

Design: ``docs/handover/handover_edge_id_axis_2026-09-18.md``, "Seed arm: score what
volume does not explain". The most repeated result in this repo is that a derived seed
score collapses to a row sum: Tang's partial correlation controlling for total hits was
+0.006 (Wilcoxon p = 1.0, ten matrices), degree beat every low-rank score at
+0.27..+0.77, and PC1 is volume at rho = +0.989. So the orthogonalisation is the
estimator here rather than the audit:

    mass_i   = sum over the seed's canonical classes of 1 / owners(class)
    score_i  = rank(mass_i) - OLS fit on rank(total hits_i)          (the residual)
    energy_i = EPS + percentile(score_i)

``1 / owners`` is incidence, not volume (rho(owners, total) = +0.957: they look alike
and are not, which was a live defect once). Classes are ``EdgeCanonicalizer``'s, so a
45-edge duplicate chain counts once and not 45 times (F10).

The module carries its own falsification, the way ``seed_tang`` exports
``modfkv_sample_complexity``: at every refit :meth:`falsification` recomputes how much
of the score is still volume (``rho_total``, ``rho_degree``) and, when outcomes are
supplied, the partial correlation of score with a seed's productivity controlling for
volume and for degree. A partial converging to zero means the arm has become a row sum
again and should be pulled. It is logged, not thresholded: the handover gives no
threshold and this file does not invent one.

Enters only as one arm under the existing Elo dispatch (the arbiter degrades a weak arm
rather than letting it do damage). It abstains -- ``select_index`` returns None, which
hands the pick back to the caller -- while the preflight gate is closed (F11 / F1) or no
fold exists, and never degrades silently to uniform. Off by default; expected gain small;
``bench_paired.py`` with a pre-registered threshold is the only thing that turns it on
by default (P3-4).

Excluded by measurement and therefore absent: any low-rank score (SVD leverage, PC1/PC2/
PC3 -- PC2 and PC3 wait on P1-3 to replicate), l2-magnitude sampling (returns the most
crowded edges, 1.3-1.9x above uniform), GF(2) or LLL as a minimiser (133 seeds against
greedy's 31), and every id-axis statistic (F3, F12).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from fuzzer_tool.core.edge_matrix import (
    MatrixSubstrate,
    average_ranks,
    partial_rank_corr,
    rank_corr,
    residualize_ranks,
)
from fuzzer_tool.core.rand_pool import RandPool

log = logging.getLogger(__name__)

#: Energy floor: a seed at the bottom of the residual ranking keeps a nonzero pick
#: probability (a zero would make it unreachable, Tang's failure mode).
EPS = 0.05
#: Uniform-pick probability floor, widened up to 2x by the saturation signal.
EXPLORE_BASE = 0.05
#: Outcome samples needed before a partial correlation is reported.
MIN_OUTCOMES = 8


class ResidualSeedScheduler:
    """Elo-arm seed scheduler over the shared :class:`MatrixSubstrate`.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        substrate: The fold, shared with ``OpCreditScheduler``.
        outcome_fn: Optional zero-arg callable returning ``{seed_key: productivity}``
            (``Fuzzer`` passes each seed's ``coverage_edges``). Read only by the
            per-refit falsification log; never by the score.
    """

    def __init__(
        self,
        rng: RandPool | None = None,
        substrate: MatrixSubstrate | None = None,
        outcome_fn: Callable[[], dict[str, float]] | None = None,
    ):
        if rng is None:
            raise ValueError("ResidualSeedScheduler requires a RandPool (Hard Rule 16)")
        if substrate is None:
            raise ValueError("ResidualSeedScheduler requires a MatrixSubstrate")
        self._rng = rng
        self.substrate = substrate
        self._outcome_fn = outcome_fn
        self._scored_version = -1
        self._energy: dict[str, float] = {}
        self._score = np.empty(0)
        self._median = 1.0
        self.last_falsification: dict[str, float] | None = None

    # ── Scoring ───────────────────────────────────────────────────────────
    def _rescore(self) -> None:
        fold = self.substrate.fold
        if fold is None or self._scored_version == self.substrate.version:
            return
        self._score = residualize_ranks(fold.mass, fold.total)
        pct = average_ranks(self._score) / max(fold.n_seeds - 1, 1)
        energy = EPS + pct
        self._energy = dict(zip(fold.seed_keys, energy.tolist(), strict=True))
        self._median = float(np.median(energy))
        self._scored_version = self.substrate.version
        if self._outcome_fn is not None:
            try:
                self.last_falsification = self.falsification(self._outcome_fn())
            except Exception:  # a diagnostic must never take down scheduling
                log.exception("seed_residual falsification failed")
                return
            log.info("seed_residual falsification: %s", self.last_falsification)

    def available(self) -> bool:
        """True when the arm can score: preflight gate open and a fold exists."""
        return self.substrate.trusted and self.substrate.fold is not None

    def seed_energy(self, key: str) -> float:
        """Energy of *key*; a seed admitted since the last refit gets the median."""
        self._rescore()
        return self._energy.get(key, self._median)

    def weights(self, keys: list[str]) -> np.ndarray:
        self._rescore()
        return np.array([self._energy.get(k, self._median) for k in keys], dtype=np.float64)

    def select_index(self, keys: list[str]) -> int | None:
        """Index into *keys* by energy; None = abstain (gate closed / nothing fitted)."""
        if not keys or not self.available():
            return None
        w = self.weights(keys)
        total = float(w.sum())
        explore = min(1.0, EXPLORE_BASE * (1.0 + self.substrate.saturation_signal()))
        if total <= 0 or self._rng.random() < explore:
            return int(self._rng.randrange(len(keys)))
        r = self._rng.random() * total
        return min(int(np.searchsorted(np.cumsum(w), r, side="left")), len(keys) - 1)

    # ── Falsification and diagnostics ─────────────────────────────────────
    def falsification(self, outcomes: dict[str, float] | None = None) -> dict[str, float]:
        """How much of the score is still volume, and whether it predicts anything.

        Returns:
            ``rho_total`` / ``rho_degree``: Spearman of the score against hit volume
            and against classes touched. By construction of the residual these sit near
            zero; drift away from it means the fold changed shape. ``partial_total`` /
            ``partial_degree`` (only with >= MIN_OUTCOMES labelled seeds): the score's
            partial correlation with *outcomes* controlling for volume / degree. Near
            zero means the arm is a row sum again.
        """
        self._rescore()
        fold = self.substrate.fold
        if fold is None:
            return {}
        out = {
            "rho_total": rank_corr(self._score, fold.total),
            "rho_degree": rank_corr(self._score, fold.degree),
            "seeds": float(fold.n_seeds),
        }
        idx = [i for i, k in enumerate(fold.seed_keys) if outcomes and k in outcomes]
        if len(idx) >= MIN_OUTCOMES:
            y = np.array([float(outcomes[fold.seed_keys[i]]) for i in idx])
            s = self._score[idx]
            out["partial_total"] = partial_rank_corr(s, y, fold.total[idx])
            out["partial_degree"] = partial_rank_corr(s, y, fold.degree[idx])
            out["outcomes"] = float(len(idx))
        return out

    def stats(self) -> dict:
        self._rescore()
        sub, fold = self.substrate, self.substrate.fold
        out: dict = {
            "gate": sub.gate_state(),
            "fitted": fold is not None,
            "skip_reason": sub.skip_reason if fold is None else "",
        }
        if sub.distrust_reason:
            out["distrust_reason"] = sub.distrust_reason
        if fold is None:
            return out
        out.update(
            seeds=fold.n_seeds,
            edges=fold.n_edges,
            classes=fold.n_classes,
            duplicate_edges=fold.n_edges - fold.n_classes,
            saturation=sub.saturation_signal(),
        )
        if self.last_falsification:
            out["falsification"] = dict(self.last_falsification)
        return out
