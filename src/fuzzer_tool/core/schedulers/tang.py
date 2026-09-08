"""Tang's quantum-inspired recommendation algorithm over the seed x edge matrix.

Port of Ewin Tang, "A quantum-inspired classical algorithm for recommendation
systems" (arXiv:1807.04271v3). The mapping to fuzzing is the obvious one:

    users    -> corpus seeds
    products -> coverage edges
    A[i][j]  -> hit count of edge j under seed i  (EdgeTracker.seed_hit_counts)

Tang's Algorithm 3 samples an entry from row *i* of a low-rank approximation
``D = A Vhat Vhat^T``. Read as a recommender that answers "which edge should
this seed reach next"; read as a scheduler it gives a per-seed energy, which
is what :meth:`seed_energy` exposes to the ``tang`` Elo arm.

**This arm is off by default and should stay that way until an A/B says
otherwise.** It is landed because the port is small, the paper's subroutines
are independently useful, and a measured negative is worth more in-tree than
out. The measurements behind that sentence — ten independent campaigns, and
the two of our own earlier claims they overturned — are in
``docs/handover/handover_done_2026-09-06.md`` §14. The one-line
version: the score works, but controlling for ``A.sum(1)`` (total hit volume,
one row sum) it adds nothing — mean partial Spearman +0.006 over ten corpora,
Wilcoxon p=1.0.

Three things about the port are easy to get wrong.

* **ModFKV is not what runs here.** Algorithm 2 needs
  ``q = Theta(K^4 / (eta*eps^2)^2)`` sampled rows with ``K = ||A||_F^2/sigma^2``.
  On real corpora that is 8e5 rows at the *most* generous parameters and
  1e9-1e13 at useful ones, against corpora of 13-116 seeds -- the subsample
  is larger than the input. :func:`modfkv_sample_complexity` computes it so
  the number stays checkable rather than folklore. What :meth:`refit` runs is
  a dense truncated SVD, which is strictly the *better* estimator (no
  subsampling noise): the negative result below is therefore a negative on
  Tang's best case, not on a degraded stand-in.

* **The paper's sampling direction is backwards for a fuzzer.** Sampling
  ``proportional to |D_ij|^2`` samples by magnitude, and magnitude is
  popularity: measured, the drawn edge is owned by 1.3-1.9x more seeds than a
  uniform draw, on corpora where 10-19% of edges are singletons. That fights
  ``RARE_EDGE_OWNERS`` and the crowding penalty in ``seed_picker``. Inverting
  it does not help -- ``frontier_mass = 1 - covered_mass`` exactly (rows of
  ``P`` sum to 1), so the inverted score is the same measurement with the sign
  flipped, and it is anti-predictive in 10/10 campaigns. ``mode="frontier"``
  exists to keep that falsifiable, not because it is a candidate.

* **Randomness goes through RandPool** (Hard Rule 16). The Gaussian sketch
  for the randomized path draws from ``rng.gauss_list``, and the edge draw
  from ``rng.random``, so ``--seed`` determines every recommendation. This is
  the same constraint that ruled out numpy's ziggurat in ``_beta_sample``:
  the binding restriction is reproducibility, not speed.
"""

from __future__ import annotations

import numpy as np

#: Rank of the low-rank approximation. Tang's ``k`` is "constant or
#: polylog(m, n)"; 10 is where the measured reconstruction recall plateaus on
#: real corpora (rank 5 -> 0.26, rank 10 -> 0.52 on zlib under MNAR masking).
DEFAULT_RANK = 10

#: Executions between refits. The SVD is the expensive half (102ms at
#: 2000x8189, and the matrix *build* from Python sets is 658ms on top), so it
#: cannot ride the per-pick path. Matches the cadence of the seed-weight
#: recompute in ``seed_picker`` rather than inventing a third clock.
DEFAULT_REFIT_INTERVAL = 2000

#: Oversampling for the randomized range finder (Halko et al.). Below the
#: crossover the exact SVD is used instead, so this only applies to matrices
#: large enough for the sketch to pay for itself.
_OVERSAMPLE = 8

#: Above this many matrix entries, sketch instead of taking the exact SVD.
_EXACT_SVD_MAX_ENTRIES = 1 << 20


def modfkv_sample_complexity(
    frobenius_sq: float, sigma: float, eps: float, eta: float
) -> tuple[float, float]:
    """``(K, q)`` for ModFKV (Algorithm 2) at these parameters.

    ``K = ||A||_F^2 / sigma^2`` and ``q = Theta(K^4 / eps_bar^2)`` with
    ``eps_bar = eta * eps^2``, straight from the paper. Exposed as a function
    because the whole practical objection to running the real ModFKV is the
    size of this number, and a number nobody can recompute becomes folklore.

    Raises:
        ValueError: if ``sigma``, ``eps`` or ``eta`` is non-positive -- the
            formula divides by all three and a zero would return ``inf``,
            which reads as "unbounded" rather than "you passed nothing".
    """
    if sigma <= 0 or eps <= 0 or eta <= 0:
        raise ValueError(f"sigma, eps and eta must be positive (got {sigma}, {eps}, {eta})")
    k_ratio = frobenius_sq / (sigma * sigma)
    eps_bar = eta * eps * eps
    return k_ratio, (k_ratio**4) / (eps_bar * eps_bar)


def estimate_inner_product(x, y, rng, samples: int = 200, groups: int = 5) -> float:
    """Proposition 4.2: estimate ``<x, y>`` from l2-norm samples of ``x``.

    Draw ``j ~ D_x`` (that is, ``P(j) = x_j^2 / ||x||^2``) and average
    ``y_j / x_j``; the expectation is ``<x, y> / ||x||^2``, so scaling by
    ``||x||^2`` gives the estimate. The median of group means is the paper's
    variance reduction and is what makes the additive error bound hold with
    ``1 - delta`` probability rather than in expectation.

    Zero entries of ``x`` have zero probability under ``D_x``, so the division
    is safe by construction; the guard below is for the caller who hands in a
    zero vector, where ``D_x`` is undefined at all.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sq = float(x @ x)
    if sq <= 0.0:
        raise ValueError("cannot l2-sample from a zero vector")
    p = (x * x) / sq
    cdf = np.cumsum(p)
    draws = np.searchsorted(cdf, np.asarray(rng.random_list(samples)), side="right")
    draws = np.clip(draws, 0, len(x) - 1)
    vals = y[draws] / x[draws]
    groups = max(1, min(groups, len(vals)))
    means = [float(np.mean(c)) for c in np.array_split(vals, groups)]
    return sq * float(np.median(means))


def sample_from_linear_combination(columns: np.ndarray, w: np.ndarray, rng, max_iter: int = 512):
    """Proposition 4.3: sample from ``D_{Vw}`` given sample access to ``V``.

    Rejection sampling with the paper's acceptance ratio

        r_i = (Vw)_i^2 / (k * sum_j (V_ij w_j)^2)

    which Cauchy-Schwarz bounds by 1, so the procedure is well defined and
    terminates in ``k*C(V, w)`` iterations in expectation. ``r_i`` is written
    exactly as the paper writes it -- computable in ``k`` queries without ever
    forming ``||Vw||``, which is the point of the construction.

    Returns ``None`` when the budget runs out. A caller that treats that as
    "no recommendation" is correct; a caller that retries forever is not,
    because ``C(V, w)`` is unbounded when the columns are near-dependent.
    """
    n, k = columns.shape
    proposal_mass = (columns * w) ** 2
    col_weight = proposal_mass.sum(axis=0)
    total = float(col_weight.sum())
    if total <= 0.0:
        return None
    col_cdf = np.cumsum(col_weight / total)
    row_denom = proposal_mass.sum(axis=1)
    target = columns @ w
    for _ in range(max_iter):
        j = int(np.searchsorted(col_cdf, rng.random(), side="right"))
        j = min(j, k - 1)
        col_sq = proposal_mass[:, j]
        col_total = float(col_sq.sum())
        if col_total <= 0.0:
            continue
        i = int(np.searchsorted(np.cumsum(col_sq / col_total), rng.random(), side="right"))
        i = min(i, n - 1)
        denom = k * row_denom[i]
        if denom <= 0.0:
            continue
        if rng.random() <= (target[i] ** 2) / denom:
            return i
    return None


class TangRecommendationScheduler:
    """Low-rank seed scorer following Tang's Algorithm 3.

    Lifecycle mirrors ``KatzChannel``: the fuzzer holds one instance, calls
    :meth:`maybe_refit` on the stats path, and reads :meth:`seed_energy` from
    the ``tang`` Elo arm. Everything is a no-op until the first successful
    refit, so an enabled-but-unfitted scheduler returns neutral scores rather
    than raising.

    Args:
        rank: Rank of the approximation (Tang's ``k``).
        refit_interval: Executions between refits.
        mode: ``"tang"`` for the paper's score (l2 mass on the seed's own
            covered edges) or ``"frontier"`` for the inverted variant. See the
            module docstring for why ``"frontier"`` is not a candidate.
        rng: A ``RandPool``. Required -- Hard Rule 16, and passing ``None``
            would silently move recommendations off the ``--seed`` stream.
    """

    MODES = ("tang", "frontier")

    def __init__(
        self,
        rng,
        rank: int = DEFAULT_RANK,
        refit_interval: int = DEFAULT_REFIT_INTERVAL,
        mode: str = "tang",
    ):
        if rng is None:
            raise ValueError("TangRecommendationScheduler requires a RandPool (Hard Rule 16)")
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Use one of {self.MODES}")
        self.rng = rng
        self.rank = max(1, int(rank))
        self.refit_interval = max(1, int(refit_interval))
        self.mode = mode
        self._basis: np.ndarray | None = None  # (n_edges, k) approximate right singular vectors
        self._edge_index: dict[int, int] = {}
        self._owner_count: np.ndarray | None = None
        self._energies: dict[str, float] = {}
        self._last_refit_exec = -(1 << 60)
        self.refits = 0
        self.last_shape: tuple[int, int] = (0, 0)
        self.last_effective_rank = 0

    # ------------------------------------------------------------------ fit

    def maybe_refit(self, edge_tracker, exec_count: int) -> bool:
        """Refit if ``refit_interval`` executions have passed. Returns whether it did.

        The interval gate lives here rather than at the call site so that the
        two callers (the stats path and a forced refit in tests) cannot drift
        apart on the cadence -- the same failure the ``katz_channel`` comment
        at :72 warned about and got wrong in the other direction.
        """
        if exec_count - self._last_refit_exec < self.refit_interval:
            return False
        self._last_refit_exec = exec_count
        return self.refit(edge_tracker)

    def refit(self, edge_tracker) -> bool:
        """Rebuild the low-rank basis from ``edge_tracker``. Returns success.

        Returns False (leaving any previous basis in place) when there is not
        enough matrix to factor: fewer than two seeds, no edges, or an
        all-zero matrix. Keeping the stale basis is deliberate -- a corpus that
        momentarily shrinks below the threshold should not blank the arm.
        """
        seed_edges = getattr(edge_tracker, "seed_edges", None)
        if not seed_edges:
            return False
        hit_counts = getattr(edge_tracker, "seed_hit_counts", None) or {}
        keys = sorted(seed_edges)
        edges = sorted(set().union(*seed_edges.values())) if seed_edges else []
        if len(keys) < 2 or not edges:
            return False

        index = {e: i for i, e in enumerate(edges)}
        matrix = np.zeros((len(keys), len(edges)), dtype=np.float64)
        for row, key in enumerate(keys):
            counts = hit_counts.get(key) or {}
            for edge in seed_edges[key]:
                # Absent from seed_hit_counts means "covered, count unknown";
                # 1.0 is the incidence value, not a sentinel. Do NOT read the
                # count with a str key: to_dict/from_dict round-trips edge ids
                # through JSON-ish string keys and EdgeTracker.from_dict is
                # what casts them back, so anything reading the raw dict gets
                # a silently binary matrix. That bug cost a full analysis pass.
                matrix[row, index[edge]] = float(counts.get(edge, 1.0))

        if not matrix.any():
            return False

        basis = self._right_singular_basis(matrix)
        if basis is None:
            return False

        self._basis = basis
        self._edge_index = index
        self._owner_count = np.maximum((matrix > 0).sum(axis=0), 1).astype(np.float64)
        self.last_shape = matrix.shape
        self.last_effective_rank = basis.shape[1]
        self.refits += 1
        self._energies = {key: self._score_row(matrix[row]) for row, key in enumerate(keys)}
        return True

    def _right_singular_basis(self, matrix: np.ndarray) -> np.ndarray | None:
        """Top-``rank`` right singular vectors, exactly or by random sketch.

        The exact SVD is used while the matrix is small because it is both
        faster and deterministic there; above the crossover the sketch is the
        only thing that keeps a refit off the tail of the pick path. Hard Rule
        14 in reverse: the vectorized form is kept, and the slower-but-exact
        one is kept too, for the range where it wins.
        """
        rank = min(self.rank, *matrix.shape)
        if rank < 1:
            return None
        if matrix.size <= _EXACT_SVD_MAX_ENTRIES:
            _u, sv, vt = np.linalg.svd(matrix, full_matrices=False)
        else:
            width = min(rank + _OVERSAMPLE, matrix.shape[1])
            sketch = np.asarray(
                self.rng.gauss_list(0.0, 1.0, matrix.shape[1] * width), dtype=np.float64
            ).reshape(matrix.shape[1], width)
            q, _r = np.linalg.qr(matrix @ sketch)
            _u, sv, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
        # Relative, not absolute: an absolute 1e-12 floor is scale-dependent,
        # and hit-count matrices span several orders of magnitude between
        # targets, so it would admit numerical-noise directions on a large
        # matrix and reject real ones on a small.
        keep = min(rank, int((sv > 1e-12 * sv[0]).sum())) if sv.size else 0
        if keep < 1:
            return None
        return vt[:keep].T

    # --------------------------------------------------------------- scoring

    def _project(self, row: np.ndarray) -> np.ndarray:
        """``D_i = A_i Vhat Vhat^T`` -- the paper's low-rank row."""
        return (row @ self._basis) @ self._basis.T

    def _score_row(self, row: np.ndarray) -> float:
        projected = self._project(row)
        mass = projected * projected
        total = float(mass.sum())
        if total <= 0.0:
            return 0.0
        covered = row > 0
        if self.mode == "frontier":
            return float((mass[~covered] / self._owner_count[~covered]).sum() / total)
        return float(mass[covered].sum() / total)

    def seed_energy(self, seed_key: str) -> float:
        """Score in ``[0, 1]`` for a seed, or 0.0 when unfitted or unseen.

        A seed admitted since the last refit is unseen and scores 0.0 rather
        than being projected on the fly: the projection needs its coverage row,
        which the caller does not hold, and fabricating one would make the arm
        prefer or avoid new seeds for a reason unrelated to the model.
        """
        return self._energies.get(seed_key, 0.0)

    @property
    def fitted(self) -> bool:
        return self._basis is not None

    # -------------------------------------------------------- recommendation

    def recommend(self, edge_row, count: int = 1) -> list[int]:
        """l2-sample ``count`` edge ids from ``D_i`` -- Algorithm 3's output.

        This is the recommender proper, and it is what the measurements say
        points the wrong way for scheduling; it is exposed because the *edge*
        answer is the paper's actual contribution and is the thing a future
        directed-fuzzing consumer would want, not because ``seed_energy``
        needs it.
        """
        if self._basis is None or count < 1:
            return []
        row = self._row_from(edge_row)
        if row is None:
            return []
        mass = self._project(row) ** 2
        total = float(mass.sum())
        if total <= 0.0:
            return []
        cdf = np.cumsum(mass / total)
        draws = np.searchsorted(cdf, np.asarray(self.rng.random_list(count)), side="right")
        draws = np.clip(draws, 0, len(cdf) - 1)
        reverse = {i: e for e, i in self._edge_index.items()}
        return [reverse[int(d)] for d in draws]

    def _row_from(self, edge_row) -> np.ndarray | None:
        """Coerce a set/dict/array of edges into a column-aligned dense row."""
        if isinstance(edge_row, np.ndarray):
            return edge_row if edge_row.shape[0] == len(self._edge_index) else None
        row = np.zeros(len(self._edge_index), dtype=np.float64)
        items = edge_row.items() if isinstance(edge_row, dict) else ((e, 1.0) for e in edge_row)
        for edge, value in items:
            idx = self._edge_index.get(edge)
            if idx is not None:
                row[idx] = float(value)
        return row if row.any() else None

    def stats(self) -> dict:
        """Display payload; mirrors the shape the other schedulers report."""
        return {
            "fitted": self.fitted,
            "mode": self.mode,
            "rank": self.rank,
            "effective_rank": self.last_effective_rank,
            "refits": self.refits,
            "seeds": self.last_shape[0],
            "edges": self.last_shape[1],
        }
