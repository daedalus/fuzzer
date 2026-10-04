"""PositionSaliencyScheduler: NEUZZ-style learned byte importance as a position arm.

Every other position arm pools evidence by offset bin and never looks at what the
bytes *are*. This one fits a small net from seed content to the edges that seed
covers (``core/nn_saliency.py``) and proposes offsets where the input gradient of
a few target edges is large -- the bytes the model thinks decide those edges.
Idea from NEUZZ (S&P'19) / Neuzz++ (FSE'23), re-derived for numpy; no code is shared.

Training set, supplied by the caller as ``samples_fn() -> [(seed bytes, edge ids)]``
(the engine passes the corpus and ``EdgeTracker.seed_edges``)::

    keep edges hit by some but not all of the sampled seeds (a constant column has
    no gradient), merge columns with identical seed sets (same canonical class),
    thin to ``MAX_TARGETS`` evenly across the support-sorted list

Target choice follows the frontier, not NEUZZ's "never covered" edges (those have
only negative labels, so their gradient says nothing): a target column is drawn
with weight ``1 / support``, so rare edges steer most proposals. A ``target_selector``
replaces that weight; ``gt_rarity_selector`` (below) uses Simple Good-Turing adjusted
counts, which separate singletons from doubletons more sharply than the raw count::

    weight(offset) = mean over T drawn targets of |d logit / d byte|
    P(offset)      = (1 - EXPLORE) * weight / sum(weight) + EXPLORE / n

Bytes past ``INPUT_CAP`` have no saliency; they keep their uniform share (a draw
lands there with probability ``(buf_len - cap) / buf_len``). Declines (``None``)
until a fit succeeds. Refit on a cadence of ``refit_interval`` calls, skipped while
the sample is unchanged. Off-policy extra: ``record`` only counts ticks, the model
learns from the corpus, not from round outcomes. Not persisted.

Cost: one fit is ``epochs`` full-batch passes over ``<= MAX_SEEDS x input_cap``
floats; ``stats()['fit_seconds']`` reports it. A proposal is one forward pass and
one ``(input_cap x hidden) @ (hidden x T)`` product.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import numpy as np

from fuzzer_tool.core.nn_saliency import TinyMLP, encode_inputs
from fuzzer_tool.core.schedulers.pos_base import Outcome

log = logging.getLogger(__name__)

SamplesFn = Callable[[], Iterable[tuple[bytes, Collection[int]]]]

INPUT_CAP = 512  # bytes of each seed the net sees
HIDDEN = 64
MAX_SEEDS = 256  # newest training samples kept
MIN_SEEDS = 8
MAX_TARGETS = 128  # output columns
N_TARGETS = 8  # target columns averaged per proposal
EPOCHS = 150
LR = 3e-2
EXPLORE = 0.2  # uniform mixture so no byte is ever starved
REFIT_INTERVAL = 2000
RETRY_INTERVAL = 200  # while no model yet

#: Per-edge Good-Turing rarity floor so the cumulative draw never
#: sees an all-zero vector (saturated: Q1 = 0).
MIN_WEIGHT = 1e-9


TargetSelector = Callable[[np.ndarray, dict[int, list[int]], int], np.ndarray]
"""Per-column target weight: ``(edge_ids, by_edge, n_seeds) -> weights``.

Args:
    edge_ids: one representative edge id per output column (the smallest id in the
        column's seed-set class), so ``by_edge[int(edge_ids[j])]`` is that column's seeds.
    by_edge: edge id -> indices of the training seeds that hit it (every edge, including
        constant and merged ones, so selectors can see the whole frequency spectrum).
    n_seeds: number of training seeds.

Returns:
    Non-negative weights, same length as ``edge_ids``. They REPLACE the default
    ``1 / support``; a larger weight makes the column a likelier gradient target.
"""


def _adjusted_counts(counts: np.ndarray, spectrum: dict[int, int]) -> np.ndarray:
    """Simple Good-Turing adjusted counts c* (Gale & Sampson 1995), log-log smoothed.

    ``spectrum`` is N_c, the number of edges seen exactly c times. Raw
    ``(c+1) N_{c+1} / N_c`` is erratic on a small corpus (N_{c+1} is often 0), so N_c is
    smoothed by a least-squares line ``log N_c = a + b log c`` over the observed c,
    with Gale-Sampson's Z_c gap averaging. ``c* = c (1 + 1/c)^(b + 1)``.
    Falls back to ``c* = c`` (no adjustment) unless the slope is below -1, which is the
    condition for c* to grow with c; so the resulting rarity order is never inverted.
    """
    cs = np.array(sorted(spectrum), dtype=np.float64)
    if len(cs) < 2:
        return counts.astype(np.float64)
    nc = np.array([spectrum[int(c)] for c in cs], dtype=np.float64)
    lower = np.concatenate(([0.0], cs[:-1]))
    upper = np.concatenate((cs[1:], [2.0 * cs[-1] - cs[-2]]))
    z = nc / (0.5 * (upper - lower))
    slope, _ = np.polyfit(np.log(cs), np.log(z), 1)
    if not np.isfinite(slope) or slope >= -1.0:
        return counts.astype(np.float64)
    c = counts.astype(np.float64)
    return c * (1.0 + 1.0 / c) ** (slope + 1.0)


def gt_rarity_selector(
    edge_ids: np.ndarray, by_edge: dict[int, list[int]], n_seeds: int
) -> np.ndarray:
    """Target weight ``1 / c*``: the inverse Good-Turing adjusted hit count of the column.

    The default ``1 / support`` treats the raw count as the edge's hit rate; Good-Turing
    estimates that rate from the frequency-of-frequencies instead, which is what makes a
    singleton (the frontier) stand out against edges seen a few times. Weights are NOT
    multiplied by ``1 / support``: that would count rarity twice.
    """
    if len(edge_ids) == 0:
        return np.ones(0, np.float64)
    spectrum: dict[int, int] = {}
    for rows in by_edge.values():
        spectrum[len(rows)] = spectrum.get(len(rows), 0) + 1
    support = np.array([len(by_edge.get(int(e), ())) for e in edge_ids], dtype=np.float64)
    ok = support > 0
    cstar = np.ones(len(edge_ids), np.float64)
    if ok.any():
        cstar[ok] = _adjusted_counts(support[ok], spectrum)
    w = np.zeros(len(edge_ids), np.float64)
    w[ok] = 1.0 / np.maximum(cstar[ok], MIN_WEIGHT)
    return w


class PositionSaliencyScheduler:
    """Position arm: offsets drawn by learned input-gradient magnitude."""

    name = "saliency"

    #: Position proposers take no Beta priors (mirrors the other pos arms).
    supports_priors = False

    def __init__(
        self,
        rng: Any,
        samples_fn: SamplesFn,
        input_cap: int = INPUT_CAP,
        hidden: int = HIDDEN,
        epochs: int = EPOCHS,
        refit_interval: int = REFIT_INTERVAL,
        n_targets: int = N_TARGETS,
        explore: float = EXPLORE,
        target_selector: TargetSelector | None = None,
    ) -> None:
        if rng is None:
            raise ValueError("PositionSaliencyScheduler requires a RandPool (Hard Rule 16)")
        if input_cap < 1 or hidden < 1 or epochs < 1 or n_targets < 1:
            raise ValueError("input_cap, hidden, epochs and n_targets must be >= 1")
        if not 0.0 <= explore <= 1.0:
            raise ValueError("explore must be in [0, 1]")
        self._rng = rng
        self._samples_fn = samples_fn
        self._cap = input_cap
        self._hidden = hidden
        self._epochs = epochs
        self._interval = max(1, refit_interval)
        self._n_targets = n_targets
        self._explore = explore
        self._target_selector = target_selector

        self._model: TinyMLP | None = None
        self._width = 0
        self._cum_support: np.ndarray | None = None  # cumulative target weights, one per column
        self._ticks = 0
        self._last_try = -(1 << 60)
        self._stamp: tuple[int, int] | None = None
        self._fits = 0
        self._fit_seconds = 0.0
        self._last_loss = float("nan")
        self._skip_reason = "not fitted"

    # -- feedback ---------------------------------------------------------------

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Counts a tick toward the refit cadence; the net learns from the corpus.

        Also gives a first / due refit its chance, so the model exists even when the
        arena rarely draws this arm (the saliency_ladder operator needs it too).
        """
        self._ticks += 1
        self._maybe_fit()

    # -- proposal ---------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """An offset drawn by saliency; None until a model exists."""
        if buf_len <= 0 or not data:
            return None
        self._ticks += 1
        self._maybe_fit()
        model = self._model
        if model is None:
            return None

        head = min(len(data), buf_len, self._cap)
        # Bytes the net cannot see keep the uniform density.
        if buf_len > head and self._rng.random() < (buf_len - head) / buf_len:
            return int(self._rng.randint(head, buf_len - 1))

        s = self.saliency(data)[:head]
        total = float(s.sum())
        if not np.isfinite(total) or total <= 0.0:
            return int(self._rng.randint(0, head - 1))
        p = (1.0 - self._explore) * s / total + self._explore / head
        cum = np.cumsum(p)
        i = int(np.searchsorted(cum, self._rng.random() * cum[-1], side="right"))
        return min(i, head - 1)

    def warm(self) -> bool:
        """Tick, give a due refit its chance, and say whether a model exists (operator gate)."""
        self._ticks += 1
        self._maybe_fit()
        return self._model is not None

    def gradient_info(self, data: bytes, top: int = 32) -> tuple[np.ndarray, np.ndarray] | None:
        """Signed gradient of ONE drawn target edge: (offsets by |grad| desc, signs).

        NEUZZ's mutation input. One target, not the mean ``saliency`` uses: signed
        gradients of different edges point opposite ways and cancel when averaged.
        None without a model or when the gradient is identically zero.
        """
        model, cum = self._model, self._cum_support
        if model is None or cum is None or not data:
            return None
        x = encode_inputs([data], self._width)[0]
        k = self._draw_targets(cum)[0]
        g = model.input_grad(x, [k])[: min(len(data), self._cap), 0]
        order = np.argsort(-np.abs(g), kind="stable")[: max(1, top)]
        order = order[g[order] != 0.0]
        if not len(order):
            return None
        return order.astype(np.int64), np.sign(g[order]).astype(np.int64)

    def saliency(self, data: bytes) -> np.ndarray:
        """Mean |gradient| per byte (length ``min(len(data), cap)``) over freshly drawn targets."""
        model, cum = self._model, self._cum_support
        if model is None or cum is None:
            return np.zeros(min(len(data), self._cap), np.float64)
        x = encode_inputs([data], self._width)[0]
        targets = self._draw_targets(cum)
        return model.saliency(x, targets)[: min(len(data), self._cap)].astype(np.float64)

    def _draw_targets(self, cum: np.ndarray) -> list[int]:
        draws = self._rng.random_list(self._n_targets)
        idx = np.searchsorted(cum, np.asarray(draws) * cum[-1], side="right")
        return [int(min(i, len(cum) - 1)) for i in idx]

    # -- fit --------------------------------------------------------------------

    def _maybe_fit(self) -> None:
        wait = self._interval if self._model is not None else RETRY_INTERVAL
        if self._ticks - self._last_try < wait:
            return
        self._last_try = self._ticks
        try:
            self.refit()
        except Exception:  # a refit must never take the campaign down
            log.exception("saliency refit failed")
            self._skip_reason = "refit raised"

    def refit(self, force: bool = False) -> bool:
        """Rebuild the net from the current samples. False if skipped (see ``stats``)."""
        samples = [(d, e) for d, e in self._samples_fn() if d and e]
        stamp = (len(samples), sum(len(e) for _, e in samples))
        if len(samples) < MIN_SEEDS:
            self._skip_reason = f"fewer than {MIN_SEEDS} seeds"
            return False
        if not force and stamp == self._stamp and self._model is not None:
            self._skip_reason = "sample unchanged"
            return False
        samples = samples[-MAX_SEEDS:]

        n = len(samples)
        by_edge: dict[int, list[int]] = {}
        for i, (_, edges) in enumerate(samples):
            for e in edges:
                by_edge.setdefault(int(e), []).append(i)
        # seed-index set -> (support, smallest edge id in the class): identical columns
        # merge; constant columns (hit by every seed) carry no gradient and are dropped.
        cols: dict[frozenset[int], tuple[int, int]] = {}
        for e in sorted(by_edge):
            rows = by_edge[e]
            if 0 < len(rows) < n:
                cols.setdefault(frozenset(rows), (len(rows), e))
        if not cols:
            self._skip_reason = "no informative edges"
            return False

        ordered = sorted(cols.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        if len(ordered) > MAX_TARGETS:
            pick = np.linspace(0, len(ordered) - 1, MAX_TARGETS).astype(int)
            ordered = [ordered[i] for i in pick]

        support = np.array([v[0] for _, v in ordered], dtype=np.float64)
        edge_ids = np.array([v[1] for _, v in ordered], dtype=np.int64)
        weights = 1.0 / support
        if self._target_selector is not None:
            try:
                sel = np.asarray(self._target_selector(edge_ids, by_edge, n), dtype=np.float64)
                if sel.shape != weights.shape or not np.isfinite(sel).all() or (sel < 0).any():
                    raise ValueError(f"selector returned shape {sel.shape} / bad values")
                weights = sel
            except Exception:  # a bad selector must not stop the model from fitting
                log.exception("saliency target_selector failed; using 1/support")
        weights = np.maximum(weights, MIN_WEIGHT)

        y = np.zeros((n, len(ordered)), np.float32)
        for j, (rows, _) in enumerate(ordered):
            y[list(rows), j] = 1.0

        width = min(self._cap, max(len(d) for d, _ in samples))
        x = encode_inputs([d for d, _ in samples], width)
        gen = np.random.default_rng(self._rng.randint(0, 2**31 - 1))
        model = TinyMLP(width, self._hidden, len(ordered), gen)

        t0 = time.perf_counter()
        loss = model.fit(x, y, epochs=self._epochs, lr=LR)
        self._fit_seconds = time.perf_counter() - t0

        self._model, self._width = model, width
        self._cum_support = np.cumsum(weights)  # with the model: a failed fit leaves both old
        self._stamp = stamp
        self._fits += 1
        self._last_loss = loss
        self._skip_reason = "ok"
        return True

    # -- introspection ----------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._model is not None

    def stats(self) -> dict[str, Any]:
        return {
            "fits": self._fits,
            "ready": self.ready,
            "targets": 0 if self._cum_support is None else len(self._cum_support),
            "width": self._width,
            "loss": self._last_loss,
            "fit_seconds": self._fit_seconds,
            "skip_reason": self._skip_reason,
        }
