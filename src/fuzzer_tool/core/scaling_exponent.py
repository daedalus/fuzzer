"""Anomalous-diffusion scaling exponent for a directed-distance trajectory.

Where this sits
----------------
``core/structure_function.py`` (formerly ``allan_variance.py``) fits a
log-log least-squares slope to Allan deviation vs. averaging time for the
*edge-discovery-rate* series, to tell whether the discovery process is
stationary or approaching a stall. This module reuses the same idea --
mean squared displacement (MSD) at lag tau, log-log OLS slope over
several taus -- applied instead to the AFLGo directed-distance signal
(``avg_distance`` in ``services/fuzzer.py``), which the structure-function
detector has never been fed.

The question this answers, that nothing else in the tree currently can:
is AFLGo distance-guided scheduling actually producing directed progress,
or would an undirected random walk over the same reachable blocks do just
as well?

**Status note (see docs/handover/handover_thermo_stochastic_concepts_2026-09-12.md,
P3-T5):** this proposal was audited and gated on three open questions --
which estimator to reuse (moot here: this module never called into
allan_variance/structure_function, it implements its own MSD/OLS
directly), the per-seed-vs-per-tick trajectory definition (this module
takes whatever scalar series it is fed, once per tick, and leaves that
choice to the caller), and the no-``--target-functions`` constant-signal
case (handled below: a constant series returns ``alpha=None`` /
``"undefined"``, never a false classification). Wiring is restored here on
request; the gating questions above are still open and worth reading
before trusting a ``ballistic``/``trapped`` verdict on a real campaign.

For a discretely sampled real-valued process x[i]:

    MSD(tau) = mean_i (x[i+tau] - x[i])^2

and the classical anomalous-diffusion classification from the scaling
exponent alpha in MSD(tau) ~ tau^alpha:

  - alpha ~= 1   : diffusive (Brownian) -- no net directional drift beyond
                   what a random walk over reachable blocks would produce
  - alpha > ~1.2 : superdiffusive / ballistic -- distance falls faster than
                   diffusion predicts, i.e. genuine directed progress
  - alpha < ~0.8 : subdiffusive / trapped -- distance is confined near a
                   level, consistent with scheduling stuck against a
                   structural barrier

alpha is estimated by closed-form OLS on (log tau, log MSD(tau)) pairs.
No numpy/scipy dependency, matching the rest of ``core/``.

Falsifiers this was checked against (see ``tests/test_scaling_exponent.py``):
a pure symmetric random walk must recover alpha near 1; a linear drift with
small additive noise must recover alpha near 2; an i.i.d.-noise-about-a-
constant series (bounded, non-cumulative) must recover alpha near 0.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

# Classification thresholds on the fitted exponent alpha.
_SUBDIFFUSIVE_MAX = 0.8
_SUPERDIFFUSIVE_MIN = 1.2

# Minimum number of samples in the window before a verdict is attempted.
_MIN_SAMPLES = 16


def mean_squared_displacement(values: Sequence[float], tau: int) -> float | None:
    """Mean of (values[i+tau] - values[i])**2 over all valid i.

    Returns None if there are fewer than one valid pair (tau >= len(values)).
    """
    n = len(values)
    if tau <= 0 or tau >= n:
        return None
    sq_sum = 0.0
    count = 0
    for i in range(n - tau):
        d = values[i + tau] - values[i]
        sq_sum += d * d
        count += 1
    if count == 0:
        return None
    return sq_sum / count


def estimate_scaling_exponent(
    values: Sequence[float], taus: Sequence[int] | None = None
) -> float | None:
    """OLS-fit log(MSD(tau)) vs log(tau); return the slope (alpha), or None.

    None is returned when there are too few (tau, MSD) pairs to fit (fewer
    than 2), or when MSD is zero/undefined at every candidate tau (a
    perfectly constant series -- no displacement to measure at all).
    """
    n = len(values)
    if taus is None:
        # MSD(tau) for a single realization averages over only (n - tau)
        # overlapping, highly-correlated windows -- at tau close to n that
        # average is effectively one sample repeated with small shifts, not
        # an independent replication, so its variance (and the downward
        # log-bias that follows from it) blows up long before tau reaches
        # n. Keeping max_tau well below n keeps enough independent windows
        # per tau for the estimate to be trustworthy; n/10 was chosen
        # empirically (see tests/test_scaling_exponent.py) as the point
        # where a symmetric random walk's fitted alpha clusters around 1
        # instead of systematically undershooting it.
        max_tau = max(2, n // 10)
        taus = range(1, max_tau + 1)

    xs: list[float] = []
    ys: list[float] = []
    for tau in taus:
        msd = mean_squared_displacement(values, tau)
        if msd is None or msd <= 0.0:
            continue
        xs.append(math.log(tau))
        ys.append(math.log(msd))

    n_pts = len(xs)
    if n_pts < 2:
        return None

    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sxx = sum(x * x for x in xs)
    denom = n_pts * sxx - sx * sx
    if denom == 0:
        return None
    return (n_pts * sxy - sx * sy) / denom


def classify_exponent(alpha: float) -> str:
    """Map a fitted scaling exponent to a diffusion-regime label."""
    if alpha < _SUBDIFFUSIVE_MAX:
        return "trapped"
    if alpha > _SUPERDIFFUSIVE_MIN:
        return "ballistic"
    return "diffusive"


class ScalingExponentDetector:
    """Rolling-window scaling-exponent classifier for a sampled trajectory.

    Feed it one value per stats tick (e.g. the current AFLGo ``avg_distance``
    reading); call :meth:`verdict` for the current classification.
    """

    def __init__(self, window: int = 128) -> None:
        self._window = window
        self._values: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._values.append(value)

    def verdict(self) -> dict:
        """Return the current classification.

        ``{"state": "insufficient_data"}`` until at least ``_MIN_SAMPLES``
        values have been observed. ``{"state": "undefined"}`` when the
        window is long enough but the series has no measurable displacement
        at any tau (perfectly flat -- e.g. no ``--target-functions``, so
        distance is a constant).
        """
        n = len(self._values)
        if n < _MIN_SAMPLES:
            return {"state": "insufficient_data", "n": n, "alpha": None}
        alpha = estimate_scaling_exponent(list(self._values))
        if alpha is None:
            return {"state": "undefined", "n": n, "alpha": None}
        return {"state": classify_exponent(alpha), "n": n, "alpha": alpha}
