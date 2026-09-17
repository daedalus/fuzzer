"""Family-wise / false-discovery-rate correction for concurrent hypothesis tests.

Why this exists
----------------
Several analyzers run independent formal hypothesis tests on the *same*
per-tick discovery-count series (``delta`` in ``services/fuzzer.py``'s stats
loop):

- ``structure_function.DispersionIndex.dispersion_pvalue`` -- Poisson
  dispersion test (``(n-1)*D ~ chi-squared(n-1)``), one-sided, folded to
  two-sided by the caller via ``is_overdispersed``/``is_underdispersed``.
- ``discovery_uniformity.dispersion_pvalue`` -- the *same* Poisson dispersion
  statistic, independently windowed, already two-sided.
- ``garch.ljung_box`` -- a Ljung-Box portmanteau test for autocorrelation in
  conditional variance (ARCH effects). A different null than the two above,
  but run on the same input series.
- ``periodicity``'s Fisher's g-test -- spectral test for a dominant
  periodicity, run here over first-differences of the fuzzer's persistent
  ``_discovery_edges`` history (the same discovery-rate series
  ``services/report.py``'s ``_spectral_diagnostics`` scans, read the same
  way).

Each of these is well-calibrated *on its own*: run in isolation, it rejects
its null at its stated alpha the stated fraction of the time. But they are
not run in isolation -- they run every stats tick, for the life of a
campaign, and (for structure_function and discovery_uniformity in
particular) on statistically dependent input. Reading any one p-value as if
it carries its nominal false-positive rate, when several such readings are
taken and at least glanced at together, overstates how surprised you should
be by any single small p-value. This module does not change what any
detector computes; it only says, given a batch of p-values collected at the
same tick, which ones remain significant once that multiplicity is
accounted for.

Two standard procedures are provided:

- :func:`holm_bonferroni` -- controls the family-wise error rate (FWER):
  the probability of *any* false rejection in the batch. Conservative;
  appropriate when a false positive from any single test is costly enough
  that you want the classical guarantee.
- :func:`benjamini_hochberg` -- controls the false discovery rate (FDR):
  the *expected proportion* of false rejections among those rejected.
  Less conservative than Holm-Bonferroni, more suited to a monitoring
  dashboard where a small fraction of false alarms among several is
  tolerable as long as most flagged items are real.

Both take p-values with ``None`` entries allowed (a detector that hasn't
seen enough data yet returns ``None`` rather than a p-value) and pass them
through untested -- ``None`` never counts toward the family size ``m`` and
is never itself rejected.

This module intentionally does not decide *what* to do with a rejection --
it is a pure statistics utility. Wiring its output into a decision path
(e.g. gating stall detection) is a separate, deliberate step: see
``docs/handover/handover_multiple_testing_2026-09-13.md`` for why that step
is *not* taken here.

Reference
---------
Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
Scandinavian Journal of Statistics.
Benjamini, Y. and Hochberg, Y. (1995). "Controlling the false discovery
rate: a practical and powerful approach to multiple testing." JRSS-B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FuzzerLike = Any
"""Duck-typed ``Fuzzer`` -- see ``core.analyzer_registry.FuzzerLike``. Kept
as a separate alias here rather than importing that one so this module has
no dependency on ``analyzer_registry`` (or on ``services.fuzzer``, which
would be circular)."""


@dataclass(frozen=True)
class Correction:
    """One named test's outcome after correction, alongside its raw input.

    ``adjusted`` is the Benjamini-Hochberg adjusted p-value (a.k.a. q-value)
    for :func:`benjamini_hochberg` results, or ``None`` for
    :func:`holm_bonferroni` results (Holm's procedure is defined via a
    rejection/no-rejection decision at a chosen alpha, not a per-test
    adjusted value in the same closed form as BH's).
    """

    name: str
    p_value: float
    rejected: bool
    adjusted: float | None = None


def holm_bonferroni(
    named_pvalues: dict[str, float | None], alpha: float = 0.05
) -> list[Correction]:
    """Holm-Bonferroni step-down procedure. Controls family-wise error rate.

    Entries whose value is ``None`` are skipped entirely (excluded from both
    the family size ``m`` and the returned list). Returns one
    :class:`Correction` per non-``None`` input, in no particular guaranteed
    order beyond "ascending by p-value internally, then re-expressed against
    the original names" -- callers that need a stable order should sort the
    result themselves.

    Procedure: sort ascending as ``p(1) <= ... <= p(m)``. Reject ``p(1),
    ..., p(k)`` for the largest ``k`` such that ``p(i) <= alpha / (m - i +
    1)`` holds for every ``i <= k`` (equivalently: walk from the smallest
    p-value up, stop at the first ``i`` that fails its threshold, reject
    everything strictly before it -- this is the standard sequentially
    rejective form and is what makes Holm uniformly more powerful than
    plain Bonferroni while keeping the same FWER guarantee).
    """
    items = [(name, p) for name, p in named_pvalues.items() if p is not None]
    m = len(items)
    if m == 0:
        return []
    items.sort(key=lambda kv: kv[1])

    rejected_count = 0
    for i, (_, p) in enumerate(items):  # i is 0-based; Holm's i is 1-based
        threshold = alpha / (m - i)
        if p <= threshold:
            rejected_count = i + 1
        else:
            break

    return [
        Correction(name=name, p_value=p, rejected=(i < rejected_count))
        for i, (name, p) in enumerate(items)
    ]


def benjamini_hochberg(
    named_pvalues: dict[str, float | None], alpha: float = 0.05
) -> list[Correction]:
    """Benjamini-Hochberg step-up procedure. Controls false discovery rate.

    Entries whose value is ``None`` are skipped (see :func:`holm_bonferroni`
    for the same convention). Returns one :class:`Correction` per non-
    ``None`` input, each carrying its BH-adjusted p-value (q-value) in
    ``adjusted`` regardless of whether it was rejected, so callers can see
    how close a non-rejected test came.

    Procedure: sort ascending as ``p(1) <= ... <= p(m)``. Reject ``p(1),
    ..., p(k)`` for the largest ``k`` such that ``p(k) <= (k/m) * alpha``.
    Adjusted p-values (q-values) are computed via the standard
    cumulative-minimum-from-the-largest construction: ``q(i) = min(1,
    min_{j>=i} (m/j) * p(j))``, which is monotone non-decreasing as required
    of an adjusted-p-value sequence.
    """
    items = [(name, p) for name, p in named_pvalues.items() if p is not None]
    m = len(items)
    if m == 0:
        return []
    items.sort(key=lambda kv: kv[1])

    # BH-adjusted p-values: q(i) = min_{j>=i} (m/j) * p(j), clamped to 1,
    # enforced non-decreasing by taking a running minimum from the end.
    raw_adjusted = [min(1.0, (m / (i + 1)) * p) for i, (_, p) in enumerate(items)]
    adjusted = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        running_min = min(running_min, raw_adjusted[i])
        adjusted[i] = running_min

    # Largest k with p(k) <= (k/m) * alpha; reject 1..k.
    rejected_count = 0
    for i, (_, p) in enumerate(items):
        if p <= ((i + 1) / m) * alpha:
            rejected_count = i + 1

    return [
        Correction(
            name=name,
            p_value=p,
            rejected=(i < rejected_count),
            adjusted=adjusted[i],
        )
        for i, (name, p) in enumerate(items)
    ]


def _fold_two_sided(p_upper: float) -> float:
    """Fold a one-sided upper-tail p-value to a two-sided one.

    Same construction ``discovery_uniformity``/``birthday_spacings`` use:
    twice the smaller tail, clamped to ``[0, 1]``.
    """
    return min(1.0, 2.0 * min(p_upper, 1.0 - p_upper))


def _periodicity_pvalue(fuzzer: FuzzerLike) -> float | None:
    """Fisher's g-test p-value for a periodic component in the discovery-
    rate series, read the same way ``services/report.py``'s
    ``_spectral_diagnostics`` does: first-differences of
    ``fuzzer._discovery_edges`` (the persistent cumulative-edges-per-sync-
    interval history -- it does exist on the fuzzer object; an earlier
    version of this module said otherwise before this attribute was
    located).

    Returns ``None`` when numpy isn't installed (``core.periodicity``
    hard-imports it), the attribute is missing, or there aren't at least
    two points to difference -- in every such case there is nothing to
    test yet, not a computed non-finding.

    One caveat worth carrying forward rather than silently absorbing: per
    ``detect_periodicity``'s own docstring, when the series needed AR
    drift-removal first (``res.ar_order > 0``) its p-value's actual
    false-positive rate runs closer to ~0.10 than its nominal alpha --
    report.py already flags this to the reader as "a lead rather than a
    finding." Folding a p-value with a known-off calibration into a
    procedure that assumes each input alpha is honest is an approximation,
    not a rigorous combination; it is included anyway because the
    alternative (silently dropping the one test that most directly
    targets corpus-sync artifacts) is the worse approximation, and BH
    itself degrades gracefully -- a single miscalibrated input shifts
    where that one item lands, it does not invalidate the others' ranks.
    """
    try:
        edges_series = fuzzer._discovery_edges
    except AttributeError:
        return None
    if edges_series is None or len(edges_series) < 51:
        return None
    try:
        from fuzzer_tool.core.periodicity import detect_periodicity
    except ImportError:
        return None

    deltas = [
        float(b) - float(a)
        for a, b in zip(edges_series[:-1], edges_series[1:], strict=True)
    ]
    res = detect_periodicity(deltas, min_samples=50)
    return res.p_value


def collect_current_pvalues(fuzzer: FuzzerLike) -> dict[str, float | None]:
    """Gather this tick's p-values from the detectors known to test the
    same (or a directly derived) per-tick edge-discovery-count series.

    Covers ``_structure_fn`` (Poisson dispersion, folded two-sided to
    match the others), ``_discovery_uniformity`` (the same Poisson
    dispersion statistic, independently windowed), ``_garch`` (Ljung-Box
    test for ARCH effects -- a different null, same series), and
    ``periodicity``'s Fisher's g-test over first-differences of
    ``_discovery_edges`` (see :func:`_periodicity_pvalue` for the
    calibration caveat that one carries).

    Missing/not-yet-available detectors and detectors that report ``None``
    (not enough data yet) are simply absent from the returned dict, which
    is exactly what :func:`holm_bonferroni`/:func:`benjamini_hochberg`
    expect.
    """
    pvalues: dict[str, float | None] = {}

    structure_fn = getattr(fuzzer, "_structure_fn", None)
    if structure_fn is not None:
        p_upper = structure_fn.dispersion_pvalue()
        if p_upper is not None:
            pvalues["structure_function_dispersion"] = _fold_two_sided(p_upper)

    discovery_uniformity = getattr(fuzzer, "_discovery_uniformity", None)
    if discovery_uniformity is not None:
        verdict = discovery_uniformity.verdict()
        # verdict()["p"] is 1.0 (not None) before min_obs -- a legitimate
        # "cannot reject" p-value, not a missing one, so it is included
        # as-is rather than filtered like the None cases above.
        if verdict.get("n", 0) >= 2:
            pvalues["discovery_uniformity_dispersion"] = verdict["p"]

    garch = getattr(fuzzer, "_garch", None)
    if garch is not None:
        _stat, p = garch.ljung_box()
        pvalues["garch_ljung_box"] = p

    periodicity_p = _periodicity_pvalue(fuzzer)
    if periodicity_p is not None:
        pvalues["periodicity_discovery_rate"] = periodicity_p

    return pvalues


def collect_and_correct(
    fuzzer: FuzzerLike, alpha: float = 0.05
) -> list[Correction]:
    """Convenience wrapper: :func:`collect_current_pvalues` followed by
    :func:`benjamini_hochberg`.

    FDR control (not FWER) is used here deliberately: this is a monitoring
    readout across a handful of related tests, not a single make-or-break
    decision, so the less conservative BH procedure is the better default.
    Callers that specifically want the stronger FWER guarantee can call
    :func:`holm_bonferroni` directly on :func:`collect_current_pvalues`'s
    output instead.
    """
    return benjamini_hochberg(collect_current_pvalues(fuzzer), alpha=alpha)
