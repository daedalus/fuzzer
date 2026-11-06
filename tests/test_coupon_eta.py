"""Coupon-collector ETA: executions until the next k new edges.

Inverts the Good-Toulmin forecast U(h) (expected new edges in the next h
executions). Expected values come from closed forms here (Q1-only series,
the exact incidence-model expectation over planted probabilities), not from
the module under test.
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.schedulers.seed_good_turing import (
    GoodTuringSeedStrategy,
    execs_to_new,
    good_toulmin,
)
from fuzzer_tool.services.stats import StatsReporter


class _Rng:
    def weighted_choice(self, seq, weights):
        return seq[0]


def _expected_qk(probs: list[float], t: int) -> list[float]:
    """E[Q_i] = sum_e C(t,i) p^i (1-p)^(t-i)."""
    return [
        sum(math.comb(t, i) * p**i * (1 - p) ** (t - i) for p in probs) for i in range(1, t + 1)
    ]


def _true_new(probs: list[float], t: int, h: float) -> float:
    """E[new edges in h more execs] = sum_e (1-p)^t (1 - (1-p)^h)."""
    return sum((1 - p) ** t * (1 - (1 - p) ** h) for p in probs)


def _invert(fn, k: float, hi: float) -> float:
    """Independent bisection oracle on a monotone fn."""
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) < k else (lo, mid)
    return hi


# ── Pure inversion ─────────────────────────────────────────────────────


def test_singletons_only_is_linear():
    """Q1 only: U(h) = h * Q1 / t, so h = k * t / Q1."""
    assert execs_to_new([8], 100, 2.0) == pytest.approx(2.0 * 100 / 8)


def test_falsify_exact_against_planted_probabilities():
    """Fed E[Q_i], the ETA equals the exact inverse of E[new](h)."""
    probs = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.004, 0.001]
    t = 60
    qk = _expected_qk(probs, t)
    k = 0.5 * _true_new(probs, t, 0.5 * t)

    truth = _invert(lambda h: _true_new(probs, t, h), k, 0.5 * t)
    assert execs_to_new(qk, t, k) == pytest.approx(truth, rel=1e-6)


def test_oracle_control_inverts_itself():
    """Hard Rule 46: the oracle recovers a known h from its own forward map."""
    probs = [0.3, 0.01]
    t, h = 40, 7.0
    assert _invert(lambda x: _true_new(probs, t, x), _true_new(probs, t, h), t) == pytest.approx(
        h, rel=1e-9
    )


def test_inverse_property():
    """U(eta) == k for any reachable k."""
    qk, t = [40, 12, 6, 3, 1, 0, 0, 0], 100
    k = 10.0
    eta = execs_to_new(qk, t, k)
    assert eta is not None
    assert good_toulmin(qk, t, eta / t) == pytest.approx(k, rel=1e-6)


def test_beyond_horizon_is_none():
    """No extrapolation past ratio_max * t: the series is unreliable there."""
    qk, t = [4, 1], 50
    reach = good_toulmin(qk, t, 0.5)
    assert execs_to_new(qk, t, reach * 1.01, ratio_max=0.5) is None
    assert execs_to_new(qk, t, reach * 0.99, ratio_max=0.5) is not None


@pytest.mark.parametrize(
    ("qk", "t", "k"),
    [([], 10, 1.0), ([0, 0], 10, 1.0), ([5], 0, 1.0), ([5], 10, 0.0), ([5], 10, -1.0)],
)
def test_degenerate_inputs(qk, t, k):
    """Empty / saturated / no-sample / non-positive targets: None, never 0."""
    assert execs_to_new(qk, t, k) is None


def test_adversarial_non_monotone_returns_first_crossing():
    """Heavy Q2 bends U(h) down; the ETA is the earliest h reaching k.

    U(h) = h*q1/t - h(h+1)/(t(t-1)) * q2: peaks then falls. A bisection over
    the whole window would land on the falling side or miss the crossing.
    """
    qk, t = [10, 30], 20
    peak_h = max((i / 100 for i in range(1, 1001)), key=lambda h: good_toulmin(qk, t, h / t))
    k = 0.8 * good_toulmin(qk, t, peak_h / t)
    eta = execs_to_new(qk, t, k, ratio_max=0.5)

    assert eta is not None
    assert eta < peak_h
    assert good_toulmin(qk, t, eta / t) == pytest.approx(k, rel=1e-6)


def test_huge_counts_finite():
    eta = execs_to_new([10**6, 10**5, 10**4], 10**9, 10.0)
    assert eta is not None and math.isfinite(eta) and eta > 0


# ── Strategy wiring ────────────────────────────────────────────────────


def _feed(strat: GoodTuringSeedStrategy, execs: list[set[int]]) -> None:
    for edges in execs:
        strat.observe(b"s", edges)


def test_strategy_gt_matches_linear_eta():
    strat = GoodTuringSeedStrategy(_Rng(), estimator="gt", min_observations=1)
    execs = [{0, 1}, {0, 2}, {0, 3}, {0}] * 5  # edges 1..3 hit 5 times, none singleton
    execs += [{100 + i} for i in range(4)]  # 4 singletons
    _feed(strat, execs)

    t = len(execs)
    assert strat.execs_to_edges(1) == pytest.approx(t / 4)


def test_strategy_gtoul_uses_series():
    strat = GoodTuringSeedStrategy(_Rng(), estimator="gtoul", min_observations=1)
    execs = [{i, i + 1} for i in range(30)]
    _feed(strat, execs)

    g = strat._global
    assert strat.execs_to_edges(2) == pytest.approx(execs_to_new(g.qk, g.t, 2, ratio_max=0.5))


def test_strategy_saturated_is_none():
    strat = GoodTuringSeedStrategy(_Rng(), estimator="gt", min_observations=1)
    _feed(strat, [{1, 2}] * 10)
    assert strat.execs_to_edges(1) is None


# ── Summary wiring ─────────────────────────────────────────────────────


def _summary(f) -> str:
    reporter = StatsReporter.__new__(StatsReporter)
    buf = io.StringIO()
    with redirect_stdout(buf):
        reporter._print_summary_eta(f)
    return buf.getvalue()


def test_summary_prints_eta():
    strat = GoodTuringSeedStrategy(_Rng(), estimator="gt", min_observations=1)
    _feed(strat, [{0}] * 36 + [{100 + i} for i in range(4)])  # t=40, q1=4 -> 10 execs/edge

    out = _summary(SimpleNamespace(_good_turing_seed=strat))
    assert "Next 1 new edge" in out
    assert "~10 execs" in out


def test_summary_eta_beyond_horizon_says_so():
    strat = GoodTuringSeedStrategy(_Rng(), estimator="gt", min_observations=1)
    _feed(strat, [{0}] * 39 + [{100}])  # 40 execs/edge: 10 edges past 0.5 * t
    assert "> 20 execs" in _summary(SimpleNamespace(_good_turing_seed=strat))


def test_summary_eta_absent_without_arm():
    assert _summary(SimpleNamespace()) == ""
