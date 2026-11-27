"""Regression: the Monte-Carlo null budget ran on a clock.

``_null_js_quantiles`` stopped drawing replicates once 50 ms had passed on the
scheduler's clock. The Fuzzer passed its virtual clock, which never advances
inside the loop, so every refit ran all 40 replicates (19.4% of wall on
GStreamer ``--elo all``); on the wall clock the count tracked machine load,
so ``--seed`` runs diverged. The count is now fixed by the input size alone.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

SEED = 11


def _freq(positions, n, seed):
    """Per-position byte histograms of *n* observations."""
    r = random.Random(seed)
    out = {}
    for pos in range(positions):
        counts: dict[int, int] = {}
        for _ in range(n):
            v = r.randrange(32)
            counts[v] = counts.get(v, 0) + 1
        out[pos] = counts
    return out


def _mc(positions, n=10):
    mc = MonteCarloScheduler(rng=RandPool(seed=SEED))
    mc._prev_byte_freq = _freq(positions, n, 1)
    mc.byte_freq = _freq(positions, n, 2)
    return mc


def _record(monkeypatch) -> list[dict]:
    """Capture each numpy null call's replicate count and deadline."""
    calls: list[dict] = []
    real = MonteCarloScheduler._null_js_samples_numpy

    def spy(pooled, replicates, *a, **kw):
        calls.append({"replicates": replicates, "deadline": kw.get("deadline")})
        return real(pooled, replicates, *a, **kw)

    monkeypatch.setattr(MonteCarloScheduler, "_null_js_samples_numpy", staticmethod(spy))
    return calls


def _expected(positions, n=10):
    units = positions * (256 + 2 * n)
    reps = MonteCarloScheduler._JS_NULL_WORK_BUDGET // units
    return max(MonteCarloScheduler._JS_NULL_MIN_REPLICATES,
               min(MonteCarloScheduler._JS_NULL_REPLICATES, reps))  # fmt: skip


def test_regression_mc_null_work_budget(monkeypatch):
    """No deadline; the count follows the work model."""
    calls = _record(monkeypatch)
    _mc(2048)._null_js_quantiles((0.05, 0.01))
    assert calls[0]["deadline"] is None
    assert calls[0]["replicates"] == _expected(2048)


def test_quantiles_repeat_exactly():
    """Falsification: same seed and inputs, same quantiles, any load."""
    a = _mc(512)._null_js_quantiles((0.05, 0.01))
    b = _mc(512)._null_js_quantiles((0.05, 0.01))
    assert a == b and a[0.05] > 0.0


def test_huge_input_keeps_floor_small_input_keeps_cap(monkeypatch):
    """Adversarial: extreme sizes clamp to the floor and the cap."""
    calls = _record(monkeypatch)
    _mc(60_000)._null_js_quantiles((0.05,))
    _mc(4)._null_js_quantiles((0.05,))
    assert calls[0]["replicates"] == MonteCarloScheduler._JS_NULL_MIN_REPLICATES
    assert calls[1]["replicates"] == MonteCarloScheduler._JS_NULL_REPLICATES
