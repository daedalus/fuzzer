"""Good-Toulmin estimator on the Good-Turing seed arm (entropy handover 7.1).

Good-Turing reads only Q1, the t -> 0 limit of the forecast. Good-Toulmin
forecasts the distinct *new* edges in the next ``ratio * T`` executions from
Q1..QK, so doubletons pull the estimate down as edges saturate.

Expected values come from closed forms worked out here (the incidence-model
series, exact expectations over known edge probabilities, Monte Carlo against
a planted distribution), not from the module under test.
"""

from __future__ import annotations

import ast
import inspect
import math
import random
from unittest.mock import patch

import numpy as np
import pytest

from fuzzer_tool.core.schedulers.seed_good_turing import (
    MIN_WEIGHT,
    GoodTuringSeedStrategy,
    good_toulmin,
)


class _RecordingRng:
    def __init__(self, idx: int = 0) -> None:
        self.idx = idx
        self.weights: list[float] | None = None

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self.idx]


A, B, C = b"seed-a", b"seed-b", b"seed-c"


def _strategy(**kw) -> GoodTuringSeedStrategy:
    kw.setdefault("min_observations", 1)
    kw.setdefault("estimator", "gtoul")
    return GoodTuringSeedStrategy(_RecordingRng(0), **kw)


def _coef(i: int, t: int, m: float) -> float:
    """C(m+i-1, i) / C(t, i), real m, from the definition (no recursion)."""
    num = math.prod(m + k for k in range(i)) / math.factorial(i)
    return num / math.comb(t, i)


def _spec(qk: list[float], t: int, ratio: float) -> float:
    m = ratio * t
    return sum((-1) ** (i + 1) * _coef(i, t, m) * q for i, q in enumerate(qk, start=1) if i <= t)


def _expected_qk(probs: list[float], t: int, k: int) -> list[float]:
    """E[Q_i] = sum_e C(t,i) p^i (1-p)^(t-i) for edges with hit prob p."""
    return [
        sum(math.comb(t, i) * p**i * (1 - p) ** (t - i) for p in probs) for i in range(1, k + 1)
    ]


# ── Pure estimator ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("qk", "t", "ratio"),
    [
        ([5, 3, 2, 1], 20, 0.5),
        ([9, 0, 4], 7, 0.25),
        ([1, 1, 1, 1, 1, 1], 6, 1.0),
        ([40, 12, 6, 3, 1, 0, 0, 0], 100, 0.5),
    ],
)
def test_matches_closed_form_series(qk, t, ratio):
    assert good_toulmin(qk, t, ratio) == pytest.approx(_spec(qk, t, ratio))


def test_first_term_is_ratio_times_q1():
    """Only singletons: U = (m/t) Q1 = ratio * Q1."""
    assert good_toulmin([7], 50, 0.5) == pytest.approx(3.5)


def test_exact_in_expectation_over_known_probabilities():
    """Falsification: fed E[Q_i], the estimator returns the exact E[new].

    E[new in m more] = sum_e (1-p)^t (1 - (1-p)^m). The series is unbiased as
    an infinite sum, so any wrong coefficient shows up here and no sampling
    noise can hide it.
    """
    probs = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.004, 0.001]
    t, ratio = 60, 0.5
    m = ratio * t

    exact = sum((1 - p) ** t * (1 - (1 - p) ** m) for p in probs)
    est = good_toulmin(_expected_qk(probs, t, t), t, ratio)

    assert est == pytest.approx(exact, rel=1e-6)


def test_ratio_to_zero_is_good_turing_to_first_order():
    """h -> 0: U/h -> sum (-1)^(i+1) Qi / (i C(t,i)); Q1/T up to O(Q2/T^2).

    C(h+i-1, i) -> h/i as h -> 0, so the exact limit is the closed form below;
    Good-Turing's Q1/T is its first term.
    """
    qk, t, eps = [12, 5, 3, 1], 40, 1e-7
    limit = sum((-1) ** (i + 1) * q / (i * math.comb(t, i)) for i, q in enumerate(qk, 1))

    rate = good_toulmin(qk, t, eps) / (eps * t)

    assert rate == pytest.approx(limit, rel=1e-5)
    assert rate == pytest.approx(12 / 40, rel=0.02)


def test_doubletons_lower_the_forecast():
    """A saturating pool (Q2 > 0) forecasts less than singletons alone."""
    assert good_toulmin([10, 10], 100, 0.5) < good_toulmin([10, 0], 100, 0.5)


def test_beats_good_turing_on_planted_distribution():
    """Control against itself: same samples, naive ``ratio * Q1`` vs series.

    Zipf edge probabilities, t executions, then m more actually drawn. Over
    many draws Good-Toulmin must sit closer to the realised new-edge count
    than the Q1-only forecast, which ignores saturation.
    """
    rng = np.random.default_rng(20261003)
    probs = 0.4 / np.arange(1, 301) ** 0.9
    t, ratio, trials = 120, 0.5, 400
    m = int(ratio * t)
    k = 10

    truth = 0.0
    gtoul = 0.0
    naive = 0.0
    for _ in range(trials):
        draws = rng.random((t + m, probs.size)) < probs
        counts = draws[:t].sum(axis=0)
        seen = counts > 0
        truth += float((draws[t:].any(axis=0) & ~seen).sum())
        qk = [float((counts == i).sum()) for i in range(1, k + 1)]
        gtoul += good_toulmin(qk, t, ratio)
        naive += ratio * qk[0]

    truth, gtoul, naive = truth / trials, gtoul / trials, naive / trials
    assert abs(gtoul - truth) < abs(naive - truth)
    assert gtoul == pytest.approx(truth, rel=0.08)


# ── Edge cases ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("qk", "t", "ratio"),
    [([], 10, 0.5), ([3, 1], 0, 0.5), ([3, 1], -4, 0.5), ([3, 1], 10, 0.0), ([3, 1], 10, -1.0)],
)
def test_degenerate_inputs_are_zero(qk, t, ratio):
    assert good_toulmin(qk, t, ratio) == 0.0


def test_single_execution_has_no_division_by_zero():
    """t = 1: every hit edge is a singleton, nothing past i = t to divide by."""
    assert good_toulmin([5, 0, 0], 1, 0.5) == pytest.approx(2.5)


def test_terms_beyond_t_are_ignored():
    """Q_i = 0 for i > t by construction; garbage there must not leak in."""
    assert good_toulmin([4, 2, 99, 99], 2, 0.5) == pytest.approx(_spec([4, 2], 2, 0.5))


def test_finite_for_huge_t():
    out = good_toulmin([10**6, 10**5, 10**4], 10**9, 0.5)
    assert math.isfinite(out)
    # c_i -> ratio^i for t >> i: 0.5e6 - 0.25e5 + 0.125e4.
    assert out == pytest.approx(0.5e6 - 0.25e5 + 0.125e4, rel=1e-3)


# ── Incidence table ────────────────────────────────────────────────────


def _brute_qk(execs: list[set[int]], k: int) -> list[int]:
    counts: dict[int, int] = {}
    for hit in execs:
        for e in hit:
            counts[e] = counts.get(e, 0) + 1
    return [sum(1 for c in counts.values() if c == i) for i in range(1, k + 1)]


def test_incremental_qk_matches_brute_force():
    from fuzzer_tool.core.schedulers.seed_good_turing import TERMS

    rng = random.Random(11)
    s = _strategy()
    execs: list[set[int]] = []
    for _ in range(300):
        hit = {rng.randrange(30) for _ in range(rng.randrange(0, 8))}
        execs.append(hit)
        s.observe(A, hit)
        assert s.counts_by_hits(A) == _brute_qk(execs, TERMS)


def test_saturated_edges_leave_the_series():
    """An edge hit more than TERMS times is in no Q_i, not stuck in Q_K."""
    from fuzzer_tool.core.schedulers.seed_good_turing import TERMS

    s = _strategy()
    for _ in range(TERMS + 5):
        s.observe(A, {1})

    assert s.counts_by_hits(A) == [0] * TERMS
    assert s.singletons(A) == 0


# ── Strategy ───────────────────────────────────────────────────────────


def test_unfuzzed_seed_scores_global_rate():
    s = _strategy()
    for i in range(30):
        s.observe(A, {i})

    assert s.discovery_probability(B) == pytest.approx(s.residual_risk())


def test_novelty_seed_outranks_exhausted_seed():
    s = _strategy()
    for i in range(60):
        s.observe(A, {1, 2, 3})
        s.observe(B, {1000 + i})

    assert s.discovery_probability(B) > s.discovery_probability(A)


def test_rate_is_not_capped_at_one():
    """Expected new edges per exec may exceed 1; capping would flatten the rank."""
    s = _strategy(prior_strength=0.0)
    for i in range(40):
        s.observe(A, {10 * i + j for j in range(5)})

    assert s.discovery_probability(A) > 1.0


def test_saturating_seed_ranks_below_good_turing_twin():
    """Adversarial: Q2-heavy seed. GT sees Q1 only; Good-Toulmin discounts it."""
    gtoul = _strategy(prior_strength=0.0)
    gt = _strategy(prior_strength=0.0, estimator="gt")
    for strat in (gtoul, gt):
        for i in range(30):
            strat.observe(A, {i, i + 1})  # every inner edge hit twice

    assert gtoul.discovery_probability(A) < gt.discovery_probability(A)


def test_negative_forecast_is_floored_not_negative():
    """Series can dip below 0 on noisy tables; weights must stay positive."""
    s = _strategy(prior_strength=0.0)
    for _ in range(20):
        s.observe(A, {1, 2, 3, 4})
    s.observe(A, {9})

    assert s.discovery_probability(A) >= 0.0
    assert all(w >= MIN_WEIGHT for w in s.scores([A, B]))


def test_cached_rate_follows_new_observations():
    """Scores are cached per table; an observation must invalidate the cache."""
    s = _strategy(prior_strength=0.0)
    horizon = 0.5
    rng = random.Random(3)

    for _ in range(25):
        s.observe(A, {rng.randrange(20) for _ in range(4)})
        qk = s.counts_by_hits(A)
        t = s.executions(A)
        fresh = max(0.0, good_toulmin(qk, t, horizon) / (horizon * t))
        assert s.discovery_probability(A) == pytest.approx(fresh)


def test_horizon_changes_the_forecast():
    near, far = _strategy(horizon=0.1), _strategy(horizon=0.9)
    for strat in (near, far):
        for i in range(40):
            strat.observe(A, {i % 7, 100 + i})

    assert near.discovery_probability(A) != pytest.approx(far.discovery_probability(A))


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_bad_horizon_rejected(bad):
    with pytest.raises(ValueError, match="horizon"):
        _strategy(horizon=bad)


def test_stats_report_estimator_and_global_rate():
    s = _strategy()
    for i in range(30):
        s.observe(A, {2 * i, 2 * i + 1})

    st = s.stats()
    assert st["estimator"] == "gtoul"
    assert st["residual_risk"] == pytest.approx(s.residual_risk())
    assert st["residual_risk"] > 0.0


def test_live_stats_label_follows_estimator():
    """gtoul prints a rate, gt a probability: the label must not claim m0."""
    from types import SimpleNamespace

    from fuzzer_tool.services.stats import _entropy_seed_str

    for est, label in (("gtoul", "rate="), ("gt", "m0=")):
        s = _strategy(estimator=est)
        for i in range(10):
            s.observe(A, {i})
        out = _entropy_seed_str(SimpleNamespace(_good_turing_seed=s))
        assert f"good-turing: {label}" in out


def test_bounded_memory_under_flood():
    """Rule 54: per-seed tables and per-table counters stay bounded."""
    from fuzzer_tool.core.schedulers.seed_good_turing import TERMS

    s = _strategy(seed_cap=4)
    for i in range(50):
        s.observe(bytes([i]), {i, i + 1})

    assert s.stats()["seeds"] <= 4
    assert len(s.counts_by_hits(bytes([49]))) == TERMS


# ── Wiring ─────────────────────────────────────────────────────────────


def test_estimator_listed():
    from fuzzer_tool.core.schedulers.seed_good_turing import ESTIMATORS

    assert "gtoul" in ESTIMATORS


def test_fuzzer_signature_defaults():
    from fuzzer_tool.services.fuzzer import Fuzzer

    params = inspect.signature(Fuzzer.__init__).parameters
    assert params["good_turing_estimator"].default == "gt"
    assert params["good_turing_horizon"].default == 0.5


def test_cli_passes_flags():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    want = {"good_turing_estimator", "good_turing_horizon"}
    assert want <= _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    calls = [
        {k.arg for k in c.keywords}
        for c in ast.walk(ast.parse(inspect.getsource(commands.cmd_fuzz)))
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "Fuzzer"
    ]
    assert calls
    assert all(want <= k for k in calls)


def test_cli_estimator_flag_is_restricted_to_choices():
    """argparse must reject an estimator the strategy would raise on later."""
    from fuzzer_tool.cli import commands
    from fuzzer_tool.core.schedulers.seed_good_turing import ESTIMATORS

    flag_calls = [
        c
        for c in ast.walk(ast.parse(inspect.getsource(commands)))
        if isinstance(c, ast.Call)
        and c.args
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value == "--good-turing-estimator"
    ]
    assert len(flag_calls) == 1
    assert "choices" in {k.arg for k in flag_calls[0].keywords}
    assert commands.GOOD_TURING_ESTIMATORS == ESTIMATORS


@pytest.fixture
def fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            good_turing_seed=True,
            good_turing_estimator="gtoul",
            good_turing_horizon=0.25,
        )
    yield f


def test_fuzzer_builds_gtoul_strategy(fuzzer):
    s = fuzzer._good_turing_seed
    assert isinstance(s, GoodTuringSeedStrategy)
    assert s._estimator == "gtoul"
    assert s._horizon == 0.25


def test_every_round_feeds_parent_and_picker_ranks(fuzzer):
    """End to end: executed mutants credit the parent; the picker draws by forecast."""
    from fuzzer_tool.core.schedulers.seed_good_turing import MIN_OBSERVATIONS
    from fuzzer_tool.services.seed_picker import SeedPicker

    s = fuzzer._good_turing_seed
    parent = fuzzer.corpus[0]
    with (
        patch.object(fuzzer, "_dedup_mutate", return_value=b"MUTANT01"),
        patch.object(fuzzer, "_run_target", return_value=(0, "")),
    ):
        fuzzer.fuzz_one(parent)
        fuzzer.fuzz_one(parent)

    assert s.executions(parent) == 2

    for i in range(MIN_OBSERVATIONS):
        s.observe(A, {i})
    fuzzer.corpus = [A, B]
    assert SeedPicker(fuzzer)._pick_good_turing_seed() in (A, B)
