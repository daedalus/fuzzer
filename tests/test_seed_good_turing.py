"""Good-Turing seed arm: discovery probability from singleton counts.

Expected values come from closed forms worked out here (Q1/T, Chao-Jost,
the shrinkage blend), not from the module under test. The behavioural
tests plant a seed whose mutants keep finding new edges next to one that
re-treads a fixed path, and check the arm ranks them that way.
"""

from __future__ import annotations

import ast
import inspect
import math
import random
from unittest.mock import patch

import pytest

from fuzzer_tool.core.schedulers.seed_good_turing import (
    MIN_OBSERVATIONS,
    MIN_WEIGHT,
    GoodTuringSeedStrategy,
    chao_jost_m0,
    good_turing_m0,
    m0_variance,
    shrunk_m0,
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
    return GoodTuringSeedStrategy(_RecordingRng(0), **kw)


# ── Pure estimators ────────────────────────────────────────────────────


def test_good_turing_is_q1_over_t():
    assert good_turing_m0(3, 12) == pytest.approx(0.25)
    assert good_turing_m0(0, 12) == 0.0
    assert good_turing_m0(5, 0) == 0.0


def test_good_turing_clamped_to_one():
    assert good_turing_m0(9, 3) == 1.0


def test_chao_jost_matches_closed_form():
    q1, q2, t = 4, 2, 10
    expected = (q1 / t) * ((t - 1) * q1) / ((t - 1) * q1 + 2 * q2)
    assert chao_jost_m0(q1, q2, t) == pytest.approx(expected)


def test_chao_jost_collapses_to_gt_without_doubletons():
    assert chao_jost_m0(4, 0, 10) == pytest.approx(good_turing_m0(4, 10))


def test_chao_jost_zero_without_singletons():
    assert chao_jost_m0(0, 7, 10) == 0.0


def test_chao_jost_never_exceeds_good_turing():
    for q1 in range(1, 8):
        for q2 in range(0, 8):
            assert chao_jost_m0(q1, q2, 12) <= good_turing_m0(q1, 12) + 1e-12


def test_variance_closed_form():
    assert m0_variance(3, 2, 10) == pytest.approx((3 + 2 * 2) / 100)
    assert m0_variance(3, 2, 0) == 0.0


def test_shrinkage_blend():
    # (Q1 + K*prior) / (T + K)
    assert shrunk_m0(2, 10, 0.5, 20.0) == pytest.approx((2 + 10.0) / 30.0)
    # No evidence: exactly the prior.
    assert shrunk_m0(0, 0, 0.3, 20.0) == pytest.approx(0.3)
    # K = 0 is the raw estimate.
    assert shrunk_m0(2, 10, 0.5, 0.0) == pytest.approx(0.2)


# ── Incremental Q1/Q2 bookkeeping ──────────────────────────────────────


def _brute_force(execs: list[set[int]]) -> tuple[int, int, int]:
    counts: dict[int, int] = {}
    for hit in execs:
        for e in hit:
            counts[e] = counts.get(e, 0) + 1
    return (
        sum(1 for c in counts.values() if c == 1),
        sum(1 for c in counts.values() if c == 2),
        len(execs),
    )


def test_incremental_counts_match_brute_force():
    rng = random.Random(7)
    s = _strategy()
    execs = []
    for _ in range(300):
        hit = {rng.randrange(40) for _ in range(rng.randrange(0, 8))}
        execs.append(hit)
        s.observe(A, hit)
        q1, q2, t = _brute_force(execs)
        st = s.stats()
        assert (st["q1"], st["q2"], st["observed"]) == (q1, q2, t)
        assert s.singletons(A) == q1
        assert s.executions(A) == t


def test_per_seed_tables_are_independent():
    s = _strategy()
    s.observe(A, {1, 2})
    s.observe(B, {2, 3})
    assert s.singletons(A) == 2
    assert s.singletons(B) == 2
    # Globally edge 2 was hit twice, so only 1 and 3 remain singletons.
    assert s.stats()["q1"] == 2
    assert s.stats()["q2"] == 1


def test_accepts_non_set_iterables():
    s = _strategy()
    s.observe(A, [1, 1, 2])
    assert s.singletons(A) == 2


# ── Behaviour ──────────────────────────────────────────────────────────


def test_unfuzzed_seed_scores_global_rate_not_one():
    s = _strategy()
    for i in range(30):
        s.observe(A, {1, 2, 3})  # saturated: nothing new after exec 1
    assert s.residual_risk() < 0.15
    assert s.discovery_probability(C) == pytest.approx(s.residual_risk())
    assert s.discovery_probability(C) < 0.5


def test_single_exec_estimate_is_not_one():
    """T=1 makes every edge a singleton; shrinkage must stop that reading as 1.0."""
    s = _strategy()
    for i in range(40):
        s.observe(A, {1, 2, 3})
    s.observe(B, {10, 11, 12})
    assert s.discovery_probability(B) < 0.5


def test_novelty_seed_outranks_exhausted_seed():
    s = _strategy()
    for i in range(200):
        s.observe(A, {1, 2, 3})  # exhausted: same path every time
        s.observe(B, {1, 2, 3, 1000 + i})  # each mutant finds a fresh edge
    assert s.discovery_probability(B) > 5 * s.discovery_probability(A)


def test_exhausted_seed_decays_as_evidence_accrues():
    s = _strategy()
    s.observe(A, {1, 2, 3})
    early = s.discovery_probability(A)
    for _ in range(300):
        s.observe(A, {1, 2, 3})
    assert s.discovery_probability(A) < early


def test_recovers_planted_discovery_rate():
    """Mutants hit a fresh edge with probability p; M0 should track p."""
    p = 0.2
    rng = random.Random(11)
    s = _strategy(prior_strength=0.0)
    fresh = 10_000
    for _ in range(4000):
        hit = {1, 2, 3}
        if rng.random() < p:
            fresh += 1
            hit.add(fresh)
        s.observe(A, hit)
    # Every planted edge is a singleton; the shared path is never one.
    assert s.discovery_probability(A) == pytest.approx(p, abs=0.03)


def test_chao_estimator_runs_and_is_bounded():
    s = _strategy(estimator="chao")
    rng = random.Random(3)
    for _ in range(200):
        s.observe(A, {rng.randrange(30) for _ in range(5)})
    assert 0.0 <= s.discovery_probability(A) <= 1.0


def test_bad_estimator_rejected():
    with pytest.raises(ValueError):
        GoodTuringSeedStrategy(_RecordingRng(), estimator="nope")
    with pytest.raises(ValueError):
        GoodTuringSeedStrategy(_RecordingRng(), prior_strength=-1.0)


# ── Selection ──────────────────────────────────────────────────────────


def test_select_declines_until_ready():
    s = GoodTuringSeedStrategy(_RecordingRng(0))
    assert not s.ready
    for _ in range(MIN_OBSERVATIONS - 1):
        s.observe(A, {1})
    assert s.select([A, B]) is None
    s.observe(A, {1})
    assert s.ready
    assert s.select([A, B]) == A


def test_select_empty_corpus_is_none():
    assert _strategy().select([]) is None


def test_weights_floored_and_aligned():
    rng = _RecordingRng(1)
    s = GoodTuringSeedStrategy(rng, min_observations=1, prior_strength=0.0)
    for _ in range(50):
        s.observe(A, {1, 2})  # Q1 = 0 -> raw weight 0
    s.observe(B, {1, 2, 99})
    assert s.select([A, B, C]) == B
    assert rng.weights is not None
    assert len(rng.weights) == 3
    assert all(w >= MIN_WEIGHT for w in rng.weights)
    assert all(math.isfinite(w) for w in rng.weights)


def test_all_zero_campaign_still_selectable():
    """A saturated campaign (Q1 = 0 everywhere) must not hand weighted_choice zeros."""
    rng = _RecordingRng(0)
    s = GoodTuringSeedStrategy(rng, min_observations=1, prior_strength=0.0)
    for _ in range(100):
        s.observe(A, {1})
    # Edge 1 hit 100x: Q1 = 0. Both weights floor, none negative.
    assert s.select([A, B]) is not None
    assert rng.weights == [MIN_WEIGHT, MIN_WEIGHT]


def test_seed_table_lru_bounded():
    s = _strategy(seed_cap=8)
    for i in range(50):
        s.observe(f"seed-{i}".encode(), {i})
    assert s.stats()["seeds"] <= 8


# ── Wiring ─────────────────────────────────────────────────────────────


def test_registered_as_seed_strategy():
    from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES, Fuzzer

    assert "good_turing" in _SEED_STRATEGY_NAMES
    params = inspect.signature(Fuzzer.__init__).parameters
    assert params["good_turing_seed"].default is False
    assert params["good_turing_prior"].default == 20.0


def test_cli_passes_flags():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert {"good_turing_seed", "good_turing_prior"} <= dests
    calls = [
        {k.arg for k in c.keywords}
        for c in ast.walk(ast.parse(inspect.getsource(commands.cmd_fuzz)))
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "Fuzzer"
    ]
    assert calls
    assert all({"good_turing_seed", "good_turing_prior"} <= k for k in calls)


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
            good_turing_prior=5.0,
        )
    yield f


def test_fuzzer_builds_strategy(fuzzer):
    s = fuzzer._good_turing_seed
    assert isinstance(s, GoodTuringSeedStrategy)
    assert s._k == 5.0


def test_fuzzer_default_off(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "c"),
            crashes_dir=str(tmp_path / "x"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )
    assert f._good_turing_seed is None


def test_picker_dispatch_and_elo_listing(fuzzer):
    from fuzzer_tool.services.seed_picker import SeedPicker

    fuzzer.corpus = [A, B]
    s = fuzzer._good_turing_seed
    picker = SeedPicker(fuzzer)

    # Cold: declines, and the Elo pool does not list a phantom opponent.
    assert picker._pick_good_turing_seed() is None
    avail: list[str] = []
    SeedPicker._elo_entropy_arms(fuzzer, avail)
    assert "good_turing" not in avail

    for _ in range(MIN_OBSERVATIONS):
        s.observe(A, {1, 2, 3})
    assert picker._pick_good_turing_seed() in (A, B)
    avail = []
    SeedPicker._elo_entropy_arms(fuzzer, avail)
    assert "good_turing" in avail
