"""Regression tests for SeedPicker._pick_ecofuzz_seed's optional
marginal-cost penalty (handover §1 candidate #1, added after the initial
op_replicator.py wiring -- see
docs/handover/handover_decision_game_theory_survey_2026-09-13.md §1).

EcoFuzz's energy = reward_prob / cost is a *lifetime* average; these tests
check the optional penalty that divides a seed's weight when its recent
(between the last two picks) cost-per-edge has run away relative to the
corpus average, without touching any seed the tracker doesn't flag.
"""

import random

from fuzzer_tool.core.marginal_cost import MarginalCostTracker
from fuzzer_tool.services.seed_picker import SeedPicker


class _FakeRNG:
    """rng stand-in whose .random() always returns the same fixed fraction,
    so the cumulative-weight draw in _pick_ecofuzz_seed is deterministic."""

    def __init__(self, fraction):
        self._fraction = fraction

    def random(self):
        return self._fraction


def _make_fuzzer_mock(seed_metas, mc_penalty_multiplier=None, rng=None):
    class MockFuzzer:
        corpus = [f"seed_{i}".encode() for i in range(len(seed_metas))]
        seed_meta = dict(zip(corpus, seed_metas, strict=False))
        _rng = rng if rng is not None else random
        _use_ecofuzz = True
        _ecofuzz_mc_penalty_multiplier = mc_penalty_multiplier
        _profile = type("obj", (object,), {"format_signature": None})()

        def mean_exec_time(self):
            return 0.0  # forces effective_fuzz_count's raw-fuzz_count fallback

        def _seed_key(self, data):
            return data.hex()

    return MockFuzzer()


def _new_picker(f):
    sp = SeedPicker(type("o", (object,), {"__init__": lambda s: None})())
    sp.f = f
    return sp


# ── disabled by default ───────────────────────────────────────────────────


def test_mc_penalty_disabled_by_default():
    f = _make_fuzzer_mock([{"fuzz_count": 10, "coverage_edges": 8}])
    sp = _new_picker(f)
    assert getattr(f, "_ecofuzz_mc_penalty_multiplier", None) is None
    sp._pick_ecofuzz_seed()  # must not raise with the attribute absent/None


def test_disabled_multiplier_never_calls_should_stop(monkeypatch):
    f = _make_fuzzer_mock(
        [
            {"fuzz_count": 10, "coverage_edges": 8},
            {"fuzz_count": 10, "coverage_edges": 0},
        ]
    )
    sp = _new_picker(f)
    called = []
    original = MarginalCostTracker.should_stop
    monkeypatch.setattr(
        MarginalCostTracker,
        "should_stop",
        lambda self, *a, **kw: called.append(1) or original(self, *a, **kw),
    )
    sp._pick_ecofuzz_seed()
    sp._pick_ecofuzz_seed()
    assert called == []


# ── snapshots accumulate regardless of whether the penalty is enabled ───


def test_marginal_cost_none_after_one_pick():
    f = _make_fuzzer_mock([{"fuzz_count": 10, "coverage_edges": 8}])
    sp = _new_picker(f)
    sp._pick_ecofuzz_seed()
    assert sp._ecofuzz_mc_tracker.marginal_cost(f.corpus[0]) is None


def test_marginal_cost_defined_after_two_picks():
    f = _make_fuzzer_mock([{"fuzz_count": 10, "coverage_edges": 8}])
    sp = _new_picker(f)
    sp._pick_ecofuzz_seed()
    f.seed_meta[f.corpus[0]] = {"fuzz_count": 20, "coverage_edges": 16}
    sp._pick_ecofuzz_seed()
    # cost 10->20 (Δ10), coverage 8->16 (Δ8) -> MC = 10/8 = 1.25
    assert sp._ecofuzz_mc_tracker.marginal_cost(f.corpus[0]) == 1.25


# ── the penalty actually shifts selection away from the flagged seed ────


def test_penalty_shifts_pick_away_from_stale_but_historically_good_seed():
    """"expensive" keeps a decent lifetime energy (reward_prob/cost) from
    an early streak, but produces zero *new* output in the second round
    while burning a lot of additional cost -- exactly the case the
    lifetime average is slow to reflect. With the penalty enabled, the
    same fixed rng draw that lands on "expensive" without the penalty
    must land on "cheap" with it."""
    round1 = [
        {"fuzz_count": 10, "coverage_edges": 8},  # cheap
        {"fuzz_count": 10, "coverage_edges": 5},  # expensive
    ]
    round2 = [
        {"fuzz_count": 20, "coverage_edges": 16},  # cheap: kept producing
        {"fuzz_count": 100, "coverage_edges": 5},  # expensive: cost spent, no new output
    ]
    u = 0.99  # fixed draw: between the two configs' cheap-seed cumulative shares

    # Baseline: no penalty.
    f_base = _make_fuzzer_mock(round1, mc_penalty_multiplier=None, rng=_FakeRNG(u))
    sp_base = _new_picker(f_base)
    sp_base._pick_ecofuzz_seed()
    f_base.seed_meta = dict(zip(f_base.corpus, round2, strict=False))
    baseline_pick = sp_base._pick_ecofuzz_seed()

    # Gated: penalty enabled.
    f_gated = _make_fuzzer_mock(round1, mc_penalty_multiplier=2.0, rng=_FakeRNG(u))
    sp_gated = _new_picker(f_gated)
    sp_gated._pick_ecofuzz_seed()
    f_gated.seed_meta = dict(zip(f_gated.corpus, round2, strict=False))
    gated_pick = sp_gated._pick_ecofuzz_seed()

    assert baseline_pick == f_base.corpus[1]  # picks "expensive" without the penalty
    assert gated_pick == f_gated.corpus[0]  # picks "cheap" once the penalty is applied


def test_penalty_never_helps_the_flagged_seed():
    """Weight can only shrink under the penalty, never grow, for any
    multiplier -- so a seed's absolute weight (reconstructed from which
    fixed rng fraction still selects it) with the penalty on is always a
    subset of what selects it with the penalty off."""
    round1 = [
        {"fuzz_count": 10, "coverage_edges": 8},
        {"fuzz_count": 10, "coverage_edges": 5},
    ]
    round2 = [
        {"fuzz_count": 20, "coverage_edges": 16},
        {"fuzz_count": 100, "coverage_edges": 5},
    ]
    for multiplier in (1.01, 2.0, 10.0):
        for u in (0.9, 0.95, 0.99, 0.999):
            f_base = _make_fuzzer_mock(round1, mc_penalty_multiplier=None, rng=_FakeRNG(u))
            sp_base = _new_picker(f_base)
            sp_base._pick_ecofuzz_seed()
            f_base.seed_meta = dict(zip(f_base.corpus, round2, strict=False))
            baseline_pick = sp_base._pick_ecofuzz_seed()

            f_gated = _make_fuzzer_mock(
                round1, mc_penalty_multiplier=multiplier, rng=_FakeRNG(u)
            )
            sp_gated = _new_picker(f_gated)
            sp_gated._pick_ecofuzz_seed()
            f_gated.seed_meta = dict(zip(f_gated.corpus, round2, strict=False))
            gated_pick = sp_gated._pick_ecofuzz_seed()

            # If the penalty ever picks "expensive" (corpus[1]) at some u,
            # the baseline must too -- the gated cumulative share for
            # "expensive" is a subset of the baseline's.
            if gated_pick == f_gated.corpus[1]:
                assert baseline_pick == f_base.corpus[1]
