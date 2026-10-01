"""ConsolidatedV2Scheduler: v1 plus an optimistic, tempered Thompson draw.

The score is ``mean + tau * max(0, draw - mean)``. These pin that rule with
scripted draws, its limits (tau=1, all-pessimistic draws), and that v1's
category prior, cap and reward handling are inherited unchanged. Convergence
and decay recovery live in test_scheduler_convergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import ConsolidatedV1Scheduler, ConsolidatedV2Scheduler

# Two operators in different categories, so neither's evidence moves the
# other's prior.
_STRONG = "bit_flip"
_WEAK = "havoc"


class _ScriptedBeta:
    """RandPool seam: ``betavariate_array`` returns the scripted draws in order."""

    def __init__(self, *draws):
        self._draws = [np.asarray(d, dtype=float) for d in draws]

    def betavariate_array(self, alphas, betas):
        assert len(alphas) == len(betas) == len(self._draws[0])
        return self._draws.pop(0)


def _trained(cls, rng, **kwargs):
    """_STRONG at ~0.3 over 100 pulls, _WEAK at 0 over 20."""
    s = cls(rng=rng, **kwargs)
    for i in range(100):
        s.record(_STRONG, i % 10 < 3)
    for _ in range(20):
        s.record(_WEAK, False)
    return s


def _means(s):
    return s.posterior_mean(_STRONG), s.posterior_mean(_WEAK)


def test_pessimistic_draw_is_floored_at_the_mean():
    """_STRONG draws below its mean, _WEAK draws above its own mean but below
    _STRONG's. Raw Thompson (v1) takes _WEAK; v2 scores _STRONG at its mean
    and _WEAK at mean + tau * excess, which stays below it."""
    probe = _trained(ConsolidatedV2Scheduler, RandPool(1))
    mu_s, mu_w = _means(probe)
    assert mu_w < mu_s - 0.1
    draws = [mu_s - 0.2, mu_s - 0.1]

    v1 = _trained(ConsolidatedV1Scheduler, _ScriptedBeta(draws))
    v2 = _trained(ConsolidatedV2Scheduler, _ScriptedBeta(draws))

    # Control: the same draws make raw Thompson pick the other arm, so the
    # assertion below can fail.
    assert v1.select_op([_STRONG, _WEAK]) == _WEAK
    assert v2.select_op([_STRONG, _WEAK]) == _STRONG


def test_tempering_shrinks_the_optimistic_excess():
    """Both arms draw above their means. Score = mean + tau * excess, so the
    winner flips at the tau where the two scores cross."""
    probe = _trained(ConsolidatedV2Scheduler, RandPool(1))
    mu_s, mu_w = _means(probe)
    excess_s, excess_w = 0.01, (mu_s - mu_w) + 0.05
    draws = [mu_s + excess_s, mu_w + excess_w]

    # mu_s + tau*excess_s == mu_w + tau*excess_w
    tau_cross = (mu_s - mu_w) / (excess_w - excess_s)
    lo = _trained(ConsolidatedV2Scheduler, _ScriptedBeta(draws), tau=tau_cross * 0.9)
    hi = _trained(ConsolidatedV2Scheduler, _ScriptedBeta(draws), tau=min(1.0, tau_cross * 1.1))

    assert lo.select_op([_STRONG, _WEAK]) == _STRONG
    assert hi.select_op([_STRONG, _WEAK]) == _WEAK


def test_all_pessimistic_draws_select_the_best_mean():
    """Adversarial for the sampler: every draw lands far below its mean, in
    reversed order. v2 degrades to greedy on the posterior mean."""
    ops = ["bit_flip", "byte_flip", "arith_inc", "havoc"]
    rates = [0.1, 0.4, 0.2, 0.05]
    s = ConsolidatedV2Scheduler(rng=RandPool(1))
    for op, p in zip(ops, rates, strict=True):
        for i in range(100):
            s.record(op, i < p * 100)
    means = [s.posterior_mean(op) for op in ops]
    draws = [1e-6 * (len(ops) - k) for k in range(len(ops))]  # argmax would be ops[0]

    s._rng = _ScriptedBeta(draws)
    assert s.select_op(ops) == ops[int(np.argmax(means))]


def test_tau_one_is_optimistic_thompson():
    """tau=1: the score is max(draw, mean) -- an above-mean draw counts in full."""
    probe = _trained(ConsolidatedV2Scheduler, RandPool(1))
    mu_s, mu_w = _means(probe)
    draws = [mu_s - 0.2, mu_s + 0.01]
    s = _trained(ConsolidatedV2Scheduler, _ScriptedBeta(draws), tau=1.0)
    assert s.select_op([_STRONG, _WEAK]) == _WEAK


@pytest.mark.parametrize("tau", [0.0, -0.5, 1.5, float("nan")])
def test_rejects_tau_outside_unit_interval(tau):
    with pytest.raises(ValueError):
        ConsolidatedV2Scheduler(tau=tau)


def test_inherits_v1_learning():
    """Same records, same posterior: v2 changes selection only."""
    v1 = _trained(ConsolidatedV1Scheduler, RandPool(1))
    v2 = _trained(ConsolidatedV2Scheduler, RandPool(1))
    for op in (_STRONG, _WEAK, "arith_inc"):
        assert v2.posterior_mean(op) == pytest.approx(v1.posterior_mean(op))


def test_seeded_rng_reproduces_the_campaign():
    ops = ["bit_flip", "byte_flip", "arith_inc", "havoc"]

    def campaign(seed):
        s = ConsolidatedV2Scheduler(rng=RandPool(seed))
        out = []
        for i in range(300):
            op = s.select_op(ops)
            s.record(op, (i * 7 + len(op)) % 5 == 0)
            out.append(op)
        return out

    assert campaign(3) == campaign(3)
    assert campaign(3) != campaign(4)


def test_degenerate_candidate_lists():
    s = ConsolidatedV2Scheduler(rng=RandPool(1))
    assert s.select_op([]) == ""
    assert s.select_op(["havoc"]) == "havoc"


def test_bandit_stats_shape():
    s = ConsolidatedV2Scheduler(rng=RandPool(1))
    s.record("bit_flip", True)
    stats = s.bandit_stats()
    assert stats["consolidated_v2_pulls"] == 1
    assert stats["consolidated_v2_top"][0][0] == "bit_flip"
    assert not any(k.startswith("consolidated_v1") for k in stats)


def test_declares_supports_priors():
    assert ConsolidatedV2Scheduler.supports_priors is True


_TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_fuzzer_wiring_selects_ahead_of_v1(tmp_path):
    """--consolidated-v2 builds it and, without Elo, it selects ahead of v1;
    v1 still learns from the rounds through the shared fan-out."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=str(_TARGET),
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        use_coverage=True,
        consolidated_v1=True,
        consolidated_v2=True,
    )
    assert isinstance(f._consolidated_v2, ConsolidatedV2Scheduler)
    assert f._track_op_effect

    selectors = set()
    for i in range(40):
        f.fuzz_one(bytes([65 + i % 26]) * 16)
        selectors.add(f._op_selector)
    assert selectors == {"consolidated_v2"}
    assert f._consolidated_v2.bandit_stats()["consolidated_v2_pulls"] > 0
    assert f._consolidated_v1.bandit_stats()["consolidated_v1_pulls"] > 0
