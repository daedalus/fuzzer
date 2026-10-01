"""PositionConsolidatedScheduler: experts propose, cross-seed + per-seed rates score.

Covers core/schedulers/pos_consolidated.py and PositionContextScheduler.tilt.
"""

from __future__ import annotations

import logging
import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_consolidated import (
    K_UNIFORM,
    PRIOR_A,
    PRIOR_B,
    STATE_VERSION,
    PositionConsolidatedScheduler,
)
from fuzzer_tool.core.schedulers.pos_context import (
    MIN_OBS,
    W_MAX,
    W_MIN,
    PositionContextScheduler,
)
from fuzzer_tool.core.schedulers.pos_context import (
    PRIOR_A as CTX_A,
)
from fuzzer_tool.core.schedulers.pos_context import (
    PRIOR_B as CTX_B,
)

# Class runs (ALPHA | ZERO | ALPHA): boundary has content edges to propose.
SEED = b"AAAAAAAA" + bytes(8) + b"BBBBBBBB"


class ScriptedRng:
    """random() fixed high (no escapes), randint -> lower bound, argmax choice."""

    def __init__(self, candidates=None, rand=0.99):
        self._candidates = candidates
        self._rand = rand
        self.seen: list[tuple[list, list[float]]] = []

    def random(self):
        return self._rand

    def randint(self, a, _b):
        return a

    def randint_list(self, a, _b, count):
        return list(self._candidates) if self._candidates else [a] * count

    def weighted_choice(self, seq, weights):
        self.seen.append((list(seq), list(weights)))
        best = max(range(len(weights)), key=lambda i: weights[i])
        return list(seq)[best]


def _bin_tilt(n, s, total_n, total_s):
    """Clamped per-seed bin rate over the seed's pooled rate, derived by hand."""
    rate = (s + PRIOR_A) / (n + PRIOR_A + PRIOR_B)
    pooled = (total_s + PRIOR_A) / (total_n + PRIOR_A + PRIOR_B)
    return min(W_MAX, max(W_MIN, rate / pooled))


class TestProtocol:
    def test_is_a_position_scheduler(self):
        s = PositionConsolidatedScheduler(RandPool(seed=1))

        assert isinstance(s, PositionScheduler)
        assert s.name == "consolidated"

    @pytest.mark.parametrize(("data", "buf_len"), [(b"", 8), (SEED, 0)])
    def test_declines_on_empty(self, data, buf_len):
        assert PositionConsolidatedScheduler(RandPool(seed=1)).propose(data, buf_len) is None


class TestCold:
    def test_falsification_cold_is_flat_over_candidates(self):
        """No evidence: every candidate weighs 1.0, uniform + boundary only."""
        rng = ScriptedRng(candidates=[3, 5, 7, 9, 11, 13])
        s = PositionConsolidatedScheduler(rng)

        pos = s.propose(SEED, len(SEED))
        cands, weights = rng.seen[-1]

        assert len(cands) == K_UNIFORM + 1  # + boundary, no levy anchor, no bins
        assert weights == [1.0] * len(cands)
        assert pos == 3


class TestScoring:
    def test_per_seed_bins_tilt_toward_gains(self):
        rng = ScriptedRng(candidates=[20, 2, 4, 6, 8, 10])
        s = PositionConsolidatedScheduler(rng)
        for _ in range(3):
            s.record(SEED, [2], Outcome.GAIN)
        for _ in range(3):
            s.record(SEED, [20], Outcome.MISS)

        s.propose(SEED, len(SEED))
        cands, weights = next(c for c in rng.seen if 20 in c[0] and len(c[0]) > K_UNIFORM)
        by_off = dict(zip(cands, weights, strict=False))

        assert by_off[2] == pytest.approx(_bin_tilt(3, 3, 6, 3))
        assert by_off[20] == pytest.approx(_bin_tilt(3, 0, 6, 3))
        assert by_off[2] > by_off[20]

    def test_regression_memo_never_serves_a_stale_tilt(self):
        """Adversarial: a credit between proposals must reach the next score."""
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        s.record(SEED, [2, 20], Outcome.MISS)
        s.propose(SEED, len(SEED))
        before = s.weight(SEED, 2)
        s.record(SEED, [2], Outcome.GAIN)

        assert s.weight(SEED, 2) == pytest.approx(_bin_tilt(2, 1, 3, 1))
        assert s.weight(SEED, 2) != pytest.approx(before)

    def test_levy_and_bin_experts_join_after_a_gain(self):
        rng = ScriptedRng(candidates=[1, 2, 3, 4, 5, 6])
        s = PositionConsolidatedScheduler(rng)
        s.record(SEED, [9], Outcome.GAIN)

        s.propose(SEED, len(SEED))
        cands, _w = rng.seen[-1]

        assert len(cands) == K_UNIFORM + 3  # boundary, levy, bins

    def test_adversarial_tilt_bounded(self):
        """A single expert never collapses the draw: weights within W_MIN^2..W_MAX^2."""
        rng = RandPool(seed=5)
        s = PositionConsolidatedScheduler(rng)
        for i in range(2 * MIN_OBS):
            s.record(SEED, [i % len(SEED)], Outcome.GAIN if i % 7 == 0 else Outcome.MISS)

        for o in range(len(SEED)):
            w = s.weight(SEED, o)
            assert W_MIN * W_MIN <= w <= W_MAX * W_MAX


class TestContextTilt:
    def test_cold_and_out_of_range_are_neutral(self):
        c = PositionContextScheduler(RandPool(seed=1))

        assert c.tilt(SEED, 0) == 1.0
        assert c.tilt(SEED, len(SEED)) == 1.0

    def test_warm_tilt_matches_rate_over_base(self):
        """One miss spread over MIN_OBS offsets at 0, then one gain at 8."""
        c = PositionContextScheduler(RandPool(seed=1))
        c.record(SEED, [0] * MIN_OBS, Outcome.MISS)
        c.record(SEED, [8], Outcome.GAIN)
        rate = (1.0 + CTX_A) / (1.0 + CTX_A + CTX_B)  # cell of 8: one gain
        base = (1.0 + CTX_A) / (2.0 + CTX_A + CTX_B)  # totals: one gain, one miss

        assert c.tilt(SEED, 8) == pytest.approx(min(W_MAX, max(W_MIN, rate / base)))


class TestRecord:
    def test_off_policy_record_feeds_every_learner(self):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        s.record(SEED, [4, 12], Outcome.GAIN, weight=2.0)

        assert s.context_obs == 2
        assert s.anchor(SEED) in (4, 12)
        assert s.seed_count() == 1

    @pytest.mark.parametrize("weight", [math.nan, -1.0, 0.0, math.inf])
    def test_adversarial_weights_and_offsets(self, weight):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        s.record(SEED, [-5, 10**9, 3], Outcome.GAIN, weight=weight)
        s.record(SEED, [], Outcome.MISS)
        s.record(b"", [1], Outcome.GAIN)

        for buf_len in (1, 3, len(SEED), 4 * len(SEED)):
            pos = s.propose(SEED, buf_len)
            assert 0 <= pos < buf_len

    def test_regression_memory_bounded(self):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        for i in range(MAX_SEEDS + 50):
            s.record(i.to_bytes(4, "little") * 4, [1], Outcome.GAIN)

        assert s.seed_count() <= MAX_SEEDS


class TestPersistence:
    def test_round_trip(self):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        s.record(SEED, [6], Outcome.GAIN)
        t = PositionConsolidatedScheduler(RandPool(seed=2))
        t.from_dict(s.to_dict())

        assert t.to_dict() == s.to_dict()
        assert t.anchor(SEED) == 6
        assert t.context_obs == 1

    @pytest.mark.parametrize(
        "bad", [{"version": STATE_VERSION + 1}, {"version": STATE_VERSION}, "junk", 7]
    )
    def test_malformed_starts_fresh(self, bad, caplog):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        s.record(SEED, [6], Outcome.GAIN)
        with caplog.at_level(logging.WARNING):
            s.from_dict(bad)

        assert s.anchor(SEED) is None
        assert s.context_obs == 0
        assert "consolidated position state unreadable" in caplog.text

    def test_empty_is_fresh_without_warning(self, caplog):
        s = PositionConsolidatedScheduler(RandPool(seed=1))
        with caplog.at_level(logging.WARNING):
            s.from_dict({})

        assert s.context_obs == 0
        assert not caplog.text
