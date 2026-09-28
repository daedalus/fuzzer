"""PositionContextScheduler: byte-context offset selection pooled across seeds.

Covers core/schedulers/pos_context.py and core/schedulers/_bytecls.py.
"""

import logging
import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bytecls import (
    CLS_ALPHA,
    CLS_CONTROL,
    CLS_DIGIT,
    CLS_FF,
    CLS_HIGH,
    CLS_PUNCT,
    CLS_ZERO,
    NUM_CLASSES,
    byte_class,
)
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_context import (
    BOF_PREV,
    DECILES,
    DISCOUNT,
    DISCOUNT_EVERY,
    K_CANDIDATES,
    MIN_OBS,
    NUM_CTX,
    PRIOR_A,
    PRIOR_B,
    STATE_VERSION,
    W_MAX,
    W_MIN,
    PositionContextScheduler,
    ctx_index,
)

ZERO_HEAVY = bytes(1000)  # class ZERO everywhere
ALPHA_HEAVY = b"a" * 1000  # class ALPHA everywhere


class ScriptedRng:
    """Scripted candidate draws; weighted_choice is argmax and records weights."""

    def __init__(self, candidates=()):
        self._candidates = list(candidates)
        self.seen_weights: list[list[float]] = []
        self.draws: list[tuple[int, int, int]] = []

    def randint_list(self, a, b, count):
        self.draws.append((a, b, count))
        if self._candidates:
            return list(self._candidates)
        return [a] * count

    def weighted_choice(self, seq, weights):
        self.seen_weights.append(list(weights))
        return seq[max(range(len(seq)), key=weights.__getitem__)]


def _ctx(rng=None):
    return PositionContextScheduler(rng or ScriptedRng())


def _warm(sched, data=ZERO_HEAVY):
    """Reach MIN_OBS with neutral misses at offset 500 of *data*."""
    for _ in range(MIN_OBS):
        sched.record(data, [500], Outcome.MISS)


class TestByteClass:
    def test_class_boundaries_pinned(self):
        # Pinned: 0x7F (DEL) is CONTROL; 0x20 and 0x7E are PUNCT.
        cases = {
            0x00: CLS_ZERO,
            0x01: CLS_CONTROL,
            0x1F: CLS_CONTROL,
            0x20: CLS_PUNCT,
            0x2F: CLS_PUNCT,
            0x30: CLS_DIGIT,
            0x39: CLS_DIGIT,
            0x3A: CLS_PUNCT,
            0x40: CLS_PUNCT,
            0x41: CLS_ALPHA,
            0x5A: CLS_ALPHA,
            0x5B: CLS_PUNCT,
            0x60: CLS_PUNCT,
            0x61: CLS_ALPHA,
            0x7A: CLS_ALPHA,
            0x7B: CLS_PUNCT,
            0x7E: CLS_PUNCT,
            0x7F: CLS_CONTROL,
            0x80: CLS_HIGH,
            0xFE: CLS_HIGH,
            0xFF: CLS_FF,
        }
        for b, want in cases.items():
            assert byte_class(b) == want, hex(b)

    def test_printable_ascii_matches_str_predicates(self):
        # Oracle derived independently from str.isdigit / str.isalpha.
        for b in range(0x20, 0x7F):
            ch = chr(b)
            want = CLS_DIGIT if ch.isdigit() else CLS_ALPHA if ch.isalpha() else CLS_PUNCT
            assert byte_class(b) == want, hex(b)

    def test_seven_distinct_classes_cover_every_byte(self):
        seen = {byte_class(b) for b in range(256)}
        assert seen == set(range(NUM_CLASSES))
        assert NUM_CLASSES == 7


class TestCtxIndex:
    def test_index_is_a_bijection_onto_the_table(self):
        # FALSIFICATION: a colliding index would silently merge contexts.
        seen = set()
        for cls in range(NUM_CLASSES):
            for prev in range(NUM_CLASSES + 1):
                for dec in range(DECILES):
                    seen.add((cls * (NUM_CLASSES + 1) + prev) * DECILES + dec)
        assert len(seen) == NUM_CTX == 560

    def test_first_byte_uses_the_bof_prev_class(self):
        data = b"a" + bytes(99)
        want = (CLS_ALPHA * (NUM_CLASSES + 1) + BOF_PREV) * DECILES + 0
        assert ctx_index(data, 0) == want

    def test_bof_prev_differs_from_every_real_class(self):
        assert BOF_PREV not in range(NUM_CLASSES)

    def test_prev_class_is_the_preceding_byte(self):
        data = b"1" + b"a" + bytes(98)
        want = (CLS_ALPHA * (NUM_CLASSES + 1) + CLS_DIGIT) * DECILES + 0
        assert ctx_index(data, 1) == want

    def test_decile_edges(self):
        data = bytes(100)
        assert ctx_index(data, 0) % DECILES == 0
        assert ctx_index(data, 9) % DECILES == 0
        assert ctx_index(data, 10) % DECILES == 1
        assert ctx_index(data, 99) % DECILES == 9

    def test_decile_matches_floor_formula_and_caps_at_nine(self):
        # Oracle: floor(o * 10 / n) computed with exact rationals.
        from fractions import Fraction

        for n in (1, 2, 3, 7, 10, 11, 999):
            for o in range(n):
                want = min(9, int(Fraction(o * 10, n)))
                assert ctx_index(bytes(n), o) % DECILES == want, (n, o)

    def test_every_offset_stays_in_the_table(self):
        data = bytes(range(256)) * 3
        assert all(0 <= ctx_index(data, o) < NUM_CTX for o in range(len(data)))


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_ctx(), PositionScheduler)

    def test_name(self):
        assert _ctx().name == "context"


class TestPropose:
    def test_cold_declines(self):
        s = _ctx()
        assert s.propose(ZERO_HEAVY, len(ZERO_HEAVY)) is None

    def test_declines_one_observation_short_of_warm(self):
        s = _ctx()
        for _ in range(MIN_OBS - 1):
            s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.propose(ZERO_HEAVY, len(ZERO_HEAVY)) is None
        s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.propose(ZERO_HEAVY, len(ZERO_HEAVY)) is not None

    def test_empty_data_or_buffer_declines_even_when_warm(self):
        s = _ctx()
        _warm(s)
        assert s.propose(b"", 10) is None
        assert s.propose(ZERO_HEAVY, 0) is None
        assert s.propose(ZERO_HEAVY, -3) is None

    def test_draws_k_candidates_within_the_shorter_of_buffer_and_seed(self):
        rng = ScriptedRng()
        s = _ctx(rng)
        _warm(s)
        s.propose(ZERO_HEAVY, 40)  # buffer shrank
        s.propose(ZERO_HEAVY, 5000)  # buffer grew
        assert rng.draws == [(0, 39, K_CANDIDATES), (0, len(ZERO_HEAVY) - 1, K_CANDIDATES)]
        assert K_CANDIDATES == 16

    def test_hot_context_beats_cold_context(self):
        # Two 100-byte halves: 'a' block then '0' block. Gains only in 'a'.
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[750, 250] + [750] * (K_CANDIDATES - 2))
        s = _ctx(rng)
        _warm(s, data)
        for _ in range(60):
            s.record(data, [250], Outcome.GAIN)
        assert s.propose(data, len(data)) == 250

    def test_gain_in_one_context_does_not_raise_another(self):
        # FALSIFICATION: if weights leaked across contexts the cold offset
        # (750) would tie or beat the hot one.
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[750, 250])
        s = _ctx(rng)
        _warm(s, data)
        for _ in range(60):
            s.record(data, [250], Outcome.GAIN)
        s.propose(data, len(data))
        w_cold, w_hot = rng.seen_weights[-1]
        assert w_hot > w_cold

    def test_misses_lower_a_context_below_the_rest(self):
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[250, 750])
        s = _ctx(rng)
        _warm(s, data)
        for _ in range(200):
            s.record(data, [250], Outcome.MISS)
        s.record(data, [750], Outcome.GAIN)
        assert s.propose(data, len(data)) == 750

    def test_equal_evidence_gives_equal_weights(self):
        # Control: identical rates in both contexts must not tilt.
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[250, 750])
        s = _ctx(rng)
        for i in range(MIN_OBS):  # interleaved: discounting hits both alike
            for o in (250, 750):
                s.record(data, [o], Outcome.GAIN if i < 5 else Outcome.MISS)
        s.propose(data, len(data))
        a, b = rng.seen_weights[-1]
        assert a == pytest.approx(b)

    def test_weights_are_clamped(self):
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[250, 750])
        s = _ctx(rng)
        _warm(s, data)
        for _ in range(1000):
            s.record(data, [250], Outcome.GAIN)
        for _ in range(20000):
            s.record(data, [750], Outcome.MISS)  # drags the global rate down
        s.propose(data, len(data))
        hot, cold = rng.seen_weights[-1]
        assert hot == W_MAX
        assert W_MIN <= cold <= W_MAX

    def test_lower_clamp_is_reachable(self):
        data = b"a" * 500 + b"0" * 500
        rng = ScriptedRng(candidates=[250, 750])
        s = _ctx(rng)
        for _ in range(5000):
            s.record(data, [250], Outcome.MISS)
        for _ in range(5000):
            s.record(data, [750], Outcome.GAIN)
        s.propose(data, len(data))
        low, high = rng.seen_weights[-1]
        assert low == W_MIN
        assert high > 1.0

    def test_all_weights_positive_and_finite(self):
        rng = ScriptedRng(candidates=list(range(K_CANDIDATES)))
        s = _ctx(rng)
        _warm(s)
        s.propose(ZERO_HEAVY, len(ZERO_HEAVY))
        assert all(math.isfinite(w) and w > 0 for w in rng.seen_weights[-1])

    def test_result_is_always_inside_the_buffer(self):
        s = PositionContextScheduler(RandPool(seed=7))
        data = bytes(range(256)) * 2
        _warm(s, data)
        for buf_len in (1, 2, 17, len(data), len(data) + 100):
            for _ in range(20):
                off = s.propose(data, buf_len)
                assert 0 <= off < min(buf_len, len(data))

    def test_one_byte_seed(self):
        s = PositionContextScheduler(RandPool(seed=1))
        for _ in range(MIN_OBS):
            s.record(b"x", [0], Outcome.MISS)
        assert s.propose(b"x", 1) == 0

    def test_pools_evidence_across_seeds(self):
        # The point of the arm: what seed A taught applies to unseen seed B.
        a = b"a" * 500 + b"0" * 500
        b = b"b" * 500 + b"9" * 500  # same class layout, different bytes
        rng = ScriptedRng(candidates=[750, 250])
        s = _ctx(rng)
        _warm(s, a)
        for _ in range(60):
            s.record(a, [250], Outcome.GAIN)
        assert s.propose(b, len(b)) == 250


class TestRecord:
    def test_negative_and_out_of_range_offsets_are_ignored(self):
        s = _ctx()
        s.record(ZERO_HEAVY, [-1, len(ZERO_HEAVY), 10**9], Outcome.GAIN)
        assert s.obs == 0

    def test_mixed_valid_and_invalid_offsets_only_credit_valid(self):
        s = _ctx()
        s.record(ZERO_HEAVY, [-5, 500, 10**9], Outcome.GAIN)
        assert s.obs == 1
        assert s.context_counts(ZERO_HEAVY, 500) == (pytest.approx(1.0), 0.0)

    def test_weight_is_split_evenly(self):
        s = _ctx()
        s.record(ZERO_HEAVY, [100, 500], Outcome.GAIN, weight=2.0)
        assert s.context_counts(ZERO_HEAVY, 100)[0] == pytest.approx(1.0)
        assert s.context_counts(ZERO_HEAVY, 500)[0] == pytest.approx(1.0)

    def test_miss_never_credits_success(self):
        # FALSIFICATION: a MISS that bumped succ would make every round a gain.
        s = _ctx()
        s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.context_counts(ZERO_HEAVY, 500) == (0.0, 1.0)

    def test_empty_offsets_is_a_noop(self):
        s = _ctx()
        s.record(ZERO_HEAVY, [], Outcome.GAIN)
        assert s.obs == 0

    def test_empty_data_is_a_noop(self):
        s = _ctx()
        s.record(b"", [0], Outcome.GAIN)
        assert s.obs == 0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
    def test_unusable_weight_is_ignored(self, bad):
        # ADVERSARIAL: one NaN would poison every rate for the rest of the run.
        s = _ctx()
        s.record(ZERO_HEAVY, [500], Outcome.GAIN, weight=bad)
        assert s.obs == 0
        assert s.context_counts(ZERO_HEAVY, 500) == (0.0, 0.0)

    def test_no_discount_until_the_boundary(self):
        s = _ctx()
        for _ in range(DISCOUNT_EVERY - 1):
            s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.context_counts(ZERO_HEAVY, 500)[1] == pytest.approx(DISCOUNT_EVERY - 1)

    def test_discount_shrinks_counts_at_the_boundary(self):
        s = _ctx()
        for _ in range(DISCOUNT_EVERY):
            s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.context_counts(ZERO_HEAVY, 500)[1] == pytest.approx(DISCOUNT_EVERY * DISCOUNT)
        assert DISCOUNT_EVERY == 256
        assert DISCOUNT == 0.95

    def test_regression_discount_swaps_table_mid_record(self):
        # One call crossing the boundary mid-list still discounts once.
        s = _ctx()
        for _ in range(DISCOUNT_EVERY - 1):
            s.record(ZERO_HEAVY, [500], Outcome.MISS)
        s.record(ZERO_HEAVY, [500, 500], Outcome.MISS, weight=2.0)
        # 256th credit discounts (255 + 1) * DISCOUNT, then the 2nd offset adds 1.
        got = s.context_counts(ZERO_HEAVY, 500)[1]
        assert got == pytest.approx(DISCOUNT_EVERY * DISCOUNT + 1.0)

    def test_obs_is_not_discounted(self):
        s = _ctx()
        for _ in range(DISCOUNT_EVERY):
            s.record(ZERO_HEAVY, [500], Outcome.MISS)
        assert s.obs == DISCOUNT_EVERY

    def test_table_size_is_fixed(self):
        # Rule 54: memory does not grow with seeds or offsets.
        s = _ctx()
        for i in range(50):
            seed = bytes((i * 7 + j) & 0xFF for j in range(300 + i))
            s.record(seed, list(range(0, len(seed), 13)), Outcome.GAIN)
        st = s.to_dict()
        assert len(st["succ"]) == len(st["fail"]) == NUM_CTX


class TestPersistence:
    def _trained(self):
        s = _ctx()
        _warm(s)
        s.record(ZERO_HEAVY, [10, 20], Outcome.GAIN, weight=1.5)
        return s

    def test_round_trip(self):
        s = self._trained()
        t = _ctx()
        t.from_dict(s.to_dict())
        assert t.obs == s.obs
        assert t.to_dict() == s.to_dict()
        assert t.context_counts(ZERO_HEAVY, 10) == s.context_counts(ZERO_HEAVY, 10)

    def test_state_is_versioned(self):
        assert self._trained().to_dict()["version"] == STATE_VERSION

    @pytest.mark.parametrize("empty", [None, {}])
    def test_empty_payload_starts_fresh(self, empty):
        s = self._trained()
        s.from_dict(empty)
        assert s.obs == 0

    def _corrupt(self, mutate):
        payload = self._trained().to_dict()
        mutate(payload)
        return payload

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.update(version=STATE_VERSION + 1),
            lambda p: p.update(succ=p["succ"][:-1]),
            lambda p: p.update(fail=p["fail"] + [0.0]),
            lambda p: p.update(succ="not a list"),
            lambda p: p.update(succ=["x"] * NUM_CTX),
            lambda p: p["succ"].__setitem__(0, float("nan")),
            lambda p: p["fail"].__setitem__(3, float("inf")),
            lambda p: p["succ"].__setitem__(5, -1.0),
            lambda p: p.update(obs=-1),
            lambda p: p.update(obs="many"),
            lambda p: p.pop("fail"),
        ],
    )
    def test_malformed_state_resets_and_warns(self, mutate, caplog):
        # ADVERSARIAL: a corrupt resume file must never crash or poison the table.
        s = _ctx()
        with caplog.at_level(logging.WARNING):
            s.from_dict(self._corrupt(mutate))
        assert s.obs == 0
        assert s.to_dict()["succ"] == [0.0] * NUM_CTX
        assert any("context" in r.message for r in caplog.records)

    def test_non_dict_payload_resets(self, caplog):
        s = self._trained()
        with caplog.at_level(logging.WARNING):
            s.from_dict([1, 2, 3])
        assert s.obs == 0


class TestDiagnostics:
    def test_top_contexts_ranks_by_rate_with_counts(self):
        s = _ctx()
        for _ in range(20):
            s.record(ZERO_HEAVY, [500], Outcome.GAIN)
        for _ in range(20):
            s.record(ALPHA_HEAVY, [500], Outcome.MISS)
        top = s.top_contexts(1)[0]
        assert (top.cls, top.prev_cls, top.decile) == (CLS_ZERO, CLS_ZERO, 5)
        assert top.count == pytest.approx(20.0)
        assert s.worst_contexts(1)[0].cls == CLS_ALPHA

    def test_unseen_contexts_are_not_listed(self):
        s = _ctx()
        assert s.top_contexts(5) == []
        s.record(ZERO_HEAVY, [500], Outcome.GAIN)
        assert len(s.top_contexts(5)) == 1

    def test_n_bounds_the_result(self):
        s = _ctx()
        for o in range(0, 1000, 100):
            s.record(ZERO_HEAVY, [o], Outcome.GAIN)
        assert len(s.top_contexts(3)) == 3
        assert s.top_contexts(0) == []


class TestRealRng:
    def test_prior_constants_are_miss_dominated(self):
        assert PRIOR_A == 1
        assert PRIOR_B == 20

    def test_runs_on_a_real_randpool(self):
        s = PositionContextScheduler(RandPool(seed=3))
        data = bytes(range(256)) * 4
        for o in range(0, len(data), 3):
            s.record(data, [o], Outcome.GAIN if o % 9 == 0 else Outcome.MISS)
        assert s.obs >= MIN_OBS
        assert 0 <= s.propose(data, len(data)) < len(data)
