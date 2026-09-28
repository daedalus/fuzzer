"""PositionBoundaryScheduler: offsets at content-derived field boundaries.

Covers core/schedulers/pos_boundary.py and the shared byte-class table in
core/schedulers/_bytecls.py. Randomness is scripted so the escape, the tail
draw, the boundary pick and the jitter are each pinned independently; scores
are pinned on crafted buffers whose boundaries are known by construction.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import pos_boundary
from fuzzer_tool.core.schedulers._bytecls import (
    BYTE_CLASS,
    CLS_ALPHA,
    CLS_CONTROL,
    CLS_DIGIT,
    CLS_HIGH,
    CLS_PUNCT,
    CLS_ZERO,
    NUM_CLASSES,
    byte_class,
)
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_boundary import (
    DELIMS,
    EPSILON,
    MAX_SEEDS,
    SCAN_CAP,
    TOP_K,
    WINDOW,
    PositionBoundaryScheduler,
    _window_entropy,
    score_boundaries,
    top_boundaries,
)

NO_ESCAPE = 0.99  # random() draw above EPSILON and above any tail share


class ScriptedRng:
    """Scripted random(), randint() and weighted_choice().

    Unscripted random() returns ``NO_ESCAPE``; unscripted randint returns the
    midpoint of its bounds (0 for the jitter's ``(-1, 1)``); unscripted
    weighted_choice picks the highest weight (first on ties). Every
    weighted_choice call is recorded in ``picks``.
    """

    def __init__(self, randoms=(), ints=(), choices=()):
        self._randoms = list(randoms)
        self._ints = list(ints)
        self._choices = list(choices)
        self.picks = []

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        v = self._ints.pop(0) if self._ints else (a + b) // 2
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v

    def weighted_choice(self, seq, weights):
        self.picks.append((list(seq), list(weights)))
        if self._choices:
            return seq[self._choices.pop(0)]
        return seq[max(range(len(weights)), key=lambda i: (weights[i], -i))]


def _sched(**kw):
    return PositionBoundaryScheduler(ScriptedRng(**kw))


def _pseudo(n, mul=37, add=11):
    """Deterministic byte soup with no structure a boundary term would like."""
    return bytes((i * mul + add) % 251 for i in range(n))


class TestByteClass:
    def test_scoring_uses_the_shared_class_table(self):
        # One definition of the classes: the vectorised table is upstream's.
        assert pos_boundary.CLASS_LUT.tolist() == list(BYTE_CLASS)
        assert len(pos_boundary.CLASS_LUT) == 256
        assert NUM_CLASSES == 7

    @pytest.mark.parametrize(
        ("value", "cls"),
        [
            (0x00, CLS_ZERO),
            (0x1F, CLS_CONTROL),
            (0x30, CLS_DIGIT),
            (0x41, CLS_ALPHA),
            (0x2C, CLS_PUNCT),
            (0x80, CLS_HIGH),
        ],
    )
    def test_a_class_change_scores_at_least_one(self, value, cls):
        assert byte_class(value) == cls
        base = 0x41 if cls != CLS_ALPHA else 0x30
        assert score_boundaries(bytes([base, value, base, base]))[1] >= 1.0


class TestProtocol:
    def test_satisfies_protocol(self):
        assert isinstance(_sched(), PositionScheduler)
        assert _sched().name == "boundary"

    def test_record_is_a_noop(self):
        s = _sched()
        data = b"ab,cd"
        before = s.propose(data, len(data))
        s.record(data, [1], Outcome.GAIN, 1.0)
        s.record(data, [-1], Outcome.MISS)
        assert s.propose(data, len(data)) == before


class TestScoring:
    def test_offset_zero_is_never_a_boundary(self):
        assert score_boundaries(b"a,b,c,d")[0] == 0.0

    def test_tiny_inputs_score_nothing(self):
        assert len(score_boundaries(b"")) == 0
        assert score_boundaries(b"x").tolist() == [0.0]

    def test_class_change_and_delimiter_start(self):
        # b"ab,cd": i=2 alpha->punct (class change); i=3 punct->alpha (class
        # change) and the byte after a delimiter. Neither is 4-aligned.
        score = score_boundaries(b"ab,cd")
        assert score[1] == 0.0  # alpha->alpha, nothing else
        assert score[2] == pytest.approx(1.0)
        assert score[3] == pytest.approx(2.0)

    def test_delimiter_set_is_the_documented_one(self):
        assert set(DELIMS) == set(b"\x00\n\r ,:;=/<>\"{}[]")

    def test_delimiter_term_needs_a_byte_that_differs(self):
        # Inside a NUL run the previous byte is a delimiter but nothing
        # starts: only the run edges and the alignment term may score.
        data = b"x" + b"\x00" * 6 + b"y"
        score = score_boundaries(data)
        assert score[1] == pytest.approx(1.5)  # class change + run start
        assert score[2] == 0.0
        assert score[3] == 0.0
        assert score[4] == pytest.approx(0.25)  # aligned only
        assert score[5] == 0.0
        assert score[6] == 0.0
        assert score[7] == pytest.approx(2.5)  # class + delim start + run end

    def test_run_edges(self):
        # One alpha class throughout, so only run edges and alignment score.
        score = score_boundaries(b"abcdQQQQefgh")
        assert score[4] == pytest.approx(0.5 + 0.25)  # run starts, aligned
        assert score[8] == pytest.approx(0.5 + 0.25)  # run ends, aligned
        assert score[5] == score[6] == score[7] == 0.0  # inside the run

    def test_run_shorter_than_run_min_is_ignored(self):
        assert pos_boundary.RUN_MIN == 4
        score = score_boundaries(b"abcQQQdef")
        assert score[3] == 0.0 and score[6] == 0.0

    def test_run_edge_between_two_long_runs_counts_once(self):
        score = score_boundaries(b"AAAAaaaa")
        assert score[4] == pytest.approx(0.5 + 0.25)  # not 1.0 + 0.25

    def test_entropy_step_is_normalised_to_one(self):
        # 16 identical bytes then 16 distinct letters, all alpha: at i=16 the
        # window entropies are 0 and 4 bits, a step of 4 / 4 = 1.0.
        data = b"Z" * 16 + bytes(range(ord("a"), ord("a") + 16))
        score = score_boundaries(data)
        assert score[16] == pytest.approx(1.0 + 0.5 + 0.25)  # + run end + aligned

    def test_entropy_step_is_symmetric(self):
        data = bytes(range(ord("a"), ord("a") + 16)) + b"Z" * 16
        score = score_boundaries(data)
        assert score[16] == pytest.approx(1.0 + 0.5 + 0.25)  # run start this time

    def test_no_entropy_term_without_a_full_window_each_side(self):
        data = b"Z" * 15 + bytes(range(ord("a"), ord("a") + 16))
        # m = 31 < 2 * WINDOW: nothing can be scored on entropy.
        assert len(data) < 2 * WINDOW
        assert score_boundaries(data)[15] == pytest.approx(0.5)  # run end only

    def test_window_entropy_matches_the_definition(self):
        data = _pseudo(300, mul=7)
        arr = np.frombuffer(data, dtype=np.uint8)
        got = _window_entropy(arr)
        assert len(got) == len(arr) - WINDOW + 1
        for j in (0, 1, 57, 200, len(got) - 1):
            win = data[j : j + WINDOW]
            probs = [win.count(v) / WINDOW for v in set(win)]
            want = -sum(p * math.log2(p) for p in probs)
            assert got[j] == pytest.approx(want)

    def test_window_entropy_bounds(self):
        flat = _window_entropy(np.frombuffer(b"\x07" * 40, dtype=np.uint8))
        assert flat == pytest.approx(np.zeros(len(flat)))
        distinct = _window_entropy(np.arange(64, dtype=np.uint8))
        assert distinct == pytest.approx(np.full(len(distinct), math.log2(WINDOW)))

    def test_window_entropy_short_input_is_empty(self):
        assert len(_window_entropy(np.zeros(WINDOW - 1, dtype=np.uint8))) == 0

    def test_scan_cap_bounds_the_scored_prefix(self):
        data = b"ab,cd" * ((SCAN_CAP // 5) + 100)
        assert len(data) > SCAN_CAP
        assert len(score_boundaries(data)) == SCAN_CAP
        assert max(top_boundaries(data)[0]) < SCAN_CAP

    def test_top_k_bound_and_ordering(self):
        data = b"a1" * 2000  # a class change at every offset
        offsets, scores = top_boundaries(data)
        assert len(offsets) == len(scores) == TOP_K
        assert scores == sorted(scores, reverse=True)
        # Ties go to the lower offset.
        for (o1, s1), (o2, s2) in zip(
            zip(offsets, scores), zip(offsets[1:], scores[1:]), strict=False
        ):
            assert s1 > s2 or o1 < o2

    def test_only_positive_scores_are_kept(self):
        offsets, scores = top_boundaries(b"AAA")
        assert offsets == [] and scores == []

    def test_deterministic_per_seed(self):
        data = _pseudo(5000)
        assert top_boundaries(data) == top_boundaries(data)
        assert top_boundaries(data) != top_boundaries(_pseudo(5000, mul=41))


class TestPropose:
    def test_empty_data_and_empty_buffer_decline(self):
        s = _sched()
        assert s.propose(b"", 10) is None
        assert s.propose(b"ab,cd", 0) is None
        assert s.propose(b"ab,cd", -3) is None

    def test_featureless_seed_declines(self):
        for data in (b"A", b"AAA", b"\x00\x00\x00"):
            assert _sched().propose(data, len(data)) is None

    def test_uniform_escape_declines(self):
        assert _sched(randoms=[EPSILON / 2]).propose(b"ab,cd", 5) is None

    def test_no_escape_at_epsilon(self):
        assert _sched(randoms=[EPSILON]).propose(b"ab,cd", 5) == 3

    def test_picks_the_scored_boundary(self):
        # Offset 3 (score 2.0) outweighs offset 2 (score 1.0).
        rng = ScriptedRng()
        assert PositionBoundaryScheduler(rng).propose(b"ab,cd", 5) == 3
        seq, weights = rng.picks[0]
        assert len(seq) == len(weights) == 3  # offsets 2, 3 and the aligned 4
        assert sorted(weights) == [0.25, 1.0, 2.0]  # picked in proportion to score

    def test_pick_index_maps_to_offset(self):
        offsets, _ = top_boundaries(b"ab,cd")
        assert offsets == [3, 2, 4]
        for idx, want in enumerate(offsets):
            assert _sched(choices=[idx]).propose(b"ab,cd", 5) == want

    @pytest.mark.parametrize(("jitter", "want"), [(-1, 2), (0, 3), (1, 4)])
    def test_jitter_is_one_byte_either_way(self, jitter, want):
        assert _sched(ints=[jitter]).propose(b"ab,cd", 5) == want

    def test_jitter_is_clamped_to_the_buffer(self):
        # Only offsets 2 and 3 score; pick 2 with -1 -> 1, pick 3 with +1
        # into a 4-byte buffer -> 3 (not 4).
        assert _sched(choices=[0], ints=[-1]).propose(b"ab,cd", 4) == 2
        assert _sched(choices=[0], ints=[1]).propose(b"ab,cd", 4) == 3
        # A boundary at offset 1 jittered to -1 lands on 0, never below.
        data = b"a,bcdefgh"
        assert _sched(choices=[0], ints=[-1]).propose(data, len(data)) >= 0

    def test_sites_past_a_shrunken_buffer_are_dropped_not_clamped(self):
        data = b"ab,cd" + b"x" * 40 + b"," + b"y" * 20
        rng = ScriptedRng()
        pos = PositionBoundaryScheduler(rng).propose(data, 6)
        seq, weights = rng.picks[0]
        assert len(seq) < len(top_boundaries(data)[0])
        offsets = [o for o in top_boundaries(data)[0] if o < 6]
        assert len(seq) == len(offsets) == len(weights)
        assert 0 <= pos < 6

    def test_every_site_past_the_buffer_declines(self):
        # Every scored offset is >= 1, so a one-byte buffer has none left.
        assert _sched().propose(b"ab,cd", 1) is None

    def test_buffer_larger_than_seed_still_lands_on_a_seed_boundary(self):
        assert _sched().propose(b"ab,cd", 500) == 3

    def test_tail_of_a_long_seed_gets_its_share(self):
        data = b"ab,cd" * ((SCAN_CAP // 5) + 2000)
        n = len(data)
        share = (n - SCAN_CAP) / n
        rng = ScriptedRng(randoms=[NO_ESCAPE, share / 2], ints=[SCAN_CAP + 7])
        assert PositionBoundaryScheduler(rng).propose(data, n) == SCAN_CAP + 7
        assert rng.picks == []  # the boundary table was not consulted

    def test_head_draw_when_tail_draw_misses(self):
        data = b"ab,cd" * ((SCAN_CAP // 5) + 2000)
        n = len(data)
        rng = ScriptedRng(randoms=[NO_ESCAPE, (n - SCAN_CAP) / n])  # not below share
        pos = PositionBoundaryScheduler(rng).propose(data, n)
        assert pos < SCAN_CAP
        assert len(rng.picks) == 1

    def test_no_tail_draw_when_buffer_shrank_under_the_cap(self):
        data = b"ab,cd" * ((SCAN_CAP // 5) + 2000)
        rng = ScriptedRng(randoms=[NO_ESCAPE, 0.0])
        pos = PositionBoundaryScheduler(rng).propose(data, 100)
        assert 0 <= pos < 100

    def test_short_seed_never_draws_the_tail(self):
        rng = ScriptedRng()
        _ = PositionBoundaryScheduler(rng).propose(b"ab,cd" * 50, 250)
        assert len(rng.picks) == 1


class TestCache:
    def test_table_is_built_once_per_seed(self, monkeypatch):
        calls = []
        real = pos_boundary.top_boundaries
        monkeypatch.setattr(
            pos_boundary, "top_boundaries", lambda d: calls.append(d) or real(d)
        )
        s = _sched()
        for _ in range(5):
            s.propose(b"ab,cd", 5)
        assert len(calls) == 1
        s.propose(b"ef,gh", 5)
        assert len(calls) == 2

    def test_lru_is_bounded_and_evicts_the_oldest(self):
        s = _sched()
        seeds = [b"ab,%03d" % i for i in range(MAX_SEEDS + 5)]
        for d in seeds:
            s.propose(d, len(d))
        assert len(s._cache) == MAX_SEEDS

    def test_access_refreshes_recency(self, monkeypatch):
        s = _sched()
        seeds = [b"ab,%03d" % i for i in range(MAX_SEEDS)]
        for d in seeds:
            s.propose(d, len(d))
        s.propose(seeds[0], len(seeds[0]))  # oldest becomes newest
        s.propose(b"zz,new", 6)  # evicts seeds[1], not seeds[0]
        calls = []
        real = pos_boundary.top_boundaries
        monkeypatch.setattr(
            pos_boundary, "top_boundaries", lambda d: calls.append(d) or real(d)
        )
        s.propose(seeds[0], len(seeds[0]))
        assert calls == []
        s.propose(seeds[1], len(seeds[1]))
        assert calls == [seeds[1]]


class TestRealRng:
    def test_proposals_stay_in_range_and_near_a_boundary(self):
        data = b'{"len": 12, "name": "abc", "id": 7}\x00\x00\x00\x00' + _pseudo(120)
        table = set(top_boundaries(data)[0])
        s = PositionBoundaryScheduler(RandPool(seed=3))
        buf_len = len(data)
        got = [s.propose(data, buf_len) for _ in range(400)]
        declined = sum(p is None for p in got)
        assert 0 < declined < 400  # EPSILON escapes happen, but rarely
        for p in got:
            if p is not None:
                assert 0 <= p < buf_len
                assert {p - 1, p, p + 1} & table

    def test_shrunken_buffer_is_respected(self):
        data = b"ab,cd" * 200
        s = PositionBoundaryScheduler(RandPool(seed=5))
        for _ in range(200):
            p = s.propose(data, 37)
            assert p is None or 0 <= p < 37
