"""PositionCmplogScheduler: offsets from redqueen matches and Weizz spans.

Covers core/schedulers/pos_cmplog.py. Uses real StructureMap objects so the
flagged_spans contract the arm depends on is pinned here too.
"""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_cmplog import (
    EPSILON,
    MAX_SEEDS,
    MAX_TARGETS,
    W_LEN,
    PositionCmplogScheduler,
)
from fuzzer_tool.core.weizz_tags import ByteTag, StructureMap, TagFlags

SEED = bytes(range(64))
NO_ESCAPE = 0.99  # random() draw above EPSILON


class ScriptedRng:
    """Scripted random()/randint(); argmax weighted_choice (first wins ties)."""

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        # Unscripted: no jitter when 0 is in range, else the low bound.
        v = self._ints.pop(0) if self._ints else (0 if a <= 0 <= b else a)
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v

    def weighted_choice(self, seq, weights):
        return seq[max(range(len(seq)), key=weights.__getitem__)]


def _smap(n, spans):
    """StructureMap of length n; spans = [(start, end, cmp_id, flags)]."""
    tags = [ByteTag() for _ in range(n)]
    for start, end, cid, flags in spans:
        for i in range(start, end):
            tags[i] = ByteTag(cmp_id=cid, flags=flags)
    return StructureMap(tags=tags)


def _sched(rng=None, meta=None, smap=None):
    """Scheduler over one fixed meta/smap for every seed."""
    return PositionCmplogScheduler(
        rng or ScriptedRng(),
        meta_of=lambda d: meta,
        smap_of=lambda d: smap,
    )


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched(), PositionScheduler)

    def test_record_is_a_noop(self):
        s = _sched(meta={"redqueen_offsets": [5]})
        s.record(SEED, [5], Outcome.GAIN, 1.0)
        assert s.propose(SEED, len(SEED)) == 5


class TestDeclines:
    def test_no_meta(self):
        assert _sched(meta=None).propose(SEED, len(SEED)) is None

    def test_non_dict_meta(self):
        assert _sched(meta=["nope"]).propose(SEED, len(SEED)) is None

    def test_meta_without_cmplog_data(self):
        assert _sched(meta={"coverage_edges": 3}).propose(SEED, len(SEED)) is None

    def test_empty_data_and_buffer(self):
        s = _sched(meta={"redqueen_offsets": [1]})
        assert s.propose(b"", 10) is None
        assert s.propose(SEED, 0) is None

    def test_meta_of_raising_declines(self):
        def boom(d):
            raise RuntimeError("x")

        s = PositionCmplogScheduler(ScriptedRng(), meta_of=boom, smap_of=lambda d: None)
        assert s.propose(SEED, len(SEED)) is None

    def test_smap_of_raising_falls_back_to_redqueen(self):
        def boom(d):
            raise RuntimeError("x")

        s = PositionCmplogScheduler(
            ScriptedRng(), meta_of=lambda d: {"redqueen_offsets": [9]}, smap_of=boom
        )
        assert s.propose(SEED, len(SEED)) == 9

    def test_uniform_escape(self):
        s = _sched(ScriptedRng(randoms=[EPSILON / 2]), meta={"redqueen_offsets": [9]})
        assert s.propose(SEED, len(SEED)) is None

    def test_escape_boundary_is_exclusive(self):
        s = _sched(ScriptedRng(randoms=[EPSILON]), meta={"redqueen_offsets": [9]})
        assert s.propose(SEED, len(SEED)) == 9


class TestRedqueenPoints:
    def test_lands_on_offset(self):
        s = _sched(meta={"redqueen_offsets": [20]})
        assert s.propose(SEED, len(SEED)) == 20

    @pytest.mark.parametrize("jitter", [-1, 0, 1])
    def test_jitter(self, jitter):
        s = _sched(ScriptedRng(ints=[jitter]), meta={"redqueen_offsets": [20]})
        assert s.propose(SEED, len(SEED)) == 20 + jitter

    def test_jitter_clamped_low_and_high(self):
        lo = _sched(ScriptedRng(ints=[-1]), meta={"redqueen_offsets": [0]})
        assert lo.propose(SEED, len(SEED)) == 0
        hi = _sched(ScriptedRng(ints=[1]), meta={"redqueen_offsets": [63]})
        assert hi.propose(SEED, len(SEED)) == 63

    def test_junk_offsets_ignored(self):
        meta = {"redqueen_offsets": [-3, True, "7", None, 2.5, 12]}
        assert _sched(meta=meta).propose(SEED, len(SEED)) == 12

    def test_only_junk_declines(self):
        meta = {"redqueen_offsets": [-3, None, "x"]}
        assert _sched(meta=meta).propose(SEED, len(SEED)) is None

    def test_duplicates_collapse(self):
        s = _sched(meta={"redqueen_offsets": [4, 4, 4]})
        assert len(s._targets(SEED).targets) == 1

    def test_clamped_to_shrunken_buffer(self):
        # Only target starts past the live buffer: nothing to land on.
        s = _sched(meta={"redqueen_offsets": [50]})
        assert s.propose(SEED, 10) is None

    def test_shrunken_buffer_keeps_reachable_targets(self):
        s = _sched(meta={"redqueen_offsets": [50, 3]})
        assert s.propose(SEED, 10) == 3


class TestSpans:
    def test_len_span_uniform_inside(self):
        smap = _smap(64, [(8, 12, 1, TagFlags.IS_LEN)])
        s = _sched(ScriptedRng(ints=[10]), meta={"weizz_tags_len": 64}, smap=smap)
        assert s.propose(SEED, len(SEED)) == 10

    def test_span_end_is_exclusive(self):
        smap = _smap(64, [(8, 12, 1, TagFlags.IS_MAGIC)])
        s = _sched(ScriptedRng(ints=[11]), meta={"weizz_tags_len": 64}, smap=smap)
        assert s.propose(SEED, len(SEED)) == 11

    def test_len_outweighs_others(self):
        smap = _smap(
            64,
            [(0, 4, 1, TagFlags.IS_MAGIC), (20, 24, 2, TagFlags.IS_LEN)],
        )
        s = _sched(ScriptedRng(ints=[21]), meta={"weizz_tags_len": 64}, smap=smap)
        # argmax weighted_choice: the IS_LEN span (1.5) beats IS_MAGIC (1.0).
        assert s.propose(SEED, len(SEED)) == 21
        assert W_LEN > 1.0

    def test_span_with_several_flags_takes_max_weight(self):
        smap = _smap(64, [(30, 34, 1, TagFlags.IS_LEN | TagFlags.IS_CHECKSUM)])
        s = _sched(meta={"weizz_tags_len": 64}, smap=smap)
        targets = s._targets(SEED).targets
        assert [(t.start, t.end, t.weight) for t in targets] == [(30, 34, W_LEN)]

    @pytest.mark.parametrize(
        "flag",
        [
            TagFlags.IS_LEN,
            TagFlags.IS_CHECKSUM,
            TagFlags.IS_MAGIC,
            TagFlags.IS_INPUT_TO_STATE,
        ],
    )
    def test_each_targeted_flag_is_picked_up(self, flag):
        smap = _smap(64, [(5, 9, 1, flag)])
        s = _sched(meta={"weizz_tags_len": 64}, smap=smap)
        assert [(t.start, t.end) for t in s._targets(SEED).targets] == [(5, 9)]

    @pytest.mark.parametrize("flag", [TagFlags.IS_IMPL, TagFlags.NONE])
    def test_untargeted_flags_ignored(self, flag):
        smap = _smap(64, [(5, 9, 1, flag)])
        assert _sched(meta={"weizz_tags_len": 64}, smap=smap).propose(SEED, 64) is None

    def test_combined_mask_matches_any_bit(self):
        # Pins the StructureMap behaviour the module docstring relies on:
        # a combined mask selects a span carrying any of its bits.
        smap = _smap(64, [(5, 9, 1, TagFlags.IS_CHECKSUM)])
        assert smap.flagged_spans(TagFlags.IS_LEN | TagFlags.IS_CHECKSUM) == [(5, 9, 1)]
        assert smap.flagged_spans(TagFlags.IS_LEN) == []

    def test_span_clamped_to_shrunken_buffer(self):
        smap = _smap(64, [(8, 40, 1, TagFlags.IS_LEN)])
        # randint's upper bound is min(end, buf_len) - 1 == 15.
        s = _sched(ScriptedRng(ints=[15]), meta={"weizz_tags_len": 64}, smap=smap)
        assert s.propose(SEED, 16) == 15

    def test_span_past_shrunken_buffer_dropped(self):
        smap = _smap(64, [(40, 44, 1, TagFlags.IS_LEN)])
        s = _sched(meta={"weizz_tags_len": 64}, smap=smap)
        assert s.propose(SEED, 16) is None

    def test_spans_and_points_combine(self):
        smap = _smap(64, [(8, 12, 1, TagFlags.IS_LEN)])
        s = _sched(meta={"redqueen_offsets": [40], "weizz_tags_len": 64}, smap=smap)
        assert len(s._targets(SEED).targets) == 2


class TestCache:
    def test_reused_while_meta_unchanged(self):
        calls = []
        meta = {"redqueen_offsets": [1]}

        def smap_of(d):
            calls.append(d)
            return None

        s = PositionCmplogScheduler(ScriptedRng(), lambda d: meta, smap_of)
        for _ in range(5):
            s.propose(SEED, len(SEED))
        assert len(calls) == 1

    def test_rebuilt_when_tags_go_dirty(self):
        smap = _smap(64, [(8, 12, 1, TagFlags.IS_LEN)])
        meta = {"weizz_tags_len": 64}
        state = {"smap": smap}
        s = PositionCmplogScheduler(ScriptedRng(), lambda d: meta, lambda d: state["smap"])
        assert len(s._targets(SEED).targets) == 1
        # Dirty tags: the real accessor returns None and the flag flips.
        meta["weizz_tags_dirty"] = True
        state["smap"] = None
        assert s._targets(SEED).targets == []

    def test_rebuilt_when_redqueen_grows(self):
        meta = {"redqueen_offsets": [1]}
        s = _sched(meta=meta)
        assert len(s._targets(SEED).targets) == 1
        meta["redqueen_offsets"] = [1, 2, 3]
        assert len(s._targets(SEED).targets) == 3

    def test_lru_bound(self):
        s = _sched(meta={"redqueen_offsets": [1]})
        for i in range(MAX_SEEDS + 20):
            s._targets(i.to_bytes(4, "little") * 4)
        assert len(s._cache) == MAX_SEEDS

    def test_lru_keeps_recently_used(self):
        s = _sched(meta={"redqueen_offsets": [1]})
        first = b"first-seed-bytes"
        s._targets(first)
        for i in range(MAX_SEEDS - 1):
            s._targets(i.to_bytes(4, "little") * 4)
        s._targets(first)  # touch: now most recent
        s._targets(b"one-more-seed!!!")
        assert len(s._cache) == MAX_SEEDS
        import xxhash

        assert xxhash.xxh3_64_intdigest(first) in s._cache

    def test_target_cap(self):
        s = _sched(meta={"redqueen_offsets": list(range(MAX_TARGETS + 100))})
        assert len(s._targets(bytes(2000)).targets) == MAX_TARGETS


class TestRealRng:
    def test_never_raises_and_stays_in_bounds(self):
        smap = _smap(64, [(8, 12, 1, TagFlags.IS_LEN), (30, 31, 2, TagFlags.IS_MAGIC)])
        meta = {"redqueen_offsets": [0, 5, 63, 200], "weizz_tags_len": 64}
        s = _sched(RandPool(seed=3), meta=meta, smap=smap)
        seen = set()
        for buf_len in (1, 2, 10, 64, 100):
            for _ in range(300):
                p = s.propose(SEED, buf_len)
                if p is not None:
                    assert 0 <= p < buf_len
                    seen.add(p)
        assert seen  # actually proposed something
