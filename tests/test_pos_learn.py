"""PositionLearnScheduler: LearnAFL's per-seed path-preserving byte mask.

Covers core/schedulers/pos_learn.py and core/auto_dict.py::frozen_tokens.
"""

import numpy as np

from fuzzer_tool.core.auto_dict import MAX_AUTO_TOKEN, MIN_AUTO_TOKEN, frozen_tokens
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_learn import (
    INDEX_RESTORE_MAX,
    MAX_BYTES,
    MAX_SEEDS,
    STABLE_ROUNDS,
    Mode,
    PositionLearnScheduler,
)
from tests.support.scripted_rng import ScriptedRng

SEED = bytes(range(64))


def _sched(moved, randints=()):
    box = {"m": moved, "tokens": []}
    s = PositionLearnScheduler(
        ScriptedRng(randints=randints),
        moved=lambda d: box["m"],
        emit_tokens=box["tokens"].extend,
    )
    return s, box


def _stabilise(s, data, offsets):
    """One learning round, then STABLE_ROUNDS that add no new mutable byte."""
    for _ in range(STABLE_ROUNDS + 1):
        s.record(data, offsets, Outcome.MISS)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched(False)[0], PositionScheduler)

    def test_cold_seed_declines(self):
        s, _ = _sched(False)
        assert s.propose(SEED, len(SEED)) is None
        assert s.mode(SEED) is None


class TestMask:
    def test_path_kept_marks_offsets_mutable(self):
        s, _ = _sched(False)
        s.record(SEED, [3, 9], Outcome.MISS)
        assert np.flatnonzero(s.mutable(SEED)).tolist() == [3, 9]

    def test_path_moved_marks_nothing(self):
        # LearnAFL only learns from mutants that kept the seed's path.
        s, _ = _sched(True)
        s.record(SEED, [3, 9], Outcome.GAIN)
        assert not s.mutable(SEED).any()

    def test_unknown_signal_is_not_credited(self):
        s, _ = _sched(None)
        s.record(SEED, [3], Outcome.MISS)
        assert s.mutable(SEED) is None

    def test_mode_flips_after_stable_rounds(self):
        s, _ = _sched(False)
        s.record(SEED, [3], Outcome.MISS)
        assert s.mode(SEED) is Mode.EXPLORE
        _stabilise(s, SEED, [3])
        assert s.mode(SEED) is Mode.EXPLOIT

    def test_falsification_one_round_short_stays_explore(self):
        s, _ = _sched(False)
        for _ in range(STABLE_ROUNDS):
            s.record(SEED, [3], Outcome.MISS)
        assert s.mode(SEED) is Mode.EXPLORE
        s.record(SEED, [3], Outcome.MISS)
        assert s.mode(SEED) is Mode.EXPLOIT

    def test_new_mutable_byte_resets_stability(self):
        s, _ = _sched(False)
        _stabilise(s, SEED, [3])
        s.record(SEED, [4], Outcome.MISS)
        assert s.mode(SEED) is Mode.EXPLORE

    def test_moved_rounds_count_as_stable(self):
        # A probe of a frozen byte that moves the path confirms the model.
        s, box = _sched(False)
        s.record(SEED, [3], Outcome.MISS)
        box["m"] = True
        _stabilise(s, SEED, [10])
        assert s.mode(SEED) is Mode.EXPLOIT


class TestPropose:
    def test_explore_proposes_frozen_bytes(self):
        s, _ = _sched(False, randints=[0, 2])
        s.record(SEED, [0, 1, 3], Outcome.MISS)
        frozen = [i for i in range(len(SEED)) if i not in (0, 1, 3)]
        assert s.propose(SEED, len(SEED)) == frozen[0]
        assert s.propose(SEED, len(SEED)) == frozen[2]

    def test_exploit_proposes_mutable_bytes(self):
        s, _ = _sched(False, randints=[1])
        _stabilise(s, SEED, [5, 40])
        assert s.propose(SEED, len(SEED)) == 40

    def test_shrunk_buffer_keeps_offsets_inside(self):
        s, _ = _sched(False, randints=[0])
        _stabilise(s, SEED, [5, 40])
        assert s.propose(SEED, 6) == 5

    def test_adversarial_buffer_shorter_than_every_site_declines(self):
        s, _ = _sched(False)
        _stabilise(s, SEED, [40])
        assert s.propose(SEED, 10) is None
        assert s.propose(SEED, 0) is None

    def test_adversarial_offsets_past_the_mask_are_ignored(self):
        s, _ = _sched(False)
        s.record(SEED, [-1, len(SEED), len(SEED) + 100], Outcome.MISS)
        assert s.mutable(SEED) is None


class TestRestore:
    def test_exploit_copies_frozen_bytes_back(self):
        s, _ = _sched(False)
        _stabilise(s, SEED, [5])
        buf = bytearray(b"\xff" * len(SEED))
        s.restore(SEED, buf)
        expect = bytearray(SEED)
        expect[5] = 0xFF
        assert buf == expect

    def test_explore_leaves_buffer_untouched(self):
        s, _ = _sched(False)
        s.record(SEED, [5], Outcome.MISS)
        buf = bytearray(b"\xff" * len(SEED))
        s.restore(SEED, buf)
        assert buf == bytearray(b"\xff" * len(SEED))

    def test_adversarial_resized_buffer_is_untouched(self):
        # Offsets are parent coordinates: a resize shifts them, so no copy-back.
        s, _ = _sched(False)
        _stabilise(s, SEED, [5])
        buf = bytearray(b"\xff" * (len(SEED) + 1))
        s.restore(SEED, buf)
        assert buf == bytearray(b"\xff" * (len(SEED) + 1))

    def test_large_frozen_side_matches_index_copy(self):
        # Swap path (frozen > INDEX_RESTORE_MAX) vs a per-byte reference.
        data = bytes(i % 251 + 1 for i in range(MAX_BYTES + 300))
        s, _ = _sched(False)
        _stabilise(s, data, [7, 1000, MAX_BYTES - 1])
        assert (~s.mutable(data)).sum() > INDEX_RESTORE_MAX
        buf = bytearray(b"\x00" * len(data))
        s.restore(data, buf)
        mutable = set(np.flatnonzero(s.mutable(data)).tolist())
        expect = bytes(0 if i in mutable or i >= MAX_BYTES else data[i] for i in range(len(data)))
        assert bytes(buf) == expect

    def test_small_frozen_side_matches_index_copy(self):
        data = bytes(range(1, 9))
        s, _ = _sched(False)
        _stabilise(s, data, [2, 5])
        buf = bytearray(len(data))
        s.restore(data, buf)
        assert bytes(buf) == bytes(0 if i in (2, 5) else data[i] for i in range(len(data)))

    def test_buffer_stays_resizable_after_restore(self):
        s, _ = _sched(False)
        _stabilise(s, SEED, [5])
        buf = bytearray(len(SEED))
        s.restore(SEED, buf)
        buf.append(0)
        assert len(buf) == len(SEED) + 1


class TestTokens:
    def test_stable_mask_emits_frozen_runs_once(self):
        data = b"MAGIC" + b"\x00" * 5 + b"HDRX" + bytes(range(1, 12))
        s, box = _sched(False)
        mutable = list(range(5, 10)) + list(range(14, len(data)))
        _stabilise(s, data, mutable)
        assert box["tokens"] == [b"MAGIC", b"HDRX"]
        _stabilise(s, data, mutable)
        assert box["tokens"] == [b"MAGIC", b"HDRX"]


class TestFrozenTokens:
    def test_runs_between_mutable_bytes(self):
        data = b"abcd" + b"zz" + b"efg"
        mutable = np.zeros(len(data), bool)
        mutable[4:6] = True
        assert frozen_tokens(data, mutable) == [b"abcd", b"efg"]

    def test_falsification_length_bounds(self):
        short = b"ab"
        long = bytes(range(1, MAX_AUTO_TOKEN + 2))
        data = short + b"z" + long
        mutable = np.zeros(len(data), bool)
        mutable[len(short)] = True
        assert len(short) < MIN_AUTO_TOKEN
        assert frozen_tokens(data, mutable) == []

    def test_adversarial_uniform_fill_and_empty(self):
        assert frozen_tokens(b"\x00" * 8, np.zeros(8, bool)) == []
        assert frozen_tokens(b"", np.zeros(0, bool)) == []
        assert frozen_tokens(b"abcdef", np.ones(6, bool)) == []


class TestBounds:
    def test_mask_capped_at_max_bytes(self):
        data = bytes(MAX_BYTES + 10)
        s, _ = _sched(False)
        s.record(data, [MAX_BYTES - 1, MAX_BYTES + 5], Outcome.MISS)
        m = s.mutable(data)
        assert len(m) == MAX_BYTES
        assert np.flatnonzero(m).tolist() == [MAX_BYTES - 1]

    def test_lru_evicts_oldest_seed(self):
        s, _ = _sched(False)
        seeds = [i.to_bytes(4, "little") for i in range(MAX_SEEDS + 1)]
        for d in seeds:
            s.record(d, [0], Outcome.MISS)
        assert s.mutable(seeds[0]) is None
        assert s.mutable(seeds[-1]) is not None
        assert s.seed_count() == MAX_SEEDS
