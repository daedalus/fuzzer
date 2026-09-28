"""PositionLineageScheduler: offsets from the sites that produced the seed.

Covers core/schedulers/pos_lineage.py. Randomness is scripted so the
escape, the site pick, the jitter magnitude and the jitter sign are each
pinned independently.
"""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_lineage import (
    EPSILON,
    JITTER,
    MAX_SITES,
    PositionLineageScheduler,
    reflect,
)

SEED = bytes(range(200))
NO_ESCAPE = 0.99  # random() draw above EPSILON
U_ZERO_JITTER = 0.99  # magnitude floor(log(.99)/log(8/9)) == 0
POSITIVE = 0.99  # sign draw >= 0.5
NEGATIVE = 0.1  # sign draw < 0.5
DELOCALISED = frozenset({"byte_shuffle", "chunk_shuffle"})


class ScriptedRng:
    """Scripted random() (escape, magnitude u, sign) and randint (site index).

    Unscripted random() draws return ``NO_ESCAPE`` (no escape, zero jitter
    magnitude, positive sign); unscripted randint returns the low bound.
    """

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        v = self._ints.pop(0) if self._ints else a
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v


def _sched(rng=None, meta=None, delocalised=DELOCALISED):
    return PositionLineageScheduler(
        rng or ScriptedRng(), meta_of=lambda d: meta, delocalised=delocalised
    )


def _magnitude_u(k):
    """A random() draw whose jitter magnitude is exactly *k* bytes."""
    # k = floor(log(u) / log(1 - 1/(1+JITTER))): the top of the k-th bucket.
    q = 1.0 / (1.0 + JITTER)
    return math.exp((k + 0.5) * math.log1p(-q))


class TestProtocol:
    def test_satisfies_protocol(self):
        assert isinstance(_sched(), PositionScheduler)
        assert _sched().name == "lineage"

    def test_record_is_a_noop(self):
        s = _sched(meta={"parent_sites": [5]})
        s.record(SEED, [5], Outcome.GAIN, 1.0)
        s.record(SEED, [-1], Outcome.MISS)
        # Nothing learned: still proposes from the same sites.
        assert s.propose(SEED, len(SEED)) == 5


class TestDeclines:
    @pytest.mark.parametrize(
        "meta",
        [
            None,
            {},
            {"parent_sites": []},
            {"parent_sites": None},
            {"parent_sites": 7},
            {"parent_sites": "40"},
            {"parent_sites": [-1, True, "x", None, 2.5]},
            {"lineage_depth": 0},  # initial-corpus seed
        ],
    )
    def test_no_usable_sites_declines(self, meta):
        assert _sched(meta=meta).propose(SEED, len(SEED)) is None

    def test_meta_that_is_not_a_dict_declines(self):
        for bad in ([1, 2], "meta", 5, object()):
            assert _sched(meta=bad).propose(SEED, len(SEED)) is None

    def test_meta_of_raising_declines_never_raises(self):
        def boom(_):
            raise RuntimeError("no meta")

        s = PositionLineageScheduler(ScriptedRng(), meta_of=boom)
        assert s.propose(SEED, len(SEED)) is None

    def test_empty_data_and_empty_buffer_decline(self):
        s = _sched(meta={"parent_sites": [3]})
        assert s.propose(b"", 10) is None
        assert s.propose(SEED, 0) is None
        assert s.propose(SEED, -3) is None

    def test_uniform_escape_declines(self):
        s = _sched(ScriptedRng(randoms=[EPSILON / 2]), meta={"parent_sites": [40]})
        assert s.propose(SEED, len(SEED)) is None

    def test_no_escape_just_above_epsilon(self):
        s = _sched(ScriptedRng(randoms=[EPSILON]), meta={"parent_sites": [40]})
        assert s.propose(SEED, len(SEED)) == 40


class TestSites:
    def test_zero_jitter_lands_on_the_site(self):
        assert _sched(meta={"parent_sites": [40]}).propose(SEED, len(SEED)) == 40

    def test_pick_is_uniform_index_over_sites(self):
        meta = {"parent_sites": [10, 50, 90]}
        for idx, want in enumerate([10, 50, 90]):
            s = _sched(ScriptedRng(ints=[idx]), meta=meta)
            assert s.propose(SEED, len(SEED)) == want

    def test_invalid_entries_are_dropped_not_fatal(self):
        meta = {"parent_sites": [-4, True, "z", 60, None]}
        assert _sched(meta=meta).propose(SEED, len(SEED)) == 60

    def test_tuple_sites_accepted(self):
        assert _sched(meta={"parent_sites": (70,)}).propose(SEED, len(SEED)) == 70

    def test_site_list_is_capped(self):
        meta = {"parent_sites": list(range(MAX_SITES + 50))}
        s = _sched(meta=meta)
        assert len(s._sites(SEED)) == MAX_SITES

    def test_other_seeds_meta_is_not_used(self):
        metas = {SEED: {"parent_sites": [40]}}
        s = PositionLineageScheduler(ScriptedRng(), meta_of=metas.get)
        assert s.propose(SEED, len(SEED)) == 40
        assert s.propose(b"different seed", 14) is None  # no meta for it


class TestDelocalisedFilter:
    def test_aligned_ops_drop_delocalised_sites(self):
        meta = {"parent_ops": ["byte_shuffle", "bitflip"], "parent_sites": [30, 120]}
        s = _sched(meta=meta)
        assert s._sites(SEED) == [120]
        assert s.propose(SEED, len(SEED)) == 120

    def test_only_delocalised_sites_declines(self):
        meta = {"parent_ops": ["byte_shuffle", "chunk_shuffle"], "parent_sites": [30, 120]}
        assert _sched(meta=meta).propose(SEED, len(SEED)) is None

    def test_misaligned_ops_keep_every_site(self):
        meta = {"parent_ops": ["byte_shuffle"], "parent_sites": [30, 120]}
        assert _sched(meta=meta)._sites(SEED) == [30, 120]

    def test_missing_ops_keep_every_site(self):
        assert _sched(meta={"parent_sites": [30, 120]})._sites(SEED) == [30, 120]

    def test_no_delocalised_set_keeps_every_site(self):
        meta = {"parent_ops": ["byte_shuffle", "bitflip"], "parent_sites": [30, 120]}
        assert _sched(meta=meta, delocalised=())._sites(SEED) == [30, 120]

    def test_trim_path_stays_aligned(self):
        # corpus_manager appends "trim" and len(data)//2 together.
        meta = {"parent_ops": ["bitflip", "trim"], "parent_sites": [15, 100]}
        assert _sched(meta=meta)._sites(SEED) == [15, 100]


class TestJitter:
    def test_magnitude_buckets(self):
        for k in (0, 1, 5, 17):
            s = _sched(
                ScriptedRng(randoms=[NO_ESCAPE, _magnitude_u(k), POSITIVE]),
                meta={"parent_sites": [100]},
            )
            assert s.propose(SEED, len(SEED)) == 100 + k

    def test_sign_below_half_is_negative(self):
        s = _sched(
            ScriptedRng(randoms=[NO_ESCAPE, _magnitude_u(6), NEGATIVE]),
            meta={"parent_sites": [100]},
        )
        assert s.propose(SEED, len(SEED)) == 94

    def test_u_zero_guard(self):
        # random() == 0.0 must not hit log(0); magnitude is bounded (~234).
        s = _sched(ScriptedRng(randoms=[NO_ESCAPE, 0.0, POSITIVE]), meta={"parent_sites": [0]})
        pos = s.propose(SEED, len(SEED))
        assert pos is not None and 0 <= pos < len(SEED)

    def test_mean_magnitude_is_jitter(self):
        rng = RandPool(seed=3)
        s = PositionLineageScheduler(rng, meta_of=lambda d: None)
        mags = [abs(s._jitter()) for _ in range(20000)]
        mean = sum(mags) / len(mags)
        assert JITTER * 0.9 < mean < JITTER * 1.1

    def test_jitter_is_two_sided(self):
        s = PositionLineageScheduler(RandPool(seed=4), meta_of=lambda d: None)
        draws = [s._jitter() for _ in range(2000)]
        assert min(draws) < 0 < max(draws)


class TestReflect:
    def test_in_range_is_identity(self):
        assert [reflect(i, 10) for i in range(10)] == list(range(10))

    def test_reflects_at_the_low_edge(self):
        assert reflect(-1, 10) == 1
        assert reflect(-4, 10) == 4

    def test_reflects_at_the_high_edge(self):
        assert reflect(10, 10) == 8  # n-1 + 1 -> n-1 - 1
        assert reflect(12, 10) == 6

    def test_far_positions_stay_in_range(self):
        for n in (2, 3, 10, 257):
            for pos in range(-3 * n, 4 * n):
                assert 0 <= reflect(pos, n) < n

    def test_tiny_buffers(self):
        assert reflect(99, 1) == 0
        assert reflect(-5, 1) == 0
        assert reflect(7, 2) in (0, 1)

    def test_propose_reflects_at_the_edge(self):
        # Site 3, jitter -5 -> -2 -> reflected to 2.
        s = _sched(
            ScriptedRng(randoms=[NO_ESCAPE, _magnitude_u(5), NEGATIVE]),
            meta={"parent_sites": [3]},
        )
        assert s.propose(SEED, len(SEED)) == 2

    def test_propose_is_clamped_to_a_shrunken_buffer(self):
        # Site 150 in a parent of 200, but the live buffer shrank to 20.
        s = _sched(meta={"parent_sites": [150]})
        for _ in range(50):
            pos = s.propose(SEED, 20)
            assert pos is not None and 0 <= pos < 20

    def test_real_rng_never_leaves_the_buffer(self):
        s = PositionLineageScheduler(
            RandPool(seed=9), meta_of=lambda d: {"parent_sites": [0, 5, 199, 5000]}
        )
        for n in (1, 2, 7, 200):
            for _ in range(500):
                pos = s.propose(SEED, n)
                assert pos is None or 0 <= pos < n
