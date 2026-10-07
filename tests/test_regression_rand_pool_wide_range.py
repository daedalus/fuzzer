"""Ranges wider than one pool word must reach their whole range.

The pool holds uint32 words and ``randint``/``randrange`` returned
``word % width``, so any width above 2**32 never produced a value at or above
2**32: ``randint(0, 0xFFFFFFFFFFFFFFFF)`` only ever filled the low half of a
64-bit value. ``ogg._mutate_granule_position`` draws exactly that, so the
64-bit granule position never had a high word to test overflow with.

Widths up to 2**32 must keep their old single-draw behaviour so existing
seeded runs do not move.
"""

import random

import pytest

from fuzzer_tool.core.mutations.ogg import OggMutator, OggPage
from fuzzer_tool.core.rand_pool import RandPool

U32 = 1 << 32
U64 = 1 << 64


def _page(granule: int = 0) -> OggPage:
    return OggPage(0, 0, granule, 1, 0, 0, b"\x00", b"\x00")


class TestWideRandint:
    def test_full_u64_reaches_high_word(self):
        p = RandPool(seed=1)
        vals = [p.randint(0, U64 - 1) for _ in range(2000)]
        assert max(vals) >= U32
        assert all(0 <= v < U64 for v in vals)

    def test_high_word_is_about_uniform(self):
        p = RandPool(seed=2)
        vals = [p.randint(0, U64 - 1) for _ in range(4000)]
        # a uniform 64-bit value is >= 2**32 with probability 1 - 2**-32
        assert sum(v >= U32 for v in vals) >= 3990
        top = sum(v >= U64 // 2 for v in vals) / len(vals)
        assert 0.45 < top < 0.55

    def test_u64_uses_exactly_two_draws(self):
        p = RandPool(seed=3)
        p._refill()
        before = p._idx
        p.randint(0, U64 - 1)
        assert p._idx - before == 2

    def test_u64_composes_two_pool_words(self):
        p = RandPool(seed=4)
        q = RandPool(seed=4)
        hi, lo = q._draw(), q._draw()
        assert p.randint(0, U64 - 1) == (hi << 32) | lo

    @pytest.mark.parametrize("width", [U32 * 3, (1 << 40) + 7, (1 << 63) + 5, U64 - 3])
    def test_stays_inside_range(self, width):
        p = RandPool(seed=5)
        vals = [p.randint(0, width - 1) for _ in range(3000)]
        assert all(0 <= v < width for v in vals)
        assert max(vals) >= U32

    def test_just_above_one_word_stays_inside_range(self):
        # P(value >= 2**32) is 2**-32 here, so only the bound is checkable
        p = RandPool(seed=5)
        assert all(0 <= p.randint(0, U32) <= U32 for _ in range(3000))

    def test_offset_lower_bound(self):
        p = RandPool(seed=6)
        a, b = 1 << 40, (1 << 40) + (1 << 36)
        vals = [p.randint(a, b) for _ in range(2000)]
        assert min(vals) >= a
        assert max(vals) <= b
        assert max(vals) - a >= U32

    def test_wider_than_64_bits(self):
        p = RandPool(seed=7)
        width = 1 << 100
        vals = [p.randint(0, width - 1) for _ in range(500)]
        assert all(0 <= v < width for v in vals)
        assert max(vals) >= U64

    def test_rejection_is_unbiased_just_above_a_power_of_two(self):
        # width 2**33 + 1 needs 34 bits; plain `% width` over 64 bits would be
        # fine here, but the rejection loop must still cover the last value
        p = RandPool(seed=8)
        width = (1 << 33) + 1
        assert all(0 <= p.randint(0, width - 1) < width for _ in range(2000))

    def test_deterministic_for_a_seed(self):
        a = [RandPool(seed=9).randint(0, U64 - 1) for _ in range(1)]
        b = [RandPool(seed=9).randint(0, U64 - 1) for _ in range(1)]
        assert a == b
        p, q = RandPool(seed=9), RandPool(seed=9)
        assert [p.randint(0, (1 << 50) - 1) for _ in range(50)] == [
            q.randint(0, (1 << 50) - 1) for _ in range(50)
        ]


class TestWideRandrange:
    def test_randrange_wide_reaches_high_word(self):
        p = RandPool(seed=10)
        vals = [p.randrange(1 << 40) for _ in range(2000)]
        assert max(vals) >= U32
        assert all(0 <= v < (1 << 40) for v in vals)

    def test_randrange_two_pow_32_unchanged(self):
        p, q = RandPool(seed=11), RandPool(seed=11)
        assert [p.randrange(U32) for _ in range(20)] == [q._draw() for _ in range(20)]


class TestNarrowPathUnchanged:
    """Widths <= 2**32 keep the exact single-draw `% width` mapping."""

    @pytest.mark.parametrize("width", [1, 2, 3, 10, 255, 256, 257, 1000, 1 << 20, U32])
    def test_randint_matches_single_draw_mod(self, width):
        p, q = RandPool(seed=12), RandPool(seed=12)
        for _ in range(100):
            assert p.randint(0, width - 1) == q._draw() % width

    def test_randint_consumes_one_draw(self):
        p = RandPool(seed=13)
        p._refill()
        before = p._idx
        p.randint(0, U32 - 1)
        assert p._idx - before == 1

    def test_randrange_nonpositive_still_zero(self):
        p = RandPool(seed=14)
        assert p.randrange(0) == 0
        assert p.randrange(-5) == 0

    def test_randint_empty_width_still_returns_a(self):
        assert RandPool(seed=15).randint(5, 4) == 5


class TestBatchAndSampleWide:
    def test_randint_list_wide(self):
        p = RandPool(seed=16)
        vals = p.randint_list(0, U64 - 1, 500)
        assert len(vals) == 500
        assert max(vals) >= U32
        assert all(0 <= v < U64 for v in vals)

    def test_randrange_list_wide(self):
        p = RandPool(seed=17)
        vals = p.randrange_list(1 << 40, 500)
        assert len(vals) == 500
        assert max(vals) >= U32
        assert all(0 <= v < (1 << 40) for v in vals)

    def test_randint_list_narrow_unchanged(self):
        p, q = RandPool(seed=18), RandPool(seed=18)
        got = p.randint_list(0, 999, 50)
        assert got == [q._draw() % 1000 for _ in range(50)]

    def test_sample_int_population_wide(self):
        p = RandPool(seed=19)
        pop = 1 << 40
        hits = [p.sample(pop, 1)[0] for _ in range(1000)]
        assert max(hits) >= U32
        pairs = [p.sample(pop, 2) for _ in range(1000)]
        assert max(max(x) for x in pairs) >= U32
        assert all(a != b for a, b in pairs)

    def test_sample_int_population_narrow_unchanged(self):
        p, q = RandPool(seed=20), RandPool(seed=20)
        assert p.sample(1000, 1) == [q._draw() % 1000]
        a, b = q._draw() % 1000, q._draw() % 999
        assert p.sample(1000, 2) == [a, b if b < a else b + 1]


class TestFloydWide:
    @pytest.fixture(autouse=True)
    def _floyd_on(self):
        from fuzzer_tool.core import rand_pool

        old = rand_pool._RAND_FLOYD
        rand_pool.configure_rand_floyd(True)
        yield
        rand_pool.configure_rand_floyd(old)

    def test_floyd_wide_population_reaches_high_word(self):
        p = RandPool(seed=21)
        pop = 1 << 40
        got = [x for _ in range(200) for x in p.sample(pop, 5)]
        assert max(got) >= U32
        assert all(0 <= x < pop for x in got)

    def test_floyd_wide_population_is_unique(self):
        p = RandPool(seed=22)
        for _ in range(50):
            s = p.sample(1 << 40, 6)
            assert len(set(s)) == 6

    def test_floyd_narrow_population_unchanged(self):
        p, q = RandPool(seed=23), RandPool(seed=23)
        n, k = 1000, 5
        expect: set[int] = set()
        for j in range(n - k, n):
            t = q._draw() % (j + 1)
            expect.add(j if t in expect else t)
        assert set(p.sample(n, k)) == expect


class TestOggGranule:
    def test_granule_position_reaches_high_word_on_randpool(self):
        m = OggMutator()
        seen_high = False
        for seed in range(60):
            pages = [_page()]
            m._mutate_granule_position(pages, RandPool(seed=seed))
            g = pages[0].granule_position
            assert 0 <= g <= U64 - 1
            seen_high = seen_high or g >= U32
        assert seen_high

    def test_granule_position_with_stdlib_random_unchanged(self):
        m = OggMutator()
        pages = [_page()]
        m._mutate_granule_position(pages, random.Random(1))
        assert 0 <= pages[0].granule_position <= U64 - 1
