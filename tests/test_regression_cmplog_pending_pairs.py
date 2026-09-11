"""Regression: the redqueen input-to-state scan died once the pair pool filled.

``fuzz_one`` tracked how far the scan had got with ``_redqueen_index``, a
high-water mark into ``CmplogCollector.pairs``, and scanned the slice
``pairs[_redqueen_index:]``. Two things make that index meaningless:

1. The eviction site rebuilds the list wholesale (``self.pairs =
   list(self._pair_set)``), so after the first eviction the entries below the
   index are an arbitrary subset -- not the ones already scanned. Genuinely
   new pairs land below it and are skipped forever.
2. Eviction pins ``len(pairs)`` at ``_max_pairs``. Once the index catches up
   to that value the guard ``_redqueen_index < len(pairs)`` is false on every
   subsequent iteration, so the scan stops for the rest of the run -- exactly
   when the campaign reaches the deep comparisons only cmplog can solve.

Replaced with an explicit queue on the collector: ``pending_new_pairs()`` /
``consume_new_pairs(n)``. The split lets a scan that bails at its match cap
leave the unscanned tail queued instead of discarding it, which the index
also got wrong (it was set to ``len(pairs)`` whether the loop finished or
broke).
"""

from fuzzer_tool.core.cmplog import CmplogCollector


def _lines(pairs):
    return [f"CMP {a.hex()} {b.hex()} 0 {len(a)}" for a, b in pairs]


def _pairs(prefix, n, start=0):
    return [(b"%s%05d" % (prefix, i), b"x%05d" % i) for i in range(start, start + n)]


class TestPendingQueueSurvivesSaturation:
    def test_new_pairs_are_offered_after_the_pool_saturates(self):
        c = CmplogCollector(max_pairs=50, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 50)))
        c.consume_new_pairs(len(c.pending_new_pairs()))
        assert c.pending_new_pairs() == []

        # The old guard: index == len(pairs) == max_pairs, so the scan
        # would never run again from here on.
        stale_index = len(c.pairs)
        c._parse_lines(_lines(_pairs(b"R", 10)))
        assert stale_index >= len(c.pairs), "pool is expected to be pinned at the cap"

        pending = c.pending_new_pairs()
        # The old index offered nothing here, ever.
        assert len(pending) == 10
        assert all(a.startswith(b"R") for a, _ in pending)

    def test_offers_stay_available_across_many_saturated_drains(self):
        c = CmplogCollector(max_pairs=50, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 50)))
        c.consume_new_pairs(len(c.pending_new_pairs()))
        for batch in range(20):
            c._parse_lines(_lines(_pairs(b"R%02d" % batch, 5)))
            assert len(c.pending_new_pairs()) == 5, f"drain {batch}"
            c.consume_new_pairs(5)

    def test_pool_order_scramble_does_not_hide_new_pairs(self):
        """The defect the index had even before the length pinned."""
        c = CmplogCollector(max_pairs=50, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 50)))
        c.consume_new_pairs(len(c.pending_new_pairs()))
        before = set(c.pairs)
        c._parse_lines(_lines(_pairs(b"R", 10)))
        genuinely_new = set(c.pairs) - before
        assert set(c.pending_new_pairs()) == genuinely_new


class TestTheOldIndexMechanicsGoDead:
    """The defect, reproduced against the live collector as an oracle.

    There is no way to falsify this by reverting the source -- the old
    bookkeeping lived in ``fuzz_one`` and the new one is an API on the
    collector, so a revert changes which object is asked, not what the
    answer should be. So the old index is reimplemented here verbatim and
    run side by side with the queue over the same drain sequence.
    """

    def test_index_stops_offering_while_the_queue_does_not(self):
        c = CmplogCollector(max_pairs=50, max_tokens=10_000)
        redqueen_index = 0  # fuzz_one's old high-water mark

        index_offers = []
        queue_offers = []
        for batch in range(12):
            c._parse_lines(_lines(_pairs(b"B%02d" % batch, 10)))

            # Old: scan pairs[index:], then set index = len(pairs).
            offered = c.pairs[redqueen_index:] if redqueen_index < len(c.pairs) else []
            index_offers.append(len(offered))
            if redqueen_index < len(c.pairs):
                redqueen_index = len(c.pairs)

            # New: scan the queue, then retire what was scanned.
            pending = c.pending_new_pairs()
            queue_offers.append(len(pending))
            c.consume_new_pairs(len(pending))

        # The pool saturates on the sixth drain (50 / 10 per batch) and the
        # index reaches _max_pairs there; from then on it offers nothing.
        assert index_offers[-1] == 0
        assert sum(index_offers[6:]) == 0, index_offers
        # And the queue keeps offering the frontier for the whole run.
        assert all(n == 10 for n in queue_offers), queue_offers

    def test_index_skipped_new_pairs_even_before_it_pinned(self):
        """Rebuilding the list puts new pairs below a still-moving index."""
        c = CmplogCollector(max_pairs=50, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 45)))
        redqueen_index = len(c.pairs)
        c.consume_new_pairs(len(c.pending_new_pairs()))

        before = set(c.pairs)
        c._parse_lines(_lines(_pairs(b"R", 10)))
        genuinely_new = set(c.pairs) - before
        offered_by_index = set(c.pairs[redqueen_index:])

        assert genuinely_new - offered_by_index, "index missed no new pair"
        assert set(c.pending_new_pairs()) == genuinely_new


class TestPartialConsumption:
    def test_unscanned_tail_stays_queued(self):
        c = CmplogCollector(max_pairs=500, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 40)))
        assert len(c.pending_new_pairs()) == 40
        c.consume_new_pairs(15)
        rest = c.pending_new_pairs()
        assert len(rest) == 25
        assert rest[0] == (b"P00015", b"x00015")

    def test_consume_zero_is_a_noop(self):
        c = CmplogCollector(max_pairs=500, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 4)))
        c.consume_new_pairs(0)
        assert len(c.pending_new_pairs()) == 4

    def test_returned_list_is_a_copy(self):
        c = CmplogCollector(max_pairs=500, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 4)))
        got = c.pending_new_pairs()
        got.clear()
        assert len(c.pending_new_pairs()) == 4


class TestPendingQueueStaysConsistentWithThePool:
    def test_evicted_pairs_are_dropped_from_the_queue(self):
        """A queued pair the collector no longer holds is a dead offer."""
        c = CmplogCollector(max_pairs=20, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 20)))
        c._parse_lines(_lines(_pairs(b"R", 10)))
        assert c.evicted_pair_count > 0
        for pair in c.pending_new_pairs():
            assert pair in c._pair_set

    def test_queue_is_bounded_when_nobody_consumes(self):
        c = CmplogCollector(max_pairs=64, max_tokens=10_000)
        for batch in range(30):
            c._parse_lines(_lines(_pairs(b"Q%02d" % batch, 20)))
            assert len(c.pending_new_pairs()) <= 64
