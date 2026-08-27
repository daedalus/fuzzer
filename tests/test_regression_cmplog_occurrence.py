"""Regression: cmplog pair occurrence counts must accumulate across batches.

``_pair_occurrence`` was incremented from ``new_pairs``, which by construction
holds only pairs absent from ``_pair_set`` -- and each of those is inserted
into ``_pair_set`` in the same iteration. A pair could therefore never be
"new" a second time, so every count stayed pinned at 1 for the life of the
run: ``pair_confidence()`` degenerated into a membership test and
``high_confidence_pairs(min_occurrences=2)`` returned an empty list always.

The existing unit tests seeded ``_pair_occurrence`` by hand and only exercised
the accessors, so the counting path itself was never covered.
"""

import os

from fuzzer_tool.core.cmplog import CmplogCollector

PAIR = (b"ABCD", b"EFGH")
LINE = "CMP 41424344 45464748 -1 4 0xdeadbeef\n"
OTHER = "CMP 11223344 55667788 1 4 0xcafe\n"


def _collector(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c.log_path = os.path.join(str(tmp_path), "cmplog.txt")
    return c


def _write(collector, *lines):
    """Fill the log the way the shim would, then let the collector drain it."""
    with open(collector.log_path, "w") as f:
        f.writelines(lines)
    collector.collect_tokens()


class TestPairOccurrenceAccumulates:
    def test_repeat_sightings_raise_confidence(self, tmp_path):
        c = _collector(tmp_path)
        for expected in (1, 2, 3):
            _write(c, LINE)
            assert c.pair_confidence(*PAIR) == expected

    def test_high_confidence_pairs_reachable(self, tmp_path):
        c = _collector(tmp_path)
        _write(c, LINE, OTHER)
        _write(c, LINE)
        # LINE seen twice, OTHER once.
        assert c.high_confidence_pairs(min_occurrences=2) == [PAIR]

    def test_first_sighting_still_counts_once(self, tmp_path):
        c = _collector(tmp_path)
        _write(c, LINE)
        assert c.pair_confidence(*PAIR) == 1

    def test_one_increment_per_batch_not_per_line(self, tmp_path):
        """The unit is "a run exercised this comparison", not "it fired N times".

        Raw fire multiplicity is the shim counters' job; the parser dedups
        within a batch, so a burst of identical records must not be able to
        inflate confidence past one.
        """
        c = _collector(tmp_path)
        _write(c, *([LINE] * 50))
        assert c.pair_confidence(*PAIR) == 1

    def test_unseen_pair_stays_zero(self, tmp_path):
        c = _collector(tmp_path)
        _write(c, LINE)
        assert c.pair_confidence(b"ZZZZ", b"YYYY") == 0
