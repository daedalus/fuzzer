"""Regression: the checksum learner must carry its evidence across a resume.

``to_dict`` wrote a ``pair_count`` key that ``from_dict`` never read -- and
could not read, since it is a derived property over ``_pairs``, which was not
serialized at all. Recovery needs ``min_pairs`` (64) pairs before it runs, so
every resume threw away the accumulated evidence and started collection from
zero: a run stopped and resumed regularly could never reach the threshold no
matter how many pairs it had seen.
"""

import types

import pytest

from fuzzer_tool.core.checksum_learner import (
    CHECKSUM_PAIRS_MAX,
    CHECKSUM_STATE_BYTES_MAX,
    ChecksumLearner,
)


def _fuzzer():
    return types.SimpleNamespace(corpus=[], seed_meta={})


def _learner_with(n_pairs: int, data_size: int = 16) -> ChecksumLearner:
    learner = ChecksumLearner(_fuzzer())
    learner.add_pairs([(bytes([i % 251]) * data_size, i) for i in range(n_pairs)])
    return learner


def _round_trip(learner: ChecksumLearner) -> ChecksumLearner:
    return ChecksumLearner.from_dict(_fuzzer(), learner.to_dict())


class TestPairsSurviveTheRoundTrip:
    def test_pairs_are_restored(self):
        original = _learner_with(10)
        restored = _round_trip(original)
        assert restored._pairs == original._pairs

    def test_pair_count_matches_after_restore(self):
        original = _learner_with(10)
        assert _round_trip(original).pair_count == original.pair_count

    def test_threshold_state_survives(self):
        """The case the loss actually cost: a run just short of min_pairs
        used to resume at zero."""
        original = _learner_with(70)
        assert original.has_enough_pairs()
        assert _round_trip(original).has_enough_pairs()

    def test_counters_are_restored(self):
        original = _learner_with(20)
        original._pairs_attempted_at = 7
        restored = _round_trip(original)
        assert restored._total_pairs_seen == original._total_pairs_seen
        assert restored._pairs_attempted_at == 7

    def test_empty_learner_round_trips(self):
        restored = _round_trip(ChecksumLearner(_fuzzer()))
        assert restored._pairs == []
        assert restored._pairs_attempted_at == -1


class TestStateSizeIsBounded:
    def test_large_pairs_are_dropped_to_fit_the_budget(self):
        """A pair's data half is a whole checksummed region; the count cap
        alone does not bound the serialized size."""
        learner = _learner_with(CHECKSUM_PAIRS_MAX, data_size=64 * 1024)
        serialized = learner.to_dict()["pairs"]
        total = sum(len(d) + 8 for d, _ in serialized)
        assert total <= CHECKSUM_STATE_BYTES_MAX
        assert 0 < len(serialized) < CHECKSUM_PAIRS_MAX

    def test_newest_pairs_are_the_ones_kept(self):
        learner = _learner_with(CHECKSUM_PAIRS_MAX, data_size=64 * 1024)
        kept = learner.to_dict()["pairs"]
        assert kept[-1] == learner._pairs[-1]

    def test_small_pairs_all_fit(self):
        learner = _learner_with(50, data_size=16)
        assert len(learner.to_dict()["pairs"]) == 50


class TestCorruptStateIsSurvivable:
    @pytest.mark.parametrize(
        "pairs",
        [
            "not-a-list",
            [None],
            [(b"ok", "not-an-int")],
            [("not-bytes", 1)],
            [(b"ok",)],
            [42],
        ],
    )
    def test_malformed_pairs_do_not_raise(self, pairs):
        learner = ChecksumLearner.from_dict(_fuzzer(), {"pairs": pairs})
        assert all(isinstance(d, bytes) and isinstance(c, int) for d, c in learner._pairs)

    def test_good_pairs_survive_a_bad_neighbour(self):
        learner = ChecksumLearner.from_dict(
            _fuzzer(), {"pairs": [(b"good", 1), ("bad", 2), (b"also-good", 3)]}
        )
        assert learner._pairs == [(b"good", 1), (b"also-good", 3)]

    def test_missing_counters_fall_back_to_the_pair_count(self):
        learner = ChecksumLearner.from_dict(_fuzzer(), {"pairs": [(b"a", 1), (b"b", 2)]})
        assert learner._total_pairs_seen == 2
        assert learner._pairs_attempted_at == -1

    def test_restore_respects_the_pair_cap(self):
        over = [(b"x", i) for i in range(CHECKSUM_PAIRS_MAX * 2)]
        learner = ChecksumLearner.from_dict(_fuzzer(), {"pairs": over})
        assert len(learner._pairs) == CHECKSUM_PAIRS_MAX

    def test_state_without_the_pairs_key_loads_as_empty(self):
        """Backwards compatibility: a state file written before this change
        has poly/int_model but no pairs."""
        learner = ChecksumLearner.from_dict(_fuzzer(), {"poly": None, "pair_count": 12})
        assert learner._pairs == []
