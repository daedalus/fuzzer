"""The comparison-progress channel: reward for getting further, not only through.

``magic_byte_search``, ``climb_hill``, ``gradient_descent`` and
``condstmt_solve`` all optimize the same objective -- make a window of the
input equal a cmplog operand -- and until this channel existed their only
reward was the edge that arrives when the comparison is fully solved and
the branch flips. Seven of eight correct bytes paid nothing, so those arms
starved in the bandits for a reason unrelated to their quality.

An input that satisfies more comparisons of some family, in one execution,
than any input before it got further into the parser. That is a real
discovery event and edge coverage misses it entirely during a magic-bytes
plateau -- precisely the stretch where a campaign looks stalled and isn't.

The shape is deliberately the per-edge maximum count one channel over, so
the growth-factor discipline is the same: see
``ShmCoverage._update_max_counts``.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.adapters.shm import MAX_COUNT_GROWTH_FACTOR
from fuzzer_tool.services.fuzzer import Fuzzer


@pytest.fixture
def f() -> Fuzzer:
    """Just the channel's own state.

    The method reads and writes two attributes and nothing else; building
    a real Fuzzer would need a built target and would test the constructor.
    """
    obj = Fuzzer.__new__(Fuzzer)
    obj._cmp_max_asserted = {}
    obj._cmp_novelty_hits = 0
    return obj


def test_first_assert_for_a_callback_reports(f):
    """Not covered by any other signal, and bounded at twenty-seven events."""
    assert f._record_cmp_progress({"memcmp": 1}) is True
    assert f._cmp_max_asserted == {"memcmp": 1}
    assert f._cmp_novelty_hits == 1


def test_fired_without_asserted_is_not_progress(f):
    """A comparison reached and failed is the status quo, not a discovery."""
    assert f._record_cmp_progress({"memcmp": 0}) is False
    assert f._cmp_novelty_hits == 0


def test_matching_the_high_water_mark_is_not_progress(f):
    f._record_cmp_progress({"memcmp": 8})
    assert f._record_cmp_progress({"memcmp": 8}) is False
    assert f._record_cmp_progress({"memcmp": 3}) is False


def test_growth_below_the_factor_moves_the_mark_without_reporting(f):
    """A +1 climb must not report once per step.

    Each report admits an input to the corpus, so an operator that adds one
    satisfied comparison per mutation would otherwise admit on every
    mutation for as long as the climb lasts.
    """
    f._record_cmp_progress({"memcmp": 10})
    assert f._cmp_novelty_hits == 1
    for count in (11, 12, 13, 14):
        assert f._record_cmp_progress({"memcmp": count}) is False
    assert f._cmp_max_asserted["memcmp"] == 14, "the mark still tracks the climb"
    assert f._cmp_novelty_hits == 1
    # 14 * 1.5 = 21
    assert f._record_cmp_progress({"memcmp": 21}) is True


def test_report_count_for_one_callback_is_logarithmically_bounded(f):
    """The reason the threshold is multiplicative rather than strict `>`."""
    reports = sum(1 for n in range(1, 100_001) if f._record_cmp_progress({"memcmp": n}))
    assert reports == f._cmp_novelty_hits
    # log_1.5(100000) < 30, with slack for the +1 floor at small values.
    assert reports < 40, reports
    assert f._cmp_max_asserted["memcmp"] == 100_000


def test_callbacks_are_tracked_independently(f):
    f._record_cmp_progress({"memcmp": 4})
    f._cmp_novelty_hits = 0
    assert f._record_cmp_progress({"memcmp": 4, "strcmp": 1}) is True
    assert f._cmp_max_asserted == {"memcmp": 4, "strcmp": 1}
    assert f._cmp_novelty_hits == 1, "one report per execution, not per callback"


def test_empty_vector_is_inert(f):
    """cmplog off means an empty dict, which is the whole off switch."""
    assert f._record_cmp_progress({}) is False
    assert f._cmp_max_asserted == {}


def test_threshold_matches_the_shared_growth_factor(f):
    """Pinned against the constant rather than a copy of 1.5."""
    f._record_cmp_progress({"memcmp": 100})
    just_under = int(100 * MAX_COUNT_GROWTH_FACTOR) - 1
    assert f._record_cmp_progress({"memcmp": just_under}) is False
    assert f._record_cmp_progress({"memcmp": int(just_under * MAX_COUNT_GROWTH_FACTOR)}) is True
