"""record() cost: MAD from a delta histogram, classification of the touched field only.

Before: MAD sorted the whole unbounded delta history twice per record (per-record
cost grew with campaign length), and every record re-walked all fields, adding
the stride boost to every aligned field regardless of where the evidence landed.
"""

from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    TimelineEntry,
)

# Adversarial delta streams: n<3, all zero, one value, even n, negatives,
# zero-inflated heavy tail, duplicates around the median.
_STREAMS = (
    [5.0, 1.0],
    [0.0] * 9,
    [3.0] * 4,
    [0.0, 0.0, 7.0, 1.0],
    [-4.0, 0.0, 2.0, -1.0, 9.0, -4.0],
    [0.0] * 40 + [12.0, 30.0, 1.0],
    [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0, -10.0],
)


def _sorted_mad(buf):
    """Reference: the pre-histogram sort-based MAD."""
    if len(buf) < 3:
        return 0.0
    vals = sorted(buf)
    median = vals[len(vals) // 2]
    devs = sorted(abs(v - median) for v in vals)
    return devs[len(devs) // 2]


def _entry(offset, delta, op="bit_flip", edges=frozenset()):
    return TimelineEntry("h", op, offset, 1, 100, 100 + delta, set(edges), set())


def _fed(stream):
    c = FormatCluster(signature="ab")
    for d in stream:
        c._observe_delta(d)
    return c


def test_mad_matches_sorted_reference_with_control():
    for stream in _STREAMS:
        # Control: the reference agrees with itself on a reordered copy.
        assert _sorted_mad(stream) == _sorted_mad(list(reversed(stream)))
        assert _fed(stream)._median_absolute_deviation() == _sorted_mad(stream), stream


def test_mad_histogram_bounded_by_distinct_deltas():
    c = _fed([0.0, 3.0, 0.0, -2.0] * 5000)

    assert len(c._delta_counts) == 3
    assert c._median_absolute_deviation() == _sorted_mad([0.0, 3.0, 0.0, -2.0] * 5000)


def test_record_feeds_histogram_and_moments_together():
    c = FormatCluster(signature="ab")
    for d in (0, 0, 5, 0, 1):
        c.record(_entry(3, d), bytes(8), 100)

    assert sum(c._delta_counts.values()) == c._delta_moments.count


def test_regression_stride_boost_only_on_evidence():
    c = FormatCluster(signature="ab")
    c.set_record_stride(8)
    aligned = FieldHypothesis(
        offset=16, width=8, field_type="unknown", confidence=0.4, observations=3
    )
    c.hypotheses.append(aligned)

    # Records at an unrelated offset carry no evidence about `aligned`.
    for _ in range(5):
        c.record(_entry(40, 0), bytes(64), 100)

    assert aligned.confidence == 0.4


def test_touched_field_boosted_once_per_record():
    c = FormatCluster(signature="ab")
    c.set_record_stride(8)
    aligned = FieldHypothesis(
        offset=16, width=8, field_type="unknown", confidence=0.4, observations=3
    )
    c.hypotheses.append(aligned)

    # Zero delta at n<3: no effect -> -0.02, then the stride boost +0.05.
    c.record(_entry(17, 0), bytes(64), 100)

    assert abs(aligned.confidence - (0.4 - 0.02 + 0.05)) < 1e-12


def test_liveness_retypes_without_boost():
    c = FormatCluster(signature="ab")
    c.set_record_stride(8)
    h = FieldHypothesis(offset=0, width=8, field_type="magic", confidence=0.52, observations=3)
    c.hypotheses.append(h)

    c.record_liveness(0, 8, confirmed_dead=True)

    # 0.52 - 0.05 drops below the magic gate; no +0.05 stride boost undoes it.
    assert abs(h.confidence - 0.47) < 1e-12
    assert h.field_type == "unknown"


def test_full_pass_still_available():
    c = FormatCluster(signature="ab")
    c.set_record_stride(8)
    c.hypotheses.append(
        FieldHypothesis(offset=0, width=8, field_type="unknown", confidence=0.6, observations=3)
    )

    c._classify_fields()

    assert c.hypotheses[0].field_type == "magic"
    assert abs(c.hypotheses[0].confidence - 0.65) < 1e-12


def test_regression_timeline_trim_in_place():
    c = FormatCluster(signature="ab")
    timeline = c.timeline
    entries = [_entry(i, 0) for i in range(7)]
    for e in entries:
        c.record(e, bytes(8), 4)

    # Same list object (no per-record copy), holding the last 4 entries in order.
    assert c.timeline is timeline
    assert c.timeline == entries[-4:]
