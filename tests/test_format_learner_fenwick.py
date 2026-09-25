"""Fenwick coverage index over a FormatCluster's field hypotheses.

Each field [offset, offset+width) is one range-add edge; index 0 is the
null-frame root (depth 0 = no field covers the byte). The index only
filters: a zero depth skips the linear hypothesis scan.
"""

from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    TimelineEntry,
    _CoverageFenwick,
)

# (offset, width) spans: overlaps, adjacency, a 1-byte field, a span past the
# initial capacity (forces growth), and a zero width that covers nothing.
_SPANS = ((0, 4), (2, 3), (4, 1), (9, 2), (11, 1), (300, 8), (20, 0))


def _brute_depth(spans, pos):
    return sum(1 for o, w in spans if o <= pos < o + w)


def _scan(hyps, pos):
    for h in hyps:
        if h.offset <= pos < h.offset + h.width:
            return h
    return None


def _cluster(spans):
    c = FormatCluster(signature="ab")
    for o, w in spans:
        c.hypotheses.append(FieldHypothesis(offset=o, width=w, field_type="unknown"))
    return c


def _entry(offset, width, effect):
    after = 101 if effect else 100
    return TimelineEntry("h", "bit_flip", offset, width, 100, after, set(), set())


def test_depth_matches_brute_force():
    fw = _CoverageFenwick()
    for o, w in _SPANS:
        fw.add(o, w)

    for pos in range(-2, 320):
        assert fw.depth(pos) == _brute_depth(_SPANS, pos), pos


def test_covering_matches_scan_with_control():
    c = _cluster(_SPANS)
    positions = range(-2, 320)

    # Control: the reference agrees with itself, so a mismatch below is the index.
    assert all(_scan(c.hypotheses, p) is _scan(c.hypotheses, p) for p in positions)
    for p in positions:
        assert c._covering(p) is _scan(c.hypotheses, p), p


def test_falsify_null_frame_is_empty():
    fw = _CoverageFenwick()

    assert all(fw.depth(p) == 0 for p in range(-1, 70))


def test_adversarial_external_mutation_rebuilds():
    c = _cluster(((0, 2),))
    assert c._covering(50) is None

    # Direct append bypasses the index; lookup must still see it.
    late = FieldHypothesis(offset=50, width=4, field_type="unknown")
    c.hypotheses.append(late)
    assert c._covering(52) is late

    # Wholesale reassignment: old spans must vanish from the index.
    fresh = FieldHypothesis(offset=7, width=1, field_type="unknown")
    c.hypotheses = [fresh]
    assert c._covering(0) is None
    assert c._covering(7) is fresh


def test_adversarial_negative_offset_falls_back_to_scan():
    c = _cluster(((-3, 5),))

    assert c._covering(-1) is c.hypotheses[0]
    assert c._covering(1) is c.hypotheses[0]
    assert c._covering(2) is None


def test_record_path_keeps_first_in_list_order():
    c = FormatCluster(signature="ab")
    c.record(_entry(10, 4, effect=True), bytes(64), 100)
    first = c.hypotheses[0]

    # Overlapping later field: lookups inside the overlap still return the first.
    c.hypotheses.append(FieldHypothesis(offset=12, width=8, field_type="unknown"))
    c.record(_entry(12, 1, effect=True), bytes(64), 100)

    assert first.observations == 2
    assert c.hypotheses[1].observations == 0


def test_from_state_roundtrip_indexes_fields():
    c = _cluster(((5, 3), (40, 2)))
    restored = FormatCluster.from_state("ab", c.get_state())

    assert restored._covering(6).offset == 5
    assert restored._covering(41).offset == 40
    assert restored._covering(20) is None


def test_dense_frame_disables_filter_but_stays_exact():
    fw = _CoverageFenwick()
    fw.add(10, 2)
    assert fw.sparse  # mass 2 over extent 12

    fw.add(0, 11)
    assert not fw.sparse  # mass 13 over extent 12

    dense = ((0, 4), (2, 4))
    c = _cluster(dense)
    for p in range(-1, 10):
        assert c._covering(p) is _scan(c.hypotheses, p), p
