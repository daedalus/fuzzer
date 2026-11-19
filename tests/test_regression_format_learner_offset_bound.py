"""Regression: a huge mutation offset/width made the format learner allocate
proportionally to it (MemoryError at offset 1<<40).

Found by tools/fuzz.py (`learner:huge_offset`). A field hypothesis indexes
`[offset, offset+width)` in the coverage Fenwick tree, which grows to the
span's end. Bytes past the input do not exist, so they are no evidence:
spans are clipped to the input and offsets past it are ignored.
"""

from __future__ import annotations

from fuzzer_tool.core.analyzers.analyzer_format_learner import DEFAULT_SIGNATURE, FormatLearner
from fuzzer_tool.core.rand_pool import RandPool

DATA = b"ABCDEFGH"
HUGE = 1 << 40


def _record(fl: FormatLearner, offset: int, width: int) -> None:
    # Coverage moves and a new edge appears: an effect, so a field is proposed.
    fl.record_transition(DATA, "bit_flip", offset, width, 10, 12, {7}, set())


def _cluster(fl: FormatLearner):
    return fl.clusters[DEFAULT_SIGNATURE]


def test_regression_offset_past_input_ignored():
    fl = FormatLearner(rng=RandPool(1))
    _record(fl, HUGE, 1)

    assert _cluster(fl).hypotheses == []
    assert _cluster(fl)._cover._n <= 2 * max(len(DATA), 64)


def test_regression_width_clipped_to_input():
    fl = FormatLearner(rng=RandPool(1))
    _record(fl, 2, HUGE)

    (h,) = _cluster(fl).hypotheses
    assert h.offset == 2
    assert h.width == len(DATA) - 2


def test_falsify_in_bounds_field_still_learned():
    fl = FormatLearner(rng=RandPool(1))
    _record(fl, 3, 2)

    (h,) = _cluster(fl).hypotheses
    assert (h.offset, h.width) == (3, 2)


def test_adversarial_last_byte_and_one_past():
    fl = FormatLearner(rng=RandPool(1))
    _record(fl, len(DATA) - 1, 4)
    _record(fl, len(DATA), 1)

    spans = [(h.offset, h.width) for h in _cluster(fl).hypotheses]
    assert spans == [(len(DATA) - 1, 1)]
