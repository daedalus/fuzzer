"""Regression: ``compute_delta_v2`` aligned inputs it was bound to reject.

The 512-byte gate in ``save_to_corpus`` bounds the child, not the parent:
a 4 KB parent with a 300-byte child ran a 4096x300 Levenshtein (45 ms per
admission on ffmpeg) to learn the script exceeds ``len(child) // 2``. Edit
distance is at least the length difference, so that case is decided up front.
"""

import time
from unittest import mock

from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2


def test_regression_delta_v2_skips_alignment_when_length_gap_too_wide():
    parent = bytes(range(256)) * 16
    child = parent[:300]
    with mock.patch("fuzzer_tool.core.similarity.levenshtein_align") as align:
        assert compute_delta_v2(parent, child) is None
    align.assert_not_called()


def test_wide_gap_was_none_before_too():
    """Falsification: the full alignment agrees the gap case is None."""
    parent = bytes(range(256)) * 2
    child = parent[:100]
    from fuzzer_tool.core.similarity import levenshtein_align

    ops = [op for op in levenshtein_align(parent, child) if op[0] != "match"]
    assert len(ops) > len(child) // 2
    assert compute_delta_v2(parent, child) is None


def test_gap_at_threshold_still_aligns():
    """Adversarial: a gap exactly at len(child)//2 may still encode."""
    child = bytes(range(200))
    parent = child + bytes(len(child) // 2)
    diff = compute_delta_v2(parent, child)
    assert diff is not None
    assert apply_delta_v2(parent, diff) == child


def test_empty_parent_still_pure_insertion():
    child = b"abc"
    diff = compute_delta_v2(b"", child)
    assert diff is not None
    assert apply_delta_v2(b"", diff) == child


def test_wide_gap_is_fast():
    parent = bytes(4096)
    child = b"\x01" * 512
    t = time.perf_counter()
    compute_delta_v2(parent, child)
    assert time.perf_counter() - t < 0.005
