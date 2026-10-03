"""Regression: region-liveness diff folding was quadratic in the diff size.

``record_coverage_diff`` built its mask with ``bits |= 1 << (e % map_size)``
per edge, reallocating a 65536-bit int each time: 42 ms per call on ffmpeg
(860 calls, 36 s of a 574 s profile). ``_fold_bits`` folds in one pass.
"""

from fuzzer_tool.services.operators import _LIVENESS_MAP_BITS, _fold_bits


def _reference(edges, map_size):
    bits = 0
    for e in edges:
        bits |= 1 << (e % map_size)
    return bits


def test_regression_liveness_fold_bits_matches_loop():
    edges = {i * 7919 for i in range(5000)} | {0, 1, _LIVENESS_MAP_BITS - 1}
    assert _fold_bits(edges, _LIVENESS_MAP_BITS) == _reference(edges, _LIVENESS_MAP_BITS)


def test_control_reference_against_itself():
    """Hard Rule 46: the oracle must agree with a second run of itself."""
    edges = {3, 64, 65535}
    assert _reference(edges, _LIVENESS_MAP_BITS) == _reference(set(edges), _LIVENESS_MAP_BITS)


def test_empty_is_zero():
    assert _fold_bits(set(), _LIVENESS_MAP_BITS) == 0


def test_aliasing_ids_collapse():
    """Adversarial: ids past map_size wrap onto the same bit, as before."""
    edges = {5, 5 + _LIVENESS_MAP_BITS, 5 + 3 * _LIVENESS_MAP_BITS}
    assert _fold_bits(edges, _LIVENESS_MAP_BITS) == 1 << 5


def test_top_bit_lands_on_top():
    """Falsification: a byte- or bit-order slip would move this bit."""
    assert _fold_bits({_LIVENESS_MAP_BITS - 1}, _LIVENESS_MAP_BITS) == 1 << (_LIVENESS_MAP_BITS - 1)
    assert _fold_bits({9}, 16) == 1 << 9


def test_huge_ids():
    edges = {2**40 + 17, 2**62 + 3}
    assert _fold_bits(edges, _LIVENESS_MAP_BITS) == _reference(edges, _LIVENESS_MAP_BITS)
