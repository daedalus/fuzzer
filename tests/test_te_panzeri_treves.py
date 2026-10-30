"""Panzeri-Treves (1996) analytic TE bias: closed form, control, adversarial.

bias[TE] = (sum_c (m_c - 1) - sum_h (m_h - 1)) / (2 N ln 2), where m_c / m_h
are the distinct futures seen per (y_hist, x) context / per y_hist.
Diagnostic only: transfer_entropy() keeps the shuffle correction.
"""

from __future__ import annotations

import math

from fuzzer_tool.core.analyzers.analyzer_transfer_entropy import (
    TransferEntropy,
    panzeri_treves_bias,
)
from fuzzer_tool.core.rand_pool import RandPool

LN2 = math.log(2)

# k=1. y_future = target[2:], y_hist = target[1:-1], x = source[1:-1].
#   t:   0  1  2  3
#   yf:  1  0  1  1
#   yh:  0  1  0  1
#   x:   0  0  0  1
# Per hist: {0: {1}, 1: {0, 1}}           -> excess_target = 0 + 1 = 1
# Per ctx:  {(0,0): {1}, (1,0): {0}, (1,1): {1}} -> excess_both = 0
# N = 4 -> bias = (0 - 1) / (8 ln 2); raw TE = 0.5 - 0 = 0.5 bit.
TARGET = [0, 0, 1, 0, 1, 1]
SOURCE = [0, 0, 0, 0, 1, 0]
N_HAND = 4
RAW_HAND = 0.5


def _joints(source: list[int], target: list[int]):
    te = TransferEntropy(history_length=1)
    jt, jb, n, _ = te._build_joints(source, target)
    return jt, jb, n


def test_bias_closed_form_hand_case():
    jt, jb, n = _joints(SOURCE, TARGET)
    assert n == N_HAND

    expected = (0 - 1) / (2 * N_HAND * LN2)
    assert math.isclose(panzeri_treves_bias(jt, jb, n), expected)


def test_corrected_te_hand_case():
    """Falsification: raw - bias; a formula without the per-context -1
    (sum m_c - sum m_h) gives bias 0 here and 0.5 instead."""
    te = TransferEntropy(history_length=1)
    expected = RAW_HAND + 1 / (2 * N_HAND * LN2)

    assert math.isclose(te.pt_corrected_te(SOURCE, TARGET), expected)


def test_bias_positive_when_contexts_split_futures():
    """Each (hist, x) context sees both futures: excess_both = 2 > 1."""
    jt = {(0, (0,)): 2, (1, (0,)): 2}
    jb = {(0, (0,), 0): 1, (1, (0,), 0): 1, (0, (0,), 1): 1, (1, (0,), 1): 1}
    n = 4

    expected = (2 - 1) / (2 * n * LN2)
    assert math.isclose(panzeri_treves_bias(jt, jb, n), expected)


def test_control_relabel_invariant():
    """Rule 46 control: the same stream under a symbol bijection must give
    the identical bias and corrected TE (counts, not labels, drive it)."""
    rng = RandPool(seed=17)
    source = rng.randint_list(0, 7, 200)
    target = rng.randint_list(0, 7, 200)
    relabel = [(3 * v + 5) % 8 for v in source]
    te = TransferEntropy(history_length=1)

    first = te.pt_corrected_te(source, target)
    assert first == te.pt_corrected_te(source, target)
    assert math.isclose(first, te.pt_corrected_te(relabel, target))


def test_adversarial_zero_counts_ignored():
    """Zero-count keys are not observed bins and must not inflate m_c."""
    jt = {(1, (0,)): 3, (0, (0,)): 0}
    jb = {(1, (0,), 0): 3, (0, (0,), 0): 0, (0, (0,), 9): 0}

    assert panzeri_treves_bias(jt, jb, 3) == 0.0


def test_adversarial_empty_and_short():
    te = TransferEntropy(history_length=1)

    assert panzeri_treves_bias({}, {}, 0) == 0.0
    assert te.pt_corrected_te([], []) == 0.0
    assert te.pt_corrected_te([1, 2], [1, 2]) == 0.0


def test_adversarial_single_symbol_alphabets():
    """Constant target: every context has one future -> bias 0, TE 0."""
    te = TransferEntropy(history_length=1)
    target = [4] * 50
    source = RandPool(seed=3).randint_list(0, 9, 50)
    jt, jb, n = _joints(source, target)

    assert panzeri_treves_bias(jt, jb, n) == 0.0
    assert te.pt_corrected_te(source, target) == 0.0
    assert te.pt_corrected_te([7] * 50, target) == 0.0
