"""Tests for the Kuramoto order parameter / Rayleigh concentration test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.circular_stats import (
    MIN_STRIDE,
    PhaseConcentration,
    concentration,
    fold_offsets,
    order_parameter,
    rayleigh_pvalue,
)

# ── order parameter ────────────────────────────────────────────────────


def test_identical_phases_give_unit_order():
    r, psi = order_parameter([0.7, 0.7, 0.7, 0.7])
    assert r == pytest.approx(1.0)
    assert psi == pytest.approx(0.7)


def test_antipodal_pairs_cancel():
    r, _ = order_parameter([0.0, math.pi, math.pi / 2, 3 * math.pi / 2])
    assert r == pytest.approx(0.0, abs=1e-12)


def test_uniform_phases_cancel():
    n = 16
    r, _ = order_parameter([2 * math.pi * k / n for k in range(n)])
    assert r == pytest.approx(0.0, abs=1e-12)


def test_mean_phase_wraps_into_range():
    # Two phases straddling the 2*pi branch cut average to 0, not to pi.
    r, psi = order_parameter([0.1, 2 * math.pi - 0.1])
    assert 0.0 <= psi < 2 * math.pi
    assert psi == pytest.approx(0.0, abs=1e-12)
    assert r == pytest.approx(math.cos(0.1))


def test_weights_shift_the_mean():
    _, psi_flat = order_parameter([0.0, 1.0])
    _, psi_biased = order_parameter([0.0, 1.0], weights=[1.0, 99.0])
    assert psi_flat == pytest.approx(0.5)
    assert psi_biased > 0.9


def test_empty_input_is_incoherent():
    r, psi = order_parameter([])
    assert r == 0.0
    assert psi == 0.0


# ── Rayleigh p-value ───────────────────────────────────────────────────


def test_pvalue_falls_with_sample_size_at_fixed_r():
    ps = [rayleigh_pvalue(n, 0.8) for n in (5, 10, 20, 40)]
    assert ps == sorted(ps, reverse=True)


def test_pvalue_is_one_when_incoherent():
    assert rayleigh_pvalue(50, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_single_observation_is_never_significant():
    # r is 1 by construction for n == 1; the test must not call that a signal.
    assert rayleigh_pvalue(1, 1.0) > 0.1


def test_pvalue_is_calibrated_against_monte_carlo():
    # Hard Rule 46: the null is the control. Draw uniform phases, and check
    # the nominal alpha is delivered within Monte-Carlo error.
    rng = np.random.default_rng(20260919)
    n, trials, alpha = 24, 20_000, 0.05
    phases = rng.uniform(0.0, 2 * math.pi, size=(trials, n))
    rs = np.abs(np.exp(1j * phases).mean(axis=1))
    hits = sum(rayleigh_pvalue(n, float(r)) < alpha for r in rs)
    rate = hits / trials
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert abs(rate - alpha) < 4 * se, f"rejection rate {rate:.4f} off nominal {alpha}"


# ── folding ────────────────────────────────────────────────────────────


def test_fold_maps_stride_multiples_to_one_phase():
    ph = fold_offsets([4, 12, 20, 28], stride=8)
    assert np.allclose(ph, ph[0])


def test_fold_is_proportional_to_residue():
    ph = fold_offsets([0, 2, 4, 6], stride=8)
    assert np.allclose(ph, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2])


# ── concentration ──────────────────────────────────────────────────────


def test_periodic_offsets_are_significant_and_locate_the_field():
    # Offset 5 of every 16-byte record.
    offs = [5 + 16 * k for k in range(12)]
    res = concentration(offs, None, stride=16)
    assert isinstance(res, PhaseConcentration)
    assert res.r == pytest.approx(1.0)
    assert res.offset == 5
    assert res.p_value < 0.01


def test_scattered_offsets_are_not_significant():
    # FALSIFICATION: if the order parameter were dropped and the code simply
    # returned the modal residue, this would still name an offset. It must not.
    offs = list(range(0, 256, 3))  # gcd(3, 16) == 1 -> residues sweep uniformly
    res = concentration(offs, None, stride=16)
    assert res is not None
    assert res.r < 0.2
    assert res.p_value > 0.05


def test_stride_one_is_rejected():
    # ADVERSARIAL: every offset folds to phase 0 at stride 1, so r is 1 by
    # construction and the test statistic is meaningless.
    assert MIN_STRIDE == 2
    assert concentration([1, 2, 3, 4], None, stride=1) is None


def test_non_positive_stride_is_rejected():
    assert concentration([1, 2, 3], None, stride=0) is None
    assert concentration([1, 2, 3], None, stride=-8) is None


def test_no_offsets_is_rejected():
    assert concentration([], None, stride=16) is None


def test_all_zero_weights_are_rejected():
    assert concentration([1, 2, 3], [0.0, 0.0, 0.0], stride=16) is None


def test_negative_weights_are_rejected():
    assert concentration([1, 2], [1.0, -3.0], stride=16) is None


def test_one_dominant_weight_is_not_significant():
    # ADVERSARIAL: raw edge counts are wildly unequal. A single position
    # carrying ~all the weight has effective n ~ 1 and must not read as a
    # concentrated field, however tight its r.
    offs = [5 + 16 * k for k in range(12)]
    weights = [1e6] + [1.0] * 11
    res = concentration(offs, weights, stride=16)
    assert res is not None
    assert res.r == pytest.approx(1.0)
    assert res.effective_n < 2.0
    assert res.p_value > 0.05


def test_offsets_beyond_one_record_fold_together():
    # Evidence from byte 5 and byte 133 is the same field when stride is 16.
    res = concentration([5, 133], None, stride=16)
    assert res is not None
    assert res.offset == 5
    assert res.r == pytest.approx(1.0)


def test_offset_is_wrapped_into_the_record():
    # psi near 2*pi must round to 0, not to stride.
    res = concentration([0, 0, 0, 15], None, stride=16)
    assert res is not None
    assert 0 <= res.offset < 16


def test_vectorized_matches_scalar_oracle():
    # Hard Rule 14/46: keep the obvious implementation as the oracle, and
    # first prove the oracle agrees with itself on a second evaluation.
    def scalar_oracle(offs, stride):
        c = s = 0.0
        for o in offs:
            th = 2 * math.pi * (o % stride) / stride
            c += math.cos(th)
            s += math.sin(th)
        n = len(offs)
        return math.hypot(c, s) / n, math.atan2(s, c) % (2 * math.pi)

    rng = np.random.default_rng(7)
    for _ in range(25):
        stride = int(rng.integers(2, 64))
        offs = rng.integers(0, 4096, size=int(rng.integers(2, 200))).tolist()
        want_r, want_psi = scalar_oracle(offs, stride)
        control_r, control_psi = scalar_oracle(offs, stride)
        assert (control_r, control_psi) == (want_r, want_psi)  # control first

        res = concentration(offs, None, stride=stride)
        assert res is not None
        assert res.r == pytest.approx(want_r, abs=1e-12)
        if want_r > 1e-9:  # psi is undefined at r == 0
            assert math.cos(res.psi - want_psi) == pytest.approx(1.0, abs=1e-12)
