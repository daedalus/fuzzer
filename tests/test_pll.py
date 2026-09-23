"""Tests for core/pll.py.

Expectations are derived from the digital-PLL model's own definitions
(quadrature phase detector, PI loop filter, I-channel coherence lock
detector with hysteresis) rather than asserted against the
implementation's own output -- same discipline as test_kuramoto.py.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.pll import PhaseLockedLoop, PLLState


def _sine(n, freq, amp=1.0, dc=0.0, phase0=0.0):
    return [dc + amp * math.sin(2.0 * math.pi * freq * i + phase0) for i in range(n)]


def _run(pll, signal):
    return [pll.step(x) for x in signal]


class TestConstruction:
    def test_rejects_center_freq_above_nyquist(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.6)

    def test_rejects_center_freq_zero(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.0)

    def test_accepts_center_freq_at_nyquist(self):
        PhaseLockedLoop(center_freq=0.5)

    def test_rejects_inverted_lock_hysteresis(self):
        # unlock_threshold must be BELOW lock_threshold now (coherence is
        # "higher is more locked", unlike the old |error| convention).
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.1, lock_threshold=0.2, unlock_threshold=0.3)

    def test_rejects_equal_lock_hysteresis(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.1, lock_threshold=0.2, unlock_threshold=0.2)

    def test_rejects_zero_min_lock_ticks(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.1, min_lock_ticks=0)

    @pytest.mark.parametrize("alpha_name", ["dc_alpha", "amp_alpha", "detector_alpha"])
    def test_rejects_out_of_range_alphas(self, alpha_name):
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.1, **{alpha_name: 0.0})
        with pytest.raises(ValueError):
            PhaseLockedLoop(center_freq=0.1, **{alpha_name: 1.5})


class TestFromPeriod:
    def test_center_freq_is_reciprocal_of_period(self):
        pll = PhaseLockedLoop.from_period(20.0)
        assert abs(pll.center_freq - 0.05) < 1e-12

    def test_rejects_period_at_nyquist_floor(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop.from_period(2.0)

    def test_rejects_period_below_nyquist_floor(self):
        with pytest.raises(ValueError):
            PhaseLockedLoop.from_period(1.5)

    def test_forwards_kwargs(self):
        pll = PhaseLockedLoop.from_period(10.0, kp=0.02)
        assert pll._pi.kp == 0.02


class TestStepBasics:
    def test_tick_increments(self):
        pll = PhaseLockedLoop(center_freq=0.1)
        s1 = pll.step(1.0)
        s2 = pll.step(1.0)
        assert s1.tick == 1
        assert s2.tick == 2

    def test_phase_always_wrapped(self):
        pll = PhaseLockedLoop(center_freq=0.5)
        for s in _run(pll, _sine(500, freq=0.5)):
            assert -math.pi < s.phase <= math.pi + 1e-9

    def test_state_is_a_pllstate(self):
        pll = PhaseLockedLoop(center_freq=0.1)
        assert isinstance(pll.step(1.0), PLLState)

    def test_period_is_reciprocal_of_freq(self):
        pll = PhaseLockedLoop(center_freq=0.1)
        s = pll.step(1.0)
        assert abs(s.period - 1.0 / s.freq) < 1e-12

    def test_freq_never_exceeds_nyquist(self):
        pll = PhaseLockedLoop(center_freq=0.45, max_freq_correction=0.4)
        for s in _run(pll, _sine(200, freq=0.05, amp=50.0)):
            assert s.freq <= 0.5

    def test_coherence_is_nonnegative(self):
        pll = PhaseLockedLoop(center_freq=0.05)
        for s in _run(pll, _sine(500, freq=0.05, dc=3.0)):
            assert s.coherence >= 0.0

    def test_first_tick_produces_zero_ac_component(self):
        # DC is snapped to the first sample, so error/coherence on tick 1
        # must both be exactly 0 regardless of the sample's magnitude.
        pll = PhaseLockedLoop(center_freq=0.1)
        s = pll.step(12345.678)
        assert s.error == 0.0
        assert s.coherence == 0.0


class TestLockAcquisition:
    def test_locks_onto_a_clean_matched_frequency(self):
        # center_freq exactly matches the input frequency, large DC
        # offset and an arbitrary phase0 to confirm neither breaks lock.
        pll = PhaseLockedLoop(center_freq=0.05)
        states = _run(pll, _sine(3000, freq=0.05, amp=1.0, dc=10.0, phase0=1.3))
        assert any(s.locked for s in states)
        # A nonzero phase0 with theta starting at 0 forces a real
        # acquisition transient (pull-in), which can cycle-slip through a
        # brief false lock before settling -- a known real PLL phenomenon,
        # not a bug. The claim this test makes is the honest one: by the
        # end of a long enough run it has settled and stayed locked, not
        # that it never wavers between the first touch of lock_threshold
        # and full settling.
        assert all(s.locked for s in states[-200:])

    def test_tracks_a_frequency_offset_from_center(self):
        # True frequency is 10% off center_freq -- the loop filter's
        # integral term must absorb that steady-state offset to lock.
        pll = PhaseLockedLoop(center_freq=0.05, max_freq_correction=0.02)
        states = _run(pll, _sine(3000, freq=0.055, amp=1.0, dc=5.0))
        assert any(s.locked for s in states)
        locked_states = [s for s in states if s.locked]
        assert abs(locked_states[-1].freq - 0.055) < 0.003

    def test_bootstrap_from_detect_periodicity_style_period_then_locks(self):
        # Mirrors the intended real usage: take a period estimate the way
        # periodicity.detect_periodicity(...).dominant_period would supply
        # it, and confirm the resulting loop actually acquires lock.
        dominant_period = 25.0
        pll = PhaseLockedLoop.from_period(dominant_period)
        states = _run(pll, _sine(3000, freq=1.0 / dominant_period, dc=1.0))
        assert any(s.locked for s in states)

    @pytest.mark.parametrize("seed", range(5))
    def test_never_locks_on_white_noise(self, seed):
        rng = random.Random(seed)
        pll = PhaseLockedLoop(center_freq=0.1)
        states = [pll.step(rng.uniform(-1.0, 1.0)) for _ in range(3000)]
        assert not any(s.locked for s in states)

    def test_never_locks_on_a_constant_series(self):
        # No AC component at all after the first-sample DC snap -- both
        # mixer channels are pinned to exactly 0 forever (see the
        # `denom` comment in pll.py), so coherence can never cross
        # lock_threshold.
        pll = PhaseLockedLoop(center_freq=0.1)
        states = [pll.step(7.0) for _ in range(1000)]
        assert not any(s.locked for s in states)
        assert all(s.coherence == 0.0 for s in states)

    def test_min_lock_ticks_delays_the_first_lock(self):
        signal = _sine(3000, freq=0.05, amp=1.0, dc=0.0)
        fast = PhaseLockedLoop(center_freq=0.05, min_lock_ticks=1)
        slow = PhaseLockedLoop(center_freq=0.05, min_lock_ticks=200)
        fast_first = next(
            (i for i, s in enumerate(_run(fast, signal)) if s.locked), None
        )
        slow_first = next(
            (i for i, s in enumerate(_run(slow, signal)) if s.locked), None
        )
        assert fast_first is not None and slow_first is not None
        assert slow_first > fast_first


class TestHysteresis:
    def test_coherence_between_thresholds_holds_current_lock_state(self):
        # Force a locked loop and feed one sample whose coherence lands
        # inside the hysteresis band (between unlock_threshold and
        # lock_threshold) -- lock must be retained, not dropped, since
        # only falling BELOW unlock_threshold clears it.
        pll = PhaseLockedLoop(
            center_freq=0.1, lock_threshold=0.4, unlock_threshold=0.2, min_lock_ticks=1
        )
        pll.locked = True
        pll._i_lp = 0.3  # inside the hysteresis band
        pll._consec_locked = 5
        s = pll.step(0.0)  # zero AC input nudges _i_lp toward 0, not out of the band yet
        assert s.locked is True

    def test_coherence_below_unlock_threshold_clears_lock(self):
        pll = PhaseLockedLoop(
            center_freq=0.1, lock_threshold=0.4, unlock_threshold=0.2, min_lock_ticks=1
        )
        pll.locked = True
        pll._i_lp = 0.05  # below unlock_threshold
        pll._consec_locked = 5
        s = pll.step(0.0)
        assert s.locked is False


class TestReset:
    def test_reset_clears_lock_state(self):
        pll = PhaseLockedLoop(center_freq=0.05)
        for x in _sine(3000, freq=0.05, dc=3.0):
            pll.step(x)
        assert pll.locked is True
        pll.reset()
        assert pll.locked is False
        s = pll.step(0.0)
        assert s.tick == 1

    def test_reset_can_recenter_frequency(self):
        pll = PhaseLockedLoop(center_freq=0.05)
        pll.reset(center_freq=0.2)
        assert pll.center_freq == 0.2

    def test_reset_rejects_invalid_recenter(self):
        pll = PhaseLockedLoop(center_freq=0.05)
        with pytest.raises(ValueError):
            pll.reset(center_freq=0.9)
