"""Digital phase-locked loop for online tracking of a periodic scalar series.

Standalone diagnostic utility, in the same spirit as ``kuramoto.py``: not
wired into ``select_op``, the timeout logic, or any scheduler. It exists to
answer one question ``core/periodicity.py`` cannot: when a periodic
component in a fuzzer time series (per-execution wall-clock time,
discovery-rate deltas) is *non-stationary* -- its frequency drifts over the
course of a long campaign, e.g. thermal throttling slowly stretching a
GC/JIT cadence -- can that drift be tracked tick-by-tick, cheaply, instead
of re-running a whole-history FFT every report tick and averaging the drift
away?

Relationship to ``core/kuramoto.py``
-------------------------------------
Do not confuse the two. Kuramoto is *mutual* synchronization: N oscillators
coupled to each other, no external reference, order parameter r measures
how well they agree with one another. A PLL is the other topology: **one**
oscillator (here, a numerically-controlled phase accumulator, the "NCO")
locking onto an **external reference** signal (the fuzzer time series) via
closed-loop feedback -- phase detector, loop filter, NCO. Nothing here
reuses or extends ``kuramoto.py``; the two solve genuinely different
problems.

Relationship to ``core/periodicity.py``
-----------------------------------------
``detect_periodicity`` is the right tool for "is there a significant
periodic component in this window, and what period is it" -- a one-shot,
whole-window FFT scored against Fisher's g-test's exact null distribution,
with an AR-drift removal pass for the "not quite stationary" case. It is
not built to *track* a frequency that keeps changing across a multi-hour
campaign: each call restarts from nothing, and a period that shifts within
one window shows up as spectral leakage rather than a clean estimate. This
module is the complementary tool for that regime: an O(1)-per-sample
tracker that follows frequency drift online and reports a continuous
lock/unlock signal, at the cost of assuming the periodic component is
roughly sinusoidal (single dominant frequency) rather than handling
arbitrary harmonic content the way ``detect_periodicity``'s harmonic-
binning mode does. Bootstrap the tracker's starting frequency from a prior
``detect_periodicity`` call via :meth:`PhaseLockedLoop.from_period` --
the two are meant to be used together, not as substitutes.

Model
-----
Standard discrete-time, type-2 digital PLL with I/Q (in-phase/quadrature)
demodulation -- textbook material in digital carrier-recovery / phaselock
and lock-in-amplifier literature (e.g. Gardner, *Phaselock Techniques*);
nothing here is novel machinery, only its application to fuzzer telemetry
is:

1. **DC removal.** An EMA tracks the series' running mean, snapped to the
   first sample on tick 1 rather than blended in from an arbitrary 0.0
   starting point (see :meth:`step`'s comment); the rest of the pipeline
   operates on the mean-subtracted signal. Fuzzer series (exec times,
   discovery counts) are never zero-mean, unlike a textbook carrier.
2. **Quadrature mixers.** Two multiplier phase detectors against the NCO's
   own sine and cosine: ``q = x_ac * sin(theta_hat)``,
   ``i = x_ac * cos(theta_hat)``. If the input truly contains
   ``A*cos(theta_true - phi0)`` and ``theta_hat`` is near ``theta_true``,
   the quadrature (Q) channel expands to a slowly-varying
   ``(A/2)*sin(theta_hat - theta_true)`` term -- the phase-error signal --
   plus a double-frequency term; the in-phase (I) channel expands to
   ``(A/2)*cos(theta_hat - theta_true)`` plus a double-frequency term --
   near a fixed nonzero amplitude exactly when locked, near zero
   otherwise. Both channels are normalized (next point) and then
   low-pass filtered (EMA, rate ``detector_alpha``) to average out the
   double-frequency terms, which a bare multiplier output does not filter
   on its own -- an earlier version of this module skipped this LPF stage
   on the (false) assumption the loop filter's own integral action would
   absorb it; empirically it does not at gains fast enough to track drift,
   producing a self-sustaining off-frequency limit cycle instead of true
   lock.
3. **Normalization.** Both mixer outputs are divided by
   ``max(amplitude_ema, |x_ac|, eps)`` -- the larger of a lagging EMA of
   ``|x_ac|`` and this tick's own instantaneous value -- so neither
   channel needs re-tuned gains per series scale (execution-time
   microseconds vs. discovery-edge counts), and so the ratio stays
   bounded even during the handful of ticks before the amplitude EMA has
   converged, rather than dividing by a near-zero floor and producing a
   transient blowup.
4. **Loop filter.** :class:`~fuzzer_tool.core.pi_controller.PIController`,
   reused unmodified, driven by the filtered Q channel only. This is not
   a convenience import: a PI filter *is* the standard type-2 PLL loop
   filter -- proportional action gives an instantaneous phase kick,
   integral action is exactly what tracks a steady frequency offset from
   ``center_freq``, which is precisely what "the period is drifting"
   means here. ``PIController``'s anti-windup (needed for the temperature
   knob it was built for) is a bonus, not a requirement, but a correction
   output pinned at a rail is exactly as undesirable here as it is there.
5. **NCO.** ``theta += 2*pi*freq`` each tick, wrapped to ``(-pi, pi]``,
   where ``freq = center_freq + loop_filter_output``.
6. **Lock detector.** The filtered I channel is a *coherence* statistic,
   not an error to be minimized: it measures how much of the
   (amplitude-normalized) input is correlated with the NCO's own phase.
   Genuinely locked, it sits near a fixed nonzero level (empirically
   ~0.55-0.65 for a clean sinusoid with these default gains); for
   incoherent input (white noise, or a genuinely flat series where the
   normalization floor forces both channels to exactly 0) it stays near
   0. This is the reason the lock statistic is the I channel and not the
   Q channel the loop filter drives toward zero -- Q *converging to zero*
   is indistinguishable between "locked, near-zero residual phase error"
   and "totally incoherent, product time-averages to zero anyway"; I
   does not have that ambiguity. Hysteresis (separate lock/unlock
   thresholds, plus a minimum run of above-threshold ticks before
   flipping to locked) follows the same chatter-avoidance shape used by
   ``analyzer_kuramoto_sync.py``'s consumer, ``CriticalSlowingDown``.

What this does *not* claim
---------------------------
No claim that fuzzer exec-time or discovery-rate series actually contain
a genuine, sustained sinusoidal component worth tracking -- that is the
same open empirical question ``kuramoto.py``'s docstring leaves for its
own model. This module only provides the machinery (NCO, loop filter,
lock detector, a bootstrap from ``detect_periodicity``) needed to run that
experiment: feed it ``f._exec_time_tracker``'s times or
``f._discovery_edges``'s deltas tick by tick and see whether lock/unlock
transitions correlate with anything the fuzzer already cares about (a
stall, a corpus-sync artifact, a thermal-throttle window). Gains and
thresholds (``kp=0.005``, ``ki=0.0002``, ``detector_alpha=0.05``,
``lock_threshold=0.45``, ``unlock_threshold=0.25``) were tuned against
synthetic sinusoids and white/constant noise in this module's own test
suite -- enough to give a working default, not values swept against a
real campaign's actual exec-time or discovery-rate statistics, flagged
here the same way ``op_kuramoto.py`` flags its borrowed
``explore_floor=0.06``.

Nyquist applies: a period must exceed 2 samples for any frequency
estimate to be meaningful, so ``center_freq`` is constrained to
``(0, 0.5]`` cycles/sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.pi_controller import PIController


@dataclass(frozen=True)
class PLLState:
    """Snapshot returned by :meth:`PhaseLockedLoop.step`.

    Args:
        tick: 1-indexed count of :meth:`step` calls so far.
        phase: NCO phase estimate, radians, wrapped to ``(-pi, pi]``.
        freq: Current tracked frequency, cycles/sample.
        period: ``1/freq`` in samples.
        error: Filtered, normalized quadrature (Q) channel -- what the
            loop filter is driving toward zero. Near zero at lock *or*
            at total incoherence; see ``coherence`` to tell those apart.
        coherence: Filtered, normalized in-phase (I) channel -- the
            lock-detector statistic (see the module docstring). Near a
            fixed nonzero level at genuine lock, near zero otherwise.
        locked: Lock-detector state after this tick (hysteresis + minimum
            run applied -- see the module docstring).
    """

    tick: int
    phase: float
    freq: float
    period: float
    error: float
    coherence: float
    locked: bool


class PhaseLockedLoop:
    """Track the frequency and phase of a periodic component in a scalar
    time series, online, tick by tick.

    Args:
        center_freq: Initial/nominal frequency, cycles/sample, in
            ``(0, 0.5]`` (Nyquist). Prefer :meth:`from_period` when the
            starting point is a period estimate (e.g. from
            ``periodicity.detect_periodicity``) rather than a raw
            frequency.
        kp: Loop-filter proportional gain, forwarded to the internal
            :class:`PIController`.
        ki: Loop-filter integral gain, forwarded to the internal
            :class:`PIController`.
        max_freq_correction: Bound on how far the loop filter may move
            ``freq`` away from ``center_freq`` in one tick, cycles/sample.
            Also used as the ``PIController``'s ``integral_limit``.
        dc_alpha: EMA rate for the running-mean (DC) estimate.
        amp_alpha: EMA rate for the running amplitude estimate used to
            normalize the mixer outputs.
        detector_alpha: EMA rate for low-pass filtering the I/Q mixer
            outputs -- needs to be slow enough, relative to
            ``center_freq``, to average out the double-frequency term
            (see the module docstring's Model section, point 2). Default
            0.05 (~20-sample time constant) is tuned for periods around
            15-30 samples in this module's own tests; a much shorter or
            longer target period likely needs a different value.
        lock_threshold: Smoothed coherence (I channel) must rise to or
            above this to begin counting toward lock.
        unlock_threshold: Smoothed coherence falling below this
            immediately clears lock and the lock-run counter. Must be
            below ``lock_threshold`` (hysteresis).
        min_lock_ticks: Consecutive at-or-above-``lock_threshold`` ticks
            required before ``locked`` flips ``True``.
    """

    def __init__(
        self,
        center_freq: float,
        kp: float = 0.005,
        ki: float = 0.0002,
        max_freq_correction: float = 0.4,
        dc_alpha: float = 0.05,
        amp_alpha: float = 0.05,
        detector_alpha: float = 0.05,
        lock_threshold: float = 0.45,
        unlock_threshold: float = 0.25,
        min_lock_ticks: int = 20,
    ):
        if not (0.0 < center_freq <= 0.5):
            raise ValueError(
                f"center_freq {center_freq!r} must be in (0, 0.5] cycles/sample (Nyquist)"
            )
        if not (
            0.0 < dc_alpha <= 1.0 and 0.0 < amp_alpha <= 1.0 and 0.0 < detector_alpha <= 1.0
        ):
            raise ValueError("dc_alpha, amp_alpha, detector_alpha must be in (0, 1]")
        if not (0.0 <= unlock_threshold < lock_threshold):
            raise ValueError(
                f"unlock_threshold {unlock_threshold!r} must be >= 0 and below "
                f"lock_threshold {lock_threshold!r}"
            )
        if min_lock_ticks < 1:
            raise ValueError(f"min_lock_ticks {min_lock_ticks!r} must be >= 1")

        self.center_freq = float(center_freq)
        self.dc_alpha = float(dc_alpha)
        self.amp_alpha = float(amp_alpha)
        self.detector_alpha = float(detector_alpha)
        self.lock_threshold = float(lock_threshold)
        self.unlock_threshold = float(unlock_threshold)
        self.min_lock_ticks = int(min_lock_ticks)
        self._max_freq_correction = float(max_freq_correction)

        self._pi = PIController(
            kp=kp,
            ki=ki,
            out_min=-max_freq_correction,
            out_max=max_freq_correction,
            integral_limit=max_freq_correction,
        )
        self._theta = 0.0
        self._dc = 0.0
        self._amp = 0.0
        self._q_lp = 0.0
        self._i_lp = 0.0
        self._consec_locked = 0
        self._tick = 0
        self.locked = False

    @classmethod
    def from_period(cls, period_samples: float, **kwargs: Any) -> PhaseLockedLoop:
        """Build a loop with ``center_freq = 1/period_samples``.

        Convenience bridge from a period estimate -- typically
        ``periodicity.detect_periodicity(...).dominant_period`` -- to this
        module's frequency convention. Deliberately takes a bare float
        rather than a ``PeriodicityResult`` to avoid coupling this module
        to ``periodicity.py``'s result type for a single field extraction.

        Raises:
            ValueError: if ``period_samples`` is not strictly greater than
                2 (Nyquist: a period of 2 samples/cycle is the fastest
                frequency this loop can represent, ``center_freq=0.5``).
        """
        if not (period_samples > 2.0):
            raise ValueError(
                f"period_samples {period_samples!r} must exceed 2 samples/cycle (Nyquist)"
            )
        return cls(center_freq=1.0 / period_samples, **kwargs)

    def step(self, x: float) -> PLLState:
        """Advance one tick with observation ``x`` and return the new state."""
        x = float(x)
        # DC snap on the very first sample: without this, self._dc starts
        # at 0.0 and a series with a large true mean (e.g. exec-time
        # microseconds, never zero-centered) leaves a huge x_ac spike on
        # tick 1 that the amplitude EMA (also starting at 0.0) hasn't
        # caught up to yet -- exactly the cold-start blowup the `denom`
        # bound below exists to prevent structurally, but there's no
        # reason to let it happen at all when the fix is free.
        if self._tick == 0:
            self._dc = x
        else:
            self._dc += self.dc_alpha * (x - self._dc)
        self._tick += 1
        x_ac = x - self._dc

        self._amp += self.amp_alpha * (abs(x_ac) - self._amp)
        # denom is the larger of the (lagging) amplitude EMA and this
        # tick's own |x_ac| -- not just the EMA floored at a tiny epsilon.
        # Bounding by the instantaneous value caps both mixer outputs at
        # magnitude 1 unconditionally, including during the ~1/amp_alpha
        # ticks before the EMA has converged, rather than relying on a
        # floor constant that is either too large (dead zone once
        # converged) or too small (amplifies noise into a blowup before
        # it has). For a genuinely flat series (x_ac == 0 always), both
        # amp and |x_ac| are 0, denom falls to the eps floor, and both
        # mixer outputs come out exactly 0 -- never contributing false
        # coherence.
        denom = max(self._amp, abs(x_ac), 1e-12)

        q = (x_ac * math.sin(self._theta)) / denom
        i = (x_ac * math.cos(self._theta)) / denom
        self._q_lp += self.detector_alpha * (q - self._q_lp)
        self._i_lp += self.detector_alpha * (i - self._i_lp)

        correction = self._pi.update(self._q_lp)
        freq = self.center_freq + correction
        # Defensive clamp: out_min/out_max already bound `correction`, but
        # a center_freq near either Nyquist rail plus a full-scale
        # correction could still push freq outside the representable
        # range. 1e-6 rather than 0.0 keeps `period = 1/freq` finite.
        freq = min(0.5, max(1e-6, freq))

        self._theta += 2.0 * math.pi * freq
        self._theta = (self._theta + math.pi) % (2.0 * math.pi) - math.pi
        if self._theta <= -math.pi:
            # The modulo above lands in [-pi, pi); the single point -pi
            # itself falls outside this module's (-pi, pi] convention.
            self._theta += 2.0 * math.pi

        coherence = abs(self._i_lp)
        if coherence >= self.lock_threshold:
            self._consec_locked += 1
            if not self.locked and self._consec_locked >= self.min_lock_ticks:
                self.locked = True
        elif coherence < self.unlock_threshold:
            self._consec_locked = 0
            self.locked = False
        # else: inside the hysteresis band -- hold current state and
        # counter, the whole point of separating lock/unlock thresholds.

        period = 1.0 / freq
        return PLLState(
            tick=self._tick,
            phase=self._theta,
            freq=freq,
            period=period,
            error=self._q_lp,
            coherence=coherence,
            locked=self.locked,
        )

    def reset(self, center_freq: float | None = None) -> None:
        """Reset all running state. Optionally re-centers the frequency."""
        if center_freq is not None:
            if not (0.0 < center_freq <= 0.5):
                raise ValueError(
                    f"center_freq {center_freq!r} must be in (0, 0.5] cycles/sample (Nyquist)"
                )
            self.center_freq = float(center_freq)
        self._pi.reset()
        self._theta = 0.0
        self._dc = 0.0
        self._amp = 0.0
        self._q_lp = 0.0
        self._i_lp = 0.0
        self._consec_locked = 0
        self._tick = 0
        self.locked = False
