"""Linear extended state observer (ESO) — the estimator half of ADRC.

Active disturbance rejection control extends the plant model with one extra,
fictitious state standing for *everything the model leaves out*: unmodelled
dynamics plus external disturbance, lumped into a single "total disturbance"
estimated online and then subtracted in the control signal. The point is that
no plant model is required — only an order and a bandwidth.

That premise is why this is here rather than a transfer-function design.
There is no model of ``d(discovery rate)/d(temperature)`` for a fuzzing
campaign and there will not be one: it depends on the target's coverage
landscape, which is unknown and non-stationary by construction. Everything
that moves the discovery rate and is not the knob — a region unlocking, a
region exhausting, ``_maybe_prune`` evicting seeds, an SHM resize, a corpus
reload on resume — is what the extra state is for.

Read :meth:`ExtendedStateObserver.compensated` before relying on it: the
rejection it provides is transient, not permanent, and the reason is
structural rather than a tuning failure.

Relation to ``core.kalman``: ``RobustKF`` already runs a 2-state
constant-velocity filter (value, rate) over the discovery rate, with Huber
innovation gating and adaptive measurement noise. This adds the third state.
It is *not* a Kalman filter: the gains are fixed, set from a single bandwidth
parameter rather than from a noise model, which is the standard linear-ESO
parameterisation and the reason ADRC is described as model-free. A Kalman
filter would need Q and R for a plant whose statistics are the thing we do
not know.

The observer also serves ADRC's *tracking differentiator* purpose, which
matters for this signal specifically: it obtains a rate estimate by
integration rather than by differencing, so a Poisson count sequence does not
get its high-frequency noise amplified the way a derivative term would. That
is also why nothing downstream of this module has a D term.

Reference: Han, "From PID to Active Disturbance Rejection Control", IEEE
Trans. Industrial Electronics 56(3), 2009.
"""

from __future__ import annotations

# Observer bandwidth, in units of 1/tick, where a tick is one control period
# (see TEMPERATURE_CONTROL_EXECS). The standard linear-ESO tuning sets the
# three gains from this one number as [3*w, 3*w^2, w^3], so bandwidth is the
# only knob: higher tracks the disturbance faster and passes more measurement
# noise through to the estimate.
#
# 0.30 is chosen, not derived. It puts the observer's settling time at roughly
# 10-15 ticks, which has to sit below the sensing chain's own dead time (
# measured at a median 2-34 ticks with p10=1/p90~36 -- see the handover's
# 2.1) or the observer would be claiming resolution the measurement does not
# have. It is deliberately at the slow end for that reason.
DEFAULT_ESO_BANDWIDTH = 0.30

# Hard bound on |disturbance estimate| as a multiple of the largest observed
# value. Without it, a long plateau feeding zeros lets the disturbance state
# integrate without limit, and the first real discovery afterwards arrives
# against a wildly wound-up estimate. Same failure as integral windup in a
# PID, and it needs the same treatment.
DISTURBANCE_CLAMP_FACTOR = 4.0


class ExtendedStateObserver:
    """Third-order linear ESO over a scalar measurement.

    States are ``[value, rate, disturbance]``. ``value`` tracks the
    measurement, ``rate`` its first difference obtained by integration, and
    ``disturbance`` the lumped unmodelled term.

    Args:
        bandwidth: Observer bandwidth in 1/tick. Gains are derived from it.
        b0: Nominal control gain — how much one unit of control input is
            believed to move the measurement per tick. Only its *scale*
            matters; ADRC tolerates it being wrong by a factor of a few,
            which is the entire reason it is usable here, where the true
            value is unmeasured (see the handover's 5: if it is near zero,
            the loop is not worth closing and this module should be deleted
            rather than retuned).
        clamp_factor: Disturbance bound, as a multiple of the largest
            absolute measurement seen so far.
    """

    def __init__(
        self,
        bandwidth: float = DEFAULT_ESO_BANDWIDTH,
        b0: float = 1.0,
        clamp_factor: float = DISTURBANCE_CLAMP_FACTOR,
    ):
        if bandwidth <= 0.0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth!r}")
        if clamp_factor <= 0.0:
            raise ValueError(f"clamp_factor must be positive, got {clamp_factor!r}")
        self.bandwidth = float(bandwidth)
        self.b0 = float(b0)
        self.clamp_factor = float(clamp_factor)
        # Standard linear-ESO gains for a third-order observer.
        w = self.bandwidth
        self._l1 = 3.0 * w
        self._l2 = 3.0 * w * w
        self._l3 = w * w * w
        self.value = 0.0
        self.rate = 0.0
        self.disturbance = 0.0
        self._initialized = False
        self._scale = 0.0  # largest |measurement| seen, for the clamp
        self._n = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def observations(self) -> int:
        return self._n

    def update(self, measurement: float, control: float = 0.0) -> float:
        """Advance one tick. Returns the current disturbance estimate.

        Args:
            measurement: Observed process variable this tick.
            control: Control input applied over the tick just elapsed. Pass
                the value that was actually applied, not the one about to
                be: the observer is explaining what already happened.
        """
        m = float(measurement)
        self._n += 1
        self._scale = max(self._scale, abs(m))

        if not self._initialized:
            # Snap-initialise rather than let the first innovation drive a
            # transient the whole loop then has to settle out of.
            self.value = m
            self.rate = 0.0
            self.disturbance = 0.0
            self._initialized = True
            return self.disturbance

        err = m - self.value
        # Euler integration at dt = 1 tick. The control period is a fixed
        # exec count by construction, so dt really is constant -- which is
        # the reason the controller does not inherit the stats interval,
        # whose width tracks EPS.
        self.value += self.rate + self._l1 * err
        self.rate += self.disturbance + self.b0 * float(control) + self._l2 * err
        self.disturbance += self._l3 * err

        bound = self.clamp_factor * self._scale if self._scale > 0.0 else self.clamp_factor
        if self.disturbance > bound:
            self.disturbance = bound
        elif self.disturbance < -bound:
            self.disturbance = -bound
        return self.disturbance

    def compensated(self, measurement: float) -> float:
        """Measurement with the estimated disturbance removed.

        **What this does and does not buy.** The disturbance state holds
        unexplained *acceleration*, not an unexplained level: once the value
        state has tracked a step, a constant measurement with no control
        applied implies zero disturbance, and the estimate decays back to
        zero. Measured on a 5:1 step down, the compensated and raw readings
        differ by more than 5% for 4 ticks out of 40, peaking at 12% of the
        new level, then agree.

        So this gives *transient* disturbance rejection, not permanent. That
        is a real benefit given the sensing chain's dead time -- it damps the
        controller's reaction to a sudden move it cannot yet have caused --
        but it is not the "plateaus are absorbed and never chased" property
        the control-theory handover claimed when it proposed this module.

        The stronger property is not available without measuring ``b0``. A
        persistent level change and a persistent setpoint error are the same
        signal; separating them requires knowing how much of the level the
        knob is responsible for, which is exactly the unmeasured derivative
        the handover's §5 names as the kill criterion. An observer cannot
        supply it. Anything claiming otherwise is fitting ``b0`` to noise.
        """
        return float(measurement) - self.disturbance

    def to_dict(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "b0": self.b0,
            "clamp_factor": self.clamp_factor,
            "value": self.value,
            "rate": self.rate,
            "disturbance": self.disturbance,
            "initialized": self._initialized,
            "scale": self._scale,
            "n": self._n,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExtendedStateObserver:
        obs = cls(
            bandwidth=data.get("bandwidth", DEFAULT_ESO_BANDWIDTH),
            b0=data.get("b0", 1.0),
            clamp_factor=data.get("clamp_factor", DISTURBANCE_CLAMP_FACTOR),
        )
        obs.value = data.get("value", 0.0)
        obs.rate = data.get("rate", 0.0)
        obs.disturbance = data.get("disturbance", 0.0)
        obs._initialized = bool(data.get("initialized", False))
        obs._scale = data.get("scale", 0.0)
        obs._n = data.get("n", 0)
        return obs
