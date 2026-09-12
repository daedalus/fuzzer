"""PI controller with anti-windup, for regulating a fuzzer knob.

PI, not PID. The derivative term is excluded on purpose and not as a
simplification: the process variable here is a discovery count per fixed exec
window, i.e. Poisson, and derivative action amplifies exactly the
high-frequency component that dominates such a signal. A PI controller is the
standard choice where derivative action would be noise-sensitive but the
integral term is still needed, because proportional action alone leaves a
steady-state offset — it needs a standing error to produce any output at all,
so a P-only exploration knob would sit permanently off target.

Anti-windup is mandatory here rather than a refinement. Both ends of the
actuator saturate (the temperature knob is clamped to [0.1, 1.0]) and the
process variable can sit at zero for an entire plateau, which is the textbook
windup case: the integral accumulates an error larger than the regulation
variable can express, and the output then overshoots until it unwinds. Two
mechanisms are used together — conditional integration (freeze the
accumulator when the output is saturated and the error would push it further
into saturation) and a hard clamp on the integral's own contribution.

The accumulator stores ``Ki * integral``, not ``integral``, so changing Ki
mid-run does not produce a discontinuous jump in output. That is the partial
bumpless-transfer trick; it covers Ki changes, not Kp ones.
"""

from __future__ import annotations


class PIController:
    """Discrete PI controller with conditional integration and output clamp.

    ``dt`` is fixed at one tick and is not a parameter. The caller is
    expected to run this on a constant-period clock; a controller whose
    sampling period varies with the plant's own throughput has a drifting
    integral gain, which is the specific failure this omission prevents.

    Args:
        kp: Proportional gain.
        ki: Integral gain, per tick.
        out_min: Lower output bound.
        out_max: Upper output bound.
        integral_limit: Hard bound on the magnitude of the integral's
            contribution to the output. Defaults to the wider of
            ``|out_min|`` and ``|out_max|`` — enough for the integral to
            reach either rail on its own, and no more.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
        integral_limit: float | None = None,
    ):
        if out_min >= out_max:
            raise ValueError(f"out_min {out_min!r} must be below out_max {out_max!r}")
        if kp < 0.0 or ki < 0.0:
            # Negative gains would make this reverse-acting, which is a
            # different controller and should be spelled that way at the
            # call site by negating the error, not hidden in a gain sign.
            raise ValueError(f"gains must be non-negative, got kp={kp!r} ki={ki!r}")
        self.kp = float(kp)
        self.ki = float(ki)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.integral_limit = (
            float(integral_limit)
            if integral_limit is not None
            else max(abs(self.out_min), abs(self.out_max))
        )
        # Stores ki * integral, not integral -- see the module docstring.
        self._accum = 0.0
        self.output = 0.0
        self.last_error = 0.0
        self.saturated = False
        self._n = 0

    @property
    def integral_term(self) -> float:
        """The integral's current contribution to the output."""
        return self._accum

    @property
    def updates(self) -> int:
        return self._n

    def update(self, error: float) -> float:
        """Advance one tick with the given error (setpoint minus measured).

        Returns the clamped control output.
        """
        e = float(error)
        self._n += 1

        candidate = self._accum + self.ki * e
        if candidate > self.integral_limit:
            candidate = self.integral_limit
        elif candidate < -self.integral_limit:
            candidate = -self.integral_limit

        raw = self.kp * e + candidate
        clamped = min(self.out_max, max(self.out_min, raw))

        # Conditional integration: only keep the new accumulator value if it
        # did not push an already-saturated output further into its rail.
        # Checked against the *candidate's* output, not the previous one, so
        # an error that moves the output back off the rail is still
        # integrated -- freezing unconditionally while saturated is the
        # version of this that never recovers.
        pushing_further = (raw > self.out_max and e > 0.0) or (raw < self.out_min and e < 0.0)
        if not pushing_further:
            self._accum = candidate

        self.output = clamped
        self.last_error = e
        self.saturated = raw != clamped
        return clamped

    def reset(self, output: float = 0.0) -> None:
        """Clear the accumulator, optionally re-seeding the output.

        Used on a regime change where the accumulated history has stopped
        describing the current plant — the same reasoning as freezing the
        integral when a furnace door opens.
        """
        self._accum = min(self.integral_limit, max(-self.integral_limit, float(output)))
        self.output = min(self.out_max, max(self.out_min, float(output)))
        self.last_error = 0.0
        self.saturated = False

    def to_dict(self) -> dict:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "out_min": self.out_min,
            "out_max": self.out_max,
            "integral_limit": self.integral_limit,
            "accum": self._accum,
            "output": self.output,
            "last_error": self.last_error,
            "n": self._n,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PIController:
        ctl = cls(
            kp=data.get("kp", 0.0),
            ki=data.get("ki", 0.0),
            out_min=data.get("out_min", -1.0),
            out_max=data.get("out_max", 1.0),
            integral_limit=data.get("integral_limit"),
        )
        ctl._accum = data.get("accum", 0.0)
        ctl.output = data.get("output", 0.0)
        ctl.last_error = data.get("last_error", 0.0)
        ctl._n = data.get("n", 0)
        return ctl
