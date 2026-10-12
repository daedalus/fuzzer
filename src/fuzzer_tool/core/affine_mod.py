"""Affine-modular compares: ``(a*x + b) % m == t``.

Cmplog logs ``v = (a*x + b) % m`` next to ``t``. ``v`` copies no input
bytes, so Redqueen finds nothing to patch, and a mutator hits one residue
class of ``m`` per ``m`` tries. The map is learnable: probe the field ``x``
and read ``v`` back.

Unit steps ``x = xs, xs+1, ...`` give ``v_i = (a*i + v_0) mod m``, so::

    step_i = v_{i+1} - v_i  is  s  or  s - m          (s = a mod m)

    steps differ  -> every difference of steps is 0 or +-m:  m = gcd of them
    steps equal   -> no wrap yet: gallop + bisect to the first wrap x_w;
                     the ramp predicts v_w + m (or - m): m = |pred - v_w|

Then ``a = s mod m`` and ``b = v_0 - a*xs``. The fit is checked on every
sample and on spread-out confirm probes; a map that is not affine mod m
(``x*x % m``, a 32-bit overflow before the ``%``) fails closed.

Solving ``a*x = t - b (mod m)``, ``g = gcd(a, m)``: no solution unless
``g | t - b``; else ``x = x_0 (mod m/g)``, one residue class. A field
that never wrapped over its range is a plain ramp, solved by division.

    probes ──▶ (x_i, v_i) ──fit──▶ AffineMod / Ramp ──solve──▶ x with v(x) == t
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import gcd

from fuzzer_tool.core.numtheory import modinv

# x -> operand ``v`` read from one run with the field set to x; None when
# the compare was not reached.
Observe = Callable[[int], int | None]

# Unit-step samples. A step wraps with probability s/m, so 12 samples see a
# wrap when s > m/12; smaller s falls to the gallop path.
WINDOW = 12

# A modulus of 1 makes every value 0; nothing to learn.
MIN_MODULUS = 2

# Default probe budget for one fit: window + gallop + bisect + confirms.
# A 32-bit field costs ~60 in the gallop case.
MAX_FIT_EXECS = 96

# Residue-class members returned per solve: nearest to the current value,
# smallest, largest.
MAX_SOLUTIONS = 3

# Probes outside the sample window that must agree with the fit.
CONFIRMS = 3


class Outcome(Enum):
    FITTED = "fitted"  # AffineMod recovered
    RAMP = "ramp"  # no wrap anywhere in the field's range
    FLAT = "flat"  # unit steps do not move the operand
    BAD_FIT = "bad_fit"  # not affine mod m; fail closed
    BLIND = "blind"  # compare not reached
    BUDGET = "budget"  # probe budget spent


@dataclass(frozen=True)
class AffineMod:
    a: int
    b: int
    m: int

    def __call__(self, x: int) -> int:
        return (self.a * x + self.b) % self.m


@dataclass(frozen=True)
class Ramp:
    """``v = v_ref + step * (x - x_ref)`` on ``lo <= x <= hi``: no wrap seen there.

    Only the side of the window the gallop walked is known to be wrap-free.
    """

    x_ref: int
    v_ref: int
    step: int
    lo: int
    hi: int

    def __call__(self, x: int) -> int:
        return self.v_ref + self.step * (x - self.x_ref)


@dataclass(frozen=True)
class Fit:
    outcome: Outcome
    model: AffineMod | Ramp | None
    execs: int


class _Meter:
    """Counted, memoised probe; ``dead`` once a read fails."""

    def __init__(self, observe: Observe, budget: int):
        self._observe = observe
        self._budget = budget
        self.execs = 0
        self.seen: dict[int, int] = {}
        self.outcome: Outcome | None = None

    def at(self, x: int) -> int | None:
        if x in self.seen:
            return self.seen[x]

        if self.execs >= self._budget:
            self.outcome = Outcome.BUDGET
            return None

        self.execs += 1
        v = self._observe(x)
        if v is None:
            self.outcome = Outcome.BLIND
            return None

        self.seen[x] = v
        return v

    def stop(self) -> Fit:
        return Fit(self.outcome or Outcome.BLIND, None, self.execs)

    def done(self, outcome: Outcome, model: AffineMod | Ramp | None) -> Fit:
        return Fit(outcome, model, self.execs)


def _consistent(vs: list[int], a: int, m: int) -> bool:
    """Every sample is below m and sits on ``v_0 + a*i (mod m)``."""
    return all(v < m and (v - vs[0] - a * i) % m == 0 for i, v in enumerate(vs))


def _confirm(meter: _Meter, model: AffineMod, xs: int, top: int) -> Outcome | None:
    """Probe spread-out x outside the window; None when all agree."""
    spots = {top, top // 2 + 1, xs + 2 * WINDOW}
    for x in sorted(spots - set(meter.seen)):
        if not 0 <= x <= top:
            continue

        v = meter.at(x)
        if v is None:
            return meter.outcome
        if v != model(x):
            return Outcome.BAD_FIT

    return None


def _finish(meter: _Meter, model: AffineMod, xs: int, top: int) -> Fit:
    verdict = _confirm(meter, model, xs, top)
    if verdict is not None:
        return meter.done(verdict, None)
    return meter.done(Outcome.FITTED, model)


def _from_wraps(meter: _Meter, vs: list[int], steps: list[int], xs: int, top: int) -> Fit:
    """Steps differ: m is the gcd of the step differences."""
    m = 0
    for s in steps:
        m = gcd(m, abs(s - steps[0]))

    a = steps[0] % m
    if m < MIN_MODULUS or not _consistent(vs, a, m):
        return meter.done(Outcome.BAD_FIT, None)

    return _finish(meter, AffineMod(a, (vs[0] - a * xs) % m, m), xs, top)


def _first_break(meter: _Meter, ramp: Ramp, good: int, limit: int) -> int | None:
    """First x past ``good`` (toward ``limit``) off the ramp; None if none.

    Off the ramp is monotone: once ``v`` wraps it is ``ramp -+ k*m``, k >= 1,
    never the ramp value again. Gallop to bracket, then bisect.
    """
    sign = 1 if limit > good else -1
    stride = WINDOW
    lo = good
    hi = None

    while lo != limit and hi is None:
        x = lo + sign * stride
        if (x - limit) * sign > 0:
            x = limit

        v = meter.at(x)
        if v is None:
            return None

        if v == ramp(x):
            lo = x
            stride *= 2
        else:
            hi = x

    if hi is None:
        return limit + sign  # sentinel: ran off the field

    while abs(hi - lo) > 1:
        mid = (lo + hi) // 2
        v = meter.at(mid)
        if v is None:
            return None

        if v == ramp(mid):
            lo = mid
        else:
            hi = mid

    return hi


def _from_ramp(meter: _Meter, v0: int, step: int, xs: int, top: int) -> Fit:
    """Equal steps: no wrap in the window; find one past it."""
    end = xs + WINDOW - 1
    # Window abuts the top of the field: look downward instead.
    good, limit = (end, top) if end < top else (xs, 0)
    ramp = Ramp(xs, v0, step, min(xs, limit), max(end, limit))

    wrap = _first_break(meter, ramp, good, limit)
    if wrap is None:
        return meter.stop()

    if not 0 <= wrap <= top:
        return meter.done(Outcome.RAMP, ramp)

    # One unit step wraps at most once (|step| < m), so the jump is exactly m.
    m = abs(ramp(wrap) - meter.seen[wrap])
    if m < MIN_MODULUS or max(meter.seen.values()) >= m:
        return meter.done(Outcome.BAD_FIT, None)

    a = step % m
    model = AffineMod(a, (v0 - a * xs) % m, m)
    if model(wrap) != meter.seen[wrap]:
        return meter.done(Outcome.BAD_FIT, None)

    return _finish(meter, model, xs, top)


def fit(observe: Observe, x0: int, width_bits: int, max_execs: int = MAX_FIT_EXECS) -> Fit:
    """Recover the map behind *observe* for a ``width_bits`` field at value *x0*.

    ``observe(x)`` runs the target with the field set to ``x`` and returns
    the compare's varying operand. Samples start at *x0* (a value the
    target already reached the compare with), shifted down only to keep the
    window inside the field.
    """
    top = (1 << width_bits) - 1
    xs = max(0, min(x0, top - WINDOW + 1))
    meter = _Meter(observe, max_execs)

    vs: list[int] = []
    for i in range(WINDOW):
        v = meter.at(xs + i)
        if v is None:
            return meter.stop()
        vs.append(v)

    steps = [b - a for a, b in zip(vs, vs[1:], strict=False)]
    if not any(steps):
        return meter.done(Outcome.FLAT, None)

    if len(set(steps)) > 1:
        return _from_wraps(meter, vs, steps, xs, top)

    return _from_ramp(meter, vs[0], steps[0], xs, top)


def _class_members(x0: int, step: int, top: int, near: int) -> list[int]:
    """Nearest-to-*near*, smallest, largest member of ``x0 (mod step)`` in range."""
    if x0 > top:
        return []

    last = (top - x0) // step
    k = min(max((near - x0 + step // 2) // step, 0), last)
    out: list[int] = []
    for member in (x0 + k * step, x0, x0 + last * step):
        if member not in out:
            out.append(member)

    return out[:MAX_SOLUTIONS]


def solve(model: AffineMod | Ramp, target: int, width_bits: int, near: int = 0) -> list[int]:
    """Field values ``x`` in range with ``model(x) == target``; may be empty."""
    top = (1 << width_bits) - 1

    if isinstance(model, Ramp):
        num = target - model.v_ref
        if num % model.step:
            return []

        x = model.x_ref + num // model.step
        return [x] if max(model.lo, 0) <= x <= min(model.hi, top) else []

    m = model.m
    if not 0 <= target < m:
        return []

    a = model.a % m
    g = gcd(a, m)
    rhs = (target - model.b) % m
    if rhs % g:
        return []

    period = m // g
    x0 = (rhs // g) * modinv(a // g, period) % period if period > 1 else 0
    return _class_members(x0, period, top, near)
