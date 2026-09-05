"""Steady continuum diagnostics over the coverage frontier.

Sits beside ``core/percolation.py``: pure math, no scheduler interface.
Invasion percolation is the quasi-static, capillary-dominated limit of
porous-media flow; this module supplies the fields one rung up — a pressure
scalar, its gradient, a viscosity from operator failure, a predicted flux,
and a dimensionless advective/diffusive ratio.

Why there is no time step, and why that is the design
-----------------------------------------------------
Tao, *Finite time blowup for an averaged three-dimensional Navier-Stokes
equation* (arXiv:1402.0290), constructs an averaged nonlinearity that keeps
the cancellation law ``<B(u,u), u> = 0`` — hence the full energy identity —
and essentially every function-space upper bound the true nonlinearity
obeys, and still admits solutions that blow up in finite time.  The result
formalises the supercriticality barrier: abstract structure does not
determine behaviour.

Two consequences bind this module.

1. Replacing the nonlinear term with "something analogous on a graph" lands
   squarely in Tao's class of counterexamples.  Qualitative behaviour is not
   inherited across averaging, so "prototype a surrogate flux now and swap
   in a real finite-volume step later" is not a continuity assumption
   anyone is entitled to.  Numbers measured on the surrogate say nothing
   about the PDE, and vice versa.

2. This setting is worse placed than Tao's, not better.  There is no Leray
   projection on the horizon graph — the sparse solve that would supply one
   is exactly the cost the design forbids — so incompressibility is never
   enforced and the energy identity does not even hold.  Tao keeps the
   identity and still gets blowup; an unprojected graph advection has
   strictly less to stand on.

So nothing here integrates.  Every quantity is a bounded function of the
*current* state: no accumulator, no state to diverge, no clamp standing in
for physics that was never conserved.  A lattice-Boltzmann step, an
advective update, or any real discretisation would reintroduce the whole
problem and is deliberately absent — see the adversarial test in
``tests/test_navier_stokes.py``.

What ``reynolds`` is and is not
-------------------------------
It is the dimensionless ratio of the advective weight to the diffusive one
on the graph.  It is *not* a fluid Reynolds number: the graph carries no
length in any metric sense (edge ids are ``prev_loc ^ cur_loc``, an XOR with
no metric structure), so the length scale is a hop count.  A high reading is
not evidence of turbulence in the fluid sense, only that discovery is
outrunning damping relative to this campaign's own history.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# Flux is a bounded score, not a physical rate.  The cap keeps an operator
# with a perfect record from swamping the ranking, and mirrors the optimism
# invasion already gives unobserved arms (see ``_resistance``).
MAX_FLUX = 2.0

# Viscosity floor: zero viscosity would make the Reynolds ratio unbounded.
_VISCOSITY_FLOOR = 0.05

# Occupancy below this reads as "no coverage here" and takes full pressure.
_OCCUPANCY_FLOOR = 0.0

# Fallback graph length scale when no adjacency is available.
_UNIT_LENGTH = 1.0


def pressure_field(occupancy: Mapping) -> dict:
    """Pressure ``p(v)`` in ``[0, 1]``, high where coverage is scarce.

    ``p(v) = 1 - log1p(occ) / log1p(max_occ)`` — monotone decreasing in
    occupancy, bounded by construction, and log-scaled because ownership
    counts are heavy-tailed (the same axis change that fixed the edge ground
    metric).  Negative counts are clamped rather than trusted.

    Inputs are numbers the seed picker already computes (edge ownership,
    rarity, coverage density); no new instrumentation is required.
    """
    if not occupancy:
        return {}

    clamped = {k: max(float(v), _OCCUPANCY_FLOOR) for k, v in occupancy.items()}
    ceiling = math.log1p(max(clamped.values()))
    if ceiling <= 0.0:
        return dict.fromkeys(clamped, 1.0)

    return {k: 1.0 - math.log1p(v) / ceiling for k, v in clamped.items()}


def gradient_magnitude(pressure: Mapping, adjacency: Mapping) -> dict:
    """``|grad p|(v)``: the largest pressure drop to any known neighbour.

    Nodes with no adjacency get ``0.0`` — the common case, since the ICFG /
    horizon graph is only available when the target carries trace_pc
    instrumentation and the Katz channel builds.  A zero gradient degrades
    the flux ranking to pure resistance, which is the existing behaviour.
    """
    out = {}
    for node, p in pressure.items():
        neighbours = adjacency.get(node) or ()
        drops = [abs(p - pressure[n]) for n in neighbours if n in pressure]
        out[node] = max(drops) if drops else 0.0

    return out


def viscosity(failure_rate: float) -> float:
    """Damping from the recent operator failure rate, in ``(0, 1]``.

    High failure means the medium resists: unproductive lineages are damped
    harder.  Floored so :func:`reynolds` cannot divide by zero.
    """
    rate = min(max(float(failure_rate), 0.0), 1.0)
    return _VISCOSITY_FLOOR + (1.0 - _VISCOSITY_FLOOR) * rate


def reynolds(velocity: float, length: float, visc: float) -> float:
    """``U * L / nu`` — advective weight over diffusive weight on the graph.

    Read the module docstring before attaching fluid meaning to this.
    """
    nu = max(float(visc), _VISCOSITY_FLOOR)
    return max(float(velocity), 0.0) * max(float(length), 0.0) / nu


def flux(successes: float, failures: float, pressure_gradient: float, visc: float) -> float:
    """Predicted flux for one operator arm, in ``[0, MAX_FLUX]``.

    The continuum generalisation of invasion's ``1 / success_rate``
    resistance: the same success signal, driven by the local pressure
    gradient and dragged by viscosity.  An unobserved arm gets the maximum,
    matching the zero-resistance optimism ``_resistance`` already applies so
    that ranking by flux and ranking by resistance agree on unseen arms.
    """
    total = successes + failures
    if total <= 0:
        return MAX_FLUX

    rate = successes / total
    drive = 1.0 + min(max(float(pressure_gradient), 0.0), 1.0)
    drag = 1.0 + max(float(visc), 0.0)

    return min(MAX_FLUX, MAX_FLUX * rate * drive / drag)


@dataclass(frozen=True)
class ContinuumDiagnostics:
    """One steady snapshot.  Frozen: there is nothing to advance."""

    pressure_gradient: float
    velocity: float
    viscosity: float
    reynolds: float


class ContinuumField:
    """Steady pressure / flux view of the current frontier.

    Recomputed from the caller's scalars on each :meth:`observe`; holds no
    integrator and no history, so repeated observation of the same state is
    idempotent.
    """

    def __init__(self) -> None:
        self._diag: ContinuumDiagnostics | None = None
        self._gradient: float = 0.0

    def observe(
        self,
        occupancy: Mapping,
        adjacency: Mapping,
        velocity: float,
        failure_rate: float,
    ) -> None:
        """Recompute the fields from the current frontier scalars.

        Args:
            occupancy: node -> coverage occupancy (ownership count, hits).
            adjacency: node -> neighbour nodes; empty when no graph exists.
            velocity: discovery flux magnitude, e.g. new edges per tick.
            failure_rate: recent operator failure fraction, in ``[0, 1]``.
        """
        pressure = pressure_field(occupancy)
        grads = gradient_magnitude(pressure, adjacency)
        self._gradient = sum(grads.values()) / len(grads) if grads else 0.0

        visc = viscosity(failure_rate)
        length = _length_scale(adjacency)

        self._diag = ContinuumDiagnostics(
            pressure_gradient=self._gradient,
            velocity=max(float(velocity), 0.0),
            viscosity=visc,
            reynolds=reynolds(velocity, length, visc),
        )

    def flux_map(self, operator_stats: Mapping) -> dict:
        """name -> predicted flux, for the shape ``bandit_stats()`` returns.

        Empty before the first :meth:`observe`, which keeps callers on their
        existing resistance path until a field actually exists.
        """
        if self._diag is None:
            return {}

        return {
            op: flux(s, f, self._gradient, self._diag.viscosity)
            for op, (s, f) in operator_stats.items()
        }

    @property
    def diagnostics(self) -> ContinuumDiagnostics | None:
        return self._diag

    def reset(self) -> None:
        self._diag = None
        self._gradient = 0.0

    def save(self) -> dict:
        if self._diag is None:
            return {}
        return {
            "pressure_gradient": self._diag.pressure_gradient,
            "velocity": self._diag.velocity,
            "viscosity": self._diag.viscosity,
            "reynolds": self._diag.reynolds,
        }

    def load(self, data: dict) -> None:
        if not data:
            return

        self._gradient = data.get("pressure_gradient", 0.0)
        self._diag = ContinuumDiagnostics(
            pressure_gradient=self._gradient,
            velocity=data.get("velocity", 0.0),
            viscosity=data.get("viscosity", _VISCOSITY_FLOOR),
            reynolds=data.get("reynolds", 0.0),
        )


def _length_scale(adjacency: Mapping) -> float:
    """Mean degree of the frontier graph, in hops; 1.0 without a graph."""
    if not adjacency:
        return _UNIT_LENGTH

    degrees = [len(v or ()) for v in adjacency.values()]
    return max(sum(degrees) / len(degrees), _UNIT_LENGTH)
