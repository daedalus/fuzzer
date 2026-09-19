"""Kuramoto phase-oscillator diagnostics over the operator transition graph.

This is a standalone diagnostic utility, in the same spirit as
``centrality.py`` -- not wired into any scheduler, ``distance.py``, or the
CLI. It exists to answer one question quantitatively: if the fuzzer's
operators were modeled as coupled phase oscillators rather than as
independent Elo/bandit arms, would the transition graph the fuzzer
*already builds* (``op_katz.build_transition_matrix``) predict a
synchronization transition, and if so at what coupling strength?

Background (Kuramoto, 1975; network generalization: Restrepo, Ott & Hunt,
Phys. Rev. E 71, 036151, 2005): N phase oscillators θ_i, each with a
natural frequency ω_i, coupled through a weighted network A:

    dθ_i/dt = ω_i + (K/N) * sum_j A_ij * sin(θ_j - θ_i)

The collective state is summarized by the order parameter

    r * e^{iψ} = (1/N) * sum_i e^{iθ_i}

r=0 means the phases are spread incoherently around the circle; r=1 means
every oscillator has locked to the same phase. Below a critical coupling
K_c the population stays incoherent (r fluctuates near a small value that
shrinks as N grows); above K_c a macroscopic cluster locks together and r
rises sharply. This is a genuine second-order phase transition, with the
same critical-slowing-down precursor (rising variance/autocorrelation
just below K_c) that ``analyzer_critical_slowing.py`` already looks for
in the discovery-rate series, and the same "distinguish real structure
from noise near a threshold" problem ``corpus_flux.py`` solves for corpus
size.

Restrepo-Ott-Hunt's headline result (their Eq. for the eigenvalue
approximation) is that for a network with coupling matrix A and a
unimodal, symmetric natural-frequency density g(ω), the transition
occurs at

    K_c = K_0 / Λ_1,   K_0 = 2 / (π * g(0))

where Λ_1 is the largest eigenvalue (spectral radius) of A. That is
*exactly* the quantity ``op_katz.classical_katz_scores`` already computes
via ``np.linalg.eigvals`` to pick its own alpha -- this module reuses the
same computation for a different purpose (predicting a coupling
threshold instead of bounding a Neumann series).

Nothing here claims the fuzzer's operators actually behave like phase
oscillators; that would need an empirical test against a real campaign
(assign each operator a phase advancing at a rate tied to its firing
rate, couple through the discovery-transition graph, and check whether
the *measured* order parameter tracks a real regime the fuzzer cares
about -- e.g. "operators near-synchronizing" correlating with stalls or
with productive bursts). This module only provides the machinery
(stepping, order parameter, K_c estimate) needed to run that test.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Order parameter
# ---------------------------------------------------------------------------


def order_parameter(phases: np.ndarray) -> tuple[float, float]:
    """Kuramoto order parameter (r, ψ) for a set of phases (radians).

    r*e^{iψ} = mean_i e^{iθ_i}. r in [0, 1]: 0 = phases uniformly spread
    around the circle (incoherent), 1 = all phases identical (fully
    synchronized). ψ is the mean phase, in (-π, π]; meaningless when r is
    near 0 (no dominant direction) but returned regardless -- callers
    that only care about coherence should look at r alone.

    Empty input returns (0.0, 0.0) rather than raising, so a caller
    computing this once per tick over an operator pool that starts empty
    doesn't need a special-cased guard.
    """
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0, 0.0
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z))


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def kuramoto_step(
    phases: np.ndarray,
    omega: np.ndarray,
    coupling: np.ndarray,
    k: float,
    dt: float = 0.05,
) -> np.ndarray:
    """One explicit-Euler step of the networked Kuramoto ODE.

    dθ_i/dt = ω_i + (K/N) * sum_j coupling[i,j] * sin(θ_j - θ_i)

    ``coupling`` need not be symmetric or row-normalized -- it is used
    exactly as given, so a directed, weighted graph like
    ``op_katz.build_transition_matrix``'s output (row-normalized rates,
    not symmetric) plugs in directly rather than needing to be
    symmetrized first. N in the (K/N) normalization is ``len(phases)``,
    the standard Kuramoto convention that keeps K's critical value
    roughly independent of network size for a fixed mean coupling
    strength.

    Explicit Euler, not an adaptive integrator: this module is a
    diagnostic (does the transition graph predict synchronization at
    all?), not a physically precise simulator, and dt is exposed so a
    caller can shrink it if a specific coupling matrix's fastest mode
    needs it. Returned phases are not wrapped to (-π, π] -- unwrapped
    phase is what ``order_parameter`` needs anyway (it only ever
    consumes phases through sin/cos), and wrapping here would just be
    extra arithmetic every step for no behavioral difference.
    """
    phases = np.asarray(phases, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    n = phases.shape[0]
    if n == 0:
        return phases.copy()
    # theta_diff[i, j] = theta_j - theta_i
    theta_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling_term = (coupling * np.sin(theta_diff)).sum(axis=1) * (k / n)
    return phases + dt * (omega + coupling_term)


def simulate(
    phases0: np.ndarray,
    omega: np.ndarray,
    coupling: np.ndarray,
    k: float,
    steps: int,
    dt: float = 0.05,
) -> np.ndarray:
    """Run ``steps`` Euler steps, returning r(t) for t=0..steps inclusive.

    Convenience wrapper around :func:`kuramoto_step` for the common case
    of "does this network converge to a synchronized state at this K" --
    exactly what a caller needs to sanity-check a :func:`critical_coupling`
    estimate against the actual ODE rather than trusting the closed-form
    approximation blindly (see the module docstring's caveat about the
    approximation's own assumptions, e.g. large minimum degree, that a
    small or sparse operator graph won't satisfy).
    """
    phases = np.asarray(phases0, dtype=np.float64).copy()
    r_trace = np.empty(steps + 1, dtype=np.float64)
    r_trace[0], _ = order_parameter(phases)
    for t in range(1, steps + 1):
        phases = kuramoto_step(phases, omega, coupling, k, dt=dt)
        r_trace[t], _ = order_parameter(phases)
    return r_trace


# ---------------------------------------------------------------------------
# Critical coupling (Restrepo, Ott & Hunt 2005 eigenvalue approximation)
# ---------------------------------------------------------------------------


def spectral_radius(coupling: np.ndarray) -> float:
    """Largest-magnitude eigenvalue of ``coupling`` (Λ_1).

    Same computation ``op_katz.classical_katz_scores`` already performs
    on this exact matrix shape (``np.linalg.eigvals`` + ``max(abs(.))``)
    to pick its own alpha -- pulled out as its own function here because
    :func:`critical_coupling` needs it as a named, reusable quantity
    rather than an inline step of a different formula. Not imported from
    ``op_katz`` to avoid a diagnostic-utility-imports-scheduler
    dependency in either direction; if the two ever need to guarantee
    bit-identical results, this is the one to hoist into a shared spot.
    """
    coupling = np.asarray(coupling, dtype=np.float64)
    if coupling.size == 0:
        return 0.0
    eigvals = np.linalg.eigvals(coupling)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def frequency_density_at_zero(omega: np.ndarray) -> float:
    """Gaussian-kernel density estimate of g(0) for natural frequencies ω.

    Restrepo-Ott-Hunt's K_c formula assumes a unimodal, symmetric density
    g(ω) and only needs its value at ω=0 (i.e. at the population's own
    mean, since ω is conventionally measured relative to the mean
    frequency -- callers should center ``omega`` themselves if their
    frequencies aren't already mean-zero; this function does not
    recenter for them, since silently shifting the caller's own
    frequency assignment is more likely to hide a bug than help).

    Silverman's rule of thumb picks the bandwidth: h = 1.06 * std * n^-1/5.
    A single-point KDE rather than a full density estimate because K_c
    only ever needs g(0), not the whole curve.

    Degenerate input (fewer than 2 points, or zero variance -- every
    oscillator given the identical frequency) has no well-defined
    density; returns 0.0 rather than raising or dividing by zero, which
    callers should read as "g(0) undefined from this input" rather than
    "the density is genuinely zero here."
    """
    omega = np.asarray(omega, dtype=np.float64)
    n = omega.shape[0]
    if n < 2:
        return 0.0
    std = float(np.std(omega))
    if std < 1e-12:
        return 0.0
    h = 1.06 * std * n ** (-1 / 5)
    kernel = np.exp(-0.5 * (omega / h) ** 2) / np.sqrt(2 * np.pi)
    return float(np.sum(kernel) / (n * h))


def critical_coupling(coupling: np.ndarray, omega: np.ndarray) -> float:
    """Restrepo-Ott-Hunt eigenvalue estimate of the synchronization threshold.

    K_c = K_0 / Λ_1, K_0 = 2 / (π * g(0)), Λ_1 = spectral_radius(coupling).

    Returns ``float("inf")`` when either factor needed to form a finite
    K_c is unavailable: a coupling matrix with spectral radius 0 (no
    cycles/no edges -- see :func:`spectral_radius`; Λ_1=0 would divide by
    zero) can never synchronize any population regardless of K, and a
    degenerate frequency density (see :func:`frequency_density_at_zero`)
    means the approximation's own precondition (unimodal g(ω) with a
    defined g(0)) doesn't hold, so no finite estimate is defensible.
    Both cases mean "this network/frequency combination doesn't fit the
    approximation," which is meaningfully different from "K_c is small"
    -- collapsing them to e.g. 0.0 would read backwards to a caller
    checking ``k > critical_coupling(...)``.

    This is an approximation with real preconditions the caller is
    responsible for judging (see the module docstring): large minimum
    degree, unimodal symmetric g(ω), and (per Restrepo-Ott-Hunt's own
    finite-size-effects discussion) low-degree nodes push the *true*
    threshold above this estimate. Small or sparse graphs -- likely for
    an operator pool of only a few dozen operators -- are exactly the
    regime the source paper flags as where the approximation degrades;
    treat the return value as a starting point for :func:`simulate`, not
    a verified threshold.
    """
    rho = spectral_radius(coupling)
    if rho <= 1e-12:
        return float("inf")
    g0 = frequency_density_at_zero(omega)
    if g0 <= 0.0:
        return float("inf")
    k0 = 2.0 / (np.pi * g0)
    return float(k0 / rho)
