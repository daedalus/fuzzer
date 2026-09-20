"""OpKuramotoScheduler: phase-coherence bandit over the operator pool.

``core/kuramoto.py`` is deliberately a standalone diagnostic (see its module
docstring and ``docs/handover/handover_kuramoto_oscillators_2026-09-19.md``):
it provides the order parameter, the stepping ODE, and the Restrepo-Ott-Hunt
critical-coupling estimate, but takes no position on what an operator's
"phase" or "natural frequency" should be, because that is an empirical
question the diagnostic module itself says nobody has answered yet. This
module makes that choice, explicitly and with the same discipline every
other unproven exploratory arm in this package carries (``op_katz``,
``op_tang``, ``WhittleIndexScheduler``): off by default, Elo-only, absent
from ``_FALLBACK_PRECEDENCE``, and honest in its own docstring about which
parts are measured and which are a documented guess.

Model
-----
Every operator ``i`` gets a phase ``theta_i`` on the circle and a natural
frequency ``omega_i``. Two choices had no existing answer in the repo and
are made here:

- **Natural frequency = the operator's own raw success rate**, scaled by
  ``omega_scale``. This reuses ``op_katz.py``'s beta convention exactly
  (``op_katz``'s module docstring: beta is the op's raw success rate, not
  its complement, because operator selection wants to exploit a working
  arm rather than seek an unexplored one the way seed selection does) --
  a more successful operator spins faster. An operator with no track
  record yet gets omega=0 (free rotation at its initial random phase
  until it has outcomes to derive a rate from).
- **Firing an operator does not perturb its phase directly.** Phases
  advance only through the shared Kuramoto ODE (:func:`kuramoto_step`),
  batched the same way ``WhittleIndexScheduler``/``op_tang`` batch their
  own expensive recomputation (``recompute_batch`` / ``op_tang_refit_interval``):
  running a stepping update on every tracked arm after every single
  ``record()`` call is O(pool size) work per outcome for no benefit over
  doing it every ``recompute_batch`` calls instead.

**Coupling** is the exact object ``op_katz.build_transition_matrix`` already
builds from this scheduler's own ``transition_counts`` (discovery-linked
transitions: an edge is recorded prev_op -> op only when ``op``'s own call
succeeds, same semantics as ``OpKatzScheduler.record`` and
``MonteCarloScheduler.transition_counts``) -- reused directly rather than
reimplemented, since this is exactly the "run the actual experiment"
follow-up ``handover_kuramoto_oscillators_2026-09-19.md`` suggested (its
own ``test_kuramoto.py`` already builds a matrix this way to test
``critical_coupling`` against a hand-computed value on this exact object).

**Selection score** for an operator is its own success rate amplified by
how phase-aligned it is with the population's coherent cluster:

    score_i = rate_i * (1 + r * cos(theta_i - psi))

where ``(r, psi)`` is the population's order parameter restricted to the
offered ops. This is deliberately the same "own rate, amplified by a
structural signal" frame ``OpKatzScheduler.scores`` uses (rate amplified by
centrality there, by phase coherence here) rather than a different scoring
philosophy invented from scratch. The ``r *`` factor is not cosmetic: per
``core/kuramoto.py``'s own ``order_parameter`` docstring, ``psi`` is
"meaningless when r is near 0" (no dominant phase direction to be aligned
with), so multiplying the alignment term by ``r`` makes it vanish exactly
when the diagnostic module says it should, degrading cleanly to plain
exploitation-by-rate for an incoherent or cold-start population instead of
chasing a meaningless angle.

``k`` (coupling strength) is a plain constructor argument, not
auto-derived from :func:`critical_coupling`. That estimate is exposed only
through :meth:`diagnostics` for inspection -- ``core/kuramoto.py`` already
flags the eigenvalue approximation as degrading on small/sparse graphs
(likely true of an operator pool of a few dozen arms), so silently driving
selection off a number the diagnostic module itself distrusts would repeat
the exact kind of unverified-paper-fidelity mistake documented in
``docs/handover/handover_kl_ducb_paper_fidelity_2026-09-14.md``.

What this does *not* claim
---------------------------
Nothing here claims operators actually behave like phase oscillators, or
that phase-alignment is a better exploitation signal than plain rate. That
is exactly the empirical question the diagnostic module's handover left
open (their "suggested next steps" 1-2): does a real campaign's transition
graph actually predict anything resembling synchronization, and does that
correlate with anything the fuzzer cares about? This scheduler makes the
question answerable by an A/B run (``tools/bench_paired.py``,
``tests/support/bandit_env.py``'s convergence harness) rather than
answering it here. Until that run happens this is exactly as unproven as
``op_katz``/``op_tang``/``WhittleIndexScheduler`` were before their own
harness runs -- reach it only through Elo explicitly choosing it.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.kuramoto import (
    critical_coupling,
    kuramoto_step,
    order_parameter,
    spectral_radius,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_katz import build_transition_matrix

DEFAULT_K = 1.0
DEFAULT_OMEGA_SCALE = 1.0
DEFAULT_DT = 0.05
DEFAULT_STEPS_PER_BATCH = 5
DEFAULT_RECOMPUTE_BATCH = 25


class OpKuramotoScheduler:
    """Elo-arm operator scheduler: Kuramoto phase coherence over discovery transitions.

    Lifecycle mirrors ``OpKatzScheduler``/``WhittleIndexScheduler``: the
    fuzzer holds one instance, calls :meth:`record` on every outcome, and
    :meth:`select_op` picks from the offered arm list.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16). Required, like
            ``OpKatzScheduler`` -- raises rather than silently defaulting.
        k: Kuramoto coupling strength fed to :func:`kuramoto_step`. Not
            auto-derived from :func:`critical_coupling` -- see module
            docstring.
        omega_scale: Natural frequency = ``omega_scale * success_rate``.
        dt: Euler step size (default matches ``core/kuramoto.py``'s own
            default).
        steps_per_batch: Euler steps run per batched advance.
        recompute_batch: Advance phases only after this many ``record()``
            calls, batched for the same cost reason
            ``WhittleIndexScheduler``/``op_tang`` batch their own
            recomputation (see module docstring).
    """

    supports_priors = False

    def __init__(
        self,
        rng: RandPool | None = None,
        k: float = DEFAULT_K,
        omega_scale: float = DEFAULT_OMEGA_SCALE,
        dt: float = DEFAULT_DT,
        steps_per_batch: int = DEFAULT_STEPS_PER_BATCH,
        recompute_batch: int = DEFAULT_RECOMPUTE_BATCH,
    ):
        if rng is None:
            raise ValueError("OpKuramotoScheduler requires a RandPool (Hard Rule 16)")
        if steps_per_batch < 1:
            raise ValueError(f"steps_per_batch must be >= 1, got {steps_per_batch!r}")
        if recompute_batch < 1:
            raise ValueError(f"recompute_batch must be >= 1, got {recompute_batch!r}")

        self._rng = rng
        self.k = k
        self.omega_scale = omega_scale
        self.dt = dt
        self.steps_per_batch = steps_per_batch
        self.recompute_batch = recompute_batch

        self.transition_counts: dict[str, dict[str, int]] = {}
        self.successes: dict[str, float] = {}
        self.attempts: dict[str, float] = {}
        self._phase: dict[str, float] = {}
        self._prev_op: str | None = None
        self._pending = 0

    def init_arm(self, name: str) -> None:
        """Register an operator at a uniform-random phase (standard Kuramoto
        initial condition) with no track record yet."""
        if name in self._phase:
            return
        self._phase[name] = self._rng.random() * 2.0 * math.pi
        self.attempts.setdefault(name, 0.0)
        self.successes.setdefault(name, 0.0)

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """Record one outcome. Mirrors ``OpKatzScheduler.record``'s
        ``record(op, success, weight=...)`` contract and its discovery-linked
        transition semantics exactly (see that method's docstring); phases
        themselves are not touched here, only batched via :meth:`_maybe_advance`.
        """
        self.init_arm(op)
        self.attempts[op] = self.attempts.get(op, 0.0) + 1.0
        if success:
            self.successes[op] = self.successes.get(op, 0.0) + max(weight, 0.0)
            if self._prev_op is not None and self._prev_op != op:
                row = self.transition_counts.setdefault(self._prev_op, {})
                row[op] = row.get(op, 0) + 1
        self._prev_op = op
        self._pending += 1

    def _rate(self, op: str) -> float:
        attempts = self.attempts.get(op, 0.0)
        return self.successes.get(op, 0.0) / attempts if attempts > 0 else 0.0

    def _maybe_advance(self, ops: list[str]) -> None:
        """Batched Euler advance of every tracked op's phase.

        Runs at most once every ``recompute_batch`` ``record()`` calls,
        over every op tracked so far (not just the ``ops`` currently
        offered) so an op's phase keeps evolving even while it is
        temporarily out of the offered set -- the same reasoning
        ``WhittleIndexScheduler._drift_idle_arms`` uses to advance idle
        arms rather than freezing them.
        """
        for op in ops:
            self.init_arm(op)
        if self._pending < self.recompute_batch:
            return
        self._pending = 0

        all_ops = list(self._phase.keys())
        if len(all_ops) < 2:
            return
        phases = np.array([self._phase[op] for op in all_ops], dtype=np.float64)
        omega = np.array([self._rate(op) * self.omega_scale for op in all_ops], dtype=np.float64)
        coupling = build_transition_matrix(self.transition_counts, all_ops)
        for _ in range(self.steps_per_batch):
            phases = kuramoto_step(phases, omega, coupling, self.k, dt=self.dt)
        # Keep phases in a bounded range so long runs don't accumulate an
        # unbounded float; order_parameter only ever consumes them through
        # sin/cos so wrapping here changes no computed quantity.
        phases = np.mod(phases, 2.0 * math.pi)
        for op, theta in zip(all_ops, phases.tolist(), strict=True):
            self._phase[op] = theta

    def scores(self, ops: list[str]) -> dict[str, float]:
        """Per-op score: own rate amplified by phase-alignment with the
        offered set's coherent cluster. See module docstring for the
        ``rate * (1 + r * cos(theta - psi))`` formula and why the ``r``
        factor is required, not optional.
        """
        self._maybe_advance(ops)
        phases = np.array([self._phase[op] for op in ops], dtype=np.float64)
        r, psi = order_parameter(phases)
        scores = {}
        for op, theta in zip(ops, phases.tolist(), strict=True):
            rate = self._rate(op)
            scores[op] = rate * (1.0 + r * math.cos(theta - psi))
        return scores

    def select_op(self, ops: list[str]) -> str:
        """Weighted pick by score, same non-negative-shift draw
        ``OpKatzScheduler.select_op`` uses (see that method's docstring) --
        kept identical so an unseen or all-zero-score offered set degrades
        to a uniform draw instead of a divide-by-zero or a fixed pick.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            self.init_arm(ops[0])
            return ops[0]
        s = self.scores(ops)
        vals = np.array([s[op] for op in ops], dtype=np.float64)
        shifted = vals - vals.min() + 1e-9
        total = float(shifted.sum())
        probs = shifted / total if total > 0 else np.full(len(ops), 1.0 / len(ops))
        r = self._rng.random()
        cumulative = 0.0
        for op, p in zip(ops, probs.tolist(), strict=True):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]

    def diagnostics(self) -> dict:
        """Inspection surface: order parameter, per-op phase/rate, and the
        Restrepo-Ott-Hunt critical-coupling estimate against the current
        transition graph -- exposed for the A/B run the module docstring
        says is still outstanding, never consumed by :meth:`select_op`
        itself (see module docstring for why ``critical_coupling`` must
        not silently drive behavior).
        """
        all_ops = list(self._phase.keys())
        if not all_ops:
            return {
                "r": 0.0,
                "psi": 0.0,
                "n_arms": 0,
                "critical_coupling": float("inf"),
                "spectral_radius": 0.0,
                "phases": {},
                "rates": {},
            }
        phases = np.array([self._phase[op] for op in all_ops], dtype=np.float64)
        r, psi = order_parameter(phases)
        omega = np.array([self._rate(op) * self.omega_scale for op in all_ops], dtype=np.float64)
        coupling = build_transition_matrix(self.transition_counts, all_ops)
        # critical_coupling's K_c formula measures frequencies relative to
        # the population mean (see core/kuramoto.py's frequency_density_at_zero
        # docstring: "callers should center omega themselves"); the omega
        # that actually drives the ODE above is left uncentered, since
        # only this diagnostic's K_c estimate needs the recentered version.
        centered_omega = omega - float(np.mean(omega))
        return {
            "r": r,
            "psi": psi,
            "n_arms": len(all_ops),
            "critical_coupling": critical_coupling(coupling, centered_omega),
            "spectral_radius": spectral_radius(coupling),
            "phases": dict(zip(all_ops, phases.tolist(), strict=True)),
            "rates": {op: self._rate(op) for op in all_ops},
        }
