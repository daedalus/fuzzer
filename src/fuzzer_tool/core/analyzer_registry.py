"""Analyzer dispatcher: single source of truth for pluggable analysis components.

Mirrors ``core.operator_registry`` (the mutation-operator dispatcher) but for
the fuzzer's detector/estimator components -- transfer entropy, crash-outcome
mutual information, length/edge correlation, work-functional fluctuation
tracking, Allan-variance stall detection, and so on. Historically each of
these was wired ad hoc, inline in ``Fuzzer.__init__``: a local ``from
fuzzer_tool.core.<x> import <Y>`` import, construction, an optional
state-store restore, and an optional log line, repeated with slightly
different shape at each of ~20 call sites. That made "does this analyzer get
constructed, and under what condition" a question you could only answer by
reading the whole constructor.

Every analyzer is now registered here exactly once as an :class:`AnalyzerSpec`:

- ``available(fuzzer)``  -- reads the gating flag already set on *fuzzer*
  (mirrors ``OperatorSpec.available``'s ``getattr(f, "enable_x", False)``
  style). ``None`` means always available.
- ``activate(fuzzer)``   -- constructs the analyzer, restores any saved
  state-store data, and does its one-time log/print. Called only when
  ``available`` passes.
- ``deactivate(fuzzer)`` -- sets the "off" defaults (usually ``None``).
  Called when ``available`` fails. Optional; defaults to a no-op, for
  analyzers that have no gating flag and are always constructed.

``REGISTRY.wire_all(fuzzer)`` runs every registered spec in registration
order and returns ``{name: activated_bool}`` for introspection/testing.
Call it once from ``Fuzzer.__init__``, after ``self._state_store`` and
``self.max_len`` are set (both are read by multiple factories) and after
each analyzer's own gating-flag attribute has been assigned.

Adding an analyzer means one ``REGISTRY.register(...)`` call here -- nothing
else. Fuzzer.__init__ should not import an analyzer module directly; if it
does, that analyzer belongs in this registry instead.

Migration status: this registry currently covers ``fluctuation``,
``transfer_entropy``, ``crash_mi``, ``length_tracker``, and ``allan`` -- the
cleanest, self-contained cluster (no cross-dependencies on other analyzers).
See docs/handover/handover_analyzer_registry_2026-09-07.md for the remaining
components still wired inline in ``Fuzzer.__init__`` (elo, distance, trace,
kalman, garch/critical_slowing/coverage_regime, navier_stokes,
execution_time, exec_time_anomaly, sensitivity, frameshift, format_learner,
corpus_compression, checksum_learner) and the plan to move each one here.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# The fuzzer instance is passed around duck-typed (mirrors operator_registry's
# `available: Callable[[object, bytes], bool]` predicates): analyzers reach
# into whatever attributes they need (``_state_store``, ``max_len``, their own
# gating flag, ...), so ``Any`` is the honest annotation here, not a laxness
# shortcut -- a narrower type would either be `object` (too narrow to use) or
# a `Fuzzer` import that would make this module depend on the very class that
# depends on it.
FuzzerLike = Any


@dataclass(frozen=True)
class AnalyzerSpec:
    """Static registration metadata for one analyzer component."""

    name: str
    category: str
    activate: Callable[[FuzzerLike], None]
    available: Callable[[FuzzerLike], bool] | None = None
    deactivate: Callable[[FuzzerLike], None] | None = None


class AnalyzerRegistry:
    """Registry/dispatcher of analyzer components (single source of truth)."""

    def __init__(self) -> None:
        self._specs: dict[str, AnalyzerSpec] = {}

    def register(self, spec: AnalyzerSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate analyzer registration: {spec.name!r}")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        """All registered analyzer names, in registration order."""
        return list(self._specs)

    def categories(self) -> dict[str, set[str]]:
        cats: dict[str, set[str]] = {}
        for spec in self._specs.values():
            cats.setdefault(spec.category, set()).add(spec.name)
        return cats

    def wire_all(self, fuzzer: FuzzerLike) -> dict[str, bool]:
        """Construct every available analyzer on *fuzzer*, in order.

        Returns ``{name: True}`` for each analyzer that was activated and
        ``{name: False}`` for each that was left at its off-default --
        useful for tests and for a one-line startup summary. Construction
        errors are not swallowed here: a broken analyzer should fail
        ``Fuzzer()`` the same way its inline predecessor would have.
        """
        activated: dict[str, bool] = {}
        for name, spec in self._specs.items():
            is_available = spec.available(fuzzer) if spec.available is not None else True
            if is_available:
                spec.activate(fuzzer)
            elif spec.deactivate is not None:
                spec.deactivate(fuzzer)
            activated[name] = is_available
        return activated


REGISTRY = AnalyzerRegistry()


def _activate_fluctuation(f: FuzzerLike) -> None:
    from fuzzer_tool.core.fluctuation import WorkFunctional

    f._fluctuation = WorkFunctional(beta=f._fluctuation_beta, window=f._fluctuation_window)
    data = f._state_store.get("fluctuation")
    if data is not None:
        f._fluctuation.restore(data)
        print(
            f"[*] Fluctuation tracker loaded (beta={f._fluctuation_beta}, "
            f"samples={sum(len(v) for v in f._fluctuation._states.values())})"
        )


def _deactivate_fluctuation(f: FuzzerLike) -> None:
    f._fluctuation = None


REGISTRY.register(
    AnalyzerSpec(
        name="fluctuation",
        category="regime_detection",
        available=lambda f: bool(getattr(f, "_fluctuation_requested", False)),
        activate=_activate_fluctuation,
        deactivate=_deactivate_fluctuation,
    )
)


def _activate_transfer_entropy(f: FuzzerLike) -> None:
    from fuzzer_tool.core.transfer_entropy import TransferEntropy

    f._te = TransferEntropy(history_length=1)
    f._te_input_history = []
    f._te_edge_history = []
    f._te_history_max = 500
    log.info("Transfer entropy tracking enabled")


def _deactivate_transfer_entropy(f: FuzzerLike) -> None:
    f._te = None


REGISTRY.register(
    AnalyzerSpec(
        name="transfer_entropy",
        category="mutual_information",
        available=lambda f: bool(getattr(f, "_use_transfer_entropy", False)),
        activate=_activate_transfer_entropy,
        deactivate=_deactivate_transfer_entropy,
    )
)


def _activate_crash_mi(f: FuzzerLike) -> None:
    from fuzzer_tool.core.crash_eta import CrashMITracker

    f._crash_mi = CrashMITracker(max_positions=f.max_len, min_observations=20)
    crash_mi_data = f._state_store.get("crash_mi")
    if crash_mi_data is not None:
        f._crash_mi.load(crash_mi_data)
        log.info(
            "Crash MI tracker loaded: %d execs, %d crashes",
            f._crash_mi.total_execs,
            f._crash_mi.total_crashes,
        )


REGISTRY.register(
    AnalyzerSpec(
        name="crash_mi",
        category="mutual_information",
        activate=_activate_crash_mi,
    )
)


def _activate_length_tracker(f: FuzzerLike) -> None:
    from fuzzer_tool.core.length_mi import LengthEdgeTracker

    # core.length_mi is on the mypy strictness-exemption ratchet list
    # (untyped legacy module); ignore the resulting no-untyped-call here
    # rather than exempting this module too.
    f._length_tracker = LengthEdgeTracker()  # type: ignore[no-untyped-call]
    lt_data = f._state_store.get("length_tracker")
    if lt_data is not None:
        f._length_tracker.load(lt_data)
        log.info("Length-edge tracker loaded: %d execs", f._length_tracker.total_execs)


REGISTRY.register(
    AnalyzerSpec(
        name="length_tracker",
        category="mutual_information",
        activate=_activate_length_tracker,
    )
)


def _activate_allan(f: FuzzerLike) -> None:
    # Local import of the fuzzer module (not a top-level import) to avoid
    # the circular import that a top-level one would create: services.fuzzer
    # imports this registry module at load time.
    from fuzzer_tool.core.allan_variance import AllanVarianceDetector
    from fuzzer_tool.services import fuzzer as _fuzzer_mod

    f._allan = AllanVarianceDetector(
        max_buffer_pow=_fuzzer_mod.ALLAN_BUFFER_POW,
        min_samples=_fuzzer_mod.ALLAN_MIN_SAMPLES,
    )
    f._last_allan_edge_count = 0


REGISTRY.register(
    AnalyzerSpec(
        name="allan",
        category="regime_detection",
        activate=_activate_allan,
    )
)
