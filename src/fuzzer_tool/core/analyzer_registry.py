"""Analyzer dispatcher: single source of truth for pluggable analysis components.

Mirrors ``core.operator_registry`` (the mutation-operator dispatcher) but for
the fuzzer's detector/estimator components -- transfer entropy, crash-outcome
mutual information, length/edge correlation, work-functional fluctuation
tracking, structure-function stall detection, and so on. Historically each of
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

Migration status: complete. All ~20 analyzer/detector components previously
wired inline in Fuzzer.__init__ are registered here -- fluctuation,
transfer_entropy, crash_mi, length_tracker, structure_function, sensitivity,
execution_time, exec_time_anomaly, frameshift, format_learner,
corpus_compression, elo, distance, trace, checksum_learner, csd,
coverage_homogeneity, garch, continuum, coverage_regime. The one deliberate
exception is `kalman` (core.kalman.RobustKF): every usage found is embedded
in something else's construction (a network-adapter settle-time smoother; a
separate, unconditional filter-smoothing usage in services/stats.py), not a
standalone analyzer with its own gating flag, so there's no good single
migration site. Two later, non-migration additions follow the same pattern:
`occupation` (core.occupation.LongitudinalRarity, finite-time edge-count
occupation) and `causal_sector` (core.causal_sector.CausalSectorGraph,
soft-requires `transfer_entropy`) -- see
docs/handover/handover_RoRd.md. A third, `discovery_uniformity`
(core.discovery_uniformity.DiscoveryUniformityDetector), is the
nonparametric member of the regime_detection family: a rolling Poisson
index-of-dispersion test of per-tick discovery counts, sitting next
to `garch`/`structure_function`/`csd` without sharing any of their parametric noise-model
assumptions.
construction site to migrate. See
docs/handover/handover_analyzer_registry_2026-09-07.md for the full
per-component history and verification notes.
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
    # "main" (default) is the single wire_all() call near the top of
    # Fuzzer.__init__. "early" is for the rare analyzer with a real ordering
    # constraint on something later in __init__ (currently just
    # `sensitivity`, which must exist before _init_seed_metadata() so resume
    # can restore sensitivity.json) -- wired from its own earlier call.
    phase: str = "main"
    # True only for analyzers whose original inline construction was itself
    # wrapped in try/except (currently just `checksum_learner`). wire_all()
    # catches and logs instead of propagating, exactly reproducing that
    # analyzer's original fail-open behaviour -- this is NOT a general
    # error-swallowing switch; every other analyzer still fails Fuzzer()
    # the way its inline predecessor would have.
    swallow_errors: bool = False


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

    def wire_all(self, fuzzer: FuzzerLike, phase: str = "main") -> dict[str, bool]:
        """Construct every available *phase*-matching analyzer, in order.

        Returns ``{name: True}`` for each analyzer that was activated and
        ``{name: False}`` for each that was left at its off-default --
        useful for tests and for a one-line startup summary. Construction
        errors propagate (a broken analyzer fails ``Fuzzer()`` the same way
        its inline predecessor would have) unless the spec sets
        ``swallow_errors=True``, in which case they're logged and treated
        as "not activated" instead.
        """
        activated: dict[str, bool] = {}
        for name, spec in self._specs.items():
            if spec.phase != phase:
                continue
            is_available = spec.available(fuzzer) if spec.available is not None else True
            if is_available and spec.swallow_errors:
                try:
                    spec.activate(fuzzer)
                except Exception as exc:  # noqa: BLE001 - mirrors the original inline try/except
                    log.debug("%s init failed: %s", name, exc)
                    is_available = False
                    if spec.deactivate is not None:
                        spec.deactivate(fuzzer)
            elif is_available:
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


def _activate_occupation(f: FuzzerLike) -> None:
    from fuzzer_tool.core.occupation import LongitudinalRarity

    f._occupation_rarity = LongitudinalRarity()
    f._last_occupation = None
    f._occupation_max_edges = 256
    log.info("Occupation tracking enabled")


def _deactivate_occupation(f: FuzzerLike) -> None:
    f._occupation_rarity = None
    f._last_occupation = None


REGISTRY.register(
    AnalyzerSpec(
        name="occupation",
        category="coverage",
        available=lambda f: bool(getattr(f, "_use_occupation", False)),
        activate=_activate_occupation,
        deactivate=_deactivate_occupation,
    )
)


def _activate_causal_sector(f: FuzzerLike) -> None:
    from fuzzer_tool.core.causal_sector import CausalSectorGraph

    f._causal_sector = CausalSectorGraph()
    log.info("Causal-sector graph enabled")


def _deactivate_causal_sector(f: FuzzerLike) -> None:
    f._causal_sector = None


# Registered after transfer_entropy so that within one wire_all() pass, when
# both flags are on, f._te already exists by the time this activate() runs
# (same registration-order dependency pattern as the coverage_regime cluster
# below). `available` soft-requires transfer_entropy: a causal-sector graph
# with no TE source to feed it would just sit empty.
REGISTRY.register(
    AnalyzerSpec(
        name="causal_sector",
        category="mutual_information",
        available=lambda f: (
            bool(getattr(f, "_use_causal_sector", False))
            and bool(getattr(f, "_use_transfer_entropy", False))
        ),
        activate=_activate_causal_sector,
        deactivate=_deactivate_causal_sector,
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


def _activate_structure_function(f: FuzzerLike) -> None:
    # Local import of the fuzzer module (not a top-level import) to avoid
    # the circular import that a top-level one would create: services.fuzzer
    # imports this registry module at load time.
    from fuzzer_tool.core.structure_function import StructureFunctionDetector
    from fuzzer_tool.services import fuzzer as _fuzzer_mod

    f._structure_fn = StructureFunctionDetector(
        max_buffer_pow=_fuzzer_mod.STRUCTURE_BUFFER_POW,
        min_samples=_fuzzer_mod.STRUCTURE_MIN_SAMPLES,
    )
    f._last_structure_edge_count = 0


REGISTRY.register(
    AnalyzerSpec(
        name="structure_function",
        category="regime_detection",
        activate=_activate_structure_function,
    )
)


def _activate_discovery_uniformity(f: FuzzerLike) -> None:
    from fuzzer_tool.core.discovery_uniformity import DiscoveryUniformityDetector

    f._discovery_uniformity = DiscoveryUniformityDetector()


REGISTRY.register(
    AnalyzerSpec(
        name="discovery_uniformity",
        category="regime_detection",
        activate=_activate_discovery_uniformity,
    )
)


def _activate_sensitivity(f: FuzzerLike) -> None:
    from fuzzer_tool.core.sensitivity import ByteSensitivityTracker

    f._sensitivity = ByteSensitivityTracker(max_seeds=50, max_bytes=f.max_len, sample_rate=0.02)


REGISTRY.register(
    AnalyzerSpec(
        name="sensitivity",
        category="mutation_feedback",
        activate=_activate_sensitivity,
        # Constructed before Fuzzer._init_seed_metadata() so a resumed run
        # can restore sensitivity.json -- it raised AttributeError
        # otherwise. Its own usage is separately gated at call time by the
        # `_use_sensitivity` flag (set directly in __init__, not through
        # this registry); construction itself is unconditional.
        phase="early",
    )
)


def _activate_execution_time(f: FuzzerLike) -> None:
    from fuzzer_tool.core.execution_time import ExecutionTimeTracker

    f._exec_time_tracker = ExecutionTimeTracker()


REGISTRY.register(
    AnalyzerSpec(
        name="execution_time",
        category="timing",
        activate=_activate_execution_time,
    )
)


def _activate_exec_time_anomaly(f: FuzzerLike) -> None:
    from fuzzer_tool.core.exec_time_anomaly import ExecTimeCalibrator

    f._exec_time_anomaly = ExecTimeCalibrator()


REGISTRY.register(
    AnalyzerSpec(
        name="exec_time_anomaly",
        category="timing",
        activate=_activate_exec_time_anomaly,
    )
)


def _activate_frameshift(f: FuzzerLike) -> None:
    from fuzzer_tool.core.frameshift import FrameShift

    f._frameshift = FrameShift(max_relations=64)


REGISTRY.register(
    AnalyzerSpec(
        name="frameshift",
        category="structural",
        activate=_activate_frameshift,
    )
)


def _activate_format_learner(f: FuzzerLike) -> None:
    from fuzzer_tool.core.format_learner import FormatLearner

    f._format_learner = FormatLearner(max_timeline=10000)


def _deactivate_format_learner(f: FuzzerLike) -> None:
    f._format_learner = None


REGISTRY.register(
    AnalyzerSpec(
        name="format_learner",
        category="structural",
        available=lambda f: bool(getattr(f, "_learn_format_requested", False)),
        activate=_activate_format_learner,
        deactivate=_deactivate_format_learner,
    )
)


def _activate_corpus_compression(f: FuzzerLike) -> None:
    from fuzzer_tool.core.corpus_compression import CorpusCompressor

    f._ppmd = CorpusCompressor()


def _deactivate_corpus_compression(f: FuzzerLike) -> None:
    f._ppmd = None


REGISTRY.register(
    AnalyzerSpec(
        name="corpus_compression",
        category="structural",
        available=lambda f: bool(getattr(f, "_corpus_ppmd_requested", False)),
        activate=_activate_corpus_compression,
        deactivate=_deactivate_corpus_compression,
    )
)


def _activate_quasiperiodicity(f: FuzzerLike) -> None:
    from fuzzer_tool.core.quasiperiodicity import QuasiperiodicityAnalyzer

    f._qp = QuasiperiodicityAnalyzer()


def _deactivate_quasiperiodicity(f: FuzzerLike) -> None:
    f._qp = None


REGISTRY.register(
    AnalyzerSpec(
        name="quasiperiodicity",
        category="structural",
        # Opt-in like corpus_compression, pending an A/B run (see
        # docs/handover/handover_oeis_port_candidates_2026-09-12.md, C2) --
        # this is a new, unmeasured novelty signal and should not change
        # scheduling weights for anyone until it has one, same policy as
        # --tang and --continuum (handover_done_2026-09-06.md §14).
        available=lambda f: bool(getattr(f, "_corpus_quasiperiodicity_requested", False)),
        activate=_activate_quasiperiodicity,
        deactivate=_deactivate_quasiperiodicity,
    )
)


def _activate_elo(f: FuzzerLike) -> None:
    from fuzzer_tool.core.elo import BayesianEloTracker

    # Local import: services.fuzzer defines _OPERATOR_STRATEGY_NAMES /
    # _SEED_STRATEGY_NAMES and imports this registry module at load time.
    from fuzzer_tool.services import fuzzer as _fuzzer_mod

    f._elo = BayesianEloTracker(
        initial_mu=1500,
        initial_sigma=350,
        beta=200,
        tau=5.0,
        min_matches=10,
    )
    log.info("Elo rating system enabled (k=16, decay=0.99)")
    f._elo_decay_interval = 100
    f._elo_decay_counter = 0
    f._elo_match_window = []
    elo_data = f._state_store.get("elo")
    if elo_data is not None:
        f._elo.from_dict(elo_data)
        log.info("Elo tracker loaded from state store (%d operators)", len(f._elo.mu))

    # Pre-register all strategy names so Elo can arbitrate immediately
    # (without this, select_strategy requires min_matches before considering
    # a strategy).
    for s in _fuzzer_mod._OPERATOR_STRATEGY_NAMES:
        f._elo._strategy_mu.setdefault(s, f._elo.initial_mu)
        f._elo._strategy_sigma_sq.setdefault(s, f._elo.initial_sigma**2)
        f._elo._strategy_match_count.setdefault(s, 0)
    for s in _fuzzer_mod._SEED_STRATEGY_NAMES:
        key = f"seed_{s}"
        f._elo._strategy_mu.setdefault(key, f._elo.initial_mu)
        f._elo._strategy_sigma_sq.setdefault(key, f._elo.initial_sigma**2)
        f._elo._strategy_match_count.setdefault(key, 0)


def _deactivate_elo(f: FuzzerLike) -> None:
    f._elo = None


REGISTRY.register(
    AnalyzerSpec(
        name="elo",
        category="scheduling",
        available=lambda f: bool(getattr(f, "_use_elo", False)),
        activate=_activate_elo,
        deactivate=_deactivate_elo,
    )
)


def _activate_distance(f: FuzzerLike) -> None:
    import os

    from fuzzer_tool.core.distance import TargetDistance

    f._distance = TargetDistance(
        f.target, f._distance_targets, use_cfg_cache=f._use_cfg_cache, debug=f.debug
    )
    f._dist_table_shm = None
    if f._distance.load():
        print(
            f"[*] Directed mode: {len(f._distance.target_addrs)} target(s), "
            f"{len(f._distance.functions)} functions mapped"
        )
        if f._distance._bb_value:
            try:
                from fuzzer_tool.adapters.shm import DistanceTableShm

                # Keys are trace-pc call-site addresses relative to the
                # object base (__sancov_pcs); the shim looks up
                # pc - dladdr_base.
                table = f._distance.pc_distance_table()
                if not table:
                    base = f._distance._base_addr or 0
                    table = {
                        bb_start - base: dist for bb_start, dist in f._distance._bb_value.items()
                    }
                f._dist_table_shm = DistanceTableShm(table)
                if f._dist_table_shm.shm_id >= 0:
                    os.environ["__AFL_DIST_SHM_ID"] = f._dist_table_shm.env_id
                    print(
                        f"[*] AFLGo distance table: {len(table)} sites "
                        "uploaded (SHM-tail channel active)"
                    )
            except OSError as e:
                log.warning("Distance table upload failed: %s", e)
    else:
        print("[!] Directed mode: failed to load target distances, falling back to coverage")
        f._distance = None


def _deactivate_distance(f: FuzzerLike) -> None:
    f._distance = None
    f._dist_table_shm = None


REGISTRY.register(
    AnalyzerSpec(
        name="distance",
        category="directed_fuzzing",
        available=lambda f: bool(getattr(f, "_distance_targets", None)),
        activate=_activate_distance,
        deactivate=_deactivate_distance,
    )
)


def _activate_trace(f: FuzzerLike) -> None:
    from fuzzer_tool.core.trace import CrashTracer

    f._tracer = CrashTracer(f.target)


def _deactivate_trace(f: FuzzerLike) -> None:
    f._tracer = None


REGISTRY.register(
    AnalyzerSpec(
        name="trace",
        category="crash_triage",
        available=lambda f: bool(getattr(f, "_trace_crashes_requested", False)),
        activate=_activate_trace,
        deactivate=_deactivate_trace,
    )
)


def _activate_checksum_learner(f: FuzzerLike) -> None:
    from fuzzer_tool.core.checksum_learner import ChecksumLearner

    f.checksum_learner = ChecksumLearner(f)


def _deactivate_checksum_learner(f: FuzzerLike) -> None:
    f.checksum_learner = None


REGISTRY.register(
    AnalyzerSpec(
        name="checksum_learner",
        category="format_recovery",
        activate=_activate_checksum_learner,
        deactivate=_deactivate_checksum_learner,
        # The only analyzer whose original inline construction was itself
        # wrapped in try/except: recovers unknown linear checksum
        # polynomials via Berlekamp-Massey/GCD, which can fail on inputs
        # that don't fit that model. That was never meant to fail Fuzzer().
        swallow_errors=True,
    )
)


def _activate_csd(f: FuzzerLike) -> None:
    from fuzzer_tool.core.critical_slowing import CriticalSlowingDown

    f._csd = CriticalSlowingDown(window_size=50, rise_threshold=1.5, min_observations=20)


REGISTRY.register(
    AnalyzerSpec(
        name="csd",
        category="regime_detection",
        activate=_activate_csd,
    )
)


def _activate_coverage_homogeneity(f: FuzzerLike) -> None:
    from fuzzer_tool.core.critical_slowing import CoverageHomogeneityDetector

    num_cols = max(1, f.map_size // 8192)
    f._homogeneity = CoverageHomogeneityDetector(
        num_columns=num_cols,
        window_size=10,
        homogeneity_p_threshold=0.01,
    )
    f._homogeneity_col_cumulative = [0] * num_cols


REGISTRY.register(
    AnalyzerSpec(
        name="coverage_homogeneity",
        category="regime_detection",
        activate=_activate_coverage_homogeneity,
    )
)


def _activate_garch(f: FuzzerLike) -> None:
    from fuzzer_tool.core.garch import OnlineGarch11

    f._garch = OnlineGarch11()
    f._garch.load(f._state_store.get("garch") or {})


def _deactivate_garch(f: FuzzerLike) -> None:
    f._garch = None


REGISTRY.register(
    AnalyzerSpec(
        name="garch",
        category="regime_detection",
        available=lambda f: bool(getattr(f, "_use_garch", False)),
        activate=_activate_garch,
        deactivate=_deactivate_garch,
    )
)


def _activate_temperature_control(f: FuzzerLike) -> None:
    from fuzzer_tool.core.temperature_control import TemperatureController

    saved = f._state_store.get("temperature_control") or {}
    if saved:
        f._temp_controller = TemperatureController.from_dict(saved)
    else:
        f._temp_controller = TemperatureController(
            setpoint_fraction=getattr(f, "_temp_setpoint_fraction", 0.5),
            reference_rate=getattr(f, "_temp_reference_rate", None),
        )
    print(
        "[*] Temperature control: closed loop on discovery rate "
        f"(setpoint {f._temp_controller.setpoint_fraction:.0%} of reference, "
        f"period {f._temp_controller.period_execs:,} execs)"
    )


def _deactivate_temperature_control(f: FuzzerLike) -> None:
    f._temp_controller = None


REGISTRY.register(
    AnalyzerSpec(
        name="temperature_control",
        category="regime_detection",
        available=lambda f: bool(getattr(f, "_use_temp_control", False)),
        activate=_activate_temperature_control,
        deactivate=_deactivate_temperature_control,
    )
)


def _activate_continuum(f: FuzzerLike) -> None:
    from fuzzer_tool.core.navier_stokes import ContinuumField

    f._continuum = ContinuumField()
    f._continuum_adjacency = {}
    f._continuum_graph_tick = 0


def _deactivate_continuum(f: FuzzerLike) -> None:
    f._continuum = None
    f._continuum_adjacency = {}
    f._continuum_graph_tick = 0


REGISTRY.register(
    AnalyzerSpec(
        name="continuum",
        category="regime_detection",
        available=lambda f: bool(getattr(f, "_use_continuum", False)),
        activate=_activate_continuum,
        deactivate=_deactivate_continuum,
    )
)


def _activate_coverage_regime(f: FuzzerLike) -> None:
    from fuzzer_tool.core.coverage_regime import CoverageRegimeDetector

    # Composite: depends on csd / coverage_homogeneity / garch / continuum
    # having already run. Registration order guarantees that within one
    # wire_all() pass -- see the four specs directly above.
    f._regime = CoverageRegimeDetector(
        csd=f._csd,
        homogeneity=f._homogeneity,
        stall_threshold=f._stall_threshold,
        garch=f._garch,
        continuum=f._continuum,
    )


REGISTRY.register(
    AnalyzerSpec(
        name="coverage_regime",
        category="regime_detection",
        activate=_activate_coverage_regime,
    )
)
