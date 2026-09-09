"""Regression tests: analyzer registry is the single source of truth.

Mirrors ``test_regression_operator_registry.py`` for ``core.operator_registry``.
Every analyzer that used to be wired ad hoc, inline in ``Fuzzer.__init__``, is
now constructed through ``core.analyzer_registry.REGISTRY.wire_all()``.
Guards against the failure modes that matter here: an analyzer silently not
constructed when its flag is on, an analyzer's off-path leaving a stale/wrong
default, the "early" phase (sensitivity) actually running before
_init_seed_metadata, the coverage_regime composite's dependency ordering, the
checksum_learner swallow_errors path, and the registry's own
duplicate-registration guard.
"""

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.core.analyzer_registry import REGISTRY, AnalyzerRegistry, AnalyzerSpec

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")

_ALWAYS_ON = {
    "crash_mi",
    "length_tracker",
    "allan",
    "sensitivity",
    "execution_time",
    "exec_time_anomaly",
    "frameshift",
    "csd",
    "coverage_homogeneity",
    "coverage_regime",
}
_FLAG_GATED = {
    "fluctuation",
    "transfer_entropy",
    "occupation",
    "causal_sector",
    "format_learner",
    "corpus_compression",
    "elo",
    "distance",
    "trace",
    "garch",
    "continuum",
}
_ALL_NAMES = _ALWAYS_ON | _FLAG_GATED | {"checksum_learner"}


def _build_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.TemporaryDirectory()
    corpus = Path(tmp.name) / "corpus"
    crashes = Path(tmp.name) / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    kwargs.setdefault("max_len", 4096)
    f = Fuzzer(
        target=_TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        **kwargs,
    )
    f._test_tmp = tmp  # keep the tempdir alive for the caller's lifetime
    return f


class TestRegistryContents:
    def test_every_migrated_analyzer_registered(self):
        assert set(REGISTRY.names()) == _ALL_NAMES

    def test_duplicate_registration_rejected(self):
        reg = AnalyzerRegistry()
        spec = AnalyzerSpec(name="x", category="c", activate=lambda f: None)
        reg.register(spec)
        with pytest.raises(ValueError, match="duplicate analyzer registration"):
            reg.register(spec)

    def test_categories_partition_every_registered_name(self):
        cats = REGISTRY.categories()
        seen = set()
        for names in cats.values():
            seen |= names
        assert seen == set(REGISTRY.names())

    def test_coverage_regime_cluster_registered_in_dependency_order(self):
        # coverage_regime's activate() reads f._csd / f._homogeneity /
        # f._garch / f._continuum, so those four must run first within the
        # same wire_all() pass -- registration order is what guarantees
        # that (see the module docstring on coverage_regime's spec).
        names = REGISTRY.names()
        regime_idx = names.index("coverage_regime")
        for dep in ("csd", "coverage_homogeneity", "garch", "continuum"):
            assert names.index(dep) < regime_idx


class TestUnconditionalAnalyzers:
    """No gating flag: always constructed regardless of CLI flags."""

    def test_all_always_on_analyzers_constructed(self):
        f = _build_fuzzer()
        assert type(f._crash_mi).__name__ == "CrashMITracker"
        assert type(f._length_tracker).__name__ == "LengthEdgeTracker"
        assert type(f._allan).__name__ == "AllanVarianceDetector"
        assert f._last_allan_edge_count == 0
        assert type(f._sensitivity).__name__ == "ByteSensitivityTracker"
        assert type(f._exec_time_tracker).__name__ == "ExecutionTimeTracker"
        assert type(f._exec_time_anomaly).__name__ == "ExecTimeCalibrator"
        assert type(f._frameshift).__name__ == "FrameShift"
        assert type(f._csd).__name__ == "CriticalSlowingDown"
        assert type(f._homogeneity).__name__ == "CoverageHomogeneityDetector"
        assert type(f._regime).__name__ == "CoverageRegimeDetector"

    def test_crash_mi_sized_to_max_len(self):
        f = _build_fuzzer(max_len=1234)
        assert f._crash_mi.max_positions == 1234

    def test_sensitivity_sized_to_max_len(self):
        f = _build_fuzzer(max_len=2048)
        assert f._sensitivity.max_bytes == 2048

    def test_homogeneity_column_count_derived_from_map_size(self):
        f = _build_fuzzer()
        expected = max(1, f.map_size // 8192)
        assert len(f._homogeneity_col_cumulative) == expected

    def test_coverage_regime_wraps_its_four_dependencies(self):
        f = _build_fuzzer()
        assert f._regime._csd is f._csd
        assert f._regime._homogeneity is f._homogeneity
        # garch/continuum are off by default -- regime should hold None
        # for both, not fail to wire because they weren't constructed yet.
        assert f._regime._garch is None
        assert f._regime._continuum is None

    def test_sensitivity_wired_before_seed_metadata_init(self):
        # The original ordering bug this guards against: sensitivity used
        # to raise AttributeError during resume if constructed after
        # _init_seed_metadata(). Fuzzer() completing at all is the signal
        # that phase="early" ran early enough.
        f = _build_fuzzer()
        assert f._sensitivity is not None


class TestFlagGatedAnalyzers:
    def test_all_off_by_default(self):
        f = _build_fuzzer()
        assert f._fluctuation is None
        assert f._te is None
        assert f._occupation_rarity is None
        assert f._last_occupation is None
        assert f._causal_sector is None
        assert f._format_learner is None
        assert f._ppmd is None
        assert f._elo is None
        assert f._distance is None
        assert f._dist_table_shm is None
        assert f._garch is None
        assert f._continuum is None
        # checksum_learner has no gating flag (always attempted) -- its
        # off-path is the swallow_errors path, covered separately below.

    def test_fluctuation_on_when_requested(self):
        f = _build_fuzzer(fluctuation=True, fluctuation_beta=0.5, fluctuation_window=200)
        assert type(f._fluctuation).__name__ == "WorkFunctional"

    def test_transfer_entropy_off_by_default(self):
        f = _build_fuzzer()
        # _te_byte_edges is unconditional -- used regardless of the flag.
        assert f._te_byte_edges == {}

    def test_transfer_entropy_on_when_requested(self):
        f = _build_fuzzer(transfer_entropy=True)
        assert type(f._te).__name__ == "TransferEntropy"
        assert f._te_input_history == []
        assert f._te_edge_history == []
        assert f._te_history_max == 500

    def test_occupation_on_when_requested(self):
        f = _build_fuzzer(occupation=True)
        assert type(f._occupation_rarity).__name__ == "LongitudinalRarity"
        assert f._last_occupation is None
        assert f._occupation_max_edges == 256

    def test_causal_sector_requires_transfer_entropy(self):
        # `available` is a conjunction with `_use_transfer_entropy` -- the
        # graph would otherwise sit permanently empty with no TE source.
        f = _build_fuzzer(causal_sector=True)
        assert f._causal_sector is None

    def test_causal_sector_on_when_both_flags_set(self):
        f = _build_fuzzer(causal_sector=True, transfer_entropy=True)
        assert type(f._causal_sector).__name__ == "CausalSectorGraph"

    def test_causal_sector_registered_after_transfer_entropy(self):
        # activate() for causal_sector doesn't read f._te directly today, but
        # the ordering guarantee mirrors coverage_regime's dependency
        # cluster and must hold regardless.
        names = REGISTRY.names()
        assert names.index("transfer_entropy") < names.index("causal_sector")

    def test_format_learner_on_when_requested(self):
        f = _build_fuzzer(learn_format=True)
        assert type(f._format_learner).__name__ == "FormatLearner"

    def test_corpus_compression_on_when_requested(self):
        f = _build_fuzzer(corpus_ppmd=True)
        assert type(f._ppmd).__name__ == "CorpusCompressor"

    def test_elo_on_when_requested(self):
        f = _build_fuzzer(elo=True)
        assert type(f._elo).__name__ == "BayesianEloTracker"
        # Pre-registration of every operator + seed strategy name.
        assert len(f._elo._strategy_mu) > 0
        assert f._elo_decay_interval == 100
        assert f._elo_match_window == []

    def test_trace_on_when_requested(self):
        f = _build_fuzzer(trace_crashes=True)
        assert type(f._tracer).__name__ == "CrashTracer"

    def test_trace_off_when_disabled(self):
        f = _build_fuzzer(trace_crashes=False)
        assert f._tracer is None

    def test_garch_on_when_requested(self):
        f = _build_fuzzer(garch=True)
        assert type(f._garch).__name__ == "OnlineGarch11"
        assert f._regime.garch is f._garch

    def test_continuum_on_when_requested(self):
        f = _build_fuzzer(continuum=True)
        assert type(f._continuum).__name__ == "ContinuumField"
        assert f._continuum_adjacency == {}
        assert f._continuum_graph_tick == 0
        assert f._regime.continuum is f._continuum

    def test_distance_off_without_targets(self):
        f = _build_fuzzer()
        assert f._distance is None
        assert f._dist_table_shm is None


class TestChecksumLearner:
    """The only analyzer whose construction failure is swallowed, not raised."""

    def test_constructed_by_default(self):
        f = _build_fuzzer()
        assert type(f.checksum_learner).__name__ == "ChecksumLearner"

    def test_swallow_errors_flag_set_on_its_spec(self):
        spec = REGISTRY._specs["checksum_learner"]
        assert spec.swallow_errors is True

    def test_construction_failure_is_caught_not_raised(self):
        # Simulate the exact original try/except semantics: a broken
        # ChecksumLearner must not blow up Fuzzer(), just leave the
        # attribute at its off-default.
        class _Fake:
            _state_store = None

        def _boom(f):
            raise RuntimeError("simulated construction failure")

        reg = AnalyzerRegistry()
        reg.register(
            AnalyzerSpec(
                name="checksum_learner",
                category="format_recovery",
                activate=_boom,
                deactivate=lambda f: setattr(f, "checksum_learner", None),
                swallow_errors=True,
            )
        )
        fake = _Fake()
        fake.checksum_learner = "sentinel"
        activated = reg.wire_all(fake)
        assert activated["checksum_learner"] is False
        assert fake.checksum_learner is None


class TestPhases:
    def test_wire_all_default_phase_excludes_early(self):
        # sensitivity is phase="early"; a bare wire_all() (main phase) call
        # should not re-touch it.
        assert REGISTRY._specs["sensitivity"].phase == "early"
        names_main_would_touch = {
            name for name, spec in REGISTRY._specs.items() if spec.phase == "main"
        }
        assert "sensitivity" not in names_main_would_touch


class TestWireAllReturnValue:
    def test_reports_activation_per_analyzer(self):
        f = _build_fuzzer(fluctuation=True, elo=True)
        # Re-running wire_all (idempotent construction) should report the
        # same activation set as what __init__ already wired, for the main
        # phase (sensitivity, "early", is intentionally excluded here).
        activated = REGISTRY.wire_all(f)
        assert activated["fluctuation"] is True
        assert activated["transfer_entropy"] is False
        assert activated["crash_mi"] is True
        assert activated["length_tracker"] is True
        assert activated["allan"] is True
        assert activated["elo"] is True
        assert activated["distance"] is False
        assert activated["checksum_learner"] is True
