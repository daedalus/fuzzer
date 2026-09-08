"""Regression tests: analyzer registry is the single source of truth.

Covers the first slice of the analyzer-dispatcher refactor (mirrors
``test_regression_operator_registry.py`` for ``core.operator_registry``):
fluctuation, transfer_entropy, crash_mi, length_tracker, and allan are all
constructed through ``core.analyzer_registry.REGISTRY.wire_all()`` instead
of inline in ``Fuzzer.__init__``. Guards against the two failure modes that
matter here -- an analyzer silently not constructed when its flag is on, and
an analyzer's off-path leaving a stale/wrong default -- plus the registry's
own duplicate-registration guard.
"""

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.core.analyzer_registry import REGISTRY, AnalyzerRegistry, AnalyzerSpec

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


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
    def test_all_five_migrated_analyzers_registered(self):
        assert set(REGISTRY.names()) == {
            "fluctuation",
            "transfer_entropy",
            "crash_mi",
            "length_tracker",
            "allan",
        }

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


class TestUnconditionalAnalyzers:
    """crash_mi, length_tracker, allan have no gating flag: always on."""

    def test_always_constructed(self):
        f = _build_fuzzer()
        assert type(f._crash_mi).__name__ == "CrashMITracker"
        assert type(f._length_tracker).__name__ == "LengthEdgeTracker"
        assert type(f._allan).__name__ == "AllanVarianceDetector"
        assert f._last_allan_edge_count == 0

    def test_crash_mi_sized_to_max_len(self):
        f = _build_fuzzer(max_len=1234)
        assert f._crash_mi.max_positions == 1234


class TestFlagGatedAnalyzers:
    """fluctuation and transfer_entropy are off unless explicitly requested."""

    def test_fluctuation_off_by_default(self):
        f = _build_fuzzer()
        assert f._fluctuation is None

    def test_fluctuation_on_when_requested(self):
        f = _build_fuzzer(fluctuation=True, fluctuation_beta=0.5, fluctuation_window=200)
        assert type(f._fluctuation).__name__ == "WorkFunctional"

    def test_transfer_entropy_off_by_default(self):
        f = _build_fuzzer()
        assert f._te is None
        # _te_byte_edges is unconditional -- used regardless of the flag.
        assert f._te_byte_edges == {}

    def test_transfer_entropy_on_when_requested(self):
        f = _build_fuzzer(transfer_entropy=True)
        assert type(f._te).__name__ == "TransferEntropy"
        assert f._te_input_history == []
        assert f._te_edge_history == []
        assert f._te_history_max == 500


class TestWireAllReturnValue:
    def test_reports_activation_per_analyzer(self):
        f = _build_fuzzer(fluctuation=True)
        # Re-running wire_all (idempotent construction) should report the
        # same activation set as what __init__ already wired.
        activated = REGISTRY.wire_all(f)
        assert activated["fluctuation"] is True
        assert activated["transfer_entropy"] is False
        assert activated["crash_mi"] is True
        assert activated["length_tracker"] is True
        assert activated["allan"] is True
