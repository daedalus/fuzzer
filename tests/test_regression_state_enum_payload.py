"""Regression tests: a persisted enum made the WHOLE state file unloadable.

``CoverageRegimeDetector.save()`` converted ``self._regime`` to ``.value`` but
passed ``self._regime_history`` through untouched, and that history is a list
of ``(exec_count, CoverageRegime)`` tuples. The state file is read back
through ``state_store._SafeUnpickler``, whose allowlist is containers and
scalars only, so the reference to ``fuzzer_tool.core.percolation`` made the
entire file rejected -- ``corpus``, ``edge_tracker``, ``markov``,
``seed_quality``, ``crash_mi``, ``length_tracker``, ``sensitivity`` and
``coverage_contract`` along with it.

Measured before the fix, on a state file written by a 2000-exec png campaign::

    ERROR  refusing to load untrusted state file .../state.pkl.gz
    SECTIONS RECOVERED: []

Two things made it survive: the failure is a log line and an empty dict
rather than an exception, so a resume looks like a cold start; and the message
blames tampering for something this codebase wrote itself.
"""

from __future__ import annotations

import gzip
import pickle

import pytest

from fuzzer_tool.core.coverage_regime import CoverageRegimeDetector
from fuzzer_tool.core.critical_slowing import CriticalSlowingDown
from fuzzer_tool.core.percolation import CoverageRegime
from fuzzer_tool.core.state_store import StateStore, UnsafeStateError, _safe_loads


def _new_detector() -> CoverageRegimeDetector:
    return CoverageRegimeDetector(csd=CriticalSlowingDown(), homogeneity=None)


def _detector_with_history() -> CoverageRegimeDetector:
    det = _new_detector()
    det._regime_history = [
        (100, CoverageRegime.SUPERCRITICAL),
        (200, CoverageRegime.CRITICAL),
        (300, CoverageRegime.SUBCRITICAL),
    ]
    det._regime = CoverageRegime.CRITICAL
    return det


def test_saved_payload_contains_only_primitive_types() -> None:
    """Recursive type audit of the payload.

    An earlier version of this test scanned the pickle opcode stream for
    GLOBAL/STACK_GLOBAL arguments -- and passed against the broken code,
    because protocol 4 emits STACK_GLOBAL with no argument (the module and
    name come off the stack as separate string opcodes), so the set it
    checked was always empty. A test that cannot fail is not a test; Hard
    Rule 39's point exactly. This walks the object graph instead.
    """
    primitives = (bool, int, float, complex, str, bytes, bytearray, type(None))

    def offenders(obj, path="save()"):
        if isinstance(obj, primitives):
            return []
        if isinstance(obj, dict):
            out = []
            for k, v in obj.items():
                out += offenders(k, f"{path}[key]") + offenders(v, f"{path}[{k!r}]")
            return out
        if isinstance(obj, (list, tuple, set, frozenset)):
            out = []
            for i, v in enumerate(obj):
                out += offenders(v, f"{path}[{i}]")
            return out
        return [f"{path} -> {type(obj).__module__}.{type(obj).__qualname__}"]

    bad = offenders(_detector_with_history().save())
    assert not bad, "save() persisted non-primitive objects: " + "; ".join(bad)


def test_saved_payload_survives_the_safe_unpickler() -> None:
    payload = {"regime": _detector_with_history().save()}
    restored = _safe_loads(pickle.dumps(payload))
    assert restored["regime"]["regime_history"][1] == (200, "critical")


def test_a_persisted_enum_still_fails_loudly() -> None:
    """Guard the guard: _SafeUnpickler must keep rejecting the old shape.

    Without this, the fix could be 'verified' by a loosened allowlist, which
    would trade a broken resume for an unpickling code-execution vector.
    """
    with pytest.raises(UnsafeStateError):
        _safe_loads(pickle.dumps({"regime": {"regime_history": [(1, CoverageRegime.CRITICAL)]}}))


def test_whole_state_file_round_trips_not_just_this_section(tmp_path) -> None:
    """The blast radius was every section, so that is what is asserted."""
    store = StateStore(tmp_path)
    store.set("regime", _detector_with_history().save())
    store.set("corpus", {"seeds": 12})
    store.set("edge_tracker", {"edges": [1, 2, 3]})
    store.set("markov", {"transitions": {}})
    store.save()

    reloaded = StateStore(tmp_path)
    sections = reloaded.load()
    assert sorted(sections) == ["corpus", "edge_tracker", "markov", "regime"]
    assert sections["corpus"] == {"seeds": 12}


def test_history_round_trips_back_to_enum_members() -> None:
    det = _detector_with_history()
    restored = _new_detector()
    restored.load(_safe_loads(pickle.dumps(det.save())))
    assert restored._regime_history == [
        (100, CoverageRegime.SUPERCRITICAL),
        (200, CoverageRegime.CRITICAL),
        (300, CoverageRegime.SUBCRITICAL),
    ]
    assert all(isinstance(r, CoverageRegime) for _, r in restored._regime_history)


def test_load_accepts_enum_members_from_an_in_memory_roundtrip() -> None:
    """save()/load() without a pickler in between must still work."""
    det = _detector_with_history()
    restored = _new_detector()
    restored.load({"regime_history": [(1, CoverageRegime.CRITICAL)], "regime": "critical"})
    assert restored._regime_history == [(1, CoverageRegime.CRITICAL)]
    assert det._regime_history[0][1] is CoverageRegime.SUPERCRITICAL


def test_unparseable_history_entries_are_dropped_not_fatal() -> None:
    restored = _new_detector()
    restored.load(
        {
            "regime": "critical",
            "regime_history": [
                (1, "critical"),
                (2, "not-a-regime"),
                "not-a-pair",
                (3, "subcritical"),
            ],
        }
    )
    assert restored._regime_history == [
        (1, CoverageRegime.CRITICAL),
        (3, CoverageRegime.SUBCRITICAL),
    ]


def test_a_real_state_file_is_readable_end_to_end(tmp_path) -> None:
    """Belt and braces: write through gzip exactly as the store does."""
    store = StateStore(tmp_path)
    store.set("regime", _detector_with_history().save())
    store.save()
    raw = gzip.open(store.path, "rb").read()
    assert isinstance(_safe_loads(raw), dict)
