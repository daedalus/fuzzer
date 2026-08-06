"""Regression test: warn when an in-process target runs without coverage.

An in-process (.so) target selected without -c/--coverage gets no SHM
segment, so nothing populates the edge bitmap. Every coverage-guided
subsystem downstream — seed scheduling, MI/TE/sensitivity position
weighting, Elo/bandit operator scheduling, stall detection, corpus
admission — then runs on a constant-zero signal.

The failure mode is entirely silent: the run reports healthy throughput
(thousands of eps) and simply discovers nothing, which is easy to mistake
for "the target has no reachable bugs" rather than "coverage was never
switched on". These tests pin that the warning fires.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fuzzer_tool.services.fuzzer import Fuzzer


class _Bare:
    """Minimal object carrying only what _warn_no_coverage touches."""

    _warn_no_coverage = Fuzzer._warn_no_coverage


def test_warns_when_no_coverage(capsys):
    obj = _Bare()
    obj._warn_no_coverage()
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "-c/--coverage" in out


def test_warning_is_emitted_once_per_run(capsys):
    """Both in-process setup paths call this; the user should see it once."""
    obj = _Bare()
    obj._warn_no_coverage()
    obj._warn_no_coverage()
    obj._warn_no_coverage()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1


def test_warning_mentions_the_actual_consequence(capsys):
    """The message must say discovery is inactive — a bare 'no coverage'
    note is what let this go unnoticed."""
    obj = _Bare()
    obj._warn_no_coverage()
    out = capsys.readouterr().out.lower()
    assert "edge discovery" in out
    assert "inactive" in out or "blind" in out


def test_logs_as_warning_level(caplog):
    obj = _Bare()
    with caplog.at_level("WARNING"):
        obj._warn_no_coverage()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_uses_sentinel_attribute_not_shared_state():
    """The once-per-run guard must be per-instance, so a second Fuzzer in
    the same process (parallel workers, tests) still warns."""
    a, b = _Bare(), _Bare()
    a._warn_no_coverage()
    assert getattr(a, "_no_cov_warned", False) is True
    assert getattr(b, "_no_cov_warned", False) is False


def test_fuzzer_has_the_method():
    """Guard against the helper being inlined away or renamed."""
    assert callable(getattr(Fuzzer, "_warn_no_coverage", None))
    # And that both in-process paths still reference it.
    import inspect

    src = inspect.getsource(Fuzzer.__init__)
    assert src.count("_warn_no_coverage()") == 2, (
        "both the .so auto-detect and explicit --inprocess paths should warn"
    )


def test_mock_fuzzer_does_not_crash():
    """Defensive: the helper is called during __init__, before many
    attributes exist."""
    m = MagicMock()
    Fuzzer._warn_no_coverage(m)  # must not raise
