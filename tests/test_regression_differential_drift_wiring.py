"""Regression: the KL drift tracker was fed constants, not target outputs.

``Fuzzer._check_differential`` called ``diff_run()``, which returns only
``(diverged, description)`` and drops the four observations it made, then
recorded::

    rc_b, stderr_b = 0, ""
    self._diff_tracker.record(0, "", rc_b, stderr_b)

So both distributions filled with the same single category. Measured over 500
inputs through the real call site: rc_counts_a == rc_counts_b == {0: 500},
sig_counts_a == sig_counts_b == {"clean": 500}, both KL values exactly 0.0,
drift_detected permanently False, and no execution times recorded at all -- so
the KS branch was dead alongside the KL one. The module advertises two levels of
comparison; the second could not fire.

The per-input verdict was discarded too (``if diverged: pass``, with a comment
claiming diff_run logs it, which it does not), and no code outside
differential.py read drift_detected, drift_description, last_kl_* or
get_report(), so the layer was dead at both ends.

These tests assert on the distributions the tracker accumulates rather than on
the KL number: a KL of 0.0 is the correct answer for the constants that were
being passed, so an oracle on the output value cannot distinguish "computed
correctly from dead inputs" from "computed correctly from live ones".
"""

import subprocess
import sys
import textwrap
from collections import Counter
from unittest.mock import patch

from fuzzer_tool.services.differential import DifferentialTracker, DiffOutcome


def _run_in_fresh_interpreter(script: str) -> str:
    """Execute `script` in a new Python process and return its stdout.

    These assertions need real fork+exec through run_target_stdin, with no
    mocks -- that is the whole point, since every mocked test missed the stale
    arity. But the interpreter running the suite cannot be trusted to fork
    safely: tests/test_abort_override.py dlopens the AFL shim .so into this
    process via InProcessRunner(direct_lite=True), and after that any
    run_target_stdin call here dies with SIGSEGV. Reproduce with

        pytest tests/test_abort_override.py::TestAbortOverride\
    ::test_shim_so_inprocess_abort_does_not_crash \
               tests/test_regression_differential_drift_wiring.py

    which exits 139, while either file alone exits 0. pytest-randomly is
    enabled here, so ordering cannot be relied on to keep them apart. Spawning
    a clean interpreter keeps the no-mocks property without depending on what
    else has run.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return proc.stdout.strip()


class _Recorder(DifferentialTracker):
    """Tracker that keeps every record() call verbatim."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def record(self, rc_a, stderr_a, rc_b, stderr_b, time_a=0.0, time_b=0.0):
        self.calls.append((rc_a, stderr_a, rc_b, stderr_b, time_a, time_b))
        super().record(rc_a, stderr_a, rc_b, stderr_b, time_a, time_b)


class _Stub:
    """Minimal stand-in for the Fuzzer attributes _check_differential touches."""

    target = "target_a"

    def __init__(self, tracker):
        self._diff_tracker = tracker
        self._diff_target = "target_b"
        self._diff_divergences = 0

    _check_differential = None  # bound below


def _make_stub(tracker):
    from fuzzer_tool.services.fuzzer import Fuzzer

    stub = _Stub(tracker)
    stub._check_differential = Fuzzer._check_differential.__get__(stub, _Stub)
    return stub


def _outcome(rc_a, stderr_a, rc_b, stderr_b, ta=0.01, tb=0.02):
    return DiffOutcome(
        diverged=rc_a != rc_b,
        description="stub",
        rc_a=rc_a,
        stderr_a=stderr_a,
        rc_b=rc_b,
        stderr_b=stderr_b,
        time_a=ta,
        time_b=tb,
    )


class TestObservationsReachTheTracker:
    def test_both_returncodes_are_forwarded(self):
        tracker = _Recorder()
        stub = _make_stub(tracker)
        with patch(
            "fuzzer_tool.services.differential.diff_run_detailed",
            return_value=_outcome(0, "", 3, "boom"),
        ):
            stub._check_differential(b"payload")
        assert tracker.calls == [(0, "", 3, "boom", 0.01, 0.02)]

    def test_distributions_diverge_when_targets_diverge(self):
        tracker = _Recorder()
        stub = _make_stub(tracker)
        outcomes = [
            _outcome(0, "", -11, "AddressSanitizer: heap-buffer-overflow")
            if i % 10 == 0
            else _outcome(0, "", 0, "")
            for i in range(200)
        ]
        with patch(
            "fuzzer_tool.services.differential.diff_run_detailed",
            side_effect=outcomes,
        ):
            for _ in range(200):
                stub._check_differential(b"payload")

        # The defect made these two identical.
        assert tracker.rc_counts_a != tracker.rc_counts_b
        assert tracker.sig_counts_a != tracker.sig_counts_b
        assert tracker.rc_counts_a == Counter({0: 200})
        assert tracker.rc_counts_b == Counter({0: 180, -11: 20})
        assert tracker.last_kl_returncode > 0.0
        assert tracker.last_kl_signature > 0.0
        assert tracker.drift_detected

    def test_execution_times_are_recorded(self):
        """time_a/time_b defaulted to 0, so the KS branch never had samples."""
        tracker = _Recorder()
        stub = _make_stub(tracker)
        with patch(
            "fuzzer_tool.services.differential.diff_run_detailed",
            side_effect=[_outcome(0, "", 0, "") for _ in range(50)],
        ):
            for _ in range(50):
                stub._check_differential(b"payload")
        assert len(tracker.exec_times_a) == 50
        assert len(tracker.exec_times_b) == 50

    def test_identical_targets_leave_kl_at_zero(self):
        """The fix must not manufacture drift where there is none."""
        tracker = _Recorder()
        stub = _make_stub(tracker)
        with patch(
            "fuzzer_tool.services.differential.diff_run_detailed",
            side_effect=[_outcome(2, "same", 2, "same", 0.01, 0.01) for _ in range(100)],
        ):
            for _ in range(100):
                stub._check_differential(b"payload")
        assert tracker.rc_counts_a == tracker.rc_counts_b
        assert tracker.last_kl_returncode == 0.0
        assert not tracker.drift_detected

    def test_divergence_verdict_is_counted(self):
        """`if diverged: pass` dropped the per-input result."""
        tracker = _Recorder()
        stub = _make_stub(tracker)
        with patch(
            "fuzzer_tool.services.differential.diff_run_detailed",
            side_effect=[_outcome(0, "", 1, "x"), _outcome(0, "", 0, "")],
        ):
            stub._check_differential(b"a")
            stub._check_differential(b"b")
        assert stub._diff_divergences == 1


class TestDiffOutcome:
    def test_diff_run_verdict_is_unchanged(self, tmp_path):
        """diff_run keeps its 2-tuple contract for existing callers."""
        a = tmp_path / "a.sh"
        a.write_text("#!/bin/sh\nexit 0\n")
        b = tmp_path / "b.sh"
        b.write_text("#!/bin/sh\nexit 7\n")
        for p in (a, b):
            p.chmod(0o755)
        out = _run_in_fresh_interpreter(f"""
            from fuzzer_tool.services.differential import diff_run
            print(diff_run({str(a)!r}, {str(b)!r}, b"x"))
        """)
        assert out == "(True, 'returncode: 0 vs 7')"

    def test_detailed_carries_observations(self, tmp_path):
        a = tmp_path / "a.sh"
        a.write_text("#!/bin/sh\nexit 0\n")
        b = tmp_path / "b.sh"
        b.write_text("#!/bin/sh\necho boom >&2\nexit 7\n")
        for p in (a, b):
            p.chmod(0o755)
        out = _run_in_fresh_interpreter(f"""
            from fuzzer_tool.services.differential import diff_run_detailed
            o = diff_run_detailed({str(a)!r}, {str(b)!r}, b"x")
            print(o.rc_a, o.rc_b, "boom" in o.stderr_b,
                  o.time_a > 0.0, o.time_b > 0.0,
                  o.as_verdict() == (o.diverged, o.description))
        """)
        assert out == "0 7 True True True True"
