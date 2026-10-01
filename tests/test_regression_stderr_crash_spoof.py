"""Regression: crash markers in stderr must not turn a clean exit into a crash.

The target owns its stderr. A process that prints "Segmentation fault" and
exits 0 (AntiFuzz-style handlers, or any program logging the word) used to
be filed as a crash by every oracle that scans stderr. A clean exit is now
authoritative; markers only corroborate a non-zero exit.
"""

import pytest

from fuzzer_tool.adapters.process import CRASH_STDERR_MARKERS, stderr_crash_marker
from fuzzer_tool.services.runner import TargetRunner


class _Fuzzer:
    extra_crash_codes: set[int] = set()
    last_report = None


@pytest.fixture
def runner() -> TargetRunner:
    return TargetRunner(_Fuzzer())  # type: ignore[arg-type]


@pytest.mark.parametrize("marker", CRASH_STDERR_MARKERS)
def test_regression_clean_exit_marker_not_crash(runner: TargetRunner, marker: str) -> None:
    # Adversarial: the target forges every marker, then exits cleanly.
    stderr = f"handled: {marker}\n"

    assert stderr_crash_marker(0, stderr) is None
    assert not runner.is_crash(0, stderr)
    assert not runner.is_interesting(0, stderr)


@pytest.mark.parametrize("marker", CRASH_STDERR_MARKERS)
def test_nonzero_exit_marker_is_crash(runner: TargetRunner, marker: str) -> None:
    # Falsification: a failing exit that reports a fault is still a crash.
    stderr = f"{marker}\n"

    assert stderr_crash_marker(1, stderr) == marker
    assert runner.is_crash(1, stderr)
    assert runner.is_interesting(1, stderr)


def test_sanitizer_report_survives_clean_exit(runner: TargetRunner) -> None:
    # Control: halt_on_error=0 builds exit 0 with a real report; keep it.
    stderr = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1234\n"

    assert runner.is_crash(0, stderr)


def test_timeout_sentinels_never_crash() -> None:
    assert stderr_crash_marker(-1, "Segmentation fault") is None
    assert stderr_crash_marker(-2, "Segmentation fault") is None
