"""Regression: the stderr cap kept the head, so a chatty target's report was cut off."""

from __future__ import annotations

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.core.sanitizer import SanitizerReport

_REPORT = "==7==ERROR: AddressSanitizer: heap-use-after-free on address 0x1"

# Noise well past the cap, then the report the sanitizer prints last.
_NOISE_BYTES = 4 * process._STDERR_CAP


@pytest.fixture
def chatty(tmp_path):
    script = tmp_path / "chatty.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"head -c {_NOISE_BYTES} /dev/zero | tr '\\0' 'n' >&2\n"
        f"echo >&2\necho '{_REPORT}' >&2\nexit 1\n"
    )
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def quiet(tmp_path):
    script = tmp_path / "quiet.sh"
    script.write_text(f"#!/bin/sh\necho '{_REPORT}' >&2\nexit 1\n")
    script.chmod(0o755)
    return str(script)


def _runs(target: str):
    """stderr of the stdin path and the posix_spawn fast path."""
    _, a, _ = process.run_target_stdin(target, b"x", 10.0)
    _, b, _ = process.run_target_fast(target, b"x", timeout=10.0)
    return a, b


def test_regression_report_after_cap_survives(chatty: str) -> None:
    for stderr in _runs(chatty):
        assert len(stderr) <= process._STDERR_CAP
        assert SanitizerReport.parse(stderr) is not None


def test_short_stderr_kept_whole(quiet: str) -> None:
    # Falsification: under the cap, nothing is dropped.
    for stderr in _runs(quiet):
        assert stderr == _REPORT + "\n"


def test_tail_buffer_keeps_last_bytes() -> None:
    # Edge: chunks straddling the cap boundary.
    cap = 10
    buf = process._TailBuffer(cap)
    for chunk in (b"abcdef", b"ghij", b"klm", b"", b"nopqrstuvwxyz0"):
        buf.add(chunk)

    expected = b"abcdefghijklmnopqrstuvwxyz0"[-cap:]
    assert buf.getvalue() == expected
