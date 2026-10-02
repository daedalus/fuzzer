"""Regression: in-process (persistent loader) runs dropped every ASAN crash.

A target built with plain ``-fsanitize=address`` (no ``-fsanitize-recover``)
cannot honour ``halt_on_error=0``: ASAN prints its report and ``_exit(1)``s
(``abort_on_error=0``). The persistent loader's fork child then exits without
writing its rc byte; the loader read EOF and replied ``rc=-2`` -- the
infrastructure-failure sentinel -- which ``is_crash`` rejects before parsing
stderr. Measured on fuzzgoat_read_asan.so: all four injected bugs came back
``(-2, <ASAN report>)`` and were never counted.

A child that exits without reporting an rc now relays its exit status, so the
sanitizer report decides.

The target here is a plain shared object that emulates ASAN's fatal path
(report on stderr, ``_exit(1)``): the code under test is the loader protocol,
not the sanitizer runtime.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.services.runner import TargetRunner
from tests.conftest import requires_clang

pytestmark = requires_clang

REPORT = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1"

# 'D': sanitizer Die() -- report, then _exit(1) mid-call.
# 'Z': target exits 0 mid-call without returning.
# 'R': returns 7.  'S': SIGSEGV.  Anything else returns 0.
TARGET_SRC = f"""
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
int fuzz_shm_run(const unsigned char *b, size_t n) {{
    if (n == 0) return 0;
    if (b[0] == 'D') {{ fprintf(stderr, "{REPORT}\\n"); fflush(stderr); _exit(1); }}
    if (b[0] == 'Z') _exit(0);
    if (b[0] == 'R') return 7;
    if (b[0] == 'P') {{ printf("0"); fflush(stdout); return 0; }}
    if (b[0] == 'S') {{ volatile int *p = 0; return *p; }}
    return 0;
}}
"""


@pytest.fixture
def runner(tmp_path):
    src = tmp_path / "die_target.c"
    so = tmp_path / "die_target.so"
    src.write_text(TARGET_SRC)
    subprocess.run(
        ["clang", "-shared", "-fPIC", "-o", str(so), str(src)], check=True, capture_output=True
    )
    r = InProcessRunner(
        target=str(so),
        function_name="fuzz_shm_run",
        timeout=5.0,
        shm_size=4096,
        capture_stderr=True,
    )
    assert r._persistent, "persistent loader did not start"
    yield r
    r._persistent.stop()


def _is_crash(rc: int, err: str) -> bool:
    f = MagicMock()
    f.extra_crash_codes = ()
    return TargetRunner(f).is_crash(rc, err)


def test_regression_sanitizer_exit_is_crash(runner):
    rc, err = runner.run_one(b"D")
    assert rc == 1, f"sanitizer exit reported as {rc} (-2 is the infra sentinel)"
    assert "AddressSanitizer" in err
    assert _is_crash(rc, err)


def test_returned_rc_unchanged(runner):
    """Falsification: a normal return still reports the function's rc."""
    assert runner.run_one(b"R")[0] == 7
    assert runner.run_one(b"A")[0] == 0


def test_loader_survives_sanitizer_exit(runner):
    """Adversarial: the run after a fatal report is clean, not a stale crash."""
    runner.run_one(b"D")
    rc, err = runner.run_one(b"A")
    assert rc == 0
    assert not _is_crash(rc, err)


def test_silent_exit_is_not_infra_failure(runner):
    """Adversarial: exit(0) mid-call is the target's choice, not a loader fault."""
    rc, err = runner.run_one(b"Z")
    assert rc == 0
    assert not _is_crash(rc, err)


def test_regression_target_stdout_does_not_desync_loader(runner):
    """fuzzgoat prints on some inputs; the child shared the loader's protocol
    stdout, so its bytes prefixed the RC header and the run read as -2."""
    assert runner.run_one(b"P")[0] == 0
    assert runner.run_one(b"R")[0] == 7, "protocol desynced after target output"


def test_signal_crash_still_negative(runner):
    """Falsification: the guarded-call / signal path is untouched."""
    rc, _ = runner.run_one(b"S")
    assert rc < 0 or rc >= 128


# ── --inprocess must capture sanitizer stderr like the auto-detect path ──


def _explicit_inprocess_kwargs(monkeypatch, tmp_path, *, sanitized: bool) -> dict:
    """Build a Fuzzer via --inprocess on a fake .so; return InProcessRunner kwargs.

    Sanitizer detection goes through UBSAN: the ASAN branch would load
    libasan into the test process.
    """
    from unittest.mock import patch

    from fuzzer_tool.services import fuzzer as fuzzer_mod
    from fuzzer_tool.services.fuzzer import Fuzzer

    monkeypatch.setenv("UBSAN_OPTIONS", "")
    monkeypatch.setattr(fuzzer_mod, "_detect_asan", lambda path: False)
    monkeypatch.setattr(fuzzer_mod, "_detect_ubsan", lambda path: sanitized)
    monkeypatch.setattr(fuzzer_mod, "_detect_cmplog", lambda path: False)
    monkeypatch.setattr(fuzzer_mod, "_detect_tracecmp_target", lambda path: False)

    kwargs: dict = {}

    class _StubRunner:
        def __init__(self, *a, **k):
            self._persistent = False
            kwargs.update(k)

    monkeypatch.setattr("fuzzer_tool.adapters.inprocess.InProcessRunner", _StubRunner)
    monkeypatch.setattr(Fuzzer, "_probe_so_function", lambda self, target: "fuzz_shm_run")

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        Fuzzer(
            target=str(tmp_path / "fake_target.so"),
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            cmplog=False,
            inprocess=True,
        )
    return kwargs


def test_regression_explicit_inprocess_captures_sanitizer_stderr(monkeypatch, tmp_path):
    kw = _explicit_inprocess_kwargs(monkeypatch, tmp_path, sanitized=True)
    assert kw.get("capture_stderr") is True


def test_explicit_inprocess_unsanitized_skips_capture(monkeypatch, tmp_path):
    """Adversarial: no sanitizer, no per-call stderr pipe."""
    kw = _explicit_inprocess_kwargs(monkeypatch, tmp_path, sanitized=False)
    assert not kw.get("capture_stderr")
