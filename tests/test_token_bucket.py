"""Token bucket and the sanitizer report throttle (closes TODO: 106 MB stderr / 10k execs)."""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.adapters.inprocess import _pre_captured
from fuzzer_tool.adapters.stderr_capture import StderrCapture
from fuzzer_tool.core.token_bucket import (
    MAX_REPORT_KEYS,
    REPORT_BURST,
    ReportThrottle,
    TokenBucket,
)
from tests.test_global_snapshot import REPO_ROOT, SHIM, _build


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _report(kind: str, addr: int, pid: int = 7, func: str = "parse") -> str:
    return (
        f"=={pid}==ERROR: AddressSanitizer: {kind} on address {addr:#x}\n"
        f"    #0 {addr + 16:#x} in {func} /src/a.c:{addr % 97}:3\n"
        f"SUMMARY: AddressSanitizer: {kind}\n"
    )


def test_bucket_burst_then_refill():
    clock = _Clock()
    bucket = TokenBucket(capacity=2, rate=0.5, clock=clock)

    assert [bucket.take() for _ in range(3)] == [True, True, False]
    clock.now = 2.0  # 2 s x 0.5/s = one token
    assert [bucket.take() for _ in range(2)] == [True, False]


def test_bucket_refill_capped_at_capacity():
    clock = _Clock()
    bucket = TokenBucket(capacity=2, rate=1.0, clock=clock)
    bucket.take()
    bucket.take()
    clock.now = 1e6
    assert [bucket.take() for _ in range(3)] == [True, True, False]


def test_falsification_backwards_clock_grants_nothing():
    clock = _Clock()
    clock.now = 10.0
    bucket = TokenBucket(capacity=1, rate=1.0, clock=clock)
    bucket.take()
    clock.now = 0.0
    assert not bucket.take()


def test_repeated_report_throttled_to_burst():
    out: list[str] = []
    throttle = ReportThrottle(out.append, clock=_Clock())
    n = REPORT_BURST * 5
    for i in range(n):
        throttle.emit(_report("global-buffer-overflow", 0x1000 + 8 * i))

    assert len(out) == REPORT_BURST
    assert throttle.suppressed == n - REPORT_BURST


def test_distinct_bugs_not_throttled_by_each_other():
    out: list[str] = []
    throttle = ReportThrottle(out.append, clock=_Clock())
    for _ in range(REPORT_BURST * 3):
        throttle.emit(_report("heap-use-after-free", 0x20, func="a"))
    throttle.emit(_report("heap-use-after-free", 0x20, func="b"))

    assert out[-1].count("in b ") == 1


def test_suppressed_count_announced_on_next_pass():
    clock = _Clock()
    out: list[str] = []
    throttle = ReportThrottle(out.append, clock=clock)
    for _ in range(REPORT_BURST + 3):
        throttle.emit(_report("stack-overflow", 0x30))
    clock.now = 1e6
    throttle.emit(_report("stack-overflow", 0x30))

    assert "3 repeated" in out[-1]


def test_one_write_holds_several_reports():
    out: list[str] = []
    throttle = ReportThrottle(out.append, clock=_Clock())
    text = "".join(_report("global-buffer-overflow", 0x40) for _ in range(REPORT_BURST + 2))
    throttle.emit(text)

    assert "".join(out).count("==ERROR") == REPORT_BURST
    assert throttle.suppressed == 2


def test_adversarial_keys_bounded():
    throttle = ReportThrottle(lambda _t: None, clock=_Clock())
    for i in range(MAX_REPORT_KEYS * 2):
        throttle.emit(_report("x", 0x50, func=f"f{i}"))
    assert throttle.keys() <= MAX_REPORT_KEYS


def test_adversarial_empty_and_plain_text():
    out: list[str] = []
    throttle = ReportThrottle(out.append, clock=_Clock())
    throttle.emit("")
    throttle.emit("plain line\n")
    assert out == ["plain line\n"]


def test_pre_exec_report_goes_through_throttle():
    # A write to fd 2 during the pre-exec stage (snapshot restore) reaches the
    # sink, not the real stderr, and repeats are dropped.
    out: list[str] = []
    capture = StderrCapture()
    throttle = ReportThrottle(out.append, clock=_Clock())

    def pre(_data):
        os.write(2, _report("global-buffer-overflow", 0x60).encode())

    run = _pre_captured(pre, capture, throttle)
    try:
        for _ in range(REPORT_BURST + 4):
            run(b"x")
    finally:
        capture.close()

    assert len(out) == REPORT_BURST
    assert throttle.suppressed == 4


# ── wiring: InProcessRunner binds the throttled pre-exec stage ─────────────

_WIRING = textwrap.dedent(
    """
    import sys
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.core.run_mode import StateRestore

    so, capture = sys.argv[1], sys.argv[2] == "1"
    r = InProcessRunner(
        so, "fuzz_test", timeout=5.0, direct_lite=True,
        capture_stderr=capture, restore=StateRestore.GLOBALS,
    )
    rcs = [r.run_one(b"S")[0] for _ in range(3)]
    bound = r._pre.__qualname__  # before stop(): it unbinds the stage
    r.stop()
    print(bound, repr(rcs))
    """
)


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("throttle"), "probe", SHIM)


@pytest.mark.parametrize(("capture", "throttled"), [("1", True), ("0", False)])
def test_restore_stage_throttled_when_captured(probe_so, capture, throttled):
    proc = subprocess.run(
        [sys.executable, "-c", _WIRING, str(probe_so), capture],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    qualname, rcs = proc.stdout.strip().splitlines()[-1].split(" ", 1)

    assert ("_pre_captured" in qualname) is throttled
    fresh = 1 * 10 + 0  # restore still works: every exec sees the first one's state
    assert ast.literal_eval(rcs) == [fresh] * 3
