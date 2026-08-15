"""Tests for feeding suggested_timeout() back into the live timeout.

``ExecutionTimeTracker.suggested_timeout()`` was computed every run and
only ever printed: the timeout was fixed at construction and never
retuned. Applying it needs two things that did not exist until recently --
fractional timeouts surviving the loader handshake (c99bb27), and a way to
give the loader a new deadline without tearing down the forkserver and its
warmed target (the TIMEOUT command).

The invariant these tests exist to protect: the timeout the fuzzer reports
and the timeout the loader enforces must never diverge. That divergence is
exactly the E2 defect (Python asking for 0.04s while the loader ran 5s),
and an adaptive path that updates ``self.timeout`` optimistically would
reintroduce it from the other end.
"""

from __future__ import annotations

import subprocess
import textwrap
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.services.fuzzer import (
    ADAPTIVE_TIMEOUT_COOLDOWN_EXECS,
    ADAPTIVE_TIMEOUT_FLOOR,
    ADAPTIVE_TIMEOUT_MAX_GROWTH,
    ADAPTIVE_TIMEOUT_MIN_SAMPLES,
    Fuzzer,
)


def _fuzzer(tmp_path, **kw):
    """A Fuzzer with no target, built only far enough to retune."""
    f = Fuzzer.__new__(Fuzzer)
    f.exec_count = 0
    f.timeout = kw.get("timeout", 1.0)
    f._timeout_initial = f.timeout
    f._adaptive_timeout = kw.get("adaptive_timeout", True)
    f._timeout_retunes = []
    f._last_timeout_retune_exec = 0
    f._forkserver = None
    f._inprocess_runner = None
    f._persistent_runner = None
    f._exec_time_tracker = MagicMock()
    f._exec_time_tracker.count = ADAPTIVE_TIMEOUT_MIN_SAMPLES
    f._exec_time_tracker.p99 = 0.03
    f._exec_time_tracker.suggested_timeout.return_value = 0.04
    return f


class TestRetuneGating:
    def test_disabled_by_default_does_nothing(self, tmp_path):
        f = _fuzzer(tmp_path, adaptive_timeout=False)
        f._maybe_retune_timeout()
        assert f.timeout == 1.0
        assert f._timeout_retunes == []

    def test_waits_for_a_settled_distribution(self, tmp_path):
        """Below the sample floor the tracker describes the warm-up."""
        f = _fuzzer(tmp_path)
        f._exec_time_tracker.count = ADAPTIVE_TIMEOUT_MIN_SAMPLES - 1
        f._maybe_retune_timeout()
        assert f.timeout == 1.0

    def test_applies_once_the_distribution_is_settled(self, tmp_path):
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._maybe_retune_timeout()
        assert f.timeout == pytest.approx(0.04)
        assert f._timeout_retunes == [(5_000, 1.0, pytest.approx(0.04))]

    def test_hysteresis_suppresses_a_small_change(self, tmp_path):
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._exec_time_tracker.suggested_timeout.return_value = 1.1  # +10%
        f._maybe_retune_timeout()
        assert f.timeout == 1.0
        assert f._timeout_retunes == []

    def test_cooldown_blocks_a_second_retune(self, tmp_path):
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._maybe_retune_timeout()
        assert len(f._timeout_retunes) == 1

        f._exec_time_tracker.suggested_timeout.return_value = 0.5
        f.exec_count += ADAPTIVE_TIMEOUT_COOLDOWN_EXECS - 1
        f._maybe_retune_timeout()
        assert len(f._timeout_retunes) == 1

        f.exec_count += 1
        f._maybe_retune_timeout()
        assert len(f._timeout_retunes) == 2

    def test_clamped_to_the_floor(self, tmp_path):
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._exec_time_tracker.suggested_timeout.return_value = 1e-9
        f._maybe_retune_timeout()
        assert f.timeout == pytest.approx(ADAPTIVE_TIMEOUT_FLOOR)

    def test_clamped_to_the_growth_ceiling(self, tmp_path):
        """Loosening is allowed -- a too-tight deadline is a correctness
        problem, not just a slow one -- but not without bound."""
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._exec_time_tracker.suggested_timeout.return_value = 10_000.0
        f._maybe_retune_timeout()
        assert f.timeout == pytest.approx(1.0 * ADAPTIVE_TIMEOUT_MAX_GROWTH)


class TestRetunePropagation:
    def test_every_runner_follows(self, tmp_path):
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._inprocess_runner = MagicMock(timeout=1.0)
        f._persistent_runner = MagicMock(timeout=1.0)
        f._maybe_retune_timeout()
        assert f._inprocess_runner.timeout == pytest.approx(0.04)
        assert f._persistent_runner.timeout == pytest.approx(0.04)

    def test_loader_refusal_moves_nothing(self, tmp_path):
        """The invariant. A loader that will not take the new deadline must
        leave self.timeout, and every runner, exactly where they were."""
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._inprocess_runner = MagicMock(timeout=1.0)
        f._forkserver = MagicMock()
        f._forkserver.set_timeout.return_value = False

        f._maybe_retune_timeout()

        assert f.timeout == 1.0
        assert f._inprocess_runner.timeout == 1.0
        assert f._timeout_retunes == []
        # And it stops trying: the capability cannot appear mid-run.
        assert f._adaptive_timeout is False

    def test_the_loaders_value_wins_over_the_request(self, tmp_path):
        """The loader clamps. Recording the request instead of the echo is
        how the two sides come to disagree in the first place."""
        f = _fuzzer(tmp_path)
        f.exec_count = 5_000
        f._forkserver = MagicMock()
        f._forkserver.set_timeout.return_value = True
        f._forkserver.timeout = 0.001  # clamped by arm_timeout's floor

        f._maybe_retune_timeout()

        assert f.timeout == pytest.approx(0.001)
        assert f._timeout_retunes == [(5_000, 1.0, pytest.approx(0.001))]


# ── The loader half, exercised against the real binary ──────────────────


@pytest.fixture(scope="module")
def loader():
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    return _ensure_compiled()


@pytest.fixture(scope="module")
def slow_target(tmp_path_factory):
    """A .so that sleeps 300ms, so a deadline either fires or does not."""
    import shutil

    cc = shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("retune")
    src = d / "slow.c"
    src.write_text(
        textwrap.dedent(
            """
            #include <stddef.h>
            #include <unistd.h>
            int fuzz_slow(const unsigned char *d, size_t n) {
                (void)d; (void)n;
                usleep(300000);
                return 0;
            }
            """
        )
    )
    so = d / "slow.so"
    r = subprocess.run(
        [cc, "-O0", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:200]}")
    return str(so)


class TestLoaderRetune:
    def test_ready_advertises_the_capability(self, loader, slow_target):
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=2.0)
        try:
            assert r.start()
            assert "retune" in r.capabilities
            # The mode token is unchanged -- capabilities are additive.
            assert r.exec_mode == "dlopen"
        finally:
            r.stop()

    def test_tightening_makes_a_passing_input_time_out(self, loader, slow_target):
        """End to end: the same input, the same loader process, different
        deadline. Anything less than this can be satisfied by a Python-side
        variable that the loader never hears about."""
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=2.0)
        try:
            assert r.start()
            rc, _ = r.run_one(b"x")
            assert rc == 0, "300ms target should complete under a 2s deadline"

            assert r.set_timeout(0.05) is True
            assert r.timeout == pytest.approx(0.05)

            rc, _ = r.run_one(b"x")
            assert rc == -1, "300ms target should time out under a 50ms deadline"
        finally:
            r.stop()

    def test_loosening_again_restores_completion(self, loader, slow_target):
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=0.05)
        try:
            assert r.start()
            assert r.run_one(b"x")[0] == -1
            assert r.set_timeout(2.0) is True
            assert r.run_one(b"x")[0] == 0
        finally:
            r.stop()

    def test_protocol_stays_in_sync_after_a_retune(self, loader, slow_target):
        """TIMEOUT has a reply, so a caller that did not read it would leave
        TIMEOUT_OK in the pipe to be parsed as the next RC header."""
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=2.0)
        try:
            assert r.start()
            for _ in range(3):
                assert r.set_timeout(1.5) is True
                assert r.run_one(b"x")[0] == 0
        finally:
            r.stop()

    def test_declines_without_the_capability(self, loader, slow_target):
        """A loader built before TIMEOUT existed ignores it silently, so the
        adapter must not claim success. Simulated by clearing the parsed
        capability set -- the binary here does support it."""
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=2.0)
        try:
            assert r.start()
            r.capabilities = frozenset()
            assert r.set_timeout(0.05) is False
            assert r.timeout == 2.0
            # The run still works: nothing was written to the pipe.
            assert r.run_one(b"x")[0] == 0
        finally:
            r.stop()

    def test_rejects_a_nonpositive_timeout(self, loader, slow_target):
        r = ForkserverRunner(slow_target, function_name="fuzz_slow", timeout=2.0)
        try:
            assert r.start()
            assert r.set_timeout(0.0) is False
            assert r.set_timeout(-1.0) is False
            assert r.timeout == 2.0
        finally:
            r.stop()
