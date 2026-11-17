"""--respawn-backoff: paced restarts of a target that keeps dying at once.

Port of soksolv restart_backoff.go. Fake clock and sleep throughout: no test
sleeps for real.
"""

from __future__ import annotations

import ast
import ctypes
import inspect

import pytest

from fuzzer_tool.adapters import persistent_signal
from fuzzer_tool.adapters.forkserver import ForkserverRunner
from fuzzer_tool.adapters.persistent_signal import PersistentRunner
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.adapters.respawn_backoff import (
    BASE_SLEEP_S,
    BREAKER_QUICK,
    CAP_SLEEP_S,
    DEADLINE_MARGIN_S,
    NO_BACKOFF,
    QUICK_LIFE_S,
    Failure,
    RespawnBackoff,
    RespawnLoopError,
)
from fuzzer_tool.core.run_mode import Switch

# Inside the quick window, and well past it.
SHORT_LIFE = QUICK_LIFE_S / 10
LONG_LIFE = QUICK_LIFE_S * 4


class FakeTime:
    """Monotonic clock that only ``sleep`` and ``advance`` move."""

    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, d: float) -> None:
        self.slept.append(d)
        self.now += d

    def advance(self, d: float) -> None:
        self.now += d


def _backoff(t: FakeTime, **kw) -> RespawnBackoff:
    return RespawnBackoff(now=t.monotonic, sleep=t.sleep, **kw)


def _die_after(b: RespawnBackoff, t: FakeTime, life: float) -> float:
    """Start an instance, let it live *life* seconds, report its death."""
    b.started()
    t.advance(life)
    return b.failed(Failure.DEATH)


# --- Policy -------------------------------------------------------------------


class TestPolicy:
    def test_quick_deaths_double_then_cap(self):
        t = FakeTime()
        b = _backoff(t)
        for _ in range(8):
            _die_after(b, t, SHORT_LIFE)

        # 20 ms << quick, quick counted before the increment, capped at 1 s.
        assert t.slept == pytest.approx([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.0])

    def test_long_life_resets_and_does_not_sleep(self):
        """Falsification: a phase past the quick window never pays."""
        t = FakeTime()
        b = _backoff(t)
        _die_after(b, t, SHORT_LIFE)
        _die_after(b, t, SHORT_LIFE)

        assert _die_after(b, t, LONG_LIFE) == 0.0
        _die_after(b, t, SHORT_LIFE)

        assert t.slept == pytest.approx([BASE_SLEEP_S, 2 * BASE_SLEEP_S, BASE_SLEEP_S])

    def test_boundary_life_counts_as_long(self):
        """Edge: exactly QUICK_LIFE_S is not quick (soksolv `>=`)."""
        t = FakeTime()
        b = _backoff(t)

        assert _die_after(b, t, QUICK_LIFE_S) == 0.0
        assert t.slept == []

    def test_start_failure_always_quick(self):
        """A failed start is quick however long the start itself took."""
        t = FakeTime()
        b = _backoff(t)
        b.started()
        t.advance(LONG_LIFE)

        assert b.failed(Failure.START) == pytest.approx(BASE_SLEEP_S)

    def test_deadline_clamps_sleep(self):
        t = FakeTime()
        b = _backoff(t, deadline=t.now + DEADLINE_MARGIN_S + 0.025)
        for _ in range(3):
            _die_after(b, t, 0.0)

        # 20 ms fits under 25 ms of headroom; then 5 ms of it remain, then none.
        assert t.slept == pytest.approx([0.02, 0.005])

    def test_deadline_passed_never_sleeps_negative(self):
        """Adversarial: a deadline already behind us must not sleep at all."""
        t = FakeTime()
        b = _backoff(t, deadline=t.now - 10.0)

        assert _die_after(b, t, 0.0) == 0.0
        assert t.slept == []

    def test_cap_reached_at_six_shifts(self):
        assert BASE_SLEEP_S * (1 << 6) > CAP_SLEEP_S


class TestBreaker:
    def test_trips_after_k_quick_failures(self):
        t = FakeTime()
        b = _backoff(t)
        for _ in range(BREAKER_QUICK - 1):
            _die_after(b, t, SHORT_LIFE)
        slept = len(t.slept)

        with pytest.raises(RespawnLoopError, match="respawn"):
            _die_after(b, t, SHORT_LIFE)
        assert len(t.slept) == slept  # trips instead of sleeping

    def test_alternating_lives_never_trip(self):
        """Adversarial: one healthy phase in between resets the count."""
        t = FakeTime()
        b = _backoff(t)
        for _ in range(BREAKER_QUICK * 3):
            _die_after(b, t, SHORT_LIFE)
            _die_after(b, t, LONG_LIFE)

        assert max(t.slept) == pytest.approx(BASE_SLEEP_S)

    def test_custom_breaker(self):
        t = FakeTime()
        b = _backoff(t, breaker=2)
        b.failed(Failure.START)

        with pytest.raises(RespawnLoopError):
            b.failed(Failure.START)


class TestNullBackoff:
    def test_noop(self):
        NO_BACKOFF.started()

        assert NO_BACKOFF.failed(Failure.START) == 0.0


# --- Runner sites ---------------------------------------------------------------


class _DeadStdin:
    def write(self, _b):
        raise BrokenPipeError

    def flush(self):
        pass

    def close(self):
        pass


class _DeadProc:
    stdin = _DeadStdin()
    stdout = None
    stderr = None

    pid = None

    def poll(self):
        return 1

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 1


def _armed(runner, t: FakeTime):
    runner.attach_backoff(_backoff(t))
    t.advance(SHORT_LIFE)
    return runner


class TestForkserverSite:
    def test_dead_loader_restart_is_paced(self, monkeypatch):
        t = FakeTime()
        r = ForkserverRunner("/fake/target")
        monkeypatch.setattr(r, "start", lambda: False)
        _armed(r, t)
        r._ready = True
        r._proc = _DeadProc()

        assert r.run_one(b"x") == (-2, "")
        assert t.slept == pytest.approx([BASE_SLEEP_S])

    def test_flag_off_never_sleeps(self, monkeypatch):
        """Falsification: no backoff attached -> same result, no pause."""
        slept = []
        monkeypatch.setattr("time.sleep", slept.append)
        r = ForkserverRunner("/fake/target")
        monkeypatch.setattr(r, "start", lambda: False)
        r._ready = True
        r._proc = _DeadProc()

        assert r._backoff is NO_BACKOFF
        assert r.run_one(b"x") == (-2, "")
        assert slept == []


class TestPersistentLoaderSite:
    def test_dead_loader_restart_is_paced(self, monkeypatch):
        t = FakeTime()
        r = PersistentLoader("/fake/target.so")
        monkeypatch.setattr(r, "start", lambda: False)
        _armed(r, t)
        r._ready = True
        r._proc = _DeadProc()

        assert r.run_one(b"x") == (-2, None)
        assert t.slept == pytest.approx([BASE_SLEEP_S])

    def test_hang_restart_is_paced(self, monkeypatch):
        t = FakeTime()
        r = PersistentLoader("/fake/target.so")
        monkeypatch.setattr(r, "start", lambda: False)
        monkeypatch.setattr(r, "_kill_orphaned_child", lambda: None)
        _armed(r, t)
        r._proc = None

        assert r._on_hang() == (-1, None)
        assert t.slept == pytest.approx([BASE_SLEEP_S])


class TestPersistentSignalSite:
    def test_failed_restart_is_paced(self, monkeypatch):
        t = FakeTime()
        r = PersistentRunner("/fake/target")
        monkeypatch.setattr(r, "start", lambda: False)
        r.attach_backoff(_backoff(t))

        assert r.restart() is False
        assert t.slept == pytest.approx([BASE_SLEEP_S])

    def test_vanished_target_is_paced(self, monkeypatch):
        t = FakeTime()
        r = PersistentRunner("/fake/target")
        r.attach_backoff(_backoff(t))
        t.advance(SHORT_LIFE)
        buf = ctypes.create_string_buffer(r.map_size)
        r.shm_ptr = ctypes.addressof(buf)
        r._started, r.pid = True, 1 << 30

        def gone(_pid, _sig):
            raise ProcessLookupError

        monkeypatch.setattr(persistent_signal.os, "kill", gone)

        assert r.run_one(b"x") == (-2, "target process not found")
        assert t.slept == pytest.approx([BASE_SLEEP_S])

    def test_crash_restart_is_not_paced(self, monkeypatch):
        """Adversarial: a crash is a finding, not a respawn fault."""
        t = FakeTime()
        r = PersistentRunner("/fake/target")
        monkeypatch.setattr(r, "start", lambda: True)
        r.attach_backoff(_backoff(t))

        assert r.restart() is True
        assert t.slept == []


class TestInProcessForward:
    def test_attach_reaches_inner_runner(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        r = InProcessRunner.__new__(InProcessRunner)
        r._persistent = PersistentLoader("/fake/target.so")
        r._forkserver = None
        b = RespawnBackoff()
        r.attach_backoff(b)

        assert r._persistent._backoff is b


# --- Fuzzer + CLI wiring ----------------------------------------------------------


def _real_fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True, exist_ok=True)
    crashes.mkdir(parents=True, exist_ok=True)
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        **kw,
    )


class TestFuzzerWiring:
    def test_on_arms_every_runner(self, tmp_path):
        f = _real_fuzzer(tmp_path, respawn_backoff=Switch.ON)
        f._forkserver = ForkserverRunner("/fake/target")
        f._persistent_runner = PersistentRunner("/fake/target")
        f._attach_backoff()

        assert isinstance(f._forkserver._backoff, RespawnBackoff)
        assert isinstance(f._persistent_runner._backoff, RespawnBackoff)
        assert f._forkserver._backoff is not f._persistent_runner._backoff

    def test_off_leaves_null(self, tmp_path):
        f = _real_fuzzer(tmp_path)
        f._forkserver = ForkserverRunner("/fake/target")
        f._attach_backoff()

        assert f._forkserver._backoff is NO_BACKOFF


def test_cli_flag_reaches_fuzzer_not_hail_mary():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    assert all("respawn_backoff" in {k.arg for k in c.keywords} for c in calls)
    assert "respawn_backoff" in dests
    assert "respawn_backoff" not in commands._HAIL_MARY_FLAGS
