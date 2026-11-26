"""Regression: the Katz recompute budget ran on the virtual clock.

``KatzChannel.ensure_scores`` caps rebuild cost at 5% of elapsed time, both
read from its clock. The Fuzzer passed its decision clock, virtual by default,
which does not advance during a rebuild: every rebuild cost 0 s, the budget
never engaged, and Katz rebuilt every 50 execs. GStreamer under ``--elo all``:
47.5% of wall time in ``ensure_scores``, 13 eps against 114.
"""

from __future__ import annotations

import time

import numpy as np

from fuzzer_tool.core.clock import ClockMode
from fuzzer_tool.services.katz_channel import KatzChannel

from .test_katz_channel import _bits, _icfg
from .test_regression_katz_resume_contract import _FakeKatz, _fuzzer

SLOW_REBUILD_S = 0.02


def _capture_build(monkeypatch) -> dict:
    """Record KatzChannel.build's kwargs; return a fake channel."""
    seen: dict = {}

    def _build(*_a, **kw):
        seen.update(kw)
        return _FakeKatz()

    monkeypatch.setattr("fuzzer_tool.services.katz_channel.KatzChannel.build", staticmethod(_build))
    return seen


def _slow_channel(monkeypatch) -> KatzChannel:
    """Host-clock channel whose horizon rebuild takes SLOW_REBUILD_S."""
    import fuzzer_tool.services.katz_channel as kc

    real = kc.build_horizon_graph

    def _slow(*a, **k):
        time.sleep(SLOW_REBUILD_S)
        return real(*a, **k)

    monkeypatch.setattr(kc, "build_horizon_graph", _slow)
    ch = KatzChannel(_icfg(), {k: k for k in range(4)})
    ch.record(_bits({0}), seed_key="s")
    return ch


def test_regression_katz_host_clock(tmp_path, monkeypatch):
    """The Fuzzer builds Katz on host time even under the virtual clock."""
    seen = _capture_build(monkeypatch)
    f = _fuzzer(tmp_path)
    assert f._clock.mode is ClockMode.VIRTUAL

    clock = seen.get("clock")
    assert clock is None or clock.mode is ClockMode.WALL


def test_slow_rebuild_is_budgeted(monkeypatch):
    """Falsification: a costly rebuild holds back the next dirty one."""
    ch = _slow_channel(monkeypatch)
    ch.ensure_scores(force=True)
    assert ch._last_cost >= SLOW_REBUILD_S

    before = ch._last_recompute_exec
    ch.exec_count += 10_000
    ch.record(_bits({1}), seed_key="s2")
    ch.ensure_scores()
    assert ch._last_recompute_exec == before


def test_cheap_rebuild_keeps_exec_interval():
    """Adversarial: below the cost floor, only the exec interval gates."""
    ch = KatzChannel(_icfg(), {k: k for k in range(4)})
    ch.record(_bits({0}), seed_key="s")
    first = ch.ensure_scores(force=True)
    ch._last_cost = 0.0

    ch.exec_count += 10_000
    ch.record(np.array([False, True, False, False]), seed_key="s2")
    assert ch.ensure_scores() is not first
