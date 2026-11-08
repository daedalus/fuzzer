"""FrontierOracle.drain() must not block forever on a worker that is gone.

``_loop`` had no exception handling: one seed whose ``measure()`` raised
(llvm-profdata missing or failing) killed the daemon worker silently. Seeds
submitted afterwards stayed queued and ``drain()`` waited on
``while queue or busy`` with nobody left to wake it. The same wait hung after
``close()`` with work still queued, and when ``_fold`` raised before ``_busy``
was cleared.

No clang/llvm needed: ``Coverage`` is replaced by a fake.
"""

from __future__ import annotations

import threading

import pytest

import fuzzer_tool.adapters.frontier_cov as fc

_COND = ("t.c", 1, 1)


class _FlakyCoverage:
    """measure() fails for tags in *bad*, else reports one executed branch."""

    bad: set[str] = set()

    def __init__(self, *_a, **_k) -> None:
        pass

    def measure(self, tag, data):
        if tag in self.bad:
            raise FileNotFoundError("llvm-profdata")
        return {_COND: (1, 0)}


@pytest.fixture
def oracle_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "Coverage", _FlakyCoverage)
    made = []

    def make(bad=(), mode=fc.OracleMode.THREAD):
        _FlakyCoverage.bad = set(bad)
        o = fc.FrontierOracle(tmp_path / "bin", tmp_path / f"w{len(made)}", mode=mode)
        made.append(o)
        return o

    yield make
    _FlakyCoverage.bad = set()
    for o in made:
        o.close()


def _drain_returns(o, timeout=5.0) -> bool:
    """True if drain() came back; a regression hangs, which must fail rather than stall."""
    t = threading.Thread(target=o.drain, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def test_regression_worker_survives_measure_failure(oracle_factory):
    o = oracle_factory(bad={"s1"})  # the first seed's tag
    o.submit("a", b"x")
    assert _drain_returns(o)
    o.submit("b", b"y")  # needs a live worker; the old one was dead by now
    assert _drain_returns(o)
    assert o._thread.is_alive()
    # The unmeasurable seed is recorded with zero reach, the next one is measured.
    assert o.open_reached("a") == frozenset()
    assert o.stats()["seeds"] == 2
    assert o.stats()["pending"] == 0


def test_regression_drain_returns_after_close_with_queued_seed(oracle_factory):
    o = oracle_factory()
    o.close()  # worker exits
    o.submit("a", b"x")  # still accepted, nobody will measure it
    assert o.stats()["pending"] == 1
    assert _drain_returns(o)


def test_regression_busy_cleared_when_fold_raises(oracle_factory, monkeypatch):
    o = oracle_factory(mode=fc.OracleMode.INLINE)

    def boom(*_a, **_k):
        raise ValueError("fold failed")

    monkeypatch.setattr(o, "_fold", boom)
    o.submit("a", b"x")
    with pytest.raises(ValueError):
        o._step()
    assert o._busy is False
