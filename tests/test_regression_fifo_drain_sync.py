"""Regression: ``_FifoDrain.drain()`` returned only what the drain thread had
already read, so records still in the pipe slipped to the next collect and
`--clock virtual` had to give up the FIFO (and with it the forkserver).

The thread is replaced by an inert stub, so the test does not depend on
scheduling: every byte must come from ``drain()`` itself.
"""

from __future__ import annotations

import os

import pytest

from fuzzer_tool.core import cmplog


class _InertThread:
    """``threading.Thread`` stand-in that never runs its target."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


@pytest.fixture
def drain(tmp_path, monkeypatch):
    """A drain with no background reader; closed and unlinked afterwards."""
    monkeypatch.setattr(cmplog.threading, "Thread", _InertThread)
    made = []

    def _make(max_buffered=None):
        d = cmplog._FifoDrain(str(tmp_path / "cmplog.fifo"), max_buffered=max_buffered)
        made.append(d)
        return d

    yield _make
    for d in made:
        d.close()


def _write(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def test_regression_fifo_drain_reads_pipe(drain):
    """Bytes written before drain() are returned by it, not by a later call."""
    d = drain()
    payload = b"CMP 1 2\n" * 32
    _write(d.path, payload)

    assert d.drain() == payload
    assert d.drain() == b""


def test_falsify_drains_keep_exec_order(drain):
    """Two execs, two collects: each sees exactly its own records."""
    d = drain()
    first, second = b"A" * 100, b"B" * 100

    _write(d.path, first)
    assert d.drain() == first

    _write(d.path, second)
    assert d.drain() == second


def test_adversarial_cap_keeps_newest_bytes(drain):
    """Over the cap, the newest bytes survive however the stream was chunked."""
    cap = 64
    d = drain(max_buffered=cap)
    stream = bytes(range(256)) * 4

    _write(d.path, stream)

    assert d.drain() == stream[-cap:]
