"""Regression: CmplogCollector.stop() never closed its FIFO drain.

The FIFO sink is the default. stop() unlinked the FIFO path but left the
drain thread polling and both FIFO file descriptors open until exit.
"""

import os

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector


@pytest.fixture
def fifo_collector(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path), fifo_sink=True)
    c._ensure_log_path()
    yield c
    c.stop()


def _fd_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_regression_stop_closes_fifo_drain(fifo_collector):
    drain = fifo_collector._fifo
    assert drain._thread.is_alive()

    fifo_collector.stop()

    assert fifo_collector._fifo is None
    assert fifo_collector.log_path is None
    assert not os.path.exists(drain.path)
    assert not drain._thread.is_alive()
    assert not _fd_open(drain._rfd)
    assert not _fd_open(drain._keepalive_wfd)


def test_adversarial_stop_twice(fifo_collector):
    """A second stop() finds no drain and must not raise."""
    fifo_collector.stop()
    fifo_collector.stop()
    assert fifo_collector._fifo is None


def test_falsification_file_sink_has_no_drain(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c._ensure_log_path()
    assert c._fifo is None
    c.stop()
