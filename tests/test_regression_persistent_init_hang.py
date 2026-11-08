"""PersistentLoader.start() must not wait forever for READY.

``start()`` read the loader's answer to INIT with ``NO_DEADLINE``. A target
whose load blocks (a constructor stuck in a lock, a hung dlopen) hung the
fuzzer at startup with no message. The READY wait is now bounded; on timeout
the hung loader is killed so a later start() is not short-circuited by the
still-living process.

No compiler needed: dlopen() of a FIFO path blocks in open() until a writer
appears, which is exactly a loader that never answers.
"""

import os
import threading
import time

import pytest

import fuzzer_tool.adapters.persistent_subprocess as ps


def test_regression_start_times_out_when_loader_never_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_INIT_TIMEOUT", 1.0, raising=False)
    fifo = tmp_path / "blocks_dlopen.so"
    os.mkfifo(fifo)
    loader = ps.PersistentLoader(str(fifo))
    result: list = []
    t0 = time.monotonic()
    # Daemon thread + join timeout: a regression hangs, which must fail rather than stall pytest.
    th = threading.Thread(target=lambda: result.append(loader.start()), daemon=True)
    th.start()
    th.join(15)
    try:
        if th.is_alive():
            pytest.fail("start() still blocked waiting for READY")
        assert result == [False]
        assert time.monotonic() - t0 < 12
        # The hung loader was killed and cleared, so start() can be retried.
        assert loader._proc is None
        assert loader._ready is False
    finally:
        loader.stop()
        # Unblock a loader that a regression left stuck in open().
        fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        os.close(fd)
