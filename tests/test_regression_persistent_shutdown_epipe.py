"""Regression: PersistentLoader teardown at interpreter exit must not spew EPIPE.

A runner alive at exit takes __del__ -> stop(). stop() writes ``QUIT\\n`` to a
dead loader; the flush fails and the bytes stay buffered. _close_streams()
returned early while ``sys.is_finalizing()``, so the stdin BufferedWriter's
dealloc flushed again and printed:

    Exception ignored in: <_io.BufferedWriter name=4>
    BrokenPipeError: [Errno 32] Broken pipe
"""

import os
import subprocess
import sys


def _top_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Child: a loader whose process already exited, torn down only by a
# module-global __del__ during finalization (mirrors InProcessRunner.__del__).
_CHILD = """
import subprocess, sys
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader

class Holder:
    def __init__(self):
        self.loader = PersistentLoader("unused.so")
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        proc.wait()
        self.loader._proc = proc

    def __del__(self):
        self.loader.stop()

HOLDER = Holder()
"""


def _run_child(src: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", src],
        env={**os.environ, "PYTHONPATH": os.path.join(_top_dir(), "src")},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_regression_persistent_shutdown_epipe():
    proc = _run_child(_CHILD)

    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr
    assert "Broken pipe" not in proc.stderr


def test_explicit_stop_on_dead_loader_is_quiet():
    """Adversarial: explicit stop() on a dead loader, outside finalization."""
    src = _CHILD + "\nHOLDER.loader.stop()\nHOLDER.loader.stop()\n"
    proc = _run_child(src)

    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr
    assert "Broken pipe" not in proc.stderr
