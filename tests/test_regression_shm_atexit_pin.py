"""Regression: ``atexit.register(self.cleanup)`` pinned every SHM segment.

A bound method holds a strong ref to its instance, so ``ShmCoverage``,
``DistanceTableShm`` and ``NodeBitmapShm`` were never collected: a dropped
``Fuzzer`` left its three segments attached and un-removed until process
exit. Measured on fuzzgoat: 3 segments per ``Fuzzer`` (6 after a second one).
"""

import gc
import subprocess
import sys
import weakref

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, NodeBitmapShm, ShmCoverage

_SYSV_SHM = "/proc/sysvipc/shm"
_SHMID_COL = 1


def _live_shm_ids() -> set[int]:
    """Segment ids the kernel still holds (removed ids vanish once detached)."""
    with open(_SYSV_SHM) as fh:
        next(fh)
        return {int(line.split()[_SHMID_COL]) for line in fh if line.strip()}


_FACTORIES = {
    "coverage": ShmCoverage,
    "distance": lambda: DistanceTableShm({0x1000: 1.5}),
    "bitmap": lambda: NodeBitmapShm(num_nodes=64),
}


@pytest.mark.parametrize("kind", sorted(_FACTORIES))
def test_regression_shm_atexit_pin(kind):
    """Falsification: dropping the last ref must release the segment."""
    obj = _FACTORIES[kind]()
    shm_id = obj.shm_id
    ref = weakref.ref(obj)

    del obj
    gc.collect()
    alive = ref()
    if alive is not None:
        alive.cleanup()  # hygiene: do not leak the segment on failure

    assert alive is None, f"{kind} pinned after its last ref dropped"
    assert shm_id not in _live_shm_ids()


_EXIT_SCRIPT = """
import gc
from fuzzer_tool.adapters.shm import ShmCoverage, NodeBitmapShm
kept = ShmCoverage()
for _ in range(8):
    NodeBitmapShm(num_nodes=8)
gc.collect()
print(kept.shm_id)
"""


def test_regression_shm_atexit_survivor_cleaned():
    """Adversarial: an instance alive at exit is still removed, and exit
    handlers for already-collected instances run without error."""
    proc = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT], capture_output=True, text=True, timeout=30
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert int(proc.stdout) not in _live_shm_ids()
