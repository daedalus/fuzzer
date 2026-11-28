"""Regression: InProcessRunner.update_shm_after_resize left two stale SHM users.

* Its own inner forkserver (the .so loader fallback) kept the old
  ``__AFL_SHM_ID``: children attached a removed segment and coverage was lost.
  The Fuzzer only restarts its *own* forkserver, not this one.
* The cached ``_shm_ptr`` attachment was dropped without ``shmdt``, so every
  resize leaked a mapping of the removed segment.

Found while modelling map resize order for the TLA+ roadmap.
"""

from unittest.mock import MagicMock, patch

from fuzzer_tool.adapters import inprocess
from fuzzer_tool.adapters.inprocess import InProcessRunner

_NEW_SIZE = 1 << 17
_NEW_ID = "4242"
_OLD_PTR = 0x7F0000001000
_NEW_PTR = 0x7F0000100000


def _runner(**attrs) -> InProcessRunner:
    """A runner past __init__: no library, no shim, no persistent loader."""
    r = object.__new__(InProcessRunner)
    defaults = {
        "_shm_ptr": None,
        "shm_size": 0,
        "coverage_env_id": "1",
        "_persistent": None,
        "_forkserver": None,
        "_lib": None,
        "_shim_handle": None,
    }
    for k, v in {**defaults, **attrs}.items():
        setattr(r, k, v)
    return r


def test_regression_resize_restarts_inner_forkserver():
    fs = MagicMock()
    r = _runner(_forkserver=fs)
    r.update_shm_after_resize(_NEW_PTR, _NEW_SIZE, _NEW_ID)
    fs.update_shm_after_resize.assert_called_once_with(_NEW_ID, _NEW_SIZE)


def test_regression_resize_detaches_cached_ptr():
    r = _runner(_shm_ptr=_OLD_PTR)
    with patch.object(inprocess.libc_shm, "shmdt") as shmdt:
        r.update_shm_after_resize(_NEW_PTR, _NEW_SIZE, _NEW_ID)
    shmdt.assert_called_once_with(_OLD_PTR)
    assert r._shm_ptr is None


def test_resize_without_cache_or_loader_is_a_noop():
    """Adversarial: nothing attached, no inner loader -> no detach, no crash."""
    r = _runner()
    with patch.object(inprocess.libc_shm, "libc") as libc:
        r.update_shm_after_resize(_NEW_PTR, _NEW_SIZE, _NEW_ID)
    libc.shmdt.assert_not_called()
    assert r.coverage_env_id == _NEW_ID
    assert r.shm_size == _NEW_SIZE
