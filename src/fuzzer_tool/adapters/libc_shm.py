"""Correctly-bound ctypes handles for the System V shared-memory calls.

Every ad-hoc ``ctypes.CDLL("libc.so.6")`` in this tree that called ``shmat()``
without declaring ``restype`` was returning a **truncated** address: ctypes
defaults to ``c_int``, so the 64-bit attach address ``0x7fa8adb96000`` came back
sign-extended as ``0xffffffffadba6000``.  Dereferencing that (``memmove``,
``string_at``) segfaults or silently reads and writes the wrong page.

The truncation also hid a second hazard.  ``shmat()`` reports failure by
returning ``(void *) -1``.  Under the accidental ``c_int`` restype that arrives
as Python ``-1``, so the ``if ptr == -1`` checks at the call sites happened to
work -- *because* of the bug.  Under a correct ``c_void_p`` restype the same
value arrives as ``0xffffffffffffffff``, which is not equal to ``-1``, so
declaring the restype without also fixing the comparison converts a loud
failure into a silent one: attach errors would sail through and every
subsequent read would return garbage from an unmapped address.

Both halves belong together, in one place.  Import :func:`shmat`, :func:`shmdt`
and :func:`shmctl_rmid` from here rather than re-binding libc locally.

``adapters/shm.py`` carries its own equivalent bindings; they are correct and
predate this module, and are left alone deliberately.
"""

import ctypes
import ctypes.util

__all__ = [
    "IPC_CREAT",
    "IPC_PRIVATE",
    "IPC_RMID",
    "SHMAT_FAILED",
    "libc",
    "shmat",
    "shmctl_rmid",
    "shmdt",
    "shmget",
]

IPC_PRIVATE = 0
IPC_CREAT = 0o1000
IPC_RMID = 0

#: ``shmat()`` failure sentinel as seen through ``restype = c_void_p``: the
#: all-ones pointer-width value, i.e. ``(void *) -1`` reinterpreted unsigned.
SHMAT_FAILED = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1

_libc_name = ctypes.util.find_library("c")
libc = ctypes.CDLL(_libc_name or "libc.so.6", use_errno=True)

# key_t is int on Linux; c_long is what adapters/shm.py uses and is harmless
# for the register-width argument passing on every platform we build for.
libc.shmget.argtypes = [ctypes.c_long, ctypes.c_size_t, ctypes.c_int]
libc.shmget.restype = ctypes.c_int

libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libc.shmat.restype = ctypes.c_void_p

libc.shmdt.argtypes = [ctypes.c_void_p]
libc.shmdt.restype = ctypes.c_int

libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
libc.shmctl.restype = ctypes.c_int


def shmget(size: int, key: int = IPC_PRIVATE, perms: int = 0o600) -> int | None:
    """Allocate a ``size``-byte segment.  Returns the id, or None on failure.

    ``IPC_CREAT`` is always set: a caller reaching for this helper wants a new
    segment.  Use ``libc.shmget`` directly to attach to an existing key.
    """
    shm_id = libc.shmget(key, size, IPC_CREAT | perms)
    return None if shm_id < 0 else shm_id


def shmat(shm_id: int, flags: int = 0) -> int | None:
    """Attach ``shm_id`` and return its address, or None if the attach failed.

    Returning None rather than the ``(void *) -1`` sentinel keeps the failure
    check at every call site a plain falsiness test, which cannot be got wrong
    the way an ``== -1`` comparison can.
    """
    addr = libc.shmat(shm_id, None, flags)
    if not addr or addr == SHMAT_FAILED:
        return None
    return addr


def shmdt(addr: int | None) -> bool:
    """Detach ``addr`` if it is a real address.  Returns True if detached."""
    if not addr:
        return False
    return libc.shmdt(ctypes.c_void_p(addr)) == 0


def shmctl_rmid(shm_id: int | None) -> bool:
    """Mark ``shm_id`` for destruction.  Returns True on success."""
    if shm_id is None:
        return False
    return libc.shmctl(shm_id, IPC_RMID, None) == 0
