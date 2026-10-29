"""Name a libc function from a crash record's ``module + offset``.

The shim's raw crash record (``__AFL_CRASH_SYM_RAW``) carries module and
offset, no names. A libc frame is resolved against the libc loaded in this
process (same file, so same symbols), with ``dladdr``: exported functions
only, which is what a target calls.
"""

import ctypes
import ctypes.util
import os


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_dl = ctypes.CDLL(None)
_dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
_dl.dladdr.restype = ctypes.c_int
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")


def _dladdr(addr: int) -> _DlInfo | None:
    info = _DlInfo()
    if not _dl.dladdr(ctypes.c_void_p(addr), ctypes.byref(info)) or not info.dli_fname:
        return None
    return info


def _loaded_libc() -> tuple[str, int]:
    """(real path, load base) of this process's libc; ("", 0) if unknown."""
    info = _dladdr(ctypes.cast(_libc.free, ctypes.c_void_p).value or 0)
    if info is None or not info.dli_fbase:
        return "", 0
    return os.path.realpath(info.dli_fname.decode()), info.dli_fbase


_LIBC_PATH, _LIBC_BASE = _loaded_libc()


def libc_symbol(module: str, offset: int) -> str:
    """Exported libc function at *module* + *offset*, e.g. "free"; "" when
    *module* is not our libc or the address is outside it."""
    if not _LIBC_BASE or os.path.realpath(module) != _LIBC_PATH:
        return ""
    info = _dladdr(_LIBC_BASE + offset)
    if info is None or not info.dli_sname or info.dli_fbase != _LIBC_BASE:
        return ""
    return info.dli_sname.decode()
