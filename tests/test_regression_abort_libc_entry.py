"""Regression: glibc heap aborts in direct_lite were tiered LOW.

glibc's "free(): invalid pointer" goes to the fuzzer's own stderr in
direct_lite, never to the captured one, so the message rule could not fire:
fuzzgoat's invalid frees read LOW like a plain assert. The shim record's
module offsets name the libc function the target called; an abort inside
the allocator is heap corruption (HIGH).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import signal

from fuzzer_tool.adapters.filesystem import shim_exploitability
from fuzzer_tool.adapters.libc_symbols import libc_symbol

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
ABRT = int(signal.SIGABRT)


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _where(name: str, into: int = 0x10) -> tuple[str, int]:
    """(module path, offset) of a PC *into* bytes inside libc's *name*."""
    addr = ctypes.cast(getattr(_libc, name), ctypes.c_void_p).value
    info = _DlInfo()
    assert ctypes.CDLL(None).dladdr(ctypes.c_void_p(addr), ctypes.byref(info))
    return info.dli_fname.decode(), addr - info.dli_fbase + into


def _frame(module: str, offset: int) -> dict:
    return {"pc": f"0x{0x7F0000000000 + offset:x}", "module": module, "offset": offset}


def _sym(*frames: dict) -> dict:
    return {"signal": ABRT, "frames": list(frames)}


def test_libc_symbol_names_exported_function() -> None:
    mod, off = _where("free")

    assert libc_symbol(mod, off) in ("free", "__libc_free", "cfree")


def test_regression_abort_inside_free_high() -> None:
    abort = _frame(*_where("abort"))
    free = _frame(*_where("free"))
    target = _frame("/t/fuzzgoat.so", 0x11D6)

    assert shim_exploitability(ABRT, _sym(abort, free, target)) == "HIGH"


def test_abort_called_by_target_low() -> None:
    # Falsification: assert()/abort() straight from the target is no heap bug.
    raise_ = _frame(*_where("raise"))
    abort = _frame(*_where("abort"))
    target = _frame("/t/fuzzgoat.so", 0x11D6)

    assert shim_exploitability(ABRT, _sym(raise_, abort, target)) is None


def test_no_record_no_verdict() -> None:
    # Edge: no shim record (or no modules, old shim): no opinion.
    assert shim_exploitability(ABRT, {}) is None
    assert shim_exploitability(ABRT, _sym({"pc": "0x1", "module": "", "offset": 0})) is None


def test_other_libc_file_not_trusted() -> None:
    # Adversarial: a libc we have not loaded (container, other distro) is
    # not resolved against ours.
    assert libc_symbol("/other/root/lib/libc.so.6", 0x1234) == ""
    mod, off = _where("free")
    assert libc_symbol(mod, 1 << 40) == ""  # offset past the object
