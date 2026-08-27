"""Regression test: ctypes ``shmat()`` bindings must declare ``restype``.

``adapters/persistent.py`` and ``services/minimize.py`` (twice) each built their
own ``ctypes.CDLL("libc.so.6")`` handle and called ``shmat()`` on it without
declaring ``restype``. ctypes then defaults to ``c_int``, so a 64-bit attach
address such as ``0x7fa8adb96000`` came back sign-extended as
``0xffffffffadba6000``. ``memmove`` through that pointer (persistent mode) and
``string_at`` through it (``minimize -c``) segfault or silently touch an
unmapped page. ``adapters/shm.py`` had the binding right all along, which is
what makes this a copy-the-neighbour bug rather than a knowledge gap -- it has
since been folded into :mod:`fuzzer_tool.adapters.libc_shm` too, so there is
one binding of these four calls in the package rather than a model and a copy.

The truncation also concealed the second half of the bug. ``shmat()`` signals
failure with ``(void *) -1``; under the accidental ``c_int`` restype that
arrives as Python ``-1``, so the ``if ptr == -1`` guards at the call sites
worked *because* the address was being truncated. Declaring the restype without
rewriting those guards would have turned every attach failure silent, since
``0xffffffffffffffff != -1``. Both halves are asserted here.

The last test is the "check the neighbour file" pass: it scans the package for
any ``.shmat(`` call in a module that neither declares the restype itself nor
routes through :mod:`fuzzer_tool.adapters.libc_shm`.
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fuzzer_tool.adapters import libc_shm  # noqa: E402

SRC = Path(__file__).parent.parent / "src" / "fuzzer_tool"

MAP_SIZE = 65536


@pytest.fixture
def segment():
    """A real IPC_PRIVATE segment, destroyed on teardown."""
    shm_id = libc_shm.shmget(MAP_SIZE)
    if shm_id is None:
        pytest.skip("SysV shared memory unavailable in this environment")
    yield shm_id
    libc_shm.shmctl_rmid(shm_id)


class TestBindings:
    def test_shmat_declares_pointer_restype(self):
        assert libc_shm.libc.shmat.restype is ctypes.c_void_p

    def test_pointer_arguments_are_declared(self):
        assert libc_shm.libc.shmdt.argtypes == [ctypes.c_void_p]
        assert libc_shm.libc.shmat.argtypes[1] is ctypes.c_void_p

    def test_failure_sentinel_is_pointer_width(self):
        assert (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1 == libc_shm.SHMAT_FAILED


class TestAttachAddress:
    def test_address_is_not_truncated(self, segment):
        """The bug: high 32 bits of the attach address were being discarded."""
        addr = libc_shm.shmat(segment)
        assert addr is not None
        try:
            # A c_int-truncated address is either negative or fits in 32 bits.
            # Linux maps SysV segments high, so a correct attach sets high bits.
            assert addr > 0
            assert addr == addr & libc_shm.SHMAT_FAILED
            round_tripped = ctypes.cast(addr, ctypes.c_void_p).value
            assert round_tripped == addr
        finally:
            libc_shm.shmdt(addr)

    def test_write_then_read_through_the_pointer(self, segment):
        """memmove/string_at through the attach address must survive."""
        addr = libc_shm.shmat(segment)
        assert addr is not None
        try:
            payload = bytes(range(256)) * 4
            ctypes.memmove(addr, payload, len(payload))
            assert ctypes.string_at(addr, len(payload)) == payload
        finally:
            libc_shm.shmdt(addr)

    def test_failed_attach_reports_none_not_minus_one(self):
        """The half-fix trap: (void *) -1 is not -1 under restype=c_void_p."""
        raw = libc_shm.libc.shmat(999999, None, 0)
        assert raw != -1, "sentinel does not compare equal to -1 -- guards must not use =="
        assert raw == libc_shm.SHMAT_FAILED
        assert libc_shm.shmat(999999) is None


class TestCallSites:
    """C1 and C4: the three sites named in docs/bugreport_2026-08-21_merged.md."""

    def test_minimize_read_shm_edges_round_trips(self, segment):
        from fuzzer_tool.services.minimize import _read_shm_edges

        addr = libc_shm.shmat(segment)
        assert addr is not None
        payload = bytes(range(256)) * 4
        ctypes.memmove(addr, payload, len(payload))
        libc_shm.shmdt(addr)

        edges = _read_shm_edges(str(segment), MAP_SIZE)
        assert len(edges) == MAP_SIZE
        assert bytes(edges[: len(payload)]) == payload
        assert not any(edges[len(payload) :])

    def test_minimize_read_shm_edges_survives_bad_id(self):
        from fuzzer_tool.services.minimize import _read_shm_edges

        assert _read_shm_edges("999999", 4096) == bytearray(4096)

    def test_persistent_runner_shm_write_path(self, segment):
        """The memmove at persistent.py that segfaulted on first use."""
        import struct

        from fuzzer_tool.adapters.persistent import PersistentRunner

        runner = PersistentRunner("/bin/true", map_size=MAP_SIZE)
        runner.shm_id = segment
        runner.shm_ptr = libc_shm.shmat(segment)
        assert runner.shm_ptr is not None

        data = b"A" * 512
        buf = struct.pack("<I", len(data)) + b"\x00" * 4 + data
        ctypes.memmove(runner.shm_ptr, buf, len(buf))
        assert ctypes.string_at(runner.shm_ptr, len(buf)) == buf

        runner.shm_id = None  # fixture owns destruction
        libc_shm.shmdt(runner.shm_ptr)
        runner.shm_ptr = 0

    def test_persistent_cleanup_detaches_and_destroys(self):
        from fuzzer_tool.adapters.persistent import PersistentRunner

        runner = PersistentRunner("/bin/true", map_size=MAP_SIZE)
        runner.shm_id = libc_shm.shmget(MAP_SIZE)
        if runner.shm_id is None:
            pytest.skip("SysV shared memory unavailable in this environment")
        runner.shm_ptr = libc_shm.shmat(runner.shm_id)
        assert runner.shm_ptr is not None

        runner._cleanup_shm()
        assert runner.shm_ptr == 0
        assert runner.shm_id is None


_SHMAT_CALL = re.compile(r"\.shmat\s*\(")
_RESTYPE_SET = re.compile(r"shmat\.restype\s*=")
#: The half-fix: restype declared, guard left comparing the address to -1.
_MINUS_ONE_GUARD = re.compile(r"ptr\s*(?:==|!=)\s*-\s*1\b")


def _strip_comments(text: str) -> str:
    """Drop trailing comments so prose about the defect is not mistaken for it."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_no_module_guards_an_attach_address_against_minus_one():
    """The other half: a declared restype makes ``ptr != -1`` dead code.

    ``adapters/inprocess.py`` sat in exactly this state -- all three of its
    attach sites set ``restype = c_void_p`` (so they passed the scan below)
    and then checked ``if ptr and ptr != -1``, which a failed attach passes.
    ``reset_bitmap()`` then wrote through the sentinel and segfaulted the
    fuzzer.  ``adapters/shm.py`` used to be the form to copy -- it compared
    against ``ctypes.c_void_p(-1).value`` rather than ``-1`` -- but it now
    routes through ``libc_shm`` like everything else, so no module in the
    package spells the sentinel comparison out any more.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        # Exempt as in the scan below: this module documents the sentinel in
        # its docstring, and its own behaviour is asserted by TestBindings.
        if path.name == "libc_shm.py":
            continue
        text = path.read_text(encoding="utf-8")
        if not _SHMAT_CALL.search(text):
            continue
        if _MINUS_ONE_GUARD.search(_strip_comments(text)):
            offenders.append(str(path.relative_to(SRC)))

    assert not offenders, (
        "attach address compared to -1 in: "
        + ", ".join(offenders)
        + " -- under restype=c_void_p a failed shmat() returns the all-ones "
        "pointer value, which is truthy and != -1; use libc_shm.shmat(), "
        "which returns None"
    )


def test_no_module_calls_shmat_without_a_declared_restype():
    """Catch the recurrence: a sibling file re-binding libc and forgetting."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "libc_shm.py":
            continue
        text = path.read_text(encoding="utf-8")
        if not _SHMAT_CALL.search(text):
            continue
        if _RESTYPE_SET.search(text) or "libc_shm" in text:
            continue
        offenders.append(str(path.relative_to(SRC)))

    assert not offenders, (
        "shmat() called without a declared c_void_p restype in: "
        + ", ".join(offenders)
        + " -- import fuzzer_tool.adapters.libc_shm instead of re-binding libc"
    )
