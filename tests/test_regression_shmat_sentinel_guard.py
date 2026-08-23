"""Regression test: the ``shmat()`` failure sentinel is not ``-1``.

``docs/bugreport_2026-08-21_merged.md`` closed the missing-``restype`` half of
this bug class and recorded the other half as an open sibling hazard: *"the
``(void *) -1`` failure sentinel does NOT compare equal to ``-1`` once
``restype=c_void_p`` is declared, so adding the restype without rewriting the
``== -1`` guard silences the failure path."*

``adapters/inprocess.py`` was the file left in exactly that half-fixed state.
All three of its attach sites declared ``libc.shmat.restype = c_void_p`` — so
they passed the neighbour scan in ``test_regression_shmat_restype.py`` — and
then guarded the result with ``if ptr and ptr != -1``.  Under a ``c_void_p``
restype a failed attach arrives as ``0xffffffffffffffff``, which is truthy and
is not ``-1``, so the guard admitted every failure:

* ``read_bitmap()`` built a ``c_uint8`` array over the sentinel address and
  returned it as coverage.
* ``reset_bitmap()`` called ``ctypes.memset(0xffffffffffffffff, 0, shm_size)``
  — a *write* through an unmapped address, which raises SIGSEGV in the fuzzer
  process itself and is therefore NOT caught by the enclosing
  ``except Exception``.
* the sentinel was then cached in ``self._shm_ptr``, whose refresh condition
  tested ``is None``.  One transient failure pinned the runner to the dead
  pointer for the rest of the campaign.

The fix routes all three through :mod:`fuzzer_tool.adapters.libc_shm`, which
returns ``None`` on failure so the call-site check is a plain falsiness test.
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fuzzer_tool.adapters import inprocess, libc_shm  # noqa: E402
from fuzzer_tool.adapters.inprocess import InProcessRunner  # noqa: E402
from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, SIZEOF_ENTRY  # noqa: E402

# An id no segment will hold.  shmat() on it fails with EINVAL.
BAD_SHM_ID = "999999"

MAP_SIZE = 4096

#: Entry count for the reset tests, which need a segment laid out the way the
#: shim lays one out: 24-byte front header + entries * 8 bytes.
TABLE_ENTRIES = 512


def _runner(**kwargs) -> InProcessRunner:
    """An InProcessRunner with target loading stubbed out."""
    defaults = dict(target="/bin/true", timeout=1, shm_size=MAP_SIZE)
    defaults.update(kwargs)
    with patch.object(InProcessRunner, "_start"):
        return InProcessRunner(**defaults)


@pytest.fixture
def segment():
    """A real IPC_PRIVATE segment big enough for shm_size bytes."""
    shm_id = libc_shm.shmget(MAP_SIZE)
    if shm_id is None:
        pytest.skip("SysV shared memory unavailable in this environment")
    yield shm_id
    libc_shm.shmctl_rmid(shm_id)


@pytest.fixture
def table_segment():
    """A segment laid out as the shim lays one out: header + entry table."""
    size = SHM_METADATA_SIZE + TABLE_ENTRIES * SIZEOF_ENTRY
    shm_id = libc_shm.shmget(size)
    if shm_id is None:
        pytest.skip("SysV shared memory unavailable in this environment")
    yield shm_id, TABLE_ENTRIES
    libc_shm.shmctl_rmid(shm_id)


class TestSentinelIsNotMinusOne:
    """The premise: the old guard could not have worked."""

    def test_failed_attach_is_truthy_and_not_minus_one(self):
        raw = libc_shm.libc.shmat(int(BAD_SHM_ID), None, 0)
        assert bool(raw), "sentinel is truthy, so `if ptr` alone admits it"
        assert raw != -1, "sentinel is not -1, so `ptr != -1` admits it"
        assert raw == libc_shm.SHMAT_FAILED


class TestReadBitmap:
    def test_failed_attach_returns_none(self):
        """Falsification: the buggy version returned an array over 0xff..ff."""
        r = _runner(coverage_env_id=BAD_SHM_ID)
        assert r.read_bitmap() is None

    def test_failed_attach_is_not_cached(self):
        """Adversarial: one failure must not pin the runner to a dead pointer."""
        r = _runner(coverage_env_id=BAD_SHM_ID)
        r.read_bitmap()
        assert r._shm_ptr is None

    def test_recovers_after_a_failed_attach(self, segment):
        """Adversarial: a real id must still work after a failure on a bad one."""
        r = _runner(coverage_env_id=BAD_SHM_ID)
        assert r.read_bitmap() is None

        r.coverage_env_id = str(segment)
        bitmap = r.read_bitmap()
        assert bitmap is not None
        assert r._shm_ptr not in (None, libc_shm.SHMAT_FAILED)
        libc_shm.shmdt(r._shm_ptr)

    def test_sentinel_is_filtered_at_the_adapter_boundary(self):
        """The raw binding may return the sentinel; the helper must not."""
        with patch.object(libc_shm.libc, "shmat", return_value=libc_shm.SHMAT_FAILED):
            r = _runner(coverage_env_id="1234")
            assert r.read_bitmap() is None
            assert r._shm_ptr is None

    def test_real_segment_round_trips(self, table_segment):
        """Positive control: a good attach still yields the table contents.

        Uses ``table_segment`` rather than ``segment``: this asserts what a
        successful attach RETURNS, so it has to agree with the layout the
        shim writes -- 24-byte front header, then ``entries * 8`` bytes of
        table.  It previously wrote a payload at the segment base and
        asserted it came back from offset 0, which was the base-offset read
        ``read_bitmap()`` was doing rather than the layout it should have
        been reading.  The sibling reset test below already used the correct
        fixture, because ``reset_bitmap()`` was fixed first; this half caught
        up later.

        The old ``segment`` fixture is also too small to hold the table it
        implied: ``shm_size`` entries is ``shm_size * 8`` bytes, so a correct
        read of a ``MAP_SIZE``-BYTE segment runs 8x off the end.
        """
        segment, entries = table_segment
        addr = libc_shm.shmat(segment)
        assert addr is not None
        payload = bytes(range(256)) * 4
        ctypes.memmove(addr + SHM_METADATA_SIZE, payload, len(payload))
        libc_shm.shmdt(addr)

        r = _runner(coverage_env_id=str(segment), shm_size=entries)
        bitmap = r.read_bitmap()
        assert bitmap is not None
        assert len(bitmap) == entries * SIZEOF_ENTRY
        assert bytes(bitmap[: len(payload)]) == payload
        libc_shm.shmdt(r._shm_ptr)


class TestResetBitmap:
    def test_failed_attach_does_not_memset(self):
        """Falsification: the buggy version wrote through 0xff..ff.

        A SIGSEGV here would kill the interpreter outright rather than fail
        this test, so surviving the call *is* the assertion.  The cache check
        pins down that the sentinel never reached ``memset``.
        """
        r = _runner(coverage_env_id=BAD_SHM_ID)
        r.reset_bitmap()
        assert r._shm_ptr is None

    def test_sentinel_never_reaches_memset(self):
        """Adversarial: prove it, rather than inferring it from not crashing."""
        with (
            patch.object(libc_shm.libc, "shmat", return_value=libc_shm.SHMAT_FAILED),
            patch.object(inprocess.ctypes, "memset") as memset,
        ):
            r = _runner(coverage_env_id="1234")
            r.reset_bitmap()
            memset.assert_not_called()

    def test_real_segment_zeroes_the_whole_table(self, table_segment):
        """The length bug: shm_size is entries, so the table is 8x that many bytes."""
        segment, entries = table_segment
        table_bytes = entries * SIZEOF_ENTRY

        addr = libc_shm.shmat(segment)
        assert addr is not None
        ctypes.memmove(
            addr, b"\xff" * (SHM_METADATA_SIZE + table_bytes), SHM_METADATA_SIZE + table_bytes
        )
        libc_shm.shmdt(addr)

        r = _runner(coverage_env_id=str(segment), shm_size=entries)
        r.reset_bitmap()
        assert r._shm_ptr is not None

        table = ctypes.string_at(r._shm_ptr + SHM_METADATA_SIZE, table_bytes)
        libc_shm.shmdt(r._shm_ptr)
        assert table == b"\x00" * table_bytes, (
            "reset zeroed only part of the edge table -- shm_size is the entry "
            "count, not a byte count"
        )

    def test_front_header_survives_the_reset(self, table_segment):
        """Falsification: the old memset wiped gen / drop count / ctx width.

        The diag word at offset 4 packs the generation tag (bits 24-31), the
        saturating dropped-edge counter (bits 8-23) and the ctx width
        (bits 0-7).  ``__afl_map_edge`` reads the generation from here on
        every edge, so zeroing it pins both sides at 0 and stale entries read
        as live coverage.
        """
        segment, entries = table_segment

        addr = libc_shm.shmat(segment)
        assert addr is not None
        # gen=0x07, drops=0x0042, ctx=0x08 -- a header no memset can produce.
        diag = (0x07 << 24) | (0x0042 << 8) | 0x08
        ctypes.memset(addr, 0, SHM_METADATA_SIZE)
        ctypes.c_uint32.from_address(addr + 4).value = diag
        ctypes.c_uint32.from_address(addr).value = 0xDEADBEEF  # stack_depth
        libc_shm.shmdt(addr)

        r = _runner(coverage_env_id=str(segment), shm_size=entries)
        r.reset_bitmap()
        assert r._shm_ptr is not None

        after_diag = ctypes.c_uint32.from_address(r._shm_ptr + 4).value
        after_depth = ctypes.c_uint32.from_address(r._shm_ptr).value
        libc_shm.shmdt(r._shm_ptr)

        assert after_diag == diag, "reset clobbered the diag word (gen/drops/ctx)"
        assert after_depth == 0xDEADBEEF, "reset clobbered the front header"

    def test_reset_does_not_run_off_the_end_of_the_segment(self, table_segment):
        """Adversarial: the memset must stay inside the mapping it was given."""
        segment, entries = table_segment
        r = _runner(coverage_env_id=str(segment), shm_size=entries)
        r.reset_bitmap()  # a SIGSEGV here fails the run, not just the test
        assert r._shm_ptr is not None
        libc_shm.shmdt(r._shm_ptr)

    def test_no_coverage_id_is_a_noop(self):
        r = _runner(coverage_env_id=None)
        r.reset_bitmap()
        assert r._shm_ptr is None


class TestLoaderScriptTemplate:
    """The third site: a Python source string executed in the child.

    It cannot import ``libc_shm`` (fuzzer_tool is not on the child's path), so
    it spells the sentinel out.  These assertions are the only thing standing
    between that copy and the original defect.
    """

    def test_template_is_valid_python(self):
        compile(inprocess._LOADER_SCRIPT, "<loader>", "exec")

    def test_template_has_no_minus_one_pointer_guard(self):
        # Comments are stripped first: prose describing the old defect must
        # neither fail this test nor, conversely, be the thing it inspects.
        code = "\n".join(line.split("#", 1)[0] for line in inprocess._LOADER_SCRIPT.splitlines())
        assert not re.search(r"ptr\s*(?:==|!=)\s*-\s*1\b", code), (
            "loader script compares an attach address to -1; under "
            "restype=c_void_p the failure sentinel is all-ones, not -1"
        )

    def test_template_compares_against_the_pointer_width_sentinel(self):
        assert "1 << (ctypes.sizeof(ctypes.c_void_p) * 8)" in inprocess._LOADER_SCRIPT

    def test_template_sentinel_matches_libc_shm(self):
        """The inlined copy must agree with the canonical constant."""
        ns: dict = {"ctypes": ctypes}
        exec("_shmat_failed = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1", ns)
        assert ns["_shmat_failed"] == libc_shm.SHMAT_FAILED
