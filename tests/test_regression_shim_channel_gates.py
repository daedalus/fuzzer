"""Regressions from the second afl_shim.c review pass (2026-11-24).

Each case here failed before its fix:

  - record gates: with only $_CMPLOG_RECORDS_SHM attached (no $_CMPLOG_OUT)
    the memcmp family reached the segment but strcmp, strcasecmp, strstr,
    wmemcmp and friends were dropped -- every one of those recorders gated
    on the text fd instead of "any record sink is live"
  - segment ids: _CMPLOG_COUNTS_SHM / _CMPLOG_RECORDS_SHM / __AFL_SHM_FUZZ_ID
    were parsed with strtol, whose " 12" / "+12" prefix semantics violate the
    file's digits-only discipline (a malformed id must read as unset, not as
    its longest numeric prefix)
  - COMPCOV level 2 is documented to fire from the case-insensitive libc
    interceptors; strcasecmp / strncasecmp / wcscasecmp never marked

The distance-tail protocol fix (size acquire-ordered before the table,
mirroring __afl_map_from_home) is a hardening with no deterministic
interleaving to drive from a test; the commit message says so.

trace_switch's unclamped matched-case scan was reviewed and left alone:
test_regression_big_switch_counted pins that a case past the logging cap
still counts as matched (the clamp is a record-volume bound; the count is
compiler data).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.cmp_shm import CMP_SHM_ENV, CmpCounterShm
from fuzzer_tool.adapters.cmplog_rec_shm import CMPLOG_REC_ENV, CmpRecordShm
from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 8192

# One memcmp (control: never fd-gated) plus the recorders that were: the
# operands all differ, so no record is dropped as already-solved (result 0).
_LIBC_TARGET = r"""
#include <string.h>
#include <strings.h>
#include <wchar.h>
volatile int sink;
int main(void) {
    char a[8], b[8], h[8];
    memcpy(a, "AAAAB", 6);
    memcpy(b, "AAAAZ", 6);
    sink += memcmp(a, b, 5) == 0;
    sink += strcmp(a, b) == 0;
    sink += strcasecmp(a, b) == 0;
    memcpy(h, "hello", 6);
    sink += strstr(h, "world") == NULL;
    wchar_t wa[3] = L"AB", wb[3] = L"AZ";
    sink += wmemcmp(wa, wb, 2) != 0;
    return sink == 12345;
}
"""

_NOBUILTIN_LIBC = (
    "-fno-builtin-memcmp",
    "-fno-builtin-strcmp",
    "-fno-builtin-strcasecmp",
    "-fno-builtin-strstr",
    "-fno-builtin-wmemcmp",
)

# 7 matching bytes ("MAGICHD" vs "MAGICHDR") then a terminator mismatch,
# 6 matching bytes for the strncmp-shaped call: the compcov lower bound is
# derived from those, not echoed from the implementation.
_STRCASE_TARGET = r"""
#include <string.h>
#include <strings.h>
#include <wchar.h>
volatile int sink;
int main(void) {
    char buf[16];
    memcpy(buf, "MAGICHD", 8);
    sink += strcasecmp(buf, "MAGICHDR");
    sink += strncasecmp(buf, "MAGICHX", 6);
    sink += wcscasecmp(L"MAGICHD", L"MAGICHDR");
    return sink == 12345;
}
"""

_STRCASE_MARKS_MIN = 7 + 6


@pytest.fixture
def libc_exe(tmp_path):
    return _build(tmp_path, _LIBC_TARGET, "-D__AFL_CMPLOG=1", *_NOBUILTIN_LIBC)


@pytest.fixture
def strcase_exe(tmp_path):
    return _build(
        tmp_path,
        _STRCASE_TARGET,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin-strcasecmp",
        "-fno-builtin-strncasecmp",
        "-fno-builtin-wcscasecmp",
    )


@pytest.fixture
def rec_shm():
    # Small geometry: these tests only need a handful of slots, not the
    # production table.
    seg = CmpRecordShm(int_sites=64, int_hits=8, rec_slots=64, byte_slots=64)
    yield seg
    seg.close()


def _run(exe: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe)], capture_output=True, timeout=60, env=_clean_env(**env))


# ── Record gates: the segment is a sink too ────────────────────────────


@requires_clang
class TestRecordSegmentWithoutTextSink:
    """$_CMPLOG_RECORDS_SHM attached, $_CMPLOG_OUT unset.

    The memcmp control must pass before and after the fix; if it alone
    fails the harness is broken, not the gate under test.
    """

    def _byte_conds(self, libc_exe, rec_shm) -> list[tuple]:
        r = _run(libc_exe, **{CMPLOG_REC_ENV: rec_shm.env_id})
        assert r.returncode in (0, 1), r.stderr[-500:]
        assert rec_shm.attached, "shim never attached the records segment"
        return [c for c in rec_shm.drain() if c[4] is None]

    def test_control_memcmp_record_reaches_segment(self, libc_exe, rec_shm):
        conds = self._byte_conds(libc_exe, rec_shm)
        assert any(
            c[0].startswith(b"AAAAB") and c[1].startswith(b"AAAAZ") and c[3] != 0 for c in conds
        )

    def test_regression_strcmp_record_reaches_segment(self, libc_exe, rec_shm):
        conds = self._byte_conds(libc_exe, rec_shm)
        assert any(
            c[0].startswith(b"AAAAB\x00") and c[1].startswith(b"AAAAZ\x00") for c in conds
        ), "strcmp record lost when only the records segment is attached"

    def test_regression_strcasecmp_record_reaches_segment(self, libc_exe, rec_shm):
        conds = self._byte_conds(libc_exe, rec_shm)
        assert any(
            c[0].startswith(b"AAAAB\x00") and c[1].startswith(b"AAAAZ\x00") for c in conds
        ), "strcasecmp record lost when only the records segment is attached"

    def test_regression_strstr_record_reaches_segment(self, libc_exe, rec_shm):
        conds = self._byte_conds(libc_exe, rec_shm)
        assert any(c[0].startswith(b"hello") and c[1].startswith(b"world") for c in conds), (
            "strstr record lost when only the records segment is attached"
        )

    def test_regression_wmemcmp_record_reaches_segment(self, libc_exe, rec_shm):
        conds = self._byte_conds(libc_exe, rec_shm)
        assert any(
            c[0].startswith(b"A\x00\x00\x00B") and c[1].startswith(b"A\x00\x00\x00Z") for c in conds
        ), "wmemcmp record lost when only the records segment is attached"


# ── Segment ids: digits only, no strtol prefix ─────────────────────────


@requires_clang
class TestSegmentIdDigitsOnly:
    @pytest.mark.parametrize("prefix", [" ", "+"])
    def test_regression_counts_id_prefix_refused(self, libc_exe, prefix):
        shm = CmpCounterShm()
        try:
            r = _run(libc_exe, **{CMP_SHM_ENV: f"{prefix}{shm.env_id}"})
            assert r.returncode in (0, 1), r.stderr[-500:]
            assert not shm.attached, (
                f"'{prefix}{shm.env_id}' attached the counts segment: "
                "a malformed id must read as unset"
            )
        finally:
            shm.close()

    @pytest.mark.parametrize("prefix", [" ", "+"])
    def test_regression_records_id_prefix_refused(self, libc_exe, prefix):
        shm = CmpRecordShm(int_sites=64, int_hits=8, rec_slots=64, byte_slots=64)
        try:
            r = _run(libc_exe, **{CMPLOG_REC_ENV: f"{prefix}{shm.env_id}"})
            assert r.returncode in (0, 1), r.stderr[-500:]
            assert not shm.attached, (
                f"'{prefix}{shm.env_id}' attached the records segment: "
                "a malformed id must read as unset"
            )
        finally:
            shm.close()


# ── COMPCOV: the case-insensitive interceptors mark at level 2 ─────────


@requires_clang
class TestCompcovCaseInsensitive:
    def _edge_ids(self, strcase_exe, level: str | None) -> set[int]:
        shm = ShmCoverage(size=MAP_ENTRIES)
        try:
            env = {"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(MAP_ENTRIES)}
            if level is not None:
                env["__AFL_COMPCOV_LEVEL"] = level
            r = _run(strcase_exe, **env)
            assert r.returncode in (0, 1), r.stderr[-500:]
            return shm.get_edge_ids()
        finally:
            shm.cleanup()

    def test_regression_strcase_marks_edge_map(self, strcase_exe):
        off = self._edge_ids(strcase_exe, None)
        on = self._edge_ids(strcase_exe, "2")

        assert not off, "no instrumentation: unset level must leave the table empty"
        assert len(on) >= _STRCASE_MARKS_MIN, (
            "COMPCOV level 2 must mark strcasecmp/strncasecmp/wcscasecmp progress"
        )
        assert all(e & 0x80000000 for e in on), "compcov marks are synthetic (bit 31) ids"

    def test_regression_strcase_level_1_no_libc_marks(self, strcase_exe):
        on = self._edge_ids(strcase_exe, "1")
        assert not on, "libc interceptors fire at level 2 only; level 1 is const trace-cmp"
