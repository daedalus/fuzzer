"""Regressions from the third afl_shim.c audit (2026-10-19).

- an auxiliary segment (distance table, node bitmap) whose size could not be
  read back with IPC_STAT was accepted on trust: its header is a loop bound
- a node-bitmap header over the 256 MiB sanity limit returned without
  detaching the mapping and without counting the refusal
- the forkserver treated EINTR on its control-pipe read() as EOF and exited,
  so a signal landing on the idle server ended the campaign
- __afl_put_i64() negated INT64_MIN (signed overflow, undefined behaviour)
- characterization: a zero-capacity distance table was already refused (by
  __afl_attach_sized's head == 0 test); pinned so it stays that way
"""

from __future__ import annotations

import ctypes
import signal
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage, _alloc_segment
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import (
    _HEALTH_MAIN,
    MAP_ENTRIES,
    SHIM,
    _build,
    _clean_env,
    _Forkserver,
)

# __AFL_HEALTH_* order: attached, map_entries, seg_rejected, ...
ATTACHED, SEG_REJECTED = 0, 2

_FAIL_STAT_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/shm.h>
int shmctl(int id, int cmd, struct shmid_ds *buf) {
    static int (*real)(int, int, struct shmid_ds *);
    if (!real) real = (int (*)(int, int, struct shmid_ds *))dlsym(RTLD_NEXT, "shmctl");
    const char *f = getenv("FAIL_STAT_ID");
    if (cmd == IPC_STAT && f && atoi(f) == id) { errno = EACCES; return -1; }
    return real(id, cmd, buf);
}
"""


def _health(exe: Path, **env: str) -> list[int]:
    r = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=60, env=_clean_env(**env)
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    return [int(x) for x in r.stdout.split()]


# ── A: unverifiable segment size ───────────────────────────────────────


@requires_clang
def test_regression_aux_segment_refused_when_ipc_stat_fails(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-D__AFL_DISTANCE_MODE", "-fsanitize-coverage=trace-pc")
    src = tmp_path / "failstat.c"
    src.write_text(_FAIL_STAT_SRC)
    so = tmp_path / "failstat.so"
    r = subprocess.run(
        ["clang", "-shared", "-fPIC", str(src), "-o", str(so), "-ldl"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr

    shm = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({0x1000: 1.0})
    base = {
        "__AFL_SHM_ID": shm.env_id,
        "AFL_MAP_SIZE": str(MAP_ENTRIES),
        "__AFL_DIST_SHM_ID": table.env_id,
    }
    try:
        sane = _health(exe, **base)  # control: a valid table attaches
        failing = _health(exe, **base, LD_PRELOAD=str(so), FAIL_STAT_ID=table.env_id)
    finally:
        shm.cleanup()
        table.cleanup()

    assert sane[SEG_REJECTED] == 0, "control run must not refuse a valid table"
    assert failing[ATTACHED] == 1  # the edge table is unaffected
    assert failing[SEG_REJECTED] == 1  # but a size nobody could verify is refused


# ── B: node bitmap header over the sanity limit ────────────────────────

_NODES_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
int main(void) {
    uint64_t h[8] = {0};
    uint32_t n = __afl_shim_health(h, 8);
    struct shmid_ds ds;
    if (shmctl(atoi(getenv("__AFL_NODE_BITMAP_ID")), IPC_STAT, &ds) != 0) return 2;
    printf("%llu %llu %llu\n", (unsigned long long)(n > 2 ? h[2] : 0),
           (unsigned long long)ds.shm_nattch, 0ULL);
    return 0;
}
"""


@requires_clang
def test_regression_node_bitmap_insane_header_detached(tmp_path):
    exe = _build(tmp_path, _NODES_MAIN, "-D__AFL_DISTANCE_MODE", "-fsanitize-coverage=trace-pc")
    shm = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({0x1000: 1.0})
    over = (1 << 28) + 1
    try:
        shm_id, ptr = _alloc_segment(4 + over)
    except Exception:
        pytest.skip("SysV segment of 256 MiB not available here")
    ctypes.c_uint32.from_address(ptr).value = over  # fits the segment, fails the sanity cap
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True, timeout=60,
            env=_clean_env(
                __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
                __AFL_DIST_SHM_ID=table.env_id, __AFL_NODE_BITMAP_ID=str(shm_id),
            ),
        )  # fmt: skip
    finally:
        libc.shmdt(ctypes.c_void_p(ptr))
        libc.shmctl(shm_id, 0, None)  # IPC_RMID
        shm.cleanup()
        table.cleanup()

    assert r.returncode == 0, r.stderr
    rejected, nattch_target, _ = (int(x) for x in r.stdout.split())
    assert rejected >= 1, "the oversized header must be counted as a refusal"
    assert nattch_target == 1, (
        "the shim left the node segment mapped (only this test may be attached)"
    )


# ── C: forkserver read() interrupted by a signal ───────────────────────


@requires_clang
def test_regression_forkserver_survives_eintr_on_control_read(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <signal.h>
        static void nop(int s) { (void)s; }
        /* Before the shim's own constructor: a handler without SA_RESTART,
         * so a signal landing on the server's blocked read() fails EINTR. */
        __attribute__((constructor(101))) static void early(void) {
            struct sigaction sa;
            sa.sa_handler = nop;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;
            sigaction(SIGUSR1, &sa, 0);
        }
        int main(void) { return 3; }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    shm = ShmCoverage(size=MAP_ENTRIES)
    fs = _Forkserver(exe, [], _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES)))
    try:
        statuses = []
        for _ in range(20):
            for _ in range(3):
                fs.proc.send_signal(signal.SIGUSR1)
                time.sleep(0.005)
            statuses.append(fs.run_one())
    finally:
        fs.close()
        shm.cleanup()

    assert statuses == [3 << 8] * 20, statuses


# ── D: INT64_MIN in the integer formatter ──────────────────────────────


@requires_clang
def test_regression_put_i64_handles_int64_min(tmp_path):
    c = tmp_path / "t.c"
    c.write_text(
        textwrap.dedent(
            r"""
            #include <stdio.h>
            #include <stdint.h>
            int main(void) {
                const int64_t vals[] = {INT64_MIN, INT64_MIN + 1, -10, -1, 0, 1, INT64_MAX};
                for (unsigned i = 0; i < sizeof vals / sizeof *vals; i++) {
                    char buf[32];
                    char *e = __afl_put_i64(buf, vals[i]);
                    *e = 0;
                    puts(buf);
                }
                return 0;
            }
            """
        )
    )
    exe = tmp_path / "t"
    r = subprocess.run(
        [
            "clang", "-O1", "-g", "-D__AFL_CMPLOG=1", "-fsanitize=signed-integer-overflow",
            "-fno-sanitize-recover=all", "-include", str(SHIM), str(c), "-o", str(exe), "-ldl",
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=_clean_env())
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [
        "-9223372036854775808", "-9223372036854775807", "-10", "-1", "0", "1",
        "9223372036854775807",
    ]  # fmt: skip


# ── E: zero-capacity distance table (characterization) ─────────────────


@requires_clang
def test_regression_zero_capacity_distance_table_refused(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-D__AFL_DISTANCE_MODE", "-fsanitize-coverage=trace-pc")
    shm = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({0x1000: 1.0})
    ctypes.c_uint32.from_address(table._ptr).value = 0
    try:
        fields = _health(
            exe, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_DIST_SHM_ID=table.env_id,
        )  # fmt: skip
    finally:
        shm.cleanup()
        table.cleanup()

    assert fields[ATTACHED] == 1
    assert fields[SEG_REJECTED] == 1


# ── A (main map): unverifiable segment size ────────────────────────────


@requires_clang
def test_regression_main_segment_refused_when_ipc_stat_fails(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-fsanitize-coverage=trace-pc-guard")
    src = tmp_path / "failstat.c"
    src.write_text(_FAIL_STAT_SRC)
    so = tmp_path / "failstat.so"
    r = subprocess.run(
        ["clang", "-shared", "-fPIC", str(src), "-o", str(so), "-ldl"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    shm = ShmCoverage(size=MAP_ENTRIES)
    base = {"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(MAP_ENTRIES)}
    try:
        sane = _health(exe, **base)
        failing = _health(exe, **base, LD_PRELOAD=str(so), FAIL_STAT_ID=shm.env_id)
    finally:
        shm.cleanup()

    assert sane[ATTACHED] == 1
    assert failing[ATTACHED] == 0
    assert failing[SEG_REJECTED] >= 1
