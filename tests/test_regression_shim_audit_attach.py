"""Regressions from the afl_shim.c audit (2026-11-12), SHM attach region.

docs/handover/handover_afl_shim_audit_2026-11-12.md. Each test builds a tiny
target against the real shim (shim statics are visible: it is -included).

Race tests are deterministic: x86 single-step (EFLAGS.TF) traps after every
instruction of the reader or writer, and the SIGTRAP handler either checks
an invariant or injects the other side's whole update at step k, for every k.
"""

from __future__ import annotations

import platform
import subprocess

import pytest

from fuzzer_tool.adapters.shm import (
    BlockBitmapShm,
    DistanceTableShm,
    NodeBitmapShm,
    ShmCoverage,
)
from fuzzer_tool.core.elf import parse_sancov_guard_count
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env

OLD_ENTRIES = 64
NEW_ENTRIES = 128
MAP_ENTRIES = 8192
GUARD = "-fsanitize-coverage=trace-pc-guard"

requires_x86_64 = pytest.mark.skipif(
    platform.machine() != "x86_64", reason="single-step harness is x86-64 only"
)

# Single-step harness: STEP_ON/STEP_OFF bracket the stepped call; on_step(n)
# runs before instruction n. A handler sets step_stop to end stepping early.
_STEP = r"""
#include <signal.h>
#include <ucontext.h>
#include <sys/shm.h>
#define TF_BIT 0x100
#define STEP_ON()  __asm__ volatile("pushfq; orq $0x100,(%%rsp); popfq" ::: "memory", "cc")
#define STEP_OFF() __asm__ volatile("pushfq; andq $-257,(%%rsp); popfq" ::: "memory", "cc")
static volatile long step_n;
static volatile int step_stop;
static void on_step(long n);
static void trap(int sig, siginfo_t *si, void *ucv) {
    (void)sig; (void)si;
    ucontext_t *uc = ucv;
    on_step(step_n++);
    if (step_stop) uc->uc_mcontext.gregs[REG_EFL] &= ~TF_BIT;
}
static void step_init(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = trap;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGTRAP, &sa, NULL);
}
"""

_HEALTH = """
#include <stdio.h>
static void health(void) {
    uint64_t h[8] = {0};
    __afl_shim_health(h, 8);
    printf("%llu %llu %llu\\n", (unsigned long long)h[0], (unsigned long long)h[1],
           (unsigned long long)h[2]);
}
"""


def _run(exe, *argv: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), *argv], capture_output=True, text=True, timeout=120, env=_clean_env(**env)
    )


def _health(r: subprocess.CompletedProcess) -> list[list[int]]:
    """Rows of (attached, entries, segments refused), one per health() call."""
    return [[int(x) for x in line.split()] for line in r.stdout.splitlines() if line.strip()]


# ── 1. Resize: retired segments stay mapped, size/area pair ──────────────

_RESIZE_KEEP = (
    _HEALTH
    + r"""
#include <stdlib.h>
int main(void) {
    volatile struct __afl_entry *stale = __afl_area;   /* a writer mid-edge */
    setenv("__AFL_SHM_ID", getenv("NEW_ID"), 1);
    setenv("AFL_MAP_SIZE", getenv("NEW_ENTRIES"), 1);
    __afl_map_shm();
    stale[0].count = 7;                                  /* old code: SEGV */
    health();
    return 0;
}
"""
)


@requires_clang
def test_regression_resize_keeps_old_segment_mapped(tmp_path):
    exe = _build(tmp_path, _RESIZE_KEEP)
    old, new = ShmCoverage(size=OLD_ENTRIES), ShmCoverage(size=NEW_ENTRIES)
    try:
        r = _run(
            exe, __AFL_SHM_ID=old.env_id, AFL_MAP_SIZE=str(OLD_ENTRIES),
            NEW_ID=new.env_id, NEW_ENTRIES=str(NEW_ENTRIES),
        )  # fmt: skip
    finally:
        old.cleanup()
        new.cleanup()

    assert r.returncode == 0, r.stderr
    assert _health(r)[-1][:2] == [1, NEW_ENTRIES]


_GRAVE_BOUND = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#ifndef __AFL_SHM_GRAVE
#define __AFL_SHM_GRAVE 0  /* shim without a graveyard */
#endif
static int nattch(const char *id) {
    struct shmid_ds ds;
    return shmctl(atoi(id), IPC_STAT, &ds) == 0 ? (int)ds.shm_nattch : -1;
}
int main(void) {
    const char *ids[2] = {getenv("__AFL_SHM_ID"), getenv("NEW_ID")};
    for (int i = 0; i < 4 * __AFL_SHM_GRAVE; i++) {
        setenv("__AFL_SHM_ID", ids[(i + 1) & 1], 1);
        __afl_map_shm();
    }
    printf("%d %d %d\n", nattch(ids[0]) + nattch(ids[1]), __AFL_SHM_GRAVE, __afl_area != NULL);
    return 0;
}
"""


@requires_clang
def test_regression_resize_graveyard_is_bounded(tmp_path):
    """Adversarial: endless re-attaches keep at most GRAVE retired mappings."""
    exe = _build(tmp_path, _GRAVE_BOUND)
    a, b = ShmCoverage(size=OLD_ENTRIES), ShmCoverage(size=OLD_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=a.env_id, AFL_MAP_SIZE=str(OLD_ENTRIES), NEW_ID=b.env_id)
    finally:
        a.cleanup()
        b.cleanup()

    assert r.returncode == 0, r.stderr
    attached, grave, live = (int(x) for x in r.stdout.split())
    python_maps = 2
    assert live == 1
    assert attached - python_maps <= grave + 1  # live + retired


# Reader: __afl_map_from_home for an edge homed at the last-but-one slot of a
# full OLD table (slot 0 free), so it must wrap. Writer: a whole resize to NEW,
# injected at step k. A reader that pairs the old table with the new size
# walks past slot OLD-1 into the segment tail instead of wrapping.
_RESIZE_PAIR = (
    _STEP
    + r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define EDGE 0xABCDEu
#define HOME (OLD - 2)
static volatile long inject_at;
static volatile int injected;
static char new_id[32], old_id[32];
static void on_step(long n) {
    if (n != inject_at) return;
    setenv("__AFL_SHM_ID", new_id, 1);
    setenv("AFL_MAP_SIZE", "128", 1);
    __afl_map_shm();
    injected = 1;
    step_stop = 1;
}
int main(void) {
    snprintf(old_id, sizeof old_id, "%s", getenv("__AFL_SHM_ID"));
    snprintf(new_id, sizeof new_id, "%s", getenv("NEW_ID"));
    uint8_t *chk = shmat(atoi(old_id), NULL, 0);
    struct __afl_entry *tab = (struct __afl_entry *)(chk + SHM_TABLE_OFFSET);
    uint8_t *tail = (uint8_t *)(tab + OLD);
    step_init();
    long bad = 0, k = 0;
    for (;; k++) {
        setenv("__AFL_SHM_ID", old_id, 1);
        setenv("AFL_MAP_SIZE", "64", 1);
        __afl_map_shm();
        memset(tab, 0, OLD * sizeof *tab + SHM_TAIL_BYTES);
        for (uint32_t i = 1; i < OLD; i++) { tab[i].edge_id = 1000 + i; tab[i].count = 1; }
        step_n = 0; inject_at = k; injected = 0; step_stop = 0;
        STEP_ON();
        __afl_map_from_home(EDGE, HOME, 0);
        STEP_OFF();
        if (!injected) break;
        for (int i = 0; i < SHM_TAIL_BYTES; i++) bad += tail[i] != 0;
    }
    printf("%ld %ld\n", k, bad);
    return 0;
}
"""
)


@requires_clang
@requires_x86_64
def test_regression_resize_never_pairs_old_table_new_size(tmp_path):
    exe = _build(tmp_path, _RESIZE_PAIR, f"-DOLD={OLD_ENTRIES}", "-mno-red-zone")
    old, new = ShmCoverage(size=OLD_ENTRIES), ShmCoverage(size=NEW_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=old.env_id, AFL_MAP_SIZE=str(OLD_ENTRIES), NEW_ID=new.env_id)
    finally:
        old.cleanup()
        new.cleanup()

    assert r.returncode == 0, r.stderr
    steps, bad = (int(x) for x in r.stdout.split())
    assert steps > 0  # the injection swept the whole call
    assert bad == 0


# ── 2. dladdr cache: no torn {ra, base} ──────────────────────────────────

_CACHE_COMMON = (
    _STEP
    + r"""
#include <stdio.h>
#define FAKE_BASE ((uintptr_t)0x5150000)
int main(void);
static uintptr_t key_a, key_b, base_a;
static uint32_t slot;
static void cache_setup(void) {
    Dl_info info;
    key_a = (uintptr_t)(void *)main + 1;
    dladdr((void *)key_a, &info);
    base_a = (uintptr_t)info.dli_fbase;
    key_b = key_a + 4u * __AFL_CTX_CACHE_SIZE;  /* same slot */
    slot = (uint32_t)((key_a >> 2) & (__AFL_CTX_CACHE_SIZE - 1));
    step_init();
}
static void cache_put(uintptr_t ra, uintptr_t base) {
    __afl_ctx_base_cache[slot].ra = ra;
    __afl_ctx_base_cache[slot].base = base;
}
"""
)

_CACHE_WRITER = (
    _CACHE_COMMON
    + r"""
static long bad;
static void on_step(long n) {
    (void)n;
    uintptr_t ra = __afl_ctx_base_cache[slot].ra, b = __afl_ctx_base_cache[slot].base;
    bad += (ra == key_a && b != base_a) || (ra == key_b && b != FAKE_BASE);
}
int main(void) {
    cache_setup();
    cache_put(key_b, FAKE_BASE);
    STEP_ON();
    uintptr_t got = __afl_ctx_resolve_base((void *)key_a);
    STEP_OFF();
    printf("%ld %ld %d\n", step_n, bad, got == base_a);
    return 0;
}
"""
)


@requires_clang
@requires_x86_64
def test_regression_ctx_cache_writer_never_torn(tmp_path):
    exe = _build(tmp_path, _CACHE_WRITER, "-mno-red-zone")
    r = _run(exe)

    assert r.returncode == 0, r.stderr
    steps, bad, ok = (int(x) for x in r.stdout.split())
    assert steps > 0
    assert ok == 1
    assert bad == 0


_CACHE_READER = (
    _CACHE_COMMON
    + r"""
static volatile long inject_at;
static volatile int injected;
static void on_step(long n) {
    if (n != inject_at) return;
    cache_put(key_b, FAKE_BASE);   /* another thread's whole update */
    injected = 1;
    step_stop = 1;
}
int main(void) {
    cache_setup();
    long bad = 0, k = 0;
    for (;; k++) {
        cache_put(key_a, base_a);
        step_n = 0; inject_at = k; injected = 0; step_stop = 0;
        STEP_ON();
        uintptr_t got = __afl_ctx_resolve_base((void *)key_a);
        STEP_OFF();
        if (!injected) break;
        bad += got != base_a;
    }
    printf("%ld %ld\n", k, bad);
    return 0;
}
"""
)


@requires_clang
@requires_x86_64
def test_regression_ctx_cache_reader_rejects_torn_hit(tmp_path):
    """Adversarial: a slot rewritten between the tag check and the base load."""
    exe = _build(tmp_path, _CACHE_READER, "-mno-red-zone")
    r = _run(exe)

    assert r.returncode == 0, r.stderr
    steps, bad = (int(x) for x in r.stdout.split())
    assert steps > 0
    assert bad == 0


# ── 3. Block bitmap without the distance channel ─────────────────────────

_BLOCK_SRC = """
static volatile int sink;
__attribute__((noinline)) static void high(int a) { sink = a * 2; }
__attribute__((noinline)) static void low(int a) { sink = a; }
int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] > '3') high(argc);
    else low(argc);
    return 0;
}
"""


def _block_run(tmp_path) -> tuple[int, int, int]:
    exe = _build(tmp_path, _BLOCK_SRC, GUARD, "-D__AFL_DISTANCE_MODE=0")
    shm, bmp = ShmCoverage(size=MAP_ENTRIES), BlockBitmapShm(num_bits=1 << 12)
    try:
        r = _run(
            exe, "9", __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_BLOCK_BITMAP_ID=bmp.env_id,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        return bmp.total(), bmp.covered(), parse_sancov_guard_count(str(exe))
    finally:
        shm.cleanup()
        bmp.cleanup()


@requires_clang
def test_regression_block_bitmap_without_distance_mode(tmp_path):
    total, covered, guards = _block_run(tmp_path)

    assert total == guards > 0
    assert covered > 0


@requires_clang
def test_falsify_block_bitmap_without_distance_mode_not_all_set(tmp_path):
    """A shim marking every guard would pass the test above."""
    total, covered, _ = _block_run(tmp_path)

    assert covered < total


# ── 4/5. Distance and node segments ─────────────────────────────────────

_AUX_RESIZE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
static int nattch(const char *env) {
    struct shmid_ds ds;
    return shmctl(atoi(getenv(env)), IPC_STAT, &ds) == 0 ? (int)ds.shm_nattch : -1;
}
int main(void) {
    for (int i = 0; i < 3; i++) __afl_map_shm();        /* resize re-attaches */
    printf("%d %d", nattch("__AFL_DIST_SHM_ID"), nattch("__AFL_NODE_BITMAP_ID"));
    setenv("__AFL_DIST_SHM_ID", getenv("DIST2_ID"), 1); /* a new table */
    __afl_map_shm();
    printf(" %u\n", *(uint32_t *)((uint8_t *)__afl_dist_table - 4));
    return 0;
}
"""


def _dist_table(n: int) -> DistanceTableShm:
    return DistanceTableShm({0x1000 + 0x10 * i: 1.0 for i in range(n)}, {})


@requires_clang
def test_regression_aux_segments_attach_once_per_id(tmp_path):
    exe = _build(tmp_path, _AUX_RESIZE)
    shm, node = ShmCoverage(size=MAP_ENTRIES), NodeBitmapShm(num_nodes=64)
    dist, dist2 = _dist_table(1), _dist_table(40)
    try:
        r = _run(
            exe, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_DIST_SHM_ID=dist.env_id, __AFL_NODE_BITMAP_ID=node.env_id,
            DIST2_ID=dist2.env_id,
        )  # fmt: skip
        cap2 = dist2.capacity
    finally:
        shm.cleanup()
        node.cleanup()
        dist.cleanup()
        dist2.cleanup()

    assert r.returncode == 0, r.stderr
    dist_n, node_n, cap = (int(x) for x in r.stdout.split())
    python_maps = 1
    assert dist_n == node_n == python_maps + 1
    assert cap == cap2  # adversarial: a changed id still switches tables


_DIST_HEADER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
int main(int argc, char **argv) {
    uint32_t *head = shmat(atoi(getenv("__AFL_DIST_SHM_ID")), NULL, 0);
    if (argv[2][0] == '1') *head = 1u << 30;  /* header rewritten after attach */
    __afl_probe_distance(strtoull(argv[1], NULL, 0));
    printf("%llu\n", (unsigned long long)__afl_dist_hits);
    return 0;
}
"""
DIST_KEY = 0x1000
FAR_KEY = (1 << 29) | 0x1000


def _dist_probe(tmp_path, key: int, rewrite: str = "1") -> subprocess.CompletedProcess:
    exe = _build(tmp_path, _DIST_HEADER)
    shm, dist = ShmCoverage(size=MAP_ENTRIES), _dist_table(1)
    try:
        return _run(
            exe, hex(key), rewrite, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_DIST_SHM_ID=dist.env_id,
        )  # fmt: skip
    finally:
        shm.cleanup()
        dist.cleanup()


@requires_clang
def test_regression_dist_probe_bound_fixed_at_attach(tmp_path):
    r = _dist_probe(tmp_path, FAR_KEY)

    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["0"]


@requires_clang
def test_regression_dist_probe_hits_after_header_rewrite(tmp_path):
    r = _dist_probe(tmp_path, DIST_KEY)

    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1"]


@requires_clang
def test_dist_probe_hits_untouched_header(tmp_path):
    """Control: the oracle sees a hit without any rewrite."""
    r = _dist_probe(tmp_path, DIST_KEY, rewrite="0")

    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1"]


# ── 6/7/8. Refusals ───────────────────────────────────────────────────────

_RETRY = (
    _HEALTH
    + r"""
#include <stdlib.h>
int main(void) {
    health();
    __afl_map_shm();                        /* inprocess.py's retry */
    setenv("__AFL_SHM_ID", getenv("ID2"), 1);
    __afl_map_shm();
    health();
    return 0;
}
"""
)
LONG_ID = "x" * 40


@requires_clang
def test_regression_long_refused_id_counted_once(tmp_path):
    exe = _build(tmp_path, _RETRY, GUARD)
    r = _run(exe, __AFL_SHM_ID=LONG_ID, ID2=LONG_ID)

    assert r.returncode == 0, r.stderr
    rows = _health(r)
    assert rows[0][2] == rows[1][2] == 1
    assert r.stderr.count("not a valid segment id") == 1


@requires_clang
def test_long_ids_differing_past_32_chars_are_distinct(tmp_path):
    """Adversarial: a hash or prefix match must not merge two ids."""
    exe = _build(tmp_path, _RETRY, GUARD)
    r = _run(exe, __AFL_SHM_ID=LONG_ID, ID2=LONG_ID + "y")

    assert r.returncode == 0, r.stderr
    assert _health(r)[1][2] == 2


M32 = ["clang", "-m32", "-O1", "-include", str(SHIM)]


def _links_m32() -> bool:
    r = subprocess.run(
        [*M32[:2], "-x", "c", "-", "-o", "/dev/null"],
        input="int main(void){return 0;}", capture_output=True, text=True,
    )  # fmt: skip
    return r.returncode == 0


@requires_clang
@pytest.mark.skipif(not _links_m32(), reason="no 32-bit toolchain")
def test_regression_overflowing_map_size_is_reported(tmp_path):
    (tmp_path / "t.c").write_text(_RETRY)
    exe = tmp_path / "t"
    cc = [*M32, GUARD, str(tmp_path / "t.c"), "-o", str(exe), "-ldl"]
    assert subprocess.run(cc, capture_output=True).returncode == 0
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE="4294967295", ID2=shm.env_id)
    finally:
        shm.cleanup()

    assert "AFL_MAP_SIZE" in r.stderr
    assert _health(r)[0][:1] == [0]


@requires_clang
@pytest.mark.parametrize("prefix", [" ", "+", "\t"])
def test_regression_shm_id_digits_only(tmp_path, prefix):
    exe = _build(tmp_path, _RETRY, GUARD)
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=prefix + shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), ID2="z")
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert _health(r)[0][0] == 0
    assert "not a valid segment id" in r.stderr


@requires_clang
def test_plain_shm_id_attaches(tmp_path):
    """Control: the digits-only check accepts a real id."""
    exe = _build(tmp_path, _RETRY, GUARD)
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), ID2=shm.env_id)
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert _health(r)[0][:2] == [1, MAP_ENTRIES]
