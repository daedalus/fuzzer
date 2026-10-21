"""Shim review, group A: SHM (re)attach, write-guard escape, ctx stack bounds.

- attach published the new map size before the new table: edges fired by
  the (instrumented) attach code indexed the old table with the new size.
- a refused re-attach kept the stale table under the new size; a successful
  one never detached the old segment (one leaked mapping per resize).
- siglongjmp out of a guarded call skipped the write-guard close: the
  mprotect nesting count stayed raised and the segment writable for good.
- the mprotect nesting count was a plain ++/-- shared by threads.
- the caller-ctx walk bounded the saved frame pointer from above only: on a
  stack below the thread stack a junk frame pointer was dereferenced.
"""

from __future__ import annotations

import signal
import subprocess

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage
from fuzzer_tool.core.shim_health import ShimField, WriteGuard
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

OLD_ENTRIES = 64  # declared size of the first attach
BIG_ENTRIES = 1 << 16  # physical size of both segments, size of the re-attach
SMALL_ENTRIES = 1024  # an undersized re-attach target
SCAN_FROM = 2 * OLD_ENTRIES  # past the old table, its tail and its touched words
STRAY_ID = 0xDEADBEEF
BAD_ID = "banana"

_HEADER = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>

static unsigned nattch(int id) {
    struct shmid_ds d;
    return shmctl(id, IPC_STAT, &d) == 0 ? (unsigned)d.shm_nattch : 9999u;
}

static uint32_t guards[] = {101, 202, 303, 404, 505, 606, 707, 808};

static void fire(void) {
    for (unsigned i = 0; i < sizeof guards / sizeof guards[0]; i++)
        __sanitizer_cov_trace_pc_guard(&guards[i]);
}

static void health(uint64_t *h) {
    memset(h, 0, 16 * sizeof *h);
    __afl_shim_health(h, 16);
}
"""

# argv[1] = the constructor's segment id, then (id, size) re-attach pairs.
# One line per re-attach: attached entries nattch(old)-before nattch(old)-after.
_REATTACH = (
    _HEADER
    + """
int main(int argc, char **argv) {
    int old = atoi(argv[1]);
    fire();
    for (int i = 2; i + 1 < argc; i += 2) {
        unsigned before = nattch(old);
        setenv("__AFL_SHM_ID", argv[i], 1);
        setenv("AFL_MAP_SIZE", argv[i + 1], 1);
        __afl_map_shm();
        unsigned after = nattch(old);
        fire();
        uint64_t h[16];
        health(h);
        printf("%llu %llu %u %u\\n", (unsigned long long)h[ATTACHED],
               (unsigned long long)h[ENTRIES], before, after);
    }
    return 0;
}
""".replace("ATTACHED", str(int(ShimField.ATTACHED))).replace(
        "ENTRIES", str(int(ShimField.MAP_ENTRIES))
    )
)


@pytest.fixture(scope="module")
def reattach(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("reattach"),
        _REATTACH,
        "-fsanitize-coverage=trace-pc-guard",
        "-D__AFL_CTX_SENSITIVE=0",
    )


def _steps(exe, old: ShmCoverage, *pairs: tuple[str, int]) -> list[list[int]]:
    args = [str(exe), old.env_id]
    for seg, size in pairs:
        args += [seg, str(size)]
    env = _clean_env(__AFL_SHM_ID=old.env_id, AFL_MAP_SIZE=str(OLD_ENTRIES))
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (r.returncode, r.stderr)
    return [[int(x) for x in line.split()] for line in r.stdout.splitlines()]


def _beyond_old_table(cov: ShmCoverage) -> int:
    """Non-zero 8-byte words past the old declared table (should be none)."""
    table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    return int(np.count_nonzero(table["edge_id"][SCAN_FROM:]))


def _live(cov: ShmCoverage) -> int:
    table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    return int(np.count_nonzero(table["edge_id"]))


@pytest.fixture
def segs():
    made: list[ShmCoverage] = []

    def make(entries: int = BIG_ENTRIES) -> ShmCoverage:
        made.append(ShmCoverage(size=entries))
        return made[-1]

    yield make
    for cov in made:
        cov.cleanup()


# ── 1: size commits together with the table ─────────────────────────────


@requires_clang
def test_control_old_table_clean_without_reattach(reattach, segs):
    """Control: no re-attach, nothing lands past the declared old table."""
    old = segs()
    _steps(reattach, old)
    assert _live(old) > 0
    assert _beyond_old_table(old) == 0


@requires_clang
def test_regression_attach_size_commits_with_table(reattach, segs):
    """Edges fired during attach must not index the old table with the new size."""
    old, new = segs(), segs()
    (step,) = _steps(reattach, old, (new.env_id, BIG_ENTRIES))
    assert step[:2] == [1, BIG_ENTRIES]
    assert _beyond_old_table(old) == 0
    assert _live(new) > 0


# ── 2: re-attach detaches the old segment; refusal is consistent ────────


@requires_clang
def test_regression_reattach_detaches_old_segment(reattach, segs):
    old, new = segs(), segs()
    (step,) = _steps(reattach, old, (new.env_id, BIG_ENTRIES))
    _, _, before, after = step
    assert after == before - 1


@requires_clang
def test_reattach_same_segment_keeps_one_mapping(reattach, segs):
    """Adversarial: re-attaching the segment already mapped must not stack mappings."""
    old = segs()
    (step,) = _steps(reattach, old, (old.env_id, OLD_ENTRIES))
    attached, _, before, after = step
    assert attached == 1
    assert after == before


@requires_clang
def test_regression_refused_reattach_detaches(reattach, segs):
    """A refused re-attach leaves coverage off, not the stale table at the new size."""
    old, small = segs(), segs(SMALL_ENTRIES)
    (step,) = _steps(reattach, old, (small.env_id, BIG_ENTRIES))
    attached, _, before, after = step
    assert attached == 0
    assert after == before - 1
    assert _beyond_old_table(old) == 0
    assert _live(small) == 0


@requires_clang
def test_regression_refusal_not_sticky_across_ids(reattach, segs):
    """A refused id must not block a later, valid one (the next resize)."""
    old, new = segs(), segs()
    bad, good = _steps(reattach, old, (BAD_ID, BIG_ENTRIES), (new.env_id, BIG_ENTRIES))
    assert bad[0] == 0
    assert good[:2] == [1, BIG_ENTRIES]
    assert _live(new) > 0


# ── 3: write guard survives a siglongjmp out of a bracket ───────────────

# argv[1]: "bracket" crashes inside an open bracket (a timeout tick landing
# mid-store), "plain" crashes outside one. argv[2] "stray" then writes the
# table the way a wild store would.
_LANDING = (
    _HEADER
    + """
#include <signal.h>

static int in_bracket(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __afl_wguard_open_cold();
    raise(SIGSEGV);
    return 0;
}

static int plain(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    raise(SIGSEGV);
    return 0;
}

int main(int argc, char **argv) {
    int bracket = strcmp(argv[1], "bracket") == 0;
#if defined(__x86_64__)
    uint32_t pkru0 = __afl_wguard_mode == __AFL_WGUARD_PKEY ? __afl_rdpkru() : 0;
#else
    uint32_t pkru0 = 0;
#endif
    int rc = __afl_guarded_call(bracket ? in_bracket : plain, NULL, 0);
#if defined(__x86_64__)
    uint32_t pkru1 = __afl_wguard_mode == __AFL_WGUARD_PKEY ? __afl_rdpkru() : 0;
#else
    uint32_t pkru1 = 0;
#endif
    __afl_map_reset();          /* reads the generation word */
    fire();
    printf("%d %d %u %d\\n", __afl_wguard_mode, rc, __afl_wguard_depth, pkru0 == pkru1);
    fflush(stdout);
    if (argc > 2 && strcmp(argv[2], "stray") == 0)
        __afl_area[3].edge_id = STRAY;
    return 0;
}
""".replace("STRAY", hex(STRAY_ID))
)


@pytest.fixture(scope="module")
def landing(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("landing"),
        _LANDING,
        "-fsanitize-coverage=trace-pc-guard",
        "-D__AFL_CTX_SENSITIVE=0",
    )


def _land(exe, cov: ShmCoverage, mode: str, *args: str) -> subprocess.CompletedProcess:
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries))
    env["__AFL_WGUARD"] = mode
    return subprocess.run([str(exe), *args], env=env, capture_output=True, text=True, timeout=30)


def _ids(cov: ShmCoverage) -> set[int]:
    table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    return {int(e) for e in table["edge_id"] if e}


@requires_clang
def test_plain_crash_keeps_recording_under_guard(landing, segs):
    """Falsification: a crash outside a bracket leaves the guard balanced and
    recording working."""
    cov = segs(SMALL_ENTRIES)
    r = _land(landing, cov, "mprotect", "plain")
    assert r.returncode == 0, r.stderr
    mode, rc, depth, _ = (int(x) for x in r.stdout.split())
    assert (mode, rc, depth) == (WriteGuard.MPROTECT, -signal.SIGSEGV, 0)
    assert _live(cov) > 0


@requires_clang
def test_regression_longjmp_mid_bracket_relocks(landing, segs):
    """Adversarial: escaping from inside a bracket must re-lock the segment."""
    cov = segs(SMALL_ENTRIES)
    r = _land(landing, cov, "mprotect", "bracket", "stray")
    mode, rc, depth, _ = (int(x) for x in r.stdout.split())
    assert (mode, rc, depth) == (WriteGuard.MPROTECT, -signal.SIGSEGV, 0)
    assert r.returncode == -signal.SIGSEGV, (r.returncode, r.stderr)
    assert STRAY_ID not in _ids(cov)


@requires_clang
def test_regression_longjmp_restores_pkru(landing, segs):
    """pkey: the handler's default PKRU must not outlive the escape (PKU hosts only)."""
    cov = segs(SMALL_ENTRIES)
    r = _land(landing, cov, "pkey", "plain")
    if r.returncode == 0 and int(r.stdout.split()[0]) != WriteGuard.PKEY:
        pytest.skip("no protection keys on this host")
    assert r.returncode == 0, (r.returncode, r.stderr)
    _, rc, _, same = (int(x) for x in r.stdout.split())
    assert rc == -signal.SIGSEGV
    assert same == 1


# ── 4: nesting count is atomic ──────────────────────────────────────────

_THREADS = (
    _HEADER
    + """
#include <pthread.h>

#define ROUNDS 20000

static void *spin(void *arg) {
    (void)arg;
    for (int i = 0; i < ROUNDS; i++)
        __afl_wguard_close_cold(__afl_wguard_open_cold());
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], NULL, spin, NULL);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    printf("%d %u\\n", __afl_wguard_mode, __afl_wguard_depth);
    fflush(stdout);
    if (argc > 1) __afl_area[3].edge_id = STRAY;
    return 0;
}
""".replace("STRAY", hex(STRAY_ID))
)


@pytest.fixture(scope="module")
def threads(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("threads"), _THREADS, "-D__AFL_CTX_SENSITIVE=0", "-pthread"
    )


@requires_clang
def test_regression_wguard_depth_rmw_is_atomic(threads):
    """Deterministic: the open/close nesting count updates with lock-prefixed RMWs."""
    asm = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(threads)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for fn in ("__afl_wguard_mp_open", "__afl_wguard_mp_close"):
        body = asm.split(f"<{fn}>:", 1)[1].split("\n\n", 1)[0]
        assert "lock" in body, fn


@requires_clang
def test_concurrent_brackets_leave_segment_locked(threads, segs):
    """Adversarial: 4 threads hammer open/close; the count returns to 0 and
    the segment ends locked."""
    cov = segs(SMALL_ENTRIES)
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(SMALL_ENTRIES))
    env["__AFL_WGUARD"] = "mprotect"
    r = subprocess.run([str(threads), "stray"], env=env, capture_output=True, text=True, timeout=60)
    mode, depth = (int(x) for x in r.stdout.split())
    assert (mode, depth) == (WriteGuard.MPROTECT, 0)
    assert r.returncode == -signal.SIGSEGV, (r.returncode, r.stderr)
    assert STRAY_ID not in _ids(cov)


# ── 5: ctx walk bounded below by the thread stack ───────────────────────

# Fire a guard on a private stack below the thread stack whose frame pointer
# points into the unmapped page just above it: a coroutine / sigaltstack
# frame with rbp holding junk. argv[1] "main" fires on the thread stack only.
_CTX = (
    _HEADER
    + """
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGES 16

static uint32_t g_alt = 4242;

__attribute__((noinline)) static void fire_on(void *top, void *junk) {
#if defined(__x86_64__)
    void *t = top, *j = junk;
    uint32_t *g = &g_alt;
    __asm__ volatile(
        "mov %%rsp, %%r12\\n\\t"
        "mov %0, %%rsp\\n\\t"
        "push %%rbp\\n\\t"
        "sub $8, %%rsp\\n\\t"
        "mov %1, %%rbp\\n\\t"
        "call __sanitizer_cov_trace_pc_guard\\n\\t"
        "add $8, %%rsp\\n\\t"
        "pop %%rbp\\n\\t"
        "mov %%r12, %%rsp\\n\\t"
        : "+a"(t), "+c"(j), "+D"(g)
        :
        : "r12", "rdx", "rsi", "r8", "r9", "r10", "r11", "memory", "cc",
          "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#else
    (void)top; (void)junk;
#endif
}

int main(int argc, char **argv) {
    fire();                                   /* resolves the stack bounds */
    if (strcmp(argv[1], "main") == 0) { puts("ok"); return 0; }

    long page = sysconf(_SC_PAGESIZE);
    uint8_t *r = mmap(NULL, (PAGES + 1) * page, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r == MAP_FAILED) { puts("skip"); return 0; }
    uint8_t *top = r + PAGES * page;
    munmap(top, page);                        /* junk target: unmapped */

    pthread_attr_t a;
    void *lo = NULL; size_t sz = 0;
    pthread_getattr_np(pthread_self(), &a);
    pthread_attr_getstack(&a, &lo, &sz);
    pthread_attr_destroy(&a);
#if !defined(__x86_64__)
    puts("skip"); return 0;
#endif
    if ((uintptr_t)top + page >= (uintptr_t)lo) { puts("skip"); return 0; }

    fire_on(top, top + 64);
    puts("ok");
    return 0;
}
"""
)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("ctx"), _CTX, "-fsanitize-coverage=trace-pc-guard", "-pthread"
    )


def _ctx_run(exe, cov: ShmCoverage, arg: str) -> subprocess.CompletedProcess:
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries))
    env["__AFL_CTX_SENSITIVE"] = "1"
    return subprocess.run([str(exe), arg], env=env, capture_output=True, text=True, timeout=30)


@requires_clang
def test_ctx_walk_on_thread_stack_records(ctx, segs):
    """Falsification: the lower bound does not disable the walk on the thread stack."""
    cov = segs(SMALL_ENTRIES)
    r = _ctx_run(ctx, cov, "main")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
    assert _live(cov) > 0


@requires_clang
def test_regression_ctx_walk_off_stack_junk_fp(ctx, segs):
    """Adversarial: a junk frame pointer above an off-stack frame is not loaded."""
    cov = segs(SMALL_ENTRIES)
    r = _ctx_run(ctx, cov, "alt")
    if r.stdout.strip() == "skip":
        pytest.skip("private stack not below the thread stack here")
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.strip() == "ok"


@requires_clang
def test_ctx_canonical_check_is_not_x86_only():
    """aarch64 user stacks may carry bit 47; the walk must not reject them."""
    from tests.test_regression_shim_audit import SHIM

    assert "if (cfp >> 47)" not in SHIM.read_text()
