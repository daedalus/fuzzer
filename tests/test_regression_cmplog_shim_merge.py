"""Regression: the cmplog shim carried a second copy of the edge machinery.

``cmplog_shim.c`` shipped its own ``__afl_map_shm`` / ``__afl_map_reset`` /
``__sanitizer_cov_trace_pc_guard{,_init}`` (byte bitmap + Morris counting)
behind ``weak`` definitions. Four defects followed from that duplication;
each is pinned below.

1. **``weak`` does not protect a preloaded shim from winning.** It only
   loses to a strong definition at STATIC link time. At dynamic link time
   the first definition in the global lookup scope wins regardless of
   binding, and LD_PRELOAD precedes dependency ``.so``s -- so a preloaded
   ``cmplog_shim.so`` preempted an instrumented target's own
   ``__afl_map_shm`` and left ``__afl_area`` NULL. Measured on a ``.so``
   target built without ``-Wl,-Bsymbolic``::

       preload=none              __afl_area = 0x7f4c757d6018
       preload=cmplog_shim.so    __afl_area = (nil)      <- zero coverage
       preload=cmplog_shim.so    __afl_area = 0x7f57...  (with -Bsymbolic)

   ``-Bsymbolic`` was the only thing standing between this and silent
   total coverage loss, and four ``_tracecmp.so`` targets are built
   without it.

2. **Double attachment.** Its constructor called ``__afl_map_shm()``,
   which in a combined link resolved to the strong definition -- so the
   segment was ``shmat``'d once by each constructor. Measured 2
   attachments per exec against 1 for the shim alone.

3. **Flush on the first crash only.** Its crash handler restored the
   previous disposition permanently, so every later crash in a
   persistent/direct_lite loop lost up to 256KB of buffered records.

4. **``AFL_MAP_SIZE`` in different units.** entries here, bytes there.

The merge removes all four by construction: one definition of the edge
machinery, one constructor, one crash handler, and an LD_PRELOAD build
(``-D__AFL_PRELOAD_ONLY``) that defines none of the ``__afl_*`` symbols
and so cannot shadow anything.

Also pinned: compiling the logger into the target's own TU means
``-fsanitize-coverage=trace-cmp`` instruments the logger, whose own
comparisons call back into it. That recursed to a stack-overflow SIGSEGV
before ``__AFL_NO_COV`` and the re-entrancy flag were added.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")

NOBUILTIN = [
    "-fno-builtin-memcmp",
    "-fno-builtin-bcmp",
    "-fno-builtin-strcmp",
    "-fno-builtin-strncmp",
    "-fno-builtin-memchr",
    "-fno-builtin-strstr",
    "-fno-builtin-memmem",
]

MAP_ENTRIES = 8192
SHM_HEADER = 24
SHM_BYTES = MAP_ENTRIES * 8 + SHM_HEADER

_TARGET_SO = """
#include <stdint.h>
#include <stddef.h>
int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n) { return n > 0 && d[0] == 'A'; }
"""

# Reports how many SYSV shared segments the process has attached, and where
# the *target's own* __afl_area ended up.
_LOADER = """
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen-failed\\n"); return 1; }
    FILE *f = fopen("/proc/self/maps", "r");
    char line[512]; int n = 0;
    while (fgets(line, sizeof line, f)) if (strstr(line, "SYSV")) n++;
    void **area = (void **)dlsym(h, "__afl_area");
    printf("attachments=%d area=%p\\n", n, area ? *area : (void *)-1);
    return 0;
}
"""

# Buffers a distinguishable comparison, then crashes. Repeated three times
# through __afl_guarded_call, which siglongjmps back out.
_CRASH_LOOP = """
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
extern int __afl_guarded_call(int (*)(const uint8_t *, size_t), const uint8_t *, size_t);
static int boom(const uint8_t *d, size_t n) {
    char buf[16]; memset(buf, 'A', sizeof buf);
    volatile int r = memcmp(buf, (const char *)d, 8);
    (void)r; (void)n;
    *(volatile int *)0 = 1;
    return 0;
}
int main(void) {
    const char *needles[3] = {"CRASHAAA", "CRASHBBB", "CRASHCCC"};
    for (int i = 0; i < 3; i++)
        printf("%d\\n", __afl_guarded_call(boom, (const uint8_t *)needles[i], 8));
    return 0;
}
"""

# Exercises every Layer 1 interceptor against a constant the pool should see.
_INTERCEPTORS = """
#include <string.h>
#include <stdint.h>
int main(void) {
    char buf[64]; memset(buf, 'A', sizeof buf);
    volatile int r = 0;
    r += memcmp(buf, "MAGICHDR", 8);
    r += strcmp(buf, "SECRET_TOKEN");
    r += strncmp(buf, "PREFIX_", 7);
    r += (memchr(buf, 0x7f, 32) != NULL);
    r += (strstr(buf, "NEEDLE99") != NULL);
    r += (memmem(buf, 16, "HAYSTACK", 8) != NULL);
    return r == 12345;
}
"""


def _cc(tmp_path, name, source, *, flags=(), shared=False, out_name=None):
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    out = tmp_path / (out_name or name)
    cmd = ["gcc", "-O2", "-g", *flags]
    if shared:
        cmd += ["-shared", "-fPIC"]
    cmd += ["-o", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr.decode()[:600]
    return out


def _cc_shim(tmp_path, name, source, *, cmplog, extra=(), shared=False):
    flags = ["-include", str(SHIM), *extra]
    if cmplog:
        flags = ["-D__AFL_CMPLOG=1", *NOBUILTIN, *flags, "-ldl"]
    return _cc(tmp_path, name, source, flags=flags, shared=shared)


class _Shm:
    """A coverage segment sized the way ShmCoverage sizes it."""

    def __init__(self):
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.shmget.restype = ctypes.c_int
        self._libc.shmat.restype = ctypes.c_void_p
        self._libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        # shmid 0 is a legal id the shim rejects (it checks `<= 0`), so take
        # two and use the second — this is a test-harness detail, not a
        # property under test.
        self._ids = [self._libc.shmget(0, SHM_BYTES, 0o1000 | 0o600) for _ in range(2)]
        self.shm_id = self._ids[-1]

    def env(self, **extra):
        e = dict(os.environ, __AFL_SHM_ID=str(self.shm_id), AFL_MAP_SIZE=str(MAP_ENTRIES))
        e.pop("LD_PRELOAD", None)
        e.update(extra)
        return e

    def read(self):
        addr = self._libc.shmat(self.shm_id, None, 0)
        return bytes((ctypes.c_ubyte * SHM_BYTES).from_address(addr))

    def close(self):
        for i in self._ids:
            self._libc.shmctl(i, 0, None)


@pytest.fixture
def shm():
    s = _Shm()
    yield s
    s.close()


# ── 1. the preload artifact cannot shadow an instrumented target ──────


@needs_cc
def test_preload_build_exports_no_edge_symbols(tmp_path):
    """The defect at its root: those exports are what won the lookup."""
    so = _cc(
        tmp_path,
        "preload",
        SHIM.read_text(),
        flags=["-D__AFL_PRELOAD_ONLY", "-ldl"],
        shared=True,
        out_name="preload.so",
    )
    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(so)], capture_output=True, text=True, timeout=60
    ).stdout
    for sym in (
        "__afl_map_shm",
        "__afl_map_reset",
        "__afl_map_edge",
        "__sanitizer_cov_trace_pc_guard",
        "__afl_area",
    ):
        assert sym not in exported, (
            f"{sym} is exported by the LD_PRELOAD build — it can preempt an "
            "instrumented target's own definition and null its __afl_area"
        )
    # ...while still providing the layers the preload exists for.
    for sym in ("memcmp", "strstr", "__sanitizer_cov_trace_cmp4", "__cmplog_reset"):
        assert sym in exported, f"{sym} missing — the preload build does nothing"


@needs_cc
@pytest.mark.parametrize("bsymbolic", [False, True])
def test_preload_does_not_null_target_afl_area(tmp_path, shm, bsymbolic):
    """The measured failure. Note bsymbolic=False is the case that broke:
    four _tracecmp.so targets are built without it."""
    extra = ["-Wl,-Bsymbolic"] if bsymbolic else []
    target = _cc_shim(tmp_path, "tgt", _TARGET_SO, cmplog=False, extra=extra, shared=True)
    preload = _cc(
        tmp_path,
        "preload",
        SHIM.read_text(),
        flags=["-D__AFL_PRELOAD_ONLY", "-ldl"],
        shared=True,
        out_name="preload.so",
    )
    loader = _cc(tmp_path, "loader", _LOADER, flags=["-ldl"])

    env = shm.env(_CMPLOG_OUT=str(tmp_path / "cmp.log"), LD_PRELOAD=str(preload))
    out = subprocess.run(
        [str(loader), str(target)], capture_output=True, text=True, env=env, timeout=60
    ).stdout
    area = out.split("area=")[-1].strip()
    assert area not in ("(nil)", "0x0"), (
        "the preloaded shim preempted the target's __afl_map_shm — the run would record zero edges"
    )


# ── 2. one constructor, one attachment ───────────────────────────────


@needs_cc
def test_cmplog_build_attaches_shm_once(tmp_path, shm):
    src = """
#include <stdio.h>
#include <string.h>
int main(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    char l[512]; int n = 0;
    while (fgets(l, sizeof l, f)) if (strstr(l, "SYSV")) n++;
    printf("attachments=%d\\n", n);
    return 0;
}
"""
    exe = _cc_shim(tmp_path, "att", src, cmplog=True)
    env = shm.env(_CMPLOG_OUT=str(tmp_path / "cmp.log"))
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60).stdout
    assert "attachments=1" in out, (
        f"expected a single shmat, got {out.strip()} — a second constructor "
        "is re-entering __afl_map_shm"
    )


# ── 3. the buffer is flushed on every crash ──────────────────────────


@needs_cc
def test_cmplog_flushes_on_every_crash_not_just_the_first(tmp_path, shm):
    exe = _cc_shim(tmp_path, "crash", _CRASH_LOOP, cmplog=True)
    log = tmp_path / "crash.log"
    env = shm.env(_CMPLOG_OUT=str(log))
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.split() == ["-11", "-11", "-11"], (
        f"guarded_call did not recover from all three crashes: {r.stdout!r}"
    )
    body = log.read_text()
    missing = [n for n in ("CRASHAAA", "CRASHBBB", "CRASHCCC") if n.encode().hex() not in body]
    assert not missing, (
        f"records from {missing} never reached disk — the flush ran on the "
        "first crash only, which is what the old second crash handler did"
    )


# ── 4. both channels populate from one binary ────────────────────────


@needs_cc
def test_interceptors_and_coverage_share_one_segment(tmp_path, shm):
    exe = _cc_shim(tmp_path, "both", _INTERCEPTORS, cmplog=True)
    log = tmp_path / "both.log"
    subprocess.run([str(exe)], capture_output=True, env=shm.env(_CMPLOG_OUT=str(log)), timeout=60)

    raw = shm.read()
    _stack, diag, _path_hash, edge_count = struct.unpack("<IIQQ", raw[:SHM_HEADER])
    drops = diag >> 8
    assert drops == 0, f"{drops} edges dropped on a trivial target — map is saturated"

    operands = set()
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "CMP":
            for h in (parts[1], parts[2]):
                with contextlib.suppress(ValueError):
                    operands.add(bytes.fromhex(h))
    for needle in (b"MAGICHDR", b"SECRET_TOKEN", b"PREFIX_", b"NEEDLE99", b"HAYSTACK"):
        assert any(needle in op for op in operands), f"{needle!r} never reached the pool"

    # edge_count is only written by the trace-pc-guard path, which gcc does
    # not emit; assert the header is intact rather than that it is non-zero.
    assert edge_count < 2**32, "header looks like it was overwritten by a byte bitmap"


# ── 5. the logger must not be instrumented into recursing ────────────


@needs_cc
def test_tracecmp_instrumentation_does_not_recurse(tmp_path, shm):
    """-include puts the logger in the target's TU, so trace-cmp instruments
    it and its own comparisons call back into it. Stack-overflow SIGSEGV at
    startup before __AFL_NO_COV / the re-entrancy flag."""
    exe = _cc_shim(
        tmp_path,
        "tc",
        _INTERCEPTORS,
        cmplog=True,
        extra=["-fsanitize-coverage=trace-cmp"],
    )
    log = tmp_path / "tc.log"
    r = subprocess.run(
        [str(exe)], capture_output=True, env=shm.env(_CMPLOG_OUT=str(log)), timeout=60
    )
    assert r.returncode != -11, "SIGSEGV — the record writer is recursing through its own callbacks"
    lines = log.read_text().splitlines()
    assert any(len(line.split()) == 6 for line in lines), (
        "no pc-bearing records — the trace-cmp layer produced nothing"
    )


# ── 6. the gate keeps _detect_cmplog a real signal ───────────────────


@needs_cc
def test_gate_off_emits_no_cmplog_symbols(tmp_path):
    """services/fuzzer.py::_detect_cmplog greps for __cmplog_reset to decide
    whether direct_lite is safe. A shim that always defined it would make
    that probe a constant."""
    plain = _cc_shim(tmp_path, "plain", _INTERCEPTORS, cmplog=False)
    gated = _cc_shim(tmp_path, "gated", _INTERCEPTORS, cmplog=True)

    def syms(path):
        return subprocess.run(["nm", str(path)], capture_output=True, text=True, timeout=60).stdout

    assert "__cmplog_reset" not in syms(plain)
    assert "__tracecmp_reset" not in syms(plain)
    assert " T memcmp" not in syms(plain), "interposers present without the gate"
    assert "__cmplog_reset" in syms(gated)
    assert "__tracecmp_reset" in syms(gated)


# ── 7. the pc field the shim writes is one the collector can read ────


@needs_cc
def test_pc_field_parses(tmp_path, shm):
    """The shim writes pc in the %p convention ("0x55f6..."). collect_tokens
    parsed it with int(s) — base 10 — so every parse raised ValueError into a
    suppress() and pc was silently None for every record ever written."""
    exe = _cc_shim(
        tmp_path,
        "pc",
        _INTERCEPTORS,
        cmplog=True,
        extra=["-fsanitize-coverage=trace-cmp"],
    )
    log = tmp_path / "pc.log"
    subprocess.run([str(exe)], capture_output=True, env=shm.env(_CMPLOG_OUT=str(log)), timeout=60)

    from fuzzer_tool.core.cmplog import CmplogCollector

    collector = CmplogCollector()
    collector.log_path = str(log)
    collector.collect_tokens()
    assert collector._pair_pc, (
        "no pair carried a pc — the collector still cannot parse the field the shim writes"
    )
