"""Shim state survives a sanitizer's own exit (``__sanitizer_set_death_callback``).

A sanitizer that halts on a report leaves through its runtime's raw
exit_group: no destructor, no signal, so buffered cmplog records and
unfolded inline counters were lost for exactly the crashing input. The
shim registers a death callback that runs the crash handler's flush.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.cmplog import CmplogCollector
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env

MAP_ENTRIES = 8192
MAGIC = 0x1234ABCD
CMP_FLAGS = ("-fsanitize-coverage=trace-pc-guard,trace-cmp", "-D__AFL_CMPLOG=1")
COUNTERS = "-fsanitize-coverage=inline-8bit-counters"

# argv[1]: hex value compared against MAGIC (one CMP record, buffered),
# argv[2] == "O": heap overflow after the compare.
_SRC = """
#include <stdlib.h>
#include <string.h>
volatile int sink;
char *volatile gp; /* volatile pointer: clang may not elide the malloc */
int main(int argc, char **argv) {
    unsigned v = argc > 1 ? (unsigned)strtoul(argv[1], 0, 16) : 0;
    if (v == 0x1234abcdu) sink = 1;
    if (argc > 2 && argv[2][0] == 'O') {
        volatile int i = 8;
        gp = malloc(8);
        gp[i] = 1;
        free(gp);
    }
    return 0;
}
"""


def _rt_available() -> bool:
    """clang's ASAN runtime is a separate package on Debian/Ubuntu."""
    probe = subprocess.run(
        ["clang", "-fsanitize=address", "-x", "c", "-", "-o", "/dev/null"],
        input="int main(void){return 0;}",
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


requires_asan_rt = pytest.mark.skipif(not _rt_available(), reason="no clang ASAN runtime")


def _pairs(exe: Path, tmp: Path, *argv: str, **env_extra: str):
    """(returncode, stderr, CMP pairs) of one run."""
    c = CmplogCollector()
    c.log_path = str(tmp / "cmplog.txt")
    Path(c.log_path).write_text("")
    env = _clean_env(_CMPLOG_OUT=c.log_path, **env_extra)
    r = subprocess.run([str(exe), *argv], env=env, capture_output=True, text=True, timeout=60)
    log = Path(c.log_path).read_text()  # before collect_tokens consumes it
    c.collect_tokens()
    return r.returncode, r.stderr, set(c.pairs), log


def _has_magic(pairs) -> bool:
    """MAGIC as an operand at either width (clang widens to strtoul's 8)."""
    want = {MAGIC.to_bytes(4, "little"), MAGIC.to_bytes(8, "little")}
    return any(a in want or b in want for a, b in pairs)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("death"), _SRC, *CMP_FLAGS, asan=True)


@requires_clang
@requires_asan_rt
class TestAsanDeath:
    def test_cmplog_records_survive_the_report(self, exe, tmp_path):
        rc, err, pairs, _ = _pairs(exe, tmp_path, "1", "O")
        assert rc != 0
        assert "AddressSanitizer" in err
        assert _has_magic(pairs)

    def test_control_clean_exit_logs_the_same_pair(self, exe, tmp_path):
        """Control: the destructor path already logs it; death must match it."""
        rc, _, pairs, _ = _pairs(exe, tmp_path, "1")
        assert rc == 0
        assert _has_magic(pairs)

    def test_adversarial_abort_on_error(self, exe, tmp_path):
        """The runtime aborts after the callback: records still land."""
        rc, err, pairs, _ = _pairs(
            exe, tmp_path, "1", "O", ASAN_OPTIONS="abort_on_error=1:handle_abort=0"
        )
        assert rc != 0
        assert "AddressSanitizer" in err
        assert _has_magic(pairs)

    def test_adversarial_recover_mode_logs_once(self, tmp_path):
        """halt_on_error=0: no death, the run goes on; one record, not two."""
        exe = _build(tmp_path, _SRC, *CMP_FLAGS, "-fsanitize-recover=address", asan=True)
        rc, err, pairs, log = _pairs(exe, tmp_path, "1", "O", ASAN_OPTIONS="halt_on_error=0")
        assert rc == 0
        assert "AddressSanitizer" in err
        magic = MAGIC.to_bytes(4, "little").hex()
        assert sum(line.startswith("CMP") and magic in line for line in log.splitlines()) == 1


@requires_clang
@requires_asan_rt
def test_counters_fold_on_asan_death(tmp_path):
    """inline-8bit-counters only reach the map when folded."""
    exe = _build(tmp_path, _SRC, COUNTERS, asan=True)
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        r = subprocess.run([str(exe), "1", "O"], env=env, capture_output=True, timeout=60)
        assert r.returncode != 0
        assert cov.get_edge_ids()
    finally:
        cov.cleanup()


@requires_clang
def test_falsify_no_sanitizer_still_links(tmp_path):
    """The hook is weak: a build without any runtime links and runs."""
    exe = _build(tmp_path, _SRC, *CMP_FLAGS)
    rc, _, pairs, _ = _pairs(exe, tmp_path, "1")
    assert rc == 0
    assert _has_magic(pairs)


# memcmp, not trace-cmp: a preloaded shim reaches the target only through
# interposed libc calls.
_PRELOAD_SRC = """
#include <stdlib.h>
#include <string.h>
volatile int sink;
char *volatile gp;
int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "";
    if (memcmp(in, "MAGICVAL", 8) == 0) sink = 1;
    if (argc > 2 && argv[2][0] == 'O') {
        volatile int i = 8;
        gp = malloc(8);
        gp[i] = 1;
        free(gp);
    }
    return 0;
}
"""


def _asan_rt_dir() -> Path:
    out = subprocess.run(
        ["clang", "-print-runtime-dir"], capture_output=True, text=True
    ).stdout.strip()
    return Path(out)


@requires_clang
@requires_asan_rt
def test_preload_only_build_flushes_on_death(tmp_path):
    """-D__AFL_PRELOAD_ONLY registers the cmplog flush too.

    Shared ASAN runtime first, then the shim, so the shim's memcmp wins the
    lookup and its weak hook binds to the runtime's callback setter.
    """
    rt = _asan_rt_dir() / "libclang_rt.asan-x86_64.so"
    if not rt.exists():
        pytest.skip("no shared ASAN runtime")

    preload = tmp_path / "preload.so"
    cc = ["clang", "-O1", "-fPIC", "-shared", "-D__AFL_PRELOAD_ONLY", str(SHIM), "-o", str(preload), "-ldl"]  # fmt: skip
    assert subprocess.run(cc, capture_output=True).returncode == 0

    src = tmp_path / "p.c"
    src.write_text(_PRELOAD_SRC)
    exe = tmp_path / "p"
    cc = ["clang", "-O1", "-fsanitize=address", "-shared-libasan", "-fno-builtin-memcmp", str(src), "-o", str(exe)]  # fmt: skip
    assert subprocess.run(cc, capture_output=True).returncode == 0

    log = tmp_path / "cmplog.txt"
    log.write_text("")
    env = _clean_env(
        _CMPLOG_OUT=str(log),
        LD_LIBRARY_PATH=str(rt.parent),
        LD_PRELOAD=f"{rt}:{preload}",
        ASAN_OPTIONS="verify_asan_link_order=0",
    )
    r = subprocess.run([str(exe), "XXXXXXXX", "O"], env=env, capture_output=True, timeout=60)
    assert r.returncode != 0
    assert b"MAGICVAL".hex() in log.read_text()


_UBSAN_SRC = """
#include <limits.h>
#include <stdlib.h>
volatile int sink;
int main(int argc, char **argv) {
    unsigned v = argc > 1 ? (unsigned)strtoul(argv[1], 0, 16) : 0;
    if (v == 0x1234abcdu) sink = 1;
    volatile int big = INT_MAX;
    if (argc > 2) sink = big + argc; /* signed overflow */
    return 0;
}
"""


@requires_clang
def test_ubsan_fatal_report_flushes(tmp_path):
    """UBSAN without recover dies through the same Die() path."""
    probe = subprocess.run(
        ["clang", "-fsanitize=undefined", "-x", "c", "-", "-o", "/dev/null"],
        input="int main(void){return 0;}",
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.skip("no clang UBSAN runtime")

    src = tmp_path / "u.c"
    src.write_text(_UBSAN_SRC)
    exe = tmp_path / "u"
    san = ["-fsanitize=signed-integer-overflow", "-fno-sanitize-recover=all"]
    cc = ["clang", "-O1", *san, *CMP_FLAGS, "-include", str(SHIM), str(src), "-o", str(exe), "-ldl"]  # fmt: skip
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]

    rc, err, pairs, _ = _pairs(exe, tmp_path, "1", "X")
    assert rc != 0
    assert "runtime error" in err
    assert _has_magic(pairs)
