"""Compares inside sanitizer interceptors reach cmplog (``__sanitizer_weak_hook_*``).

The shim's strong memcmp/strcmp only see direct calls. A caller that lands
on the runtime's ``__interceptor_*`` (another DSO, a direct call) bypassed
them; the runtime reports those compares through its weak hooks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env
from tests.test_sanitizer_death_callback import requires_asan_rt

CMP_FLAGS = ("-fsanitize-coverage=trace-pc-guard", "-D__AFL_CMPLOG=1")
TOKEN = b"MAGICTOKEN"

# argv[1] selects the interceptor called directly, bypassing the shim's
# strong override; argv[2] is the operand compared against TOKEN.
_SRC = """
#include <stdlib.h>
#include <string.h>
int __interceptor_memcmp(const void *, const void *, size_t);
int __interceptor_strcmp(const char *, const char *);
int __interceptor_strncmp(const char *, const char *, size_t);
int __interceptor_strcasecmp(const char *, const char *);
int __interceptor_strncasecmp(const char *, const char *, size_t);
char *__interceptor_strstr(const char *, const char *);
char *__interceptor_strcasestr(const char *, const char *);
void *__interceptor_memmem(const void *, size_t, const void *, size_t);
volatile long sink;
int main(int argc, char **argv) {
    const char *t = "MAGICTOKEN";
    char *in = malloc(16);
    strncpy(in, argc > 2 ? argv[2] : "", 15);
    in[15] = 0;
    switch (argv[1][0]) {
    case 'm': sink = __interceptor_memcmp(in, t, 10); break;
    case 'c': sink = __interceptor_strcmp(in, t); break;
    case 'n': sink = __interceptor_strncmp(in, t, 10); break;
    case 'C': sink = __interceptor_strcasecmp(in, t); break;
    case 'N': sink = __interceptor_strncasecmp(in, t, 10); break;
    case 's': sink = (long)__interceptor_strstr(in, t); break;
    case 'S': sink = (long)__interceptor_strcasestr(in, t); break;
    case 'M': sink = (long)__interceptor_memmem(in, 15, t, 10); break;
    case 'd': sink = memcmp(in, t, 10); break;   /* direct: shim override */
    case 'o': {                                  /* 2-byte heap operand */
        char *p = malloc(2);
        p[0] = 'h'; p[1] = 'x';
        sink = __interceptor_strncmp(p, "http:", 5);
        free(p);
        break;
    }
    }
    free(in);
    return 0;
}
"""

MAP_ENTRIES = 8192
ROUTES = ("m", "c", "n", "C", "N", "s", "S", "M")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("hooks"), _SRC, *CMP_FLAGS, asan=True)


def _run(exe: Path, tmp: Path, *argv: str, **env_extra: str) -> tuple[int, str, str]:
    """(returncode, cmplog text, counts text) of one run."""
    log, counts = tmp / "cmplog.txt", tmp / "counts.txt"
    log.write_text("")
    counts.write_text("")
    env = _clean_env(_CMPLOG_OUT=str(log), _CMPLOG_COUNTS=str(counts), **env_extra)
    r = subprocess.run([str(exe), *argv], env=env, capture_output=True, timeout=60)
    return r.returncode, log.read_text(), counts.read_text()


def _records_with(log: str, needle: bytes) -> int:
    hexed = needle.hex()
    return sum(line.startswith("CMP ") and hexed in line for line in log.splitlines())


def _fired(counts: str, name: str) -> int:
    """Sum of CNT <name> fired deltas over every dump."""
    total = 0
    for line in counts.splitlines():
        parts = line.split()
        if parts[:2] == ["CNT", name]:
            total += int(parts[2])
    return total


@requires_clang
@requires_asan_rt
class TestWeakHooks:
    @pytest.mark.parametrize("route", ROUTES)
    def test_interceptor_compare_is_logged(self, exe, tmp_path, route):
        """Falsification: a compare that never touches the override logs a pair."""
        rc, log, _ = _run(exe, tmp_path, route, "XAGICTOKEN")
        assert rc == 0
        assert _records_with(log, TOKEN[:4]) >= 1

    def test_satisfied_compare_not_logged(self, exe, tmp_path):
        """Solved compares stay out of the pool, as for the override."""
        rc, log, _ = _run(exe, tmp_path, "m", TOKEN.decode())
        assert rc == 0
        assert _records_with(log, TOKEN) == 0

    def test_control_override_logs_once(self, exe, tmp_path):
        """Control: the override path already logs; the hook must not add one."""
        rc, log, _ = _run(exe, tmp_path, "d", "XAGICTOKEN")
        assert rc == 0
        assert _records_with(log, TOKEN[1:5]) == 1

    def test_adversarial_override_counts_once(self, exe, tmp_path):
        """memcmp -> __interceptor_memcmp -> hook is one compare, not two."""
        rc, _, counts = _run(exe, tmp_path, "d", "XAGICTOKEN")
        assert rc == 0
        assert _fired(counts, "memcmp") == 1

    def test_hook_counts_under_libc_name(self, exe, tmp_path):
        rc, _, counts = _run(exe, tmp_path, "m", "XAGICTOKEN")
        assert rc == 0
        assert _fired(counts, "memcmp") == 1

    def test_adversarial_short_heap_operand(self, exe, tmp_path):
        """strncmp's n is a bound, not a length: the hook reads no redzone."""
        r = subprocess.run(
            [str(exe), "o"],
            env=_clean_env(_CMPLOG_OUT=str(tmp_path / "log.txt")),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stderr[-800:]
        assert "AddressSanitizer" not in r.stderr

    def test_no_sink_is_silent(self, exe, tmp_path):
        """No _CMPLOG_OUT: the hook runs, logs nothing, crashes nothing."""
        r = subprocess.run([str(exe), "m", "X"], env=_clean_env(), capture_output=True, timeout=60)
        assert r.returncode == 0


def _edges(exe: Path, operand: str) -> set[int]:
    """Edge ids of one COMPCOV level-2 run; context term off for stable ids."""
    from fuzzer_tool.adapters.shm import ShmCoverage

    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(
            __AFL_SHM_ID=cov.env_id,
            AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_COMPCOV_LEVEL="2",
            __AFL_CTX_SENSITIVE="0",
        )
        r = subprocess.run([str(exe), "m", operand], env=env, capture_output=True, timeout=60)
        assert r.returncode == 0
        return cov.get_edge_ids()
    finally:
        cov.cleanup()


@requires_clang
@requires_asan_rt
def test_compcov_marks_through_hook(exe):
    """COMPCOV level 2 folds hook-reported progress into the edge map."""
    miss, again, partial = (_edges(exe, op) for op in ("XXXXXXXXXX", "XXXXXXXXXX", "MAGICXXXXX"))
    assert miss == again  # control: same input, same map
    assert partial - miss


@requires_clang
@requires_asan_rt
def test_regression_cmplog_so_dlopens(tmp_path):
    """The hook mute's TLS must not force static TLS: the .so must dlopen."""
    from tests.test_alloc_size_signal import _HOST, _TARGET_SO

    so = _build(tmp_path, _TARGET_SO, *CMP_FLAGS, shared=True, asan=True)
    host_c = tmp_path / "host.c"
    host_c.write_text(_HOST)
    host = tmp_path / "host"
    cc = ["clang", "-O1", "-fsanitize=address", str(host_c), "-o", str(host), "-ldl"]
    assert subprocess.run(cc, capture_output=True).returncode == 0

    r = subprocess.run(
        [str(host), str(so), "t", "16"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-800:]
