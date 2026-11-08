"""Allocation-size feedback (``__sanitizer_install_malloc_and_free_hooks``).

With ``__AFL_ALLOC_SIZES=1`` every user allocation the sanitizer runtime
serves inside the target marks one synthetic edge per log2 size class, so an
input reaching a new allocation size is new coverage.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.shim_health import ALLOC_SIZES_ENV, export_alloc_sizes
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env
from tests.test_sanitizer_death_callback import requires_asan_rt

MAP_ENTRIES = 8192
SMALL, LARGE = 16, 1 << 20
EDGE_FLAGS = ("-fsanitize-coverage=trace-pc-guard",)

# argv[1]: allocation size. Same path for every size.
_SRC = """
#include <stdlib.h>
char *volatile gp;
int main(int argc, char **argv) {
    gp = malloc(strtoul(argv[1], 0, 10));
    free(gp);
    return 0;
}
"""


def _edges(cmd: list[str], **env_extra: str) -> set[int]:
    """Edge ids of one run; context term off so ids are stable across runs."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(
            __AFL_SHM_ID=cov.env_id,
            AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_CTX_SENSITIVE="0",
            **env_extra,
        )
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-800:]
        return cov.get_edge_ids()
    finally:
        cov.cleanup()


@pytest.fixture(scope="module")
def asan_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("alloc"), _SRC, *EDGE_FLAGS, asan=True)


@requires_clang
@requires_asan_rt
class TestAllocSizes:
    def test_size_class_is_new_coverage(self, asan_exe):
        """Falsification: same path, different size class, different map."""
        on = {ALLOC_SIZES_ENV: "1"}
        small = _edges([str(asan_exe), str(SMALL)], **on)
        again = _edges([str(asan_exe), str(SMALL)], **on)
        large = _edges([str(asan_exe), str(LARGE)], **on)
        assert small == again  # control: one size, one map
        assert large - small

    def test_same_class_same_map(self, asan_exe):
        """17 and 31 share 16's class (bit length 5): no novelty."""
        on = {ALLOC_SIZES_ENV: "1"}
        assert _edges([str(asan_exe), "17"], **on) == _edges([str(asan_exe), "31"], **on)

    def test_off_by_default(self, asan_exe):
        small = _edges([str(asan_exe), str(SMALL)])
        large = _edges([str(asan_exe), str(LARGE)])
        assert small == large

    def test_adversarial_zero_size(self, asan_exe):
        """malloc(0) must not crash the hook; ASAN serves it as 1 byte."""
        on = {ALLOC_SIZES_ENV: "1"}
        assert _edges([str(asan_exe), "0"], **on) == _edges([str(asan_exe), "1"], **on)


@requires_clang
def test_adversarial_no_runtime_is_inert(tmp_path):
    """No sanitizer runtime: the weak installer is NULL, the env is a no-op."""
    exe = _build(tmp_path, _SRC, *EDGE_FLAGS)
    on = {ALLOC_SIZES_ENV: "1"}
    assert _edges([str(exe), str(SMALL)], **on) == _edges([str(exe), str(LARGE)], **on)


# In-process shape: a host process dlopens the target .so. Host allocations
# are not the target's; only allocations under __afl_guarded_call count.
_TARGET_SO = """
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
char *volatile gp;
int entry(const uint8_t *d, size_t n) {
    size_t sz;
    memcpy(&sz, d, sizeof sz);
    gp = malloc(sz);
    free(gp);
    return (int)n - (int)sizeof sz;
}
"""

_HOST = """
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*entry_t)(const uint8_t *, size_t);
typedef int (*guard_t)(entry_t, const uint8_t *, size_t);
char *volatile gp;
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\\n", dlerror()); return 2; }
    guard_t guard = (guard_t)dlsym(h, "__afl_guarded_call");
    entry_t entry = (entry_t)dlsym(h, "entry");
    size_t sz = strtoul(argv[3], 0, 10);
    if (argv[2][0] == 'h') {          /* host allocates, target idles */
        gp = malloc(sz);
        free(gp);
        sz = 16;
    }
    return guard(entry, (const uint8_t *)&sz, sizeof sz);
}
"""


@pytest.fixture(scope="module")
def inproc(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("inproc")
    so = _build(tmp, _TARGET_SO, *EDGE_FLAGS, shared=True, asan=True)
    host_c = tmp / "host.c"
    host_c.write_text(_HOST)
    host = tmp / "host"
    cc = ["clang", "-O1", "-fsanitize=address", str(host_c), "-o", str(host), "-ldl"]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return host, so


@requires_clang
@requires_asan_rt
class TestInProcessGate:
    def test_guarded_target_allocation_counts(self, inproc):
        host, so = inproc
        on = {ALLOC_SIZES_ENV: "1"}
        small = _edges([str(host), str(so), "t", str(SMALL)], **on)
        large = _edges([str(host), str(so), "t", str(LARGE)], **on)
        assert large - small

    def test_adversarial_host_allocation_ignored(self, inproc):
        """The host's own malloc outside the guard is not target coverage."""
        host, so = inproc
        on = {ALLOC_SIZES_ENV: "1"}
        small = _edges([str(host), str(so), "h", str(SMALL)], **on)
        large = _edges([str(host), str(so), "h", str(LARGE)], **on)
        assert small == large


class TestExport:
    def test_export_sets_env(self, monkeypatch):
        monkeypatch.setenv(
            ALLOC_SIZES_ENV, "0"
        )  # registers the undo; delenv of an absent var does not
        export_alloc_sizes(True)
        assert os.environ[ALLOC_SIZES_ENV] == "1"

    def test_export_off_clears_stale_value(self, monkeypatch):
        """Off must not inherit a parent's =1."""
        monkeypatch.setenv(ALLOC_SIZES_ENV, "1")
        export_alloc_sizes(False)
        assert ALLOC_SIZES_ENV not in os.environ


def test_shim_reads_the_exported_name():
    assert f'"{ALLOC_SIZES_ENV}"' in SHIM.read_text()


# ── CLI / wiring ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--alloc-sizes"], True)])
def test_cli_reaches_fuzzer(monkeypatch, tmp_path, flags, expected):
    from tests.test_shim_touched_atomic_env import _fuzz_kwargs

    assert _fuzz_kwargs(monkeypatch, tmp_path, flags)["alloc_sizes"] is expected


def test_hail_mary_leaves_it_off(monkeypatch, tmp_path):
    """Changes the edge set: not a strategy --hail-mary may flip."""
    from tests.test_shim_touched_atomic_env import _fuzz_kwargs

    assert _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])["alloc_sizes"] is False


def test_fuzzer_exports_env():
    """Fuzzer(alloc_sizes=True) sets the env before any target starts."""
    import inspect

    from fuzzer_tool.services import fuzzer

    assert inspect.signature(fuzzer.Fuzzer.__init__).parameters["alloc_sizes"].default is False
    assert "export_alloc_sizes(alloc_sizes)" in inspect.getsource(fuzzer.Fuzzer.__init__)


@requires_clang
def test_regression_env_parse_adds_no_compare(tmp_path):
    """Parsing the env must not run through the shim's own strcmp override.

    It did, and every run with the env set counted one phantom strcmp.
    """
    exe = _build(tmp_path, _SRC, *EDGE_FLAGS, "-D__AFL_CMPLOG=1")
    counts = tmp_path / "counts.txt"
    env = _clean_env(_CMPLOG_COUNTS=str(counts), **{ALLOC_SIZES_ENV: "1"})
    r = subprocess.run([str(exe), "16"], env=env, capture_output=True, timeout=60)
    assert r.returncode == 0
    assert "strcmp" not in counts.read_text()
