"""``-D__AFL_TRACE_FIRES=1``: the shim's per-fire edge-id log (P0-1).

F2 (docs/handover/handover_edge_id_axis_2026-09-18.md) was a path-hash
difference between the first execution against a clean table and later ones.
The path hash proves the fire sequence differs, not where; this log is the
sequence. Compiled out by default, and with the gate on it writes only when
``__AFL_FIRES_OUT`` names a file. The driver calls the guard callback
directly, as tests/test_edge_id_stability_guard.py does, so no
``-fsanitize-coverage`` is needed.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SHIM = os.path.join(_SRC, "fuzzer_tool", "adapters", "afl_shim.c")
CC = shutil.which("clang") or shutil.which("gcc")
MASK64 = (1 << 64) - 1

# Guards 1..n, then 1..n again: repeated edges must log once per fire.
_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    uint32_t n = (uint32_t)atoi(argv[1]);
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t g = 1; g <= n; g++) { uint32_t guard = g;
            __sanitizer_cov_trace_pc_guard(&guard); }
    return 0;
}
"""

GUARDS = 12


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    """{"on": gate in, ctx off; "on_ctx": gate in, ctx on; "off": default build}."""
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("fires")
    src = d / "drv.c"
    src.write_text(_DRIVER)
    out = {}
    builds = (
        ("on", ["-D__AFL_TRACE_FIRES=1", "-D__AFL_CTX_SENSITIVE=0"]),
        ("on_ctx", ["-D__AFL_TRACE_FIRES=1", "-fno-omit-frame-pointer"]),
        ("off", ["-D__AFL_CTX_SENSITIVE=0"]),
    )
    for name, flags in builds:
        exe = d / f"drv_{name}"
        r = subprocess.run(
            [CC, "-O1", *flags, "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[-1500:]
        out[name] = str(exe)
    return out


def _run(exe, cov, fires_out):
    cov.reset_edge_map()
    # FUZZER_KEEP_ASLR=1: base-relative caller context, so a ctx build's ids
    # do not move with ASLR (F1) and only F2 is under test.
    env = dict(
        os.environ,
        __AFL_SHM_ID=str(cov.shm_id),
        AFL_MAP_SIZE=str(cov.num_entries),
        FUZZER_KEEP_ASLR="1",
    )
    if fires_out is not None:
        env["__AFL_FIRES_OUT"] = str(fires_out)
    return subprocess.run([exe, str(GUARDS)], env=env, capture_output=True, timeout=10)


@pytest.fixture
def cov():
    c = ShmCoverage(size=4096)
    yield c
    c.cleanup()


def test_log_replays_the_path_hash(drivers, cov, tmp_path):
    out = tmp_path / "fires.txt"
    assert _run(drivers["on"], cov, out).returncode == 0
    ids = [int(x) for x in out.read_text().split()]
    assert len(ids) == 2 * GUARDS

    # The shim's rolling hash, recomputed from the log alone.
    h = 0
    for e in ids:
        h = ((h * 31) ^ e) & MASK64
    assert h == cov.read_path_hash()
    live = set(cov._scan_with_positions()[1].tolist())
    assert set(ids) == live


@pytest.mark.parametrize("build", ["on", "on_ctx"])
def test_clean_table_run_matches_reset_table_runs(drivers, cov, tmp_path, build):
    """F2 closed: the first execution against a fresh segment is comparable.

    Measured 2026-09-23 on fuzzgoat, all 250 inputs, ctx and context-free:
    fire streams, path hashes and id sets identical across clean, reset and
    reset again. This pins it on the driver.
    """
    seen = []
    for i in range(3):  # run 0 sees a clean table (fresh fixture), 1-2 a reset one
        out = tmp_path / f"{i}.txt"
        _run(drivers[build], cov, out)
        seen.append(
            (out.read_text(), cov.read_path_hash(), set(cov._scan_with_positions()[1].tolist()))
        )
    assert seen[0] == seen[1] == seen[2]


def test_compiled_out_by_default(drivers, cov, tmp_path):
    out = tmp_path / "fires.txt"
    assert _run(drivers["off"], cov, out).returncode == 0
    assert not out.exists()
    assert (
        "__afl_fire_fd"
        not in subprocess.run(["nm", drivers["off"]], capture_output=True, text=True).stdout
    )


def test_unset_or_unopenable_path_is_harmless(drivers, cov, tmp_path):
    assert _run(drivers["on"], cov, None).returncode == 0
    r = _run(drivers["on"], cov, tmp_path / "missing" / "dir" / "fires.txt")
    assert r.returncode == 0
    assert len(cov._scan_with_positions()[1]) > 0  # coverage unaffected
