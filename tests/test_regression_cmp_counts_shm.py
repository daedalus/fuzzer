"""Regression: cmplog counters travelled through text files every exec.

Per exec the shim formatted ``CNT``/``CNS`` lines (~1000 on ffmpeg) and
Python stat-ed, opened, seeked and parsed them (~330 us fold). With
``_CMPLOG_COUNTS_SHM`` the shim keeps both counter sets in a SysV segment;
``CmpCounterShm`` reads them through numpy views and zeroes them. Old shims
never mark the segment, so the file path stays the fallback.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.cmp_shm import CMP_NAMES, CMP_SHM_ENV, CmpCounterShm
from fuzzer_tool.core.cmplog import CmplogCollector

REPO = Path(__file__).parent.parent
AFL_SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
MEMCMP, STRCMP = CMP_NAMES.index("memcmp"), CMP_NAMES.index("strcmp")
PC_A, PC_B = 0x401000, 0x401080
SLOT_A, SLOT_B = 7, 900


@pytest.fixture
def shm():
    s = CmpCounterShm()
    yield s
    s.close()


def _collector(shm: CmpCounterShm, sites: bool = False) -> CmplogCollector:
    c = CmplogCollector(site_counts=sites)
    c.counts_shm = shm
    return c


def _site(shm: CmpCounterShm, slot: int, cb: int, pc: int, fired: int, hit: int) -> None:
    """Write one site entry the way the shim does."""
    e = shm.sites[slot]
    e["pc"], e["id"], e["fired"], e["hit"] = pc, cb, fired, hit


def test_names_match_shim():
    """CMP_NAMES must be the shim's __afl_cmp_names, index for index."""
    src = AFL_SHIM.read_text()
    body = re.search(r"__afl_cmp_names\[__AFL_CMP_SITES\] = \{(.*?)\};", src, re.S).group(1)
    assert tuple(re.findall(r'"([^"]+)"', body)) == CMP_NAMES


def test_regression_cmp_counts_shm(shm):
    """A drain reads this exec's deltas from SHM and zeroes them."""
    shm.mark_attached()
    shm.fired[MEMCMP], shm.hit[MEMCMP] = 4, 1
    shm.fired[STRCMP] = 2
    c = _collector(shm)
    assert c.collect_counts() == ({"memcmp": 4, "strcmp": 2}, {"memcmp": 1})
    assert not shm.fired.any() and not shm.hit.any()

    # Delta contract: the next drain sees only what came after.
    shm.fired[MEMCMP], shm.hit[MEMCMP] = 3, 2
    assert c.collect_counts() == ({"memcmp": 3}, {"memcmp": 2})
    assert c.comparison_stats() == {"memcmp": (7, 3), "strcmp": (2, 0)}
    assert c.collect_counts() == ({}, {})


def test_unattached_falls_back_to_files(shm, tmp_path):
    """Falsification: an old shim never marks the segment; files are read."""
    c = _collector(shm)
    c.counts_path = str(tmp_path / "counts.txt")
    Path(c.counts_path).write_text("CNT bcmp 5 5\n")
    shm.fired[MEMCMP] = 9  # stray, unmarked: must be ignored
    assert c.collect_counts() == ({"bcmp": 5}, {"bcmp": 5})


def _lines_collector(tmp_path: Path, lines: list[str]) -> CmplogCollector:
    c = CmplogCollector(site_counts=True)
    c.sites_path = str(tmp_path / "sites.txt")
    Path(c.sites_path).write_text("".join(f"{x}\n" for x in lines))
    c.collect_counts()
    return c


def _site_view(c: CmplogCollector):
    return c.site_fired, c.site_asserted, c.last_site_asserted, c.site_dropped


def test_control_lines_match_themselves(tmp_path):
    """Hard Rule 46: the line oracle agrees with a second run of itself."""
    lines = [f"CNS memcmp {PC_A:x} 4 1", f"CNS strcmp {PC_B:x} 2 0", "CND 3"]
    views = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        views.append(_site_view(_lines_collector(tmp_path / name, lines)))
    assert views[0] == views[1]


def test_sites_match_line_oracle(shm, tmp_path):
    """SHM site drain folds the same totals as the equivalent CNS lines."""
    lines = [f"CNS memcmp {PC_A:x} 4 1", f"CNS strcmp {PC_B:x} 2 0", "CND 3"]
    oracle = _lines_collector(tmp_path, lines)

    shm.mark_attached()
    _site(shm, SLOT_A, MEMCMP, PC_A, 4, 1)
    _site(shm, SLOT_B, STRCMP, PC_B, 2, 0)
    shm.header["site_dropped"] = 3
    c = _collector(shm, sites=True)
    c.collect_counts()
    assert _site_view(c) == _site_view(oracle)
    # Counters zeroed, keys kept (the shim pays the insert once).
    assert shm.sites[SLOT_A]["pc"] == PC_A and shm.sites[SLOT_A]["fired"] == 0


def test_site_keys_survive_drains(shm):
    """Adversarial: a slot drained once and refilled folds into the same site."""
    shm.mark_attached()
    c = _collector(shm, sites=True)
    _site(shm, SLOT_A, MEMCMP, PC_A, 4, 1)
    c.collect_counts()
    shm.sites[SLOT_A]["fired"], shm.sites[SLOT_A]["hit"] = 2, 2
    c.collect_counts()
    assert c.site_fired == {("memcmp", PC_A): 6}
    assert c.last_site_asserted == {("memcmp", PC_A): 2}


# ── End to end against the real shim ───────────────────────────────────

TARGET = r"""
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
static int cmps(void) {
    volatile int r = 0;
    r += memcmp("HELLO_WORLD", "HELLO_WORLD", 11) == 0;
    r += memcmp("HELLO_WORLD", "GOODBYE_ABC", 11) == 0;
    r += strcmp("HELLO_WORLD", "nope") == 0;
    return r;
}
int main(int argc, char **argv) {
    int r = cmps();
    if (argc > 1) {            /* a forked child's comparisons count too */
        pid_t p = fork();
        if (p == 0) { cmps(); _exit(0); }
        waitpid(p, 0, 0);
    }
    printf("%d\n", r);
    return 0;
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory) -> Path:
    if shutil.which("clang") is None or shutil.which("setarch") is None:
        pytest.skip("clang or setarch not available")
    d = tmp_path_factory.mktemp("cmpshm")
    obj, src, out = d / "shim.o", d / "t.c", d / "t"
    src.write_text(TARGET)
    flags = ["-fno-builtin-memcmp", "-fno-builtin-strcmp"]
    for cmd in (
        ["clang", "-O2", "-fPIC", "-D__AFL_PRELOAD_ONLY", "-c", str(AFL_SHIM), "-o", str(obj)],
        ["clang", "-O1", *flags, str(src), str(obj), "-o", str(out), "-ldl"],
    ):
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            pytest.skip("shim target did not build")
    return out


def _run(exe: Path, tmp_path: Path, shm: CmpCounterShm | None, *args: str) -> CmplogCollector:
    c = CmplogCollector(site_counts=True)
    c.counts_path = str(tmp_path / "counts.txt")
    c.sites_path = str(tmp_path / "sites.txt")
    env = dict(os.environ, _CMPLOG_COUNTS=c.counts_path, _CMPLOG_SITE_COUNTS=c.sites_path)
    env.pop(CMP_SHM_ENV, None)  # file runs must not inherit a segment
    if shm is not None:
        env[CMP_SHM_ENV] = shm.env_id
        c.counts_shm = shm
    # ASLR off, as the fuzzer runs targets: site keys are absolute PCs.
    cmd = ["setarch", os.uname().machine, "-R", str(exe), *args]
    done = subprocess.run(cmd, env=env, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr
    c.collect_counts()
    return c


def _totals(c: CmplogCollector):
    return c.comparison_stats(), c.site_fired, c.site_asserted


def test_shim_shm_matches_files(exe, tmp_path, shm):
    for name in ("f1", "f2", "s"):
        (tmp_path / name).mkdir()
    files = _run(exe, tmp_path / "f1", None)
    assert _totals(files) == _totals(_run(exe, tmp_path / "f2", None))  # control
    via_shm = _run(exe, tmp_path / "s", shm)
    assert shm.attached
    assert _totals(via_shm) == _totals(files)
    assert files.comparison_stats()["memcmp"] == (2, 1)
    # SHM mode writes no sidecar text.
    for name in ("counts.txt", "sites.txt"):
        p = tmp_path / "s" / name
        assert not p.exists() or p.stat().st_size == 0


def test_forked_child_counted(exe, tmp_path, shm):
    """Adversarial: a forked child's comparisons land in the shared segment."""
    via_shm = _run(exe, tmp_path, shm, "fork")
    assert via_shm.comparison_stats()["memcmp"] == (4, 2)
