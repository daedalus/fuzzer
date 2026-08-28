"""Per-site comparison counters: which comparison, not just which family.

The per-callback counters have one bucket per interceptor, so two memcmp
call sites are one number. A target with one always-satisfied memcmp site
and one never-satisfied site reports a 75% assert rate for the memcmp
family and hides the wall completely.

Keying by program counter separates them. The call site comes from
``__builtin_return_address(0)`` evaluated inside the interceptor, so the
counting macro picks it up and no interceptor had to be edited to pass it.

Absolute addresses, matching the pc field layer-2 records already carry,
which is sound only because the fuzzer disables ASLR for the target
(``personality(ADDR_NO_RANDOMIZE)``, adapters/process.py).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector
from tests.conftest import requires_gcc

AFL_SHIM = Path("src/fuzzer_tool/adapters/afl_shim.c")

# One function per site so the compiler cannot merge them; the loop makes
# the two sites' fire counts differ, which is what proves they are separate
# buckets rather than one bucket printed twice.
_TARGET_C = """
#include <stdio.h>
#include <string.h>
int hot(const char *s)  { return memcmp(s, "AAAA", 4) == 0; }
int cold(const char *s) { return memcmp(s, "BBBB", 4) == 0; }
int main(void) {
    volatile int r = 0;
    for (int i = 0; i < 3; i++) r += hot("AAAA");   /* always satisfied */
    r += cold("ZZZZ");                              /* never satisfied */
    printf("%d\\n", r);
    return 0;
}
"""


@pytest.fixture(scope="module")
def site_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmp_sites")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "gcc",
            "-O1",
            "-fno-builtin-memcmp",
            "-fno-builtin-strcmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"site target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path, *, sites: bool) -> CmplogCollector:
    c = CmplogCollector()
    c.counts_path = str(tmp_path / "counts.txt")
    env = dict(os.environ)
    env["_CMPLOG_COUNTS"] = c.counts_path
    env.pop("_CMPLOG_OUT", None)
    if sites:
        c.sites_path = str(tmp_path / "counts.txt.sites")
        env["_CMPLOG_SITE_COUNTS"] = c.sites_path
    else:
        env.pop("_CMPLOG_SITE_COUNTS", None)
    subprocess.run([str(exe)], capture_output=True, env=env, timeout=60, check=True)
    c.collect_counts()
    return c


@requires_gcc
class TestAgainstTheRealShim:
    def test_the_callback_bucket_hides_the_wall(self, tmp_path, site_target):
        """The motivating measurement, stated as the baseline it is."""
        c = _run(site_target, tmp_path, sites=False)
        assert c.comparison_stats()["memcmp"] == (4, 3)
        # 75% satisfied: no wall is visible at this granularity, and the
        # never-satisfied site is invisible.
        assert c.comparison_walls(min_fired=1) == {}

    def test_sites_separate_what_the_bucket_merged(self, tmp_path, site_target):
        c = _run(site_target, tmp_path, sites=True)
        rows = sorted(c.site_fired.items(), key=lambda kv: -kv[1])
        assert len(rows) == 2, rows
        (hot_key, hot_fired), (cold_key, cold_fired) = rows
        assert hot_fired == 3
        assert cold_fired == 1
        assert c.site_asserted[hot_key] == 3
        assert c.site_asserted.get(cold_key, 0) == 0
        assert hot_key[0] == cold_key[0] == "memcmp"
        assert hot_key[1] != cold_key[1], "distinct call sites, distinct PCs"

    def test_the_never_satisfied_site_reads_as_a_wall(self, tmp_path, site_target):
        c = _run(site_target, tmp_path, sites=True)
        walls = c.site_walls(min_fired=1)
        assert len(walls) == 1
        (name, _pc), (fired, asserted) = next(iter(walls.items()))
        assert (name, fired, asserted) == ("memcmp", 1, 0)

    def test_counting_is_off_without_the_env_var(self, tmp_path, site_target):
        """A hash and a probe per comparison is not a default."""
        c = _run(site_target, tmp_path, sites=False)
        assert not (tmp_path / "counts.txt.sites").exists()
        assert c.site_fired == {}

    def test_per_callback_counters_are_unaffected_by_site_counting(self, tmp_path, site_target):
        with_sites = _run(site_target, tmp_path / "a", sites=True)
        without = _run(site_target, tmp_path / "b", sites=False)
        assert with_sites.comparison_stats() == without.comparison_stats()


class TestParsing:
    def _collector(self, tmp_path, *lines: str) -> CmplogCollector:
        c = CmplogCollector()
        c.sites_path = str(tmp_path / "s.txt")
        Path(c.sites_path).write_text("".join(line + "\n" for line in lines))
        c.collect_sites()
        return c

    def test_deltas_accumulate(self, tmp_path):
        c = self._collector(
            tmp_path,
            "CNS memcmp 0x4011a0 3 1",
            "CNS memcmp 0x4011a0 2 0",
            "CNS strcmp 0x401200 7 7",
        )
        assert c.site_fired[("memcmp", 0x4011A0)] == 5
        assert c.site_asserted[("memcmp", 0x4011A0)] == 1
        assert c.site_fired[("strcmp", 0x401200)] == 7

    def test_the_offset_stops_double_counting(self, tmp_path):
        c = self._collector(tmp_path, "CNS memcmp 0x4011a0 3 1")
        c.collect_sites()
        assert c.site_fired[("memcmp", 0x4011A0)] == 3

    def test_dropped_insertions_are_reported(self, tmp_path):
        """A full table means the site figures are a subset, not a census."""
        c = self._collector(tmp_path, "CNS memcmp 0x4011a0 3 1", "CND 42")
        assert c.site_dropped == 42

    def test_malformed_lines_are_skipped(self, tmp_path):
        c = self._collector(
            tmp_path,
            "CNT memcmp 4 3",  # the other channel
            "CNS memcmp notahexpc 1 1",
            "CNS memcmp 0x4011a0 3",  # truncated
            "CNS bcmp 0x402000 9 0",
        )
        assert c.site_fired == {("bcmp", 0x402000): 9}

    def test_no_sites_path_is_inert(self):
        c = CmplogCollector()
        c.collect_sites()
        assert c.site_fired == {}
