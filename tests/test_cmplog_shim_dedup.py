"""Shim-side dedup of cmplog records within one drain.

79% of ffmpeg's records repeated a record already written in the same
execution (21.2M lines over 125 seeds, 4.4M unique), so the 10k-line read
cap kept ~9% of an execution's distinct comparisons. The collector dedups
each drain anyway, so the shim now drops repeats before formatting; the
scope resets at ``__cmplog_reset`` (the drain boundary) and in a fork child.
Counters are untouched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_REPS = 100
_DISTINCT = 200_000  # with the loop's own i-vs-bound records, > dedup slots
_KEY = 0x71727374

_TARGET_C = r"""
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
void __cmplog_reset(void);
void __tracecmp_flush(void);
static char big_a[80], big_b[80];
static void once(const char *s, unsigned long v) {
    volatile int r = memcmp(s, "REPEATAA", 8);
    volatile int c = (v == 0x71727374UL);
    (void)r; (void)c;
}
int main(int argc, char **argv) {
    if (argc < 4) return 1;
    unsigned long v = strtoul(argv[2], NULL, 0);
    const char *mode = argv[3];
    if (!strcmp(mode, "repeat")) {
        for (int i = 0; i < REPS; i++) once(argv[1], v);
    } else if (!strcmp(mode, "reset")) {
        once(argv[1], v);
        __cmplog_reset();          /* drain boundary: truncates the file */
        once(argv[1], v);
    } else if (!strcmp(mode, "fork")) {
        once(argv[1], v);
        __tracecmp_flush();
        pid_t pid = fork();
        if (pid == 0) { once(argv[1], v); __tracecmp_flush(); _exit(0); }
        waitpid(pid, NULL, 0);
    } else if (!strcmp(mode, "distinct")) {
        for (unsigned long i = 0; i < DISTINCT; i++) {
            volatile int c = (v + i == 0x71727374UL);
            (void)c;
        }
    } else if (!strcmp(mode, "lengths")) {
        memset(big_a, 'A', sizeof big_a);
        memset(big_b, 'A', sizeof big_b);
        big_b[79] = 'B';
        big_a[0] = argv[1][0];     /* defeat folding */
        big_b[78] = argv[1][0] == 'Z' ? 'Y' : 'Z';
        volatile int r1 = memcmp(big_a, big_b, 79);
        volatile int r2 = memcmp(big_a, big_b, 80);
        (void)r1; (void)r2;
    }
    __tracecmp_flush();
    return 0;
}
"""


@pytest.fixture(scope="module")
def dedup_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_dedup")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O0",
            f"-DREPS={_REPS}",
            f"-DDISTINCT={_DISTINCT}",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp",
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
        pytest.skip(f"dedup target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path, mode: str) -> tuple[list[str], str]:
    records = tmp_path / f"{mode}.cmplog"
    counts = tmp_path / f"{mode}.counts"
    env = dict(os.environ, _CMPLOG_OUT=str(records), _CMPLOG_COUNTS=str(counts))
    run = subprocess.run(
        [str(exe), "XXXXXXXX", "0x1234", mode], capture_output=True, text=True, env=env, timeout=120
    )
    assert run.returncode == 0, run.stderr
    return records.read_text().splitlines(), counts.read_text()


_MEMCMP_HEX = b"REPEATAA".hex()
_KEY_HEX = _KEY.to_bytes(8, "little").hex()


def _hits(lines: list[str], needle: str) -> int:
    return sum(needle in ln for ln in lines)


@requires_clang
class TestShimDedup:
    def test_repeats_written_once(self, dedup_target, tmp_path):
        lines, _ = _run(dedup_target, tmp_path, "repeat")
        assert _hits(lines, _MEMCMP_HEX) == 1
        assert _hits(lines, _KEY_HEX) == 1

    def test_no_line_repeats_within_a_drain(self, dedup_target, tmp_path):
        lines, _ = _run(dedup_target, tmp_path, "repeat")
        assert len(lines) == len(set(lines))

    def test_counters_still_count_every_call(self, dedup_target, tmp_path):
        """Falsification: dedup must stay out of the counts channel."""
        _, counts = _run(dedup_target, tmp_path, "repeat")
        fired = sum(
            int(p[2])
            for p in (ln.split() for ln in counts.splitlines())
            if p[:2] == ["CNT", "memcmp"]
        )
        assert fired >= _REPS

    def test_reset_starts_a_new_scope(self, dedup_target, tmp_path):
        """After the drain boundary the same comparison is written again."""
        lines, _ = _run(dedup_target, tmp_path, "reset")
        assert _hits(lines, _MEMCMP_HEX) == 1
        assert _hits(lines, _KEY_HEX) == 1

    def test_fork_child_starts_clean(self, dedup_target, tmp_path):
        """Adversarial: a forkserver child must not inherit the parent's
        seen-set, or its own execution's records vanish."""
        lines, _ = _run(dedup_target, tmp_path, "fork")
        assert _hits(lines, _MEMCMP_HEX) == 2
        assert _hits(lines, _KEY_HEX) == 2

    def test_table_overflow_loses_nothing(self, dedup_target, tmp_path):
        """Adversarial: more distinct records than slots fail open."""
        lines, _ = _run(dedup_target, tmp_path, "distinct")
        assert len({ln for ln in lines if f" {_KEY_HEX} " in ln}) == _DISTINCT

    def test_length_field_distinguishes_records(self, dedup_target, tmp_path):
        """Same 64-byte truncated operands, different n: different lines."""
        lines, _ = _run(dedup_target, tmp_path, "lengths")
        ns = {ln.split()[4] for ln in lines if ln.startswith("CMP ") and len(ln.split()[1]) == 128}
        assert {"79", "80"} <= ns
