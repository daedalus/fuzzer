"""Divisors and GEP indices as input-to-state pairs.

``-fsanitize-coverage=trace-div,trace-gep`` hands the shim two operands
trace-cmp never sees: the runtime divisor of every non-constant division
and the runtime index of every GEP. Neither is a comparison, so neither
can be written as a ``CMP`` record without fabricating an opponent
operand and polluting the wall statistics that read those records.

They are written as their own record kinds instead, and the collector
turns each into I2S pairs against the values that make them interesting:
a divisor against 0 and 1 (division by zero, degenerate quotient), an
index against 0 and the all-ones word (out of bounds). The existing
redqueen mutator consumes those pairs unchanged.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.track_parser import (
    OPERAND_MIN_VALUE,
    pairs_from_operand_records,
)
from fuzzer_tool.core.cmplog import CmplogCollector
from tests.conftest import requires_clang

AFL_SHIM = Path("src/fuzzer_tool/adapters/afl_shim.c")

# The divisor and the index both come from argv so no constant folding can
# reach them: with a literal divisor clang emits no trace-div call at all.
# The table is large enough that the index clears OPERAND_MIN_VALUE while
# staying in bounds -- a small index is dropped by the floor, not a bug.
_TARGET_C = """
#include <stdlib.h>
#include <stdio.h>
static int table[65536];
int main(int argc, char **argv) {
    if (argc < 3) return 1;
    unsigned d = (unsigned)strtoul(argv[1], NULL, 0);
    unsigned long i = strtoul(argv[2], NULL, 0);
    volatile unsigned q = 1000000u / d;
    volatile int v = table[i % 65536];
    printf("%u %d\\n", q, v);
    return 0;
}
"""

_DIVISOR = 0x12345678
_INDEX = 0xABCD


@pytest.fixture(scope="module")
def operand_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("trace_operands")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            # -O0: at -O1 clang folds the array index into an addressing
            # mode before the sancov pass runs, so the GEP callback is never
            # emitted for it. The division survives either way.
            "-O0",
            # trace-div/trace-gep are modifiers, not levels: clang emits no
            # call sites for them unless a coverage level is also requested.
            "-fsanitize-coverage=trace-pc-guard,trace-div,trace-gep",
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
        pytest.skip(f"operand target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path) -> CmplogCollector:
    c = CmplogCollector()
    c.log_path = str(tmp_path / "cmplog.txt")
    Path(c.log_path).touch()
    env = dict(os.environ)
    env["_CMPLOG_OUT"] = c.log_path
    env.pop("_CMPLOG_COUNTS", None)
    env.pop("_CMPLOG_SITE_COUNTS", None)
    subprocess.run(
        [str(exe), str(_DIVISOR), str(_INDEX)],
        capture_output=True,
        env=env,
        timeout=60,
        check=True,
    )
    c.collect_tokens()
    return c


@requires_clang
class TestAgainstTheRealShim:
    def test_the_divisor_reaches_the_pair_pool(self, tmp_path, operand_target):
        c = _run(operand_target, tmp_path)
        divisor = _DIVISOR.to_bytes(4, "little")
        targets = {b for a, b in c.pairs if a == divisor}
        assert targets, f"divisor absent from {len(c.pairs)} pairs"
        assert b"\x00\x00\x00\x00" in targets

    def test_the_gep_index_reaches_the_pair_pool(self, tmp_path, operand_target):
        c = _run(operand_target, tmp_path)
        index = _INDEX.to_bytes(8, "little")  # trace-gep is pointer-sized
        targets = {b for a, b in c.pairs if a == index}
        assert targets, f"index absent from {len(c.pairs)} pairs"
        assert b"\xff" * 8 in targets

    def test_operands_are_not_counted_as_comparisons(self, tmp_path, operand_target):
        """A divisor has no opponent: it must not enter the wall statistics."""
        c = _run(operand_target, tmp_path)
        assert c.comparison_stats() == {}
        assert c.comparison_walls(min_fired=1) == {}


class TestParsing:
    def test_divisor_pairs_against_zero_and_one(self):
        val = (0x12345678).to_bytes(4, "little")
        pairs = pairs_from_operand_records([f"DIV {val.hex()} 4 0x400123"])
        assert pairs == [
            (val, (0).to_bytes(4, "little")),
            (val, (1).to_bytes(4, "little")),
        ]

    def test_index_pairs_against_zero_and_all_ones(self):
        val = (0xABCD).to_bytes(8, "little")
        pairs = pairs_from_operand_records([f"GEP {val.hex()} 8 0x400123"])
        assert pairs == [
            (val, (0).to_bytes(8, "little")),
            (val, b"\xff" * 8),
        ]

    def test_small_operands_are_dropped(self):
        """Every loop counter is a GEP index; below the floor they are noise."""
        val = (OPERAND_MIN_VALUE - 1).to_bytes(4, "little")
        assert pairs_from_operand_records([f"GEP {val.hex()} 4 0x400123"]) == []

    def test_comparison_records_are_left_alone(self):
        assert pairs_from_operand_records(["CMP 41414141 42424242 -1 4 0x400123"]) == []

    @pytest.mark.parametrize(
        "line",
        [
            "DIV",
            "DIV zzzz 4 0x400123",
            "DIV 12345678",
            "DIV 12345678 3 0x400123",  # width the shim never emits
            "DIV 1234 4 0x400123",  # operand shorter than its own width
            "GEP  7 0x400123",
            "DIVX 12345678 4 0x400123",
        ],
    )
    def test_malformed_records_are_skipped(self, line):
        assert pairs_from_operand_records([line]) == []

    def test_collector_folds_operand_pairs_in(self, tmp_path):
        val = (0x12345678).to_bytes(4, "little")
        c = CmplogCollector()
        c.log_path = str(tmp_path / "cmplog.txt")
        Path(c.log_path).write_text(f"DIV {val.hex()} 4 0x400123\n")
        c.collect_tokens()
        assert (val, (0).to_bytes(4, "little")) in c.pairs
        assert val in c.tokens
