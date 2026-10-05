"""WingFuzz few-bits-set constant pruning in the cmplog shim.

WingFuzz's ``LoadCmpTracer`` skips a compare against a constant with fewer
than two bits set or fewer than two bits clear (0, 1, -1, 0x80..., 0x7f...):
such a constant is no information for input-to-state replacement. With
``$__AFL_CMP_PRUNE_CONST`` set, the shim drops those *records*. Counters and
COMPCOV are untouched, and variable-vs-variable compares are never pruned.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_TARGET_C = r"""
#include <stdint.h>
#include <stdlib.h>
void __tracecmp_flush(void);
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    unsigned long seed = strtoul(argv[1], NULL, 0);
    volatile uint8_t  v1 = (uint8_t)seed;
    volatile uint16_t v2 = (uint16_t)seed;
    volatile uint32_t v4 = (uint32_t)seed;
    volatile uint64_t v8 = (uint64_t)seed;
    volatile uint32_t w4 = 1;
    volatile int sink = 0;

    sink += (v1 == 0x00); sink += (v1 == 0x01); sink += (v1 == 0x7f);
    sink += (v1 == 0x80); sink += (v1 == 0xfe); sink += (v1 == 0xff);
    sink += (v1 == 0x41);
    sink += (v2 == 0x0000); sink += (v2 == 0x8000); sink += (v2 == 0x4142);
    sink += (v4 == 0x00000000u); sink += (v4 == 0x00000001u);
    sink += (v4 == 0xffffffffu); sink += (v4 == 0x80000000u);
    sink += (v4 == 0x71727374u);
    sink += (v8 == 0ull); sink += (v8 == 0xffffffffffffffffull);
    sink += (v8 == 0x8000000000000000ull); sink += (v8 == 0x1122334455667788ull);
    /* ordering compare against a poor constant: dropped upstream too */
    sink += (v4 > 0x00000001u);
    /* variable vs variable: never pruned, even with 0 / 1 operands */
    volatile uint32_t z4 = 0;
    sink += (z4 == w4);
    (void)sink;
    __tracecmp_flush();
    return 0;
}
"""

# Records print operands little-endian, constant first.
_RICH = {
    "41": 1,
    "4241": 2,
    "74737271": 4,
    "8877665544332211": 8,
}
_POOR = {
    "00": 1,
    "01": 1,
    "7f": 1,
    "80": 1,
    "fe": 1,
    "ff": 1,
    "0000": 2,
    "0080": 2,
    "00000000": 4,
    "01000000": 4,
    "ffffffff": 4,
    "00000080": 4,
    "0000000000000000": 8,
    "ffffffffffffffff": 8,
    "0000000000000080": 8,
}
_VAR_VS_VAR = ("00000000", "01000000")
# The seed 0x1234 as each width prints it; shim-internal compares never carry it.
_SEED_FIELDS = {"34", "3412", "34120000", "3412000000000000"}


@pytest.fixture(scope="module")
def prune_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_const_prune")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O2",  # -O0 promotes 1/2-byte compares to int; -O2 keeps cmp1/cmp2
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
        pytest.skip(f"prune target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path, *, prune: str | None) -> tuple[list[str], str]:
    tag = f"prune_{prune}"
    records = tmp_path / f"{tag}.cmplog"
    counts = tmp_path / f"{tag}.counts"
    env = dict(os.environ, _CMPLOG_OUT=str(records), _CMPLOG_COUNTS=str(counts))
    env.pop("__AFL_CMP_PRUNE_CONST", None)
    if prune is not None:
        env["__AFL_CMP_PRUNE_CONST"] = prune
    run = subprocess.run([str(exe), "0x1234"], capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 0, run.stderr
    return records.read_text().splitlines(), counts.read_text()


def _const_operands(lines: list[str]) -> set[str]:
    """First operand of every CMP record whose second operand is the seed."""
    out = set()
    for ln in lines:
        p = ln.split()
        if p[0] == "CMP" and len(p) >= 4 and p[2] in _SEED_FIELDS:
            out.add(p[1])
    return out


def _count_total(counts: str) -> int:
    return sum(int(p[2]) for p in (ln.split() for ln in counts.splitlines()) if p[0] == "CNT")


@requires_clang
class TestConstPrune:
    def test_default_logs_poor_and_rich_constants(self, prune_target, tmp_path):
        """Falsification: with the switch off nothing is dropped."""
        lines, _ = _run(prune_target, tmp_path, prune=None)
        ops = _const_operands(lines)
        assert set(_POOR) <= ops
        assert set(_RICH) <= ops

    def test_prune_drops_poor_constants(self, prune_target, tmp_path):
        lines, _ = _run(prune_target, tmp_path, prune="1")
        assert not set(_POOR) & _const_operands(lines)

    def test_prune_keeps_rich_constants(self, prune_target, tmp_path):
        lines, _ = _run(prune_target, tmp_path, prune="1")
        assert set(_RICH) <= _const_operands(lines)

    def test_prune_leaves_variable_compares(self, prune_target, tmp_path):
        """Adversarial: a 0-vs-1 compare of two variables is not a constant compare."""
        lines, _ = _run(prune_target, tmp_path, prune="1")
        pairs = {(ln.split()[1], ln.split()[2]) for ln in lines if ln.startswith("CMP ")}
        assert _VAR_VS_VAR in pairs

    def test_counters_unchanged(self, prune_target, tmp_path):
        """Pruning is a record filter; the comparison profile must not move."""
        _, off = _run(prune_target, tmp_path, prune=None)
        _, on = _run(prune_target, tmp_path, prune="1")
        assert _count_total(on) == _count_total(off) > 0

    def test_zero_value_means_off(self, prune_target, tmp_path):
        """Adversarial: ``=0`` is the documented off spelling, not 'set'."""
        lines, _ = _run(prune_target, tmp_path, prune="0")
        assert set(_POOR) <= _const_operands(lines)
