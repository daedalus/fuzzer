"""Cmplog records in a shared table (``$_CMPLOG_RECORDS_SHM``) vs the text stream.

With the segment attached the shim writes CMP records into it as binary
operands instead of hex text. The reader must yield the same
``(op_a, op_b, width, result, pc)`` set the text parser gets from the same
run; DIV/GEP records stay on the text stream.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.cmplog_rec_shm import CMPLOG_REC_ENV, CmpRecordShm
from fuzzer_tool.adapters.track_parser import cond_tuples_from_cmplog_text
from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_LOOP = 10
# PCs are absolute: without ASLR off, two runs never agree (the Rule 46 control caught it).
_NO_ASLR = ["setarch", platform.machine(), "-R"]

_TARGET_C = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
volatile int sink;
__attribute__((noinline)) static void loop_site(uint32_t v) {
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 10; i++) sink += (v == (uint32_t)i * 7u + 1000u);
}
int main(int argc, char **argv) {
    if (argc < 3) return 1;
    unsigned long v = strtoul(argv[1], NULL, 0);
    volatile uint8_t v1 = (uint8_t)v;
    volatile uint16_t v2 = (uint16_t)v;
    volatile uint32_t v4 = (uint32_t)v;
    volatile uint64_t v8 = (uint64_t)v;
    sink += (v1 == 0x41);
    sink += (v2 == 0x4142);
    sink += (v4 == 0x71727374u);
    sink += (v8 == 0x1122334455667788ull);
    sink += (v4 > 0x10u);
    loop_site(v4);
    sink += memcmp(argv[2], "MAGICVAL", 8) == 0;
    sink += strcmp(argv[2], "hello") == 0;
    sink += 1000 / (int)(v4 | 1);   /* DIV record: stays on the text stream */
    return 0;
}
"""


@pytest.fixture(scope="module")
def rec_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_rec_shm")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O2",
            "-fno-builtin",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp,trace-div",
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
        pytest.skip(f"record target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path, tag: str, shm: CmpRecordShm | None = None) -> list[str]:
    """Run once; return the text stream's lines."""
    out = tmp_path / f"{tag}.cmplog"
    env = dict(os.environ, _CMPLOG_OUT=str(out))
    env.pop(CMPLOG_REC_ENV, None)
    if shm is not None:
        env[CMPLOG_REC_ENV] = shm.env_id
    run = subprocess.run(
        [*_NO_ASLR, str(exe), "0x1234", "MAGICVAX"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    return out.read_text().splitlines() if out.exists() else []


def _cmp_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("CMP ")]


@pytest.fixture
def shm():
    seg = CmpRecordShm()
    yield seg
    seg.close()


@requires_clang
class TestEquivalence:
    def test_control_text_matches_itself(self, rec_target, tmp_path):
        """Rule 46: two text runs agree, or the comparison below means nothing."""
        a = set(cond_tuples_from_cmplog_text(_run(rec_target, tmp_path, "t1")))
        b = set(cond_tuples_from_cmplog_text(_run(rec_target, tmp_path, "t2")))
        assert a == b and a

    def test_table_yields_text_set(self, rec_target, tmp_path, shm):
        text = set(cond_tuples_from_cmplog_text(_run(rec_target, tmp_path, "text")))
        _run(rec_target, tmp_path, "shm", shm)
        assert shm.attached
        assert set(shm.drain()) == text

    def test_no_cmp_lines_on_text_stream_when_attached(self, rec_target, tmp_path, shm):
        lines = _run(rec_target, tmp_path, "shm", shm)
        assert _cmp_lines(lines) == []

    def test_div_records_stay_on_text(self, rec_target, tmp_path, shm):
        lines = _run(rec_target, tmp_path, "shm", shm)
        assert any(ln.startswith("DIV ") for ln in lines)

    def test_drain_empties_table(self, rec_target, tmp_path, shm):
        _run(rec_target, tmp_path, "shm", shm)
        assert shm.drain()
        assert shm.drain() == []

    def test_loop_site_keeps_order(self, rec_target, tmp_path, shm):
        """Within one site, records come back in the order they were written."""
        _run(rec_target, tmp_path, "shm", shm)
        # clang passes the computed loop value as the first operand.
        loop = [
            int.from_bytes(a, "little")
            for a, b, w, r, pc in shm.drain()
            if pc is not None and w == 4 and 1000 <= int.from_bytes(a, "little") < 1000 + 7 * _LOOP
        ]
        assert loop == [1000 + 7 * i for i in range(_LOOP)]


@requires_clang
class TestBounds:
    def test_full_site_drops_and_counts(self, rec_target, tmp_path):
        seg = CmpRecordShm(int_hits=4)
        try:
            _run(rec_target, tmp_path, "cap", seg)
            conds = seg.drain()
            loop = [c for c in conds if c[2] == 4 and 1000 <= int.from_bytes(c[0], "little") < 1070]
            assert len(loop) == 4
            assert seg.take_dropped()[0] >= _LOOP - 4
        finally:
            seg.close()

    def test_adversarial_tiny_site_table_is_subset(self, rec_target, tmp_path):
        text = set(cond_tuples_from_cmplog_text(_run(rec_target, tmp_path, "text")))
        seg = CmpRecordShm(int_sites=2)
        try:
            _run(rec_target, tmp_path, "tiny", seg)
            got = set(seg.drain())
            assert got < text
            assert seg.take_dropped()[0] > 0
        finally:
            seg.close()

    def test_adversarial_bad_segment_id_falls_back_to_text(self, rec_target, tmp_path):
        out = tmp_path / "bad.cmplog"
        env = dict(os.environ, _CMPLOG_OUT=str(out), **{CMPLOG_REC_ENV: "999999999"})
        run = subprocess.run(
            [str(rec_target), "0x1234", "MAGICVAX"], env=env, capture_output=True, timeout=60
        )
        assert run.returncode == 0
        assert _cmp_lines(out.read_text().splitlines())

    def test_falsify_no_env_is_text_only(self, rec_target, tmp_path, shm):
        """Segment exists but is not offered: the shim never attaches."""
        lines = _run(rec_target, tmp_path, "plain")
        assert _cmp_lines(lines)
        assert not shm.attached
